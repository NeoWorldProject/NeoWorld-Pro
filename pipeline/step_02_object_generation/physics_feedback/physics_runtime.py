"""Actual PhysX trials, measurements, cooked proxies and rendered evidence.

Only initialization may teleport an articulation. All recorded motion is driven
by bounded effort; native zero-target viscous damping is saved in the asset.
"""
from __future__ import annotations

import json
import copy
import hashlib
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from .verify_physics_isaac import initial_report, sha256, source_hashes, write_report
from .control import BoundedJointExecutor, native_joint_axis_values, target_schedule, UNIFORM_PID_POLICY
from .trajectory_io import open_trajectory_writer
from urdf_core.native_progress import write_progress

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kinematics_validation"))
from verify_urdf_isaac import CaptureResources, capture_rgb, configure_fractional_opacity, tensor_transform, transform_points
from isaac_motion import camera_for_points, encode_video


DECOMPOSITION_ATTRIBUTES = {
    "hull_vertex_limit": "HullVertexLimit",
    "max_convex_hulls": "MaxConvexHulls",
    "voxel_resolution": "VoxelResolution",
    "error_percentage": "ErrorPercentage",
    "min_thickness_m": "MinThickness",
    "shrink_wrap": "ShrinkWrap",
}
TARGET_SPEED_FRACTION = 0.8


def native_decomposition_value(name, value):
    """Encode the logical minimum without rounding below the native bound.

    USD stores errorPercentage as float32, whereas native VHACD validates it
    against a double 0.01. Nearest rounding alone produces 0.009999999776...
    and rejects an otherwise valid profile. Move only this boundary encoding
    to the next float32; keep the requested logical profile unchanged.
    """
    if name == "error_percentage" and float(value) >= 0.01:
        encoded = np.float32(value)
        if float(encoded) < 0.01:
            return float(np.nextafter(encoded, np.float32(np.inf)))
    return value


def decomposition_settings(prim):
    """Return effective native settings, including schema defaults."""
    from pxr import PhysxSchema
    schema = PhysxSchema.PhysxConvexDecompositionCollisionAPI(prim)
    values = {name: getattr(schema, "Get" + attribute + "Attr")().Get()
              for name, attribute in DECOMPOSITION_ATTRIBUTES.items()}
    if any(value is None for value in values.values()):
        raise RuntimeError(f"Convex decomposition settings cannot be observed: {prim.GetPath()}")
    return values


def triangulate(faces):
    return np.asarray([[face[0], face[i], face[i + 1]] for face in faces for i in range(1, len(face) - 1)], dtype=int)


def load_geometry(manifest, asset_dir, key="geometry_meshes"):
    from urdf_core.urdf_validation import read_obj
    meshes = {}
    for item in manifest[key]:
        path = Path(item["path"])
        if not path.is_absolute():
            path = asset_dir / path
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Geometry mesh hash mismatch: {path}")
        mesh = read_obj(path)
        meshes.setdefault(item["link"], []).append({"vertices": mesh["vertices"], "faces": triangulate(mesh["faces"]),
                                                   "name": item.get("object", path.stem)})
    if not meshes:
        raise ValueError("No original geometry for independent observation")
    return meshes


class Contacts:
    def __init__(self):
        from omni.physx import get_physx_simulation_interface
        self.pairs, self.active_pairs, self.error = [], {}, None
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._event)

    def _event(self, headers, data):
        from pxr import PhysicsSchemaTools
        try:
            for header in headers:
                pairs = {name: str(PhysicsSchemaTools.intToSdfPath(getattr(header, name)))
                         for name in ("actor0", "actor1", "collider0", "collider1")}
                pairs.update(body0=pairs["actor0"], body1=pairs["actor1"])
                points = []
                for index in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                    point = data[index]
                    points.append({"position_m": list(point.position), "normal": list(point.normal),
                                   "impulse_Ns": list(point.impulse), "separation_m": float(point.separation)})
                event = dict(pairs, event_type=str(header.type), points=points)
                self.pairs.append(event)
                key = (pairs["collider0"], pairs["collider1"])
                if "lost" in str(header.type).lower():
                    self.active_pairs.pop(key, None)
                elif points:
                    self.active_pairs[key] = event
        except Exception as exc:
            self.error = str(exc)

    def take(self):
        if self.error:
            raise RuntimeError("Contact observation failed: " + self.error)
        pairs, self.pairs = self.pairs, []
        return {"observed": self.subscription is not None, "pairs": list(self.active_pairs.values()),
                "events": pairs, "state_semantics": "found/persist until PhysX lost event, including sleeping contacts"}

    def close(self):
        self.subscription = None


# This implementation version uses PGS for the entire object test, including
# delivery reload. Controlled native comparisons showed TGS position drift
# despite adequate static friction and near-zero reported joint velocities.
# Scene solver choice is code-owned, never an API repair parameter. Historical
# assets keep their own frozen runtime and required-scene metadata.
SCENE_SOLVER_TYPE = "PGS"
EXTERNAL_FORCES_EVERY_ITERATION = False


def scene_solver_policy():
    return {"solver_type": SCENE_SOLVER_TYPE,
            "enable_external_forces_every_iteration": EXTERNAL_FORCES_EVERY_ITERATION}


def configure_scene_solver(physx):
    policy = scene_solver_policy()
    physx.CreateSolverTypeAttr(policy['solver_type'])
    physx.CreateEnableExternalForcesEveryIterationAttr(policy['enable_external_forces_every_iteration'])
    if (physx.GetSolverTypeAttr().Get() != policy['solver_type']
            or physx.GetEnableExternalForcesEveryIterationAttr().Get() is not policy['enable_external_forces_every_iteration']):
        raise RuntimeError('Native scene attributes differ from the supported solver policy')
    return policy


def author_native_joint_friction(prim, parameters, kind):
    """Isaac 5.1 codeless multiple-apply schema; exported and verified on reload."""
    if not prim.ApplyAPI("PhysxJointAxisAPI", kind):
        raise RuntimeError("Native PhysxJointAxisAPI unavailable")
    expected = native_joint_axis_values(parameters, kind)
    for key, value in expected.items():
        attribute = prim.GetAttribute(f"physxJointAxis:{kind}:{key}")
        if not attribute or not attribute.Set(value):
            raise RuntimeError("Cannot author native joint friction: " + key)
    return expected


def verify_native_joint_friction(prim, parameters, kind):
    expected = native_joint_axis_values(parameters, kind)
    observed = {}
    for key, value in expected.items():
        attribute = prim.GetAttribute(f"physxJointAxis:{kind}:{key}")
        actual = attribute.Get() if attribute else None
        if actual is None or not math.isclose(float(actual), value, rel_tol=1e-5, abs_tol=1e-8):
            raise RuntimeError("Native joint friction differs from declared SI profile: " + key)
        observed[key] = actual
    return observed


def configure_asset(stage, root_path, spec, profile, manifest, contract=None):
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade
    while True:
        instances = [prim for prim in stage.Traverse() if prim.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    # The importer keeps mesh prototypes as active top-level scopes. Flatten
    # internal references before removing them so they cannot become phantom
    # static colliders outside the actual robot.
    stage.GetRootLayer().ImportFromString(stage.Flatten().ExportToString())
    asset_top = root_path.strip("/").split("/")[0]
    for prim in list(stage.GetPseudoRoot().GetChildren()):
        if prim.GetName() in {"colliders", "visuals", "meshes"} and prim.GetName() != asset_top:
            stage.RemovePrim(prim.GetPath())
    # OBJ's intermediate World Xform may receive CollisionAPI instead of its
    # Mesh. Author explicit per-mesh collision APIs; those are also queryable by
    # the PhysX cooked-representation interface.
    from pxr import Usd
    inherited = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI) and not p.IsA(UsdGeom.Mesh)]
    for parent in inherited:
        for prim in Usd.PrimRange(parent):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim)
        parent.RemoveAPI(UsdPhysics.CollisionAPI)
        parent.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        parent.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
    if not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse()):
        candidates = [p for p in stage.Traverse() if p.GetName() == spec["root"] and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(candidates) != 1:
            raise RuntimeError("Cannot identify the native rigid root")
        UsdPhysics.ArticulationRootAPI.Apply(candidates[0])
    adaptive = profile["collision"]["approximation"] in {"adaptive", "coacd"}
    if adaptive:
        # Reconstruct the declared proxies in link-local frames. Do not rely on
        # importer defaults, mesh naming, material splitting, or traversal order.
        for prim in stage.Traverse():
            for schema in (UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI, PhysxSchema.PhysxCollisionAPI):
                if prim.HasAPI(schema):
                    prim.RemoveAPI(schema)
    bodies, joints, collisions = {}, {}, []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if prim.GetName() in manifest["link_properties"]:
                bodies[prim.GetName()] = prim
                rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rigid.CreateDisableGravityAttr(False)
                rigid.CreateEnableCCDAttr(True)
                # Avoid undocumented simulator damping being mistaken for material damping.
                rigid.CreateLinearDampingAttr(0.0)
                rigid.CreateAngularDampingAttr(0.0)
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
                material = UsdShade.Material.Define(stage, str(prim.GetPath()) + "/ContactMaterial")
                physmat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
                parameters = profile["links"][prim.GetName()]["material"]
                physmat.CreateStaticFrictionAttr(parameters["static_friction"])
                physmat.CreateDynamicFrictionAttr(parameters["dynamic_friction"])
                physmat.CreateRestitutionAttr(parameters["restitution"])
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True)
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition")
                decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                for name, value in profile.get("collision", {}).get("decomposition", {}).items():
                    getattr(decomposition, "Create" + DECOMPOSITION_ATTRIBUTES[name] + "Attr")(
                        native_decomposition_value(name, value))
                collisions.append(prim)
            physcol = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physcol.CreateRestOffsetAttr(0.0)
            physcol.CreateContactOffsetAttr(0.001)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            art.CreateEnabledSelfCollisionsAttr(False)
            art.CreateSolverPositionIterationCountAttr(32)
            art.CreateSolverVelocityIterationCountAttr(8)
        if prim.IsA(UsdPhysics.Joint) and prim.GetName() in profile["joints"]:
            joints[prim.GetName()] = prim
    expected = {item["name"] for item in spec["links"]}
    if set(bodies) != expected:
        from urdf_core.content_errors import AssetContentError
        raise AssetContentError(f"Imported rigid links differ from spec: {set(bodies)} != {expected}", stage='physical_asset_import')
    expected_joints = {joint['name'] for joint in spec['joints'] if joint['type'] != 'fixed'}
    if not expected_joints <= joints.keys():
        from urdf_core.content_errors import AssetContentError
        raise AssetContentError(f'Missing imported joints: {sorted(expected_joints-joints.keys())}', stage='physical_asset_import')
    # Only intra-object rigid bodies are filtered. Ground and external objects
    # never appear in these relationships and keep normal native contacts.
    for name, body in bodies.items():
        UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel().SetTargets(
            [other.GetPath() for other_name, other in bodies.items() if other_name != name])
    if adaptive:
        collisions = author_collision_proxies(stage, bodies, profile, manifest)
    for joint in spec["joints"]:
        if joint["type"] == "fixed":
            continue
        prim = joints[joint["name"]]
        kind = "linear" if joint["type"] == "prismatic" else "angular"
        drive = UsdPhysics.DriveAPI.Apply(prim, kind)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        # Passive resistance belongs to native JointAxis, not a PD drive.
        drive.CreateDampingAttr(0.0)
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateMaxForceAttr(0.0)
        PhysxSchema.PhysxJointAPI.Apply(prim).CreateJointFrictionAttr(0.0)
        author_native_joint_friction(prim, profile["joints"][joint["name"]], kind)
        q0 = (contract or {}).get("joints", {}).get(joint["name"], {}).get("initial_position", 0.0)
        state = PhysxSchema.JointStateAPI.Apply(prim, kind)
        state.CreatePositionAttr(q0 if kind == "linear" else math.degrees(q0))
        state.CreateVelocityAttr(0.0)
    if not collisions:
        from urdf_core.content_errors import AssetContentError
        raise AssetContentError("No imported collision meshes", stage='physical_asset_import')
    return bodies, collisions


def author_collision_proxies(stage, bodies, profile, manifest):
    """Author each certified primitive/mesh, preserving native articulation."""
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    from urdf_core.urdf_validation import read_obj
    collisions = []
    for index, item in enumerate(manifest["collision_proxies"]):
        path = str(bodies[item["link"]].GetPath()) + f"/PhysicsColliders/part_{index:04d}"
        kind = item["representation"]
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("Collision proxy hash mismatch: " + item["path"])
        if kind == "box":
            box = item["box"]
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(2.0)
            geom.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*box["center_m"]))
            q = Rotation.from_euler("xyz", box["rpy_rad"]).as_quat()
            geom.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(q[3]), Gf.Vec3d(*q[:3])))
            geom.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(np.asarray(box["size_m"]) / 2)))
        elif kind in {"convexHull", "convexDecomposition"}:
            mesh = read_obj(Path(item["path"]))
            geom = UsdGeom.Mesh.Define(stage, path)
            geom.CreatePointsAttr(mesh["vertices"].tolist())
            geom.CreateFaceVertexCountsAttr([len(f) for f in mesh["faces"]])
            geom.CreateFaceVertexIndicesAttr(np.asarray(mesh["faces"]).ravel().tolist())
            UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr(kind)
            if kind == "convexHull":
                schema = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(geom.GetPrim())
                schema.CreateHullVertexLimitAttr(32)
                schema.CreateMinThicknessAttr(0.0)
            else:
                schema = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(geom.GetPrim())
                for name, value in profile["collision"].get("decomposition", {}).items():
                    getattr(schema, "Create" + DECOMPOSITION_ATTRIBUTES[name] + "Attr")(native_decomposition_value(name, value))
        else:
            raise ValueError("Unsupported declared collision representation: " + kind)
        prim = geom.GetPrim()
        geom.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        schema = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        schema.CreateRestOffsetAttr(0.0)
        schema.CreateContactOffsetAttr(0.001)
        prim.CreateAttribute("neoworld:collisionRepresentation", Sdf.ValueTypeNames.Token).Set(kind)
        prim.CreateAttribute("neoworld:sourceObject", Sdf.ValueTypeNames.String).Set(item["object"])
        collisions.append(prim)
    return collisions


def cooked_geometry(stage, bodies, collisions):
    """Read the actual PhysX decomposition, never substitute the source mesh."""
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysxSchema, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    output = {name: [] for name in bodies}
    evidence = []
    for prim in collisions:
        parent = prim
        while parent and parent.GetName() not in bodies:
            parent = parent.GetParent()
        if not parent:
            raise RuntimeError("Collision prim does not belong to a link")
        name = parent.GetName()
        # Work in the owning link frame directly. Inverting world transforms
        # needlessly made identical local colliders depend on root placement.
        relative, reset_stack = cache.ComputeRelativeTransform(prim, parent)
        if reset_stack:
            raise RuntimeError('Collision transform resets before its owning link')
        mesh_to_link = np.asarray(relative).T
        if prim.IsA(UsdGeom.Cube):
            # Native boxes have an exact analytic representation, no mesh cook.
            size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
            vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size / 2
            faces = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                              [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
            output[name].append({"vertices": transform_points(mesh_to_link, vertices), "faces": faces,
                                 "name": str(prim.GetPath()) + "#native_box",
                                 "source_object": prim.GetAttribute("neoworld:sourceObject").Get() or str(prim.GetPath())})
            evidence.append({"link": name, "prim_path": str(prim.GetPath()), "representation": "box",
                             "observation": "native_analytic_box", "size": size,
                             "mesh_to_link": mesh_to_link.tolist()})
            continue
        kind = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        if kind not in {"convexHull", "convexDecomposition"}:
            raise RuntimeError("Cannot observe this collision representation: " + str(kind))
        received = []
        cooking_results = []
        def on_result(result, hulls):
            cooking_results.append(result)
            if result != PhysxCollisionRepresentationResult.RESULT_VALID:
                return
            for index, hull in enumerate(hulls):
                faces = []
                for polygon in hull.polygons:
                    ids = hull.indices[polygon.index_base:polygon.index_base + polygon.num_vertices]
                    faces.extend([[ids[0], ids[i], ids[i + 1]] for i in range(1, len(ids) - 1)])
                points = np.array([list(vertex) for vertex in hull.vertices])
                received.append({"vertices": transform_points(mesh_to_link, points), "faces": np.asarray(faces),
                                 "name": str(prim.GetPath()) + f"#hull{index}",
                                 "source_object": prim.GetAttribute("neoworld:sourceObject").Get() or str(prim.GetPath())})
        get_physx_cooking_interface().request_convex_collision_representation(
            stage_id=stage_id, collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
            run_asynchronously=False, on_result=on_result)
        from urdf_core.content_errors import require_cooked_geometry
        require_cooked_geometry(cooking_results, len(received), link=name,
            part=prim.GetAttribute('neoworld:sourceObject').Get() or str(prim.GetPath()))
        output[name].extend(received)
        settings = ({"native_decomposition": decomposition_settings(prim)} if kind == "convexDecomposition" else
                    {"native_hull": {"hull_vertex_limit": PhysxSchema.PhysxConvexHullCollisionAPI(prim).GetHullVertexLimitAttr().Get(),
                                     "min_thickness_m": PhysxSchema.PhysxConvexHullCollisionAPI(prim).GetMinThicknessAttr().Get()}})
        evidence.append({"link": name, "prim_path": str(prim.GetPath()), "hull_count": len(received),
                         "representation": kind, **settings})
    if not any(output.values()):
        raise RuntimeError("Asset has no observable external collision geometry")
    return output, evidence


class Runtime:
    def __init__(self, args, report, spec, profile, contract, manifest, original):
        self.args, self.report, self.spec, self.profile = args, report, spec, profile
        self.contract, self.config, self.manifest, self.original = contract, contract["config"], manifest, original
        self.sim = None
        self.capture = CaptureResources()
        self.contacts = None
        self.delivered = args.out_dir / "object.usd"
        self.delivered_written = False
        self.declared_colliders = load_geometry(manifest, args.asset_dir, "collision_meshes")
        self.initial_positions = {name: value["initial_position"] for name, value in contract["joints"].items()}

    def close_trial(self):
        if self.contacts:
            self.contacts.close()
            self.contacts = None
        self.capture.close()
        if self.sim:
            self.sim.stop()
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
            self.sim = None

    def create_trial(self, *, fixed, reload=False, support_pose=None, joint_name=None):
        import omni.kit.commands
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.stage import create_new_stage
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
        from urdf_core.urdf_validation import read_urdf, urdf_fk
        self.close_trial()
        if reload:
            if not omni.usd.get_context().open_stage(str(self.delivered)):
                raise RuntimeError("Cannot reopen delivered USD")
            stage = omni.usd.get_context().get_stage()
            roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
            if len(roots) != 1:
                raise RuntimeError("Delivered USD lacks one native articulation")
            root_path = roots[0]
            # Inspect the saved state before adding any testing environment.
            for prim in stage.Traverse():
                if prim.IsA(UsdPhysics.FixedJoint):
                    joint = UsdPhysics.Joint(prim)
                    if not joint.GetBody0Rel().GetTargets() or not joint.GetBody1Rel().GetTargets():
                        raise RuntimeError("Delivered USD contains a fixed-to-world fixture")
                if prim.HasAPI(UsdPhysics.CollisionAPI) and not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                    raise RuntimeError("Delivered USD has disabled collision")
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) and PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get():
                    raise RuntimeError("Delivered USD disables gravity")
            bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
            collisions = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI) and (p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Cube))]
        else:
            create_new_stage()
            stage = omni.usd.get_context().get_stage()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            ok, config = omni.kit.commands.execute("URDFCreateImportConfig")
            if not ok:
                raise RuntimeError("URDFCreateImportConfig failed")
            config.merge_fixed_joints = False
            config.fix_base = fixed
            config.import_inertia_tensor = True
            config.distance_scale = 1.0
            config.collision_from_visuals = False
            config.self_collision = False
            ok, root_path = omni.kit.commands.execute("URDFParseAndImportFile",
                urdf_path=str(self.args.asset_dir / "object.urdf"), import_config=config, get_articulation_root=True)
            if not ok or not root_path:
                raise RuntimeError("URDF import failed")
            bodies, collisions = configure_asset(stage, root_path, self.spec, self.profile, self.manifest, self.contract)
            root_path = next(str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI))
            if not fixed and not self.delivered_written:
                top = stage.GetPrimAtPath(root_path)
                while top.GetParent() and not top.GetParent().IsPseudoRoot():
                    top = top.GetParent()
                stage.SetDefaultPrim(top)
                # No ground, physicsScene, camera, light, or fixture has been added.
                # Strip the viewport only in a detached export stage. Deleting
                # live RenderProducts invalidates Hydra on later trials.
                asset_stage = Usd.Stage.Open(stage.Flatten())
                for prim in list(asset_stage.GetPseudoRoot().GetChildren()):
                    if prim.GetName() != top.GetName():
                        asset_stage.RemovePrim(prim.GetPath())
                # Standalone assets require a host scene with this solver policy.
                # Keep test ground/fixtures out of the reusable object USD.
                asset_stage.GetDefaultPrim().SetCustomDataByKey("neoworld:requiredSceneSolver", scene_solver_policy())
                asset_stage.GetRootLayer().Export(str(self.delivered))
                self.delivered_written = True
                self.report.update(delivered_usd_path=str(self.delivered), delivered_usd_sha256=sha256(self.delivered))
        self.stage, self.root_path, self.bodies = stage, root_path, bodies
        self.parsed = read_urdf(self.args.asset_dir / "object.urdf")
        self.measured_fk_model = copy.deepcopy(self.parsed)
        # FK must still be measurable when the solver slightly exceeds a stop.
        # Evaluate raw measured q here; the separate frozen-limit metric rejects
        # excessive overshoot. Do not convert a physics rejection into an error.
        for joint in self.measured_fk_model["joints"]:
            if joint["type"] in {"revolute", "prismatic"}:
                joint["limit"].update(lower=-float("inf"), upper=float("inf"))
        self.root_name = self.parsed["root"]
        self.support_pose = support_pose
        self.reset_positions = dict(self.initial_positions)
        self.initial_orientation = np.array([1., 0., 0., 0.])
        self.initial_root_transform = np.eye(4)
        if support_pose is not None:
            from .support_rest_pose import support_world_matrix, validate_support_rest_pose
            from scipy.spatial.transform import Rotation
            validate_support_rest_pose(support_pose, self.contract)
            self.reset_positions = dict(support_pose["joint_positions"])
            self.initial_root_transform = support_world_matrix(support_pose)
            q = Rotation.from_matrix(self.initial_root_transform[:3, :3]).as_quat()
            self.initial_orientation = q[[3, 0, 1, 2]]
        # Saved support states have already passed the frozen q_initial check.
        # Preserve native coordinates at a stop instead of clipping roundoff.
        rest = urdf_fk(self.measured_fk_model if support_pose is not None else self.parsed,
                       self.reset_positions)
        all_points = np.concatenate([transform_points(rest[name], mesh["vertices"])
                                     for name, meshes in self.original.items() for mesh in meshes])
        declared_points = np.concatenate([transform_points(rest[name], mesh["vertices"])
                                          for name, meshes in self.declared_colliders.items() for mesh in meshes])
        lift = -float(declared_points[:, 2].min())
        self.initial_root = self.initial_root_transform[:3, 3].copy() if support_pose is not None else np.array([0.0, 0.0, lift])
        placement = self.initial_root_transform.copy()
        placement[:3, 3] = self.initial_root
        self.initial_link_poses = {name: (placement @ pose).tolist() for name, pose in rest.items()}
        # Apply initial placement before physics creation, including the fixed fixture.
        top = stage.GetPrimAtPath(root_path)
        while top.GetParent() and not top.GetParent().IsPseudoRoot():
            top = top.GetParent()
        xform = UsdGeom.Xformable(top)
        placement_op = xform.AddTransformOp(opSuffix="physicsTestPlacement")
        placement_op.Set(Gf.Matrix4d(placement.T.tolist()))
        if reload and fixed:
            fixture = UsdPhysics.FixedJoint.Define(stage, "/PhysicsTestFixedFixture")
            fixture.CreateBody1Rel().SetTargets([bodies[self.root_name].GetPath()])
            fixture.CreateLocalPos0Attr(Gf.Vec3f(*self.initial_root))
            fixture.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            # PhysX fixed articulations are rooted at their world fixture.
            # This lives only in the reload test stage, never in object.usd.
            for prim in stage.Traverse():
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                    prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            UsdPhysics.ArticulationRootAPI.Apply(fixture.GetPrim())
            art_schema = PhysxSchema.PhysxArticulationAPI.Apply(fixture.GetPrim())
            art_schema.CreateEnabledSelfCollisionsAttr(False)
            art_schema.CreateSolverPositionIterationCountAttr(32)
            art_schema.CreateSolverVelocityIterationCountAttr(8)
            root_path = str(fixture.GetPath())
            self.root_path = root_path
        scene = UsdPhysics.Scene.Define(stage, "/PhysicsTestScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(self.config["gravity_m_s2"])
        physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        solver_policy = configure_scene_solver(physx)
        physx.CreateEnableGPUDynamicsAttr(False)
        physx.CreateBroadphaseTypeAttr("MBP")
        physx.CreateEnableCCDAttr(True)
        # physicsUtils.add_ground_plane prepends the default prim even when an
        # absolute path is passed. Define world geometry explicitly so ground
        # never inherits the moving object's initial placement.
        size = max(self.contract["diagonal_m"] * 20, 10)
        ground = UsdGeom.Mesh.Define(stage, "/PhysicsTestGround")
        ground.CreatePointsAttr([Gf.Vec3f(-size, -size, 0), Gf.Vec3f(size, -size, 0),
                                Gf.Vec3f(size, size, 0), Gf.Vec3f(-size, size, 0)])
        ground.CreateFaceVertexCountsAttr([4])
        ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        ground.CreateDisplayColorAttr([Gf.Vec3f(.22, .24, .27)])
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        ground_material = UsdShade.Material.Define(stage, "/PhysicsTestGroundMaterial")
        ground_physics = UsdPhysics.MaterialAPI.Apply(ground_material.GetPrim())
        ground_physics.CreateStaticFrictionAttr(.5)
        ground_physics.CreateDynamicFrictionAttr(.5)
        ground_physics.CreateRestitutionAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_material, materialPurpose="physics")
        UsdLux.DomeLight.Define(stage, "/PhysicsTestLight").CreateIntensityAttr(1000.0)
        cooked, cooking_evidence = cooked_geometry(stage, bodies, collisions)
        if self.profile["collision"]["approximation"] in {"adaptive", "coacd"}:
            declared = self.manifest["collision_proxies"]
            if len(cooking_evidence) != len(declared):
                raise RuntimeError("Active collider count differs from the declared adaptive plan")
            observed = {item["prim_path"].rsplit("/", 1)[-1]: item for item in cooking_evidence}
            for index, item in enumerate(declared):
                actual = observed.get(f"part_{index:04d}")
                if not actual or actual["link"] != item["link"] or actual["representation"] != item["representation"]:
                    raise RuntimeError("Active collision representation differs from the declared adaptive plan")
                if item["representation"] == "convexHull" and actual["native_hull"] != {"hull_vertex_limit": 32, "min_thickness_m": 0.0}:
                    raise RuntimeError("Native hull settings changed on import/reload")
        for evidence in cooking_evidence:
            if evidence["representation"] != "convexDecomposition":
                continue
            for name, expected in self.profile.get("collision", {}).get("decomposition", {}).items():
                actual = evidence["native_decomposition"][name]
                if (actual != expected if isinstance(expected, (bool, int)) else
                        not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-10)):
                    raise RuntimeError(f"Native decomposition differs from profile: {evidence['prim_path']} "
                                       f"{name}: declared={expected}, actual={actual}")
        self.cooked = cooked
        # Native cooking can move an extreme vertex slightly. Ground placement
        # uses the actual cooked support surface; no visual/source fidelity gate.
        native_points = np.concatenate([transform_points(rest[name], mesh["vertices"])
                                        for name, meshes in cooked.items() for mesh in meshes])
        if support_pose is None:
            self.initial_root[2] = -float(native_points[:, 2].min())
        placement[:3, 3] = self.initial_root
        self.motion_scope_plan = None
        self.motion_scope_geometry_binding = None
        self.fixture_clearance_m = None
        if self.contract.get("motion_scope_policy") and joint_name is not None:
            from .motion_scope import plan_joint_motion
            definition = self.contract["joints"][joint_name]
            plan = plan_joint_motion(self.parsed, cooked, self.initial_positions,
                                     placement, joint_name, definition["lower"], definition["upper"])
            self.motion_scope_plan = plan
            geometry_path = self.args.out_dir / "motion_scopes" / (("fixed_" if fixed else "free_") + joint_name + ".json")
            geometry_record = {name: [{key: (value.tolist() if isinstance(value, np.ndarray) else value)
                                      for key, value in mesh.items()} for mesh in meshes]
                               for name, meshes in cooked.items()}
            write_report(geometry_path, geometry_record)
            self.motion_scope_geometry_binding = {"path": str(geometry_path), "sha256": sha256(geometry_path)}
            # This only positions a fixed testing fixture before initialization.
            # Source geometry, joint limits, prescribed q0 and free-root assets
            # are unchanged. The clearance is not a penetration acceptance gate.
            if fixed:
                self.fixture_clearance_m = max(.005, .01 * self.contract["diagonal_m"])
                self.initial_root[2] += max(0., self.fixture_clearance_m - plan["fixed_min_z_m"])
                placement[:3, 3] = self.initial_root
        placement_op.Set(Gf.Matrix4d(placement.T.tolist()))
        if reload and fixed:
            fixture.CreateLocalPos0Attr(Gf.Vec3f(*self.initial_root))
        self.initial_root_transform = placement.copy()
        self.initial_link_poses = {name: (placement @ pose).tolist() for name, pose in rest.items()}
        digest = hashlib.sha256()
        for name, meshes in sorted(cooked.items()):
            digest.update(name.encode())
            for mesh in meshes:
                digest.update(np.asarray(mesh["vertices"], dtype=np.float64).tobytes())
                digest.update(np.asarray(mesh["faces"], dtype=np.int64).tobytes())
        cooked_hash = digest.hexdigest()
        self.report["active_cooked_geometry"] = {"sha256": cooked_hash,
            "components_per_link": {name: len(meshes) for name, meshes in cooked.items()}}
        self.report["collision_observation_policy"] = {
            "same_object_self_collision": False, "offline_internal_intersection": "not_evaluated",
            "proxy_fidelity": "not_an_acceptance_gate", "ground_and_external_contacts": True, "ground_geometry": "native_cooked_collision"}
        native_friction = {}
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                if PhysxSchema.PhysxArticulationAPI(prim).GetEnabledSelfCollisionsAttr().Get() is not False:
                    raise RuntimeError("Delivered asset enables same-object self collision")
            if prim.IsA(UsdPhysics.Joint) and prim.GetName() in self.profile["joints"]:
                kind = "linear" if self.contract["joints"][prim.GetName()]["type"] == "prismatic" else "angular"
                native_friction[prim.GetName()] = verify_native_joint_friction(prim, self.profile["joints"][prim.GetName()], kind)
        if set(native_friction) != set(self.contract["joints"]):
            raise RuntimeError("Missing native joint friction parameters")
        for name, body in bodies.items():
            filtered = set(map(str, UsdPhysics.FilteredPairsAPI(body).GetFilteredPairsRel().GetTargets()))
            expected = {str(other.GetPath()) for other_name, other in bodies.items() if other_name != name}
            if filtered != expected:
                raise RuntimeError("Intra-object collision filters differ on import/reload")
        dt = self.config["physics_dt_s"]
        self.sim = SimulationContext(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0,
                                    physics_prim_path="/PhysicsTestScene", set_defaults=False)
        self.sim.initialize_physics()
        if self.sim.get_physics_context().get_solver_type() != SCENE_SOLVER_TYPE:
            raise RuntimeError("Initialized PhysX solver differs from the frozen engine policy")
        self.contacts = Contacts()
        self.sim.play()
        self.art = Articulation(str(root_path), name="physics_object")
        self.art.initialize()
        if not self.art.is_physics_handle_valid():
            raise RuntimeError("Imported articulation has no valid physics handle")
        actual_fixed = bool(self.art._physics_view.shared_metatype.fixed_base)
        self.fixed_trial = actual_fixed
        if actual_fixed != fixed:
            raise RuntimeError(f"PhysX root mode differs from requested trial: fixed={actual_fixed}, expected={fixed}")
        if not fixed:
            # initialize_physics may internally advance startup frames. Restore
            # the specified release state once, before the observed trial.
            self.art.set_world_poses(positions=self.initial_root[None, :], orientations=self.initial_orientation[None, :])
            self.art.set_velocities(np.zeros((1, 6), dtype=np.float32))
        self.dofs, self.links = list(self.art.dof_names), list(self.art.body_names)
        from .measurement_plan import MeasurementPlan
        self.measurement_plan = MeasurementPlan(self.measured_fk_model, self.links, self.root_name, self.cooked)
        if set(self.dofs) != set(self.contract["joints"]) or set(self.links) != set(bodies):
            raise RuntimeError("Physics tensors do not cover exactly the required links and joints")
        measured_properties = {}
        masses = np.asarray(self.art._physics_view.get_masses(), dtype=float)[0]
        coms = np.asarray(self.art._physics_view.get_coms(), dtype=float)[0]
        inertias = np.asarray(self.art._physics_view.get_inertias(), dtype=float)[0]
        for index, name in enumerate(self.links):
            expected = self.manifest["link_properties"][name]
            inertia = inertias[index].reshape(3, 3).T
            if not np.isclose(masses[index], expected["mass_kg"], rtol=1e-4, atol=1e-7):
                raise RuntimeError(f"PhysX changed the declared link mass: {name}")
            if "center_of_mass_m" in expected and not np.allclose(coms[index][:3], expected["center_of_mass_m"], rtol=1e-4, atol=1e-6):
                raise RuntimeError(f"PhysX changed the declared center of mass: {name}")
            inertia_error = None
            if "inertia_kg_m2" in expected:
                expected_inertia = np.asarray(expected["inertia_kg_m2"])
                inertia_error = float(np.linalg.norm(inertia - expected_inertia))
                # Principal-axis quaternion roundoff can dominate a nearly
                # zero off-diagonal element. Compare tensor error to the whole
                # tensor scale, not elementwise relative errors at zero.
                allowed_error = max(1e-9, float(np.linalg.norm(expected_inertia)) * 2e-5)
                if inertia_error > allowed_error:
                    raise RuntimeError(f"PhysX changed inertia {name}: expected={expected_inertia.tolist()}, "
                                       f"actual={inertia.tolist()}, error={inertia_error}, allowed={allowed_error}")
            measured_properties[name] = {"mass_kg": float(masses[index]), "center_of_mass_m": coms[index][:3].tolist(),
                                         "inertia_kg_m2": inertia.tolist(), "inertia_tensor_error_kg_m2": inertia_error}
        if self.dofs:
            zeros = np.zeros((1, len(self.dofs)), dtype=np.float32)
            # The only explicit state reset: before the first recorded physics step.
            self.art.set_joint_positions(np.asarray([[self.reset_positions[name] for name in self.dofs]], dtype=np.float32))
            self.art.set_joint_velocities(zeros)
            gains = self.art.get_gains()
            if not np.allclose(np.asarray(gains[0]), 0, atol=1e-8) or not np.allclose(np.asarray(gains[1]), 0, atol=1e-8):
                raise RuntimeError("Native drive must have zero stiffness and damping")
            observed_friction = np.asarray(self.art._physics_view.get_dof_friction_properties(), dtype=float)[0]
            expected_friction = np.asarray([[self.profile["joints"][name][key] for key in
                ("static_friction_effort", "dynamic_friction_effort", "damping")] for name in self.dofs])
            if not np.allclose(observed_friction, expected_friction, rtol=1e-4, atol=1e-7):
                raise RuntimeError(f"Native tensor friction differs from SI profile: {observed_friction.tolist()}")
            self.art.set_joint_efforts(zeros)
        self.skip_ground_acceptance = False
        self.previous_poses = None
        self.q_previous, self.q_unwrapped = {}, {}
        self.camera_points = transform_points(placement, all_points)
        self.contacts.take()  # discard startup contact events before the trial
        self.contacts.active_pairs.clear()
        from .control_diagnostics import observe_control_evidence
        control_evidence = observe_control_evidence(self.art, self.dofs, self.contract)
        from urdf_core.isaac_admission import initialized
        initialized()
        return {"fixed_root": actual_fixed, "reloaded": reload, "cooked_collisions": cooking_evidence,
                **({"motion_scope_plan": self.motion_scope_plan,
                    "motion_scope_geometry": self.motion_scope_geometry_binding,
                    "fixture_clearance_m": self.fixture_clearance_m}
                   if self.motion_scope_plan is not None else {}),
                "measured_properties": measured_properties,
                "control_discrete_linearization": control_evidence,
                "cooked_geometry_sha256": cooked_hash, "native_joint_friction": native_friction,
                "native_articulation": True, "gravity_enabled": True, "self_collision_enabled": False,
                "internal_geometry_checks": False, "proxy_fidelity_acceptance": False,
                "ground_material": {"static_friction": .5, "dynamic_friction": .5, "restitution": 0.0},
                "ground_placement_geometry": "recorded_support_rest_pose_without_regrounding" if support_pose is not None else "native_cooked_collision",
                "solver_policy": {**solver_policy,
                                  "position_iterations": PhysxSchema.PhysxArticulationAPI(stage.GetPrimAtPath(root_path)).GetSolverPositionIterationCountAttr().Get(),
                                  "velocity_iterations": PhysxSchema.PhysxArticulationAPI(stage.GetPrimAtPath(root_path)).GetSolverVelocityIterationCountAttr().Get()},
                "joint_position_writes": "initialization_only", "active_drive_stiffness": 0.0,
                "target_speed_fraction": {name: self.contract["joints"][name]["speed_fraction"] for name in self.dofs},
                "target_speed_limits": {name: self.contract["joints"][name]["max_velocity"] * self.contract["joints"][name]["speed_fraction"]
                                        for name in self.dofs},
                "passive_damping": {name: self.profile["joints"][name]["damping"] for name in self.dofs}}

    def measure(self, t, efforts, targets):
        from urdf_core.urdf_validation import urdf_fk
        transforms = np.asarray(self.art._physics_view.get_link_transforms(), dtype=float)[0]
        optimized = getattr(self.args, 'measurement_mode', 'optimized') == 'optimized'
        poses = None if optimized else {name: tensor_transform(value) for name, value in zip(self.links, transforms)}
        link_velocities = np.asarray(self.art._physics_view.get_link_velocities(), dtype=float)[0]
        if link_velocities.shape != (len(self.links), 6) or not np.isfinite(link_velocities).all():
            raise RuntimeError("Missing/nonfinite native rigid-link velocity")
        values = np.asarray(self.art.get_joint_positions()).reshape(-1) if self.dofs else np.array([])
        velocities = np.asarray(self.art.get_joint_velocities()).reshape(-1) if self.dofs else np.array([])
        positions = {}
        for name, value in zip(self.dofs, values):
            value = float(value)
            if self.contract["joints"][name]["type"] == "continuous" and name in self.q_previous:
                delta = (value - self.q_previous[name] + math.pi) % (2 * math.pi) - math.pi
                self.q_unwrapped[name] += delta
            else:
                self.q_unwrapped[name] = value
            positions[name], self.q_previous[name] = self.q_unwrapped[name], value
        geometry = {"passed": True, "complete": True, "status": "passed", "issues": [], "diagnostics": [],
                    "original": {"contacts": [], "policy": "internal_not_evaluated"},
                    "cooked_proxies": {"contacts": [], "policy": "internal_not_evaluated"}}
        if optimized:
            poses, bottom, structure_error, rotation_error = self.measurement_plan.geometry(transforms, positions)
        else:
            bottom = min(float(transform_points(poses[name], mesh["vertices"])[:, 2].min())
                         for name, meshes in self.cooked.items() for mesh in meshes)
        self.previous_poses = poses
        if bottom < 0:
            geometry["diagnostics"].append({"type": "ground_penetration", "depth_m": -bottom,
                                            "acceptance_gate": False})
        root_index = self.links.index(self.root_name)
        root = transforms[root_index]
        if not optimized:
            expected = urdf_fk(self.measured_fk_model, positions)
            structure_error = max(float(np.linalg.norm((np.linalg.inv(poses[self.root_name]) @ poses[name])[:3, 3] - expected[name][:3, 3]))
                                  for name in self.links)
            from scipy.spatial.transform import Rotation
            rotation_error = max(float(Rotation.from_matrix(expected[name][:3, :3].T @
                                 (np.linalg.inv(poses[self.root_name]) @ poses[name])[:3, :3]).magnitude()) for name in self.links)
        reaction = np.asarray(self.art.get_measured_joint_forces(), dtype=float)
        if not np.isfinite(reaction).all():
            raise RuntimeError("Non-finite solver joint reaction")
        contacts = self.contacts.take()
        ground_separation = min((point["separation_m"] for pair in contacts["pairs"]
                                 if "ground" in (pair["body0"] + pair["body1"]).lower()
                                 for point in pair["points"]), default=None)
        if ground_separation is not None and ground_separation < 0:
            geometry["diagnostics"].append({"type": "ground_contact_penetration", "separation_m": ground_separation,
                                            "acceptance_gate": False})
        result = {"time_s": t, "dt_s": self.config["physics_dt_s"],
                  "root_position": root[:3].tolist(), "root_quaternion_wxyz": [float(root[6]), *root[3:6].tolist()],
                  "link_transforms": {name: pose.tolist() for name, pose in poses.items()},
                  "link_velocities": {name: {"linear_m_s": value[:3].tolist(), "angular_rad_s": value[3:].tolist()}
                                      for name, value in zip(self.links, link_velocities)},
                  **({"initial_link_transforms": self.initial_link_poses} if t == 0 else {}),
                  "joint_positions": positions, "joint_velocities": dict(zip(self.dofs, velocities.tolist())),
                  "joint_efforts": efforts, "joint_targets": targets,
                  "target_speed_fraction": {name: self.contract["joints"][name]["speed_fraction"] for name in self.dofs},
                  "passive_efforts": {name: -self.profile["joints"][name]["damping"] * float(velocity)
                                      for name, velocity in zip(self.dofs, velocities)},
                  "passive_effort_kind": "viscous component estimate only; native static/dynamic friction is solver applied",
                  "native_joint_friction": self.profile["joints"],
                  "solver_joint_reactions": reaction.tolist(), "contacts": contacts,
                  "min_ground_separation_m": ground_separation,
                  "geometry": geometry, "ground_min_z_m": bottom, "structure_error_m": structure_error}
        result["structure_rotation_error_rad"] = rotation_error
        if t == 0:
            result["initialization"] = {
                "expected_root_position": self.initial_root.tolist(),
                "expected_root_quaternion_wxyz": self.initial_orientation.tolist(),
                "expected_joint_positions": dict(self.reset_positions),
                "mode": ("fixed_motion_fixture" if self.fixture_clearance_m is not None else
                         "support_rest_pose" if self.support_pose is not None else "cooked_ground"),
                "ground_alignment_required": self.support_pose is None and self.fixture_clearance_m is None,
                "tilt_reference_quaternion_wxyz": [1., 0., 0., 0.]}
            if self.motion_scope_plan is not None:
                result["motion_scope_plan"] = self.motion_scope_plan
                result["motion_scope_binding"] = {
                    "urdf_sha256": sha256(self.args.asset_dir / "object.urdf"),
                    "cooked_geometry_sha256": self.report["active_cooked_geometry"]["sha256"],
                    "geometry_artifact": self.motion_scope_geometry_binding,
                    "fixture_clearance_m": self.fixture_clearance_m}
        from .motion_acceptance import enabled, StructurePlan
        if enabled(self.contract, getattr(self, 'current_trial_kind', None)) and getattr(self, 'current_trial_kind', None) in ('fixed_joint', 'free_joint'):
            if not hasattr(self, '_structure_plan'):
                self._structure_plan = StructurePlan(self.contract)
            distances, angles = self._structure_plan.measure(poses, positions)
            separated = not self._structure_plan.passed(distances, angles)
            result['structural_joint_errors'] = {'translation_m': distances.tolist(), 'rotation_deg': angles.tolist()}
        else:
            separated = structure_error > max(.001, self.contract["diagonal_m"] * .001) or rotation_error > math.radians(1)
        if separated:
            geometry["passed"] = False
            geometry["status"] = "failed"
            geometry["issues"].append({"type": "structural_separation", "error_m": structure_error, "rotation_error_rad": rotation_error})
        return result

    def reset_trial_state(self):
        """One final initialization reset after render resources, before t=0."""
        if not self.fixed_trial:
            self.art.set_world_poses(positions=self.initial_root[None, :], orientations=self.initial_orientation[None, :])
            self.art.set_velocities(np.zeros((1, 6), dtype=np.float32))
        if self.dofs:
            zeros = np.zeros((1, len(self.dofs)), dtype=np.float32)
            self.art.set_joint_positions(np.asarray([[self.reset_positions[name] for name in self.dofs]], dtype=np.float32))
            self.art.set_joint_velocities(zeros)
            self.art.set_joint_efforts(zeros)
        # Refresh dependent body poses without advancing gravity or the trial clock.
        self.sim.physics_sim_view.update_articulations_kinematic()
        self.q_previous, self.q_unwrapped = {}, {}
        self.contacts.take()
        self.contacts.active_pairs.clear()

    def make_rest_candidate(self, ground_trial, ground_trace):
        from .support_rest_pose import make_support_rest_pose
        if ground_trial.get("passed") is not True:
            raise ValueError("Only a passed measured ground trial can propose a support-rest pose")
        bindings = {"usd": {"path": str(self.delivered), "sha256": sha256(self.delivered)},
                    "profile": {"path": str(self.args.profile), "sha256": sha256(self.args.profile)},
                    "contract": {"path": str(self.args.contract), "sha256": sha256(self.args.contract)},
                    "source_trajectory": {"path": ground_trial["trajectory_path"], "sha256": ground_trial["trajectory_sha256"]}}
        candidate = make_support_rest_pose(ground_trace, self.contract, bindings)
        path = self.args.out_dir / "support_rest_pose.json"
        write_report(path, candidate)
        self.report["support_rest_pose"] = {"status": "candidate", "path": str(path), "sha256": sha256(path),
            "source_trial_id": ground_trial["id"], "reload_trial_id": "reload", "verified": False}
        return candidate

    def trial(self, kind, joint_name=None, *, reload=False, support_pose=None):
        import carb.settings
        import omni.replicator.core as rep
        from PIL import Image
        trial_id = kind + ("_" + joint_name if joint_name else "")
        folder = self.args.out_dir / "trials" / trial_id
        if folder.exists() and any(folder.iterdir()):
            import uuid
            folder=folder.with_name(trial_id+'.attempt_'+uuid.uuid4().hex[:12])
        folder.mkdir(parents=True, exist_ok=True)
        self.current_trial_kind = kind
        setup = self.create_trial(fixed=kind == "fixed_joint", reload=reload,
                                  support_pose=support_pose, joint_name=joint_name)
        self.skip_ground_acceptance = kind == "reload_supported"
        duration = .25 if kind == "reload_supported" else self.config["settle_duration_s"]
        operation_scope = None
        if kind == "free_joint" and self.motion_scope_plan is not None:
            plan = self.motion_scope_plan
            if plan["free_status"] == "ground_blocked":
                # An explicitly evaluated scope, never a fake successful trace.
                return {"id": trial_id, "kind": kind, "joint_name": joint_name,
                        "status": "not_applicable_ground_blocked", "passed": None,
                        "motion_passed": None, "ground_coverage_of_full_range": 0.,
                        "setup": setup, "motion_scope_plan": plan}, []
            from .metrics import planned_free_operation_scope
            operation_scope = planned_free_operation_scope(self.contract, plan)
        from .control import scoped_motion_joint, post_operation_observation_s
        executor = BoundedJointExecutor(self.contract, operation_scope=operation_scope,
                                        effort_policy=UNIFORM_PID_POLICY)
        executor.execution_optimized = getattr(self.args, 'measurement_mode', 'optimized') == 'optimized'
        motion_started = False
        if joint_name:
            motion_joint = scoped_motion_joint(self.contract, joint_name, operation_scope)
            _, motion_duration = target_schedule(motion_joint, self.config, contract=self.contract)
            passive_duration = post_operation_observation_s(self.config, contract=self.contract)
            duration += motion_duration + passive_duration
            self.report.setdefault("post_operation_observation", {})[trial_id] = {
                "policy": "active_control_until_trial_end.v1",
                "zero_effort_duration_s": passive_duration,
                "stability_window_s": 0.,
                "active_control_duration_unchanged": False}
            # Initial settling holds q_initial. The operation gets its own clock
            # only when start(joint_name) is explicitly called after this wait.
            executor.start()
        from .control import trial_resource_budget
        workload = trial_resource_budget(duration, self.config)
        self.report.setdefault("trial_work_budgets", {})[trial_id] = workload
        camera_points = [self.camera_points]
        if joint_name:
            from urdf_core.urdf_validation import urdf_fk
            definition = motion_joint
            for value in np.linspace(definition["lower"], definition["upper"], 17):
                frames = urdf_fk(self.measured_fk_model, {**self.initial_positions, joint_name: float(value)})
                camera_points.extend(transform_points(frames[name], mesh["vertices"]) + self.initial_root
                                     for name, meshes in self.original.items() for mesh in meshes)
        camera_config = camera_for_points(np.concatenate(camera_points), resolution=(640, 480))
        deferred = getattr(self.args, "capture_mode", "deferred") == "deferred"
        if not deferred:
            camera = rep.create.camera(position=camera_config["position"], look_at=camera_config["look_at"],
                                       clipping_range=camera_config["clipping_range"], focal_length=camera_config["focal_length"],
                                       horizontal_aperture=camera_config["horizontal_aperture"])
            product = rep.create.render_product(camera, tuple(camera_config["resolution"]))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb.attach([product])
            self.capture.keep(rgb, product)
            settings = carb.settings.get_settings()
            configure_fractional_opacity(settings)
        dt, fps = self.config["physics_dt_s"], self.config["video_fps"]
        steps = workload["expected_physics_steps"]
        trace, frames, geometry_failures = [], [], []
        witnessed_geometry_failures = set()
        terminated_early = None
        trajectory = folder / "trajectory.jsonl.gz"
        started, last_frame = time.monotonic(), -1
        timing = {key: 0. for key in ('control', 'physics_step', 'measurement', 'trajectory_write',
                                     'capture', 'video_pipe', 'progress', 'video_finalize', 'keyframes')}
        from .stream_video import StreamVideo, materialize_keyframes
        self.reset_trial_state()
        from contextlib import nullcontext
        recording = nullcontext(None) if deferred else StreamVideo(folder / 'motion.mp4', fps, ffmpeg=self.args.ffmpeg)
        with recording as encoder, open_trajectory_writer(trajectory) as handle:
            initial = self.measure(0.0, {name: 0.0 for name in self.dofs}, self.initial_positions)
            trace.append(initial)
            previous_sample = initial
            handle.write(json.dumps(initial, allow_nan=False, separators=(",", ":")) + "\n")
            for step in range(steps):
                timer = time.monotonic()
                t = step * dt
                optimized = getattr(self.args, 'measurement_mode', 'optimized') == 'optimized'
                if not optimized:
                    q = np.asarray(self.art.get_joint_positions()).reshape(-1) if self.dofs else []
                    qd = np.asarray(self.art.get_joint_velocities()).reshape(-1) if self.dofs else []
                if joint_name and not motion_started and t >= self.config["settle_duration_s"]:
                    executor.start(joint_name)
                    motion_started = True
                command_t = t - self.config["settle_duration_s"] if motion_started else t
                positions = (previous_sample['joint_positions'] if optimized else
                             {name: float(self.q_unwrapped.get(name, float(position))) for name, position in zip(self.dofs, q)})
                velocities = (previous_sample['joint_velocities'] if optimized else
                              {name: float(velocity) for name, velocity in zip(self.dofs, qd)})
                from .control import ordered_native_inertia
                matrix = ordered_native_inertia(self.art, self.dofs, executor.names) if executor.needs_native_inertia else None
                efforts = executor.step(positions, velocities, t=command_t, mass_matrix=matrix, physics_dt=dt)
                if executor.controller_selection is not None:
                    self.report.setdefault("controller_selections", {}).setdefault(trial_id, {})[
                        "motion" if motion_started else "pre_motion_hold"] = dict(executor.controller_selection)
                targets = dict(executor.last_targets)
                if self.dofs:
                    self.art.set_joint_efforts(np.asarray([[efforts[name] for name in self.dofs]], dtype=np.float32))
                if step == 0:
                    self.report.update(stage="trial_dynamics", active_trial=trial_id,
                                       active_trial_time_s=t + dt, active_trial_physics_step=step)
                    write_report(self.args.out_dir / "report.json", self.report)
                timing['control'] += time.monotonic() - timer
                timer = time.monotonic()
                self.sim.step(render=False, update_fabric=True)
                timing['physics_step'] += time.monotonic() - timer
                if step == 0:
                    self.report.update(stage="trial_geometry")
                    write_report(self.args.out_dir / "report.json", self.report)
                timer = time.monotonic()
                sample = self.measure(t + dt, efforts, targets)
                previous_sample = sample
                timing['measurement'] += time.monotonic() - timer
                trace.append(sample)
                timer = time.monotonic()
                handle.write(json.dumps(sample, allow_nan=False, separators=(",", ":")) + "\n")
                if step == 0 or not sample["geometry"]["passed"]:
                    handle.flush()
                timing['trajectory_write'] += time.monotonic() - timer
                if not sample["geometry"]["passed"]:
                    evidence = {"source_contacts": sample["geometry"]["original"].get("contacts", []),
                                "proxy_contacts": sample["geometry"]["cooked_proxies"].get("contacts", []),
                                "issues": sample["geometry"].get("issues", []),
                                "complete": sample["geometry"].get("complete", False)}
                    signature = json.dumps(evidence, sort_keys=True)
                    if signature not in witnessed_geometry_failures and len(geometry_failures) < 16:
                        geometry_failures.append(dict(evidence, time_s=t + dt, physics_step=step,
                                                      positions=sample["joint_positions"]))
                        witnessed_geometry_failures.add(signature)
                frame_index = int(math.floor(t * fps + 1e-8))
                failed_observation = not sample["geometry"]["passed"]
                if not deferred and (frame_index > last_frame or step == steps - 1 or failed_observation):
                    if step == 0 or failed_observation:
                        self.report.update(stage="trial_capture", active_trial=trial_id,
                                           active_trial_time_s=t + dt, active_trial_physics_step=step)
                        write_report(self.args.out_dir / "report.json", self.report)
                    timer = time.monotonic()
                    pixels = capture_rgb(rep.orchestrator, rgb, settings=settings,
                                         rt_subframes=4 if not frames else 1)
                    timing['capture'] += time.monotonic() - timer
                    timer = time.monotonic()
                    pixel_hash = encoder.append(pixels)
                    timing['video_pipe'] += time.monotonic() - timer
                    frames.append({"rgb_sha256": pixel_hash, "time_seconds": t + dt,
                                   "frame_index": len(frames), "positions": sample["joint_positions"],
                                   "geometry_passed": sample["geometry"]["passed"],
                                   "failure_event": failed_observation})
                    last_frame = frame_index
                if failed_observation and (deferred or len(frames) >= 2):
                    terminated_early = ("geometry_failed" if sample["geometry"].get("complete")
                                        else "geometry_observation_incomplete")
                    break
                if step % 240 == 0:
                    timer = time.monotonic()
                    self.report.update(stage="trial", active_trial=trial_id, active_trial_time_s=t)
                    write_progress(self.args.out_dir / "report.json", self.report)
                    timing['progress'] += time.monotonic() - timer
            timer = time.monotonic()
            video = encoder.finish() if encoder is not None else None
            timing['video_finalize'] += time.monotonic() - timer
        if not deferred:
            self.capture.suspend(product)
        from .metrics import evaluate_trial
        raw_trial = {"name": trial_id, "kind": "ground_settle" if kind == "reload" else kind, "joint_name": joint_name,
                     "duration_s": duration, "trace": trace}
        evaluated = evaluate_trial(trace, kind=raw_trial["kind"], joint_name=joint_name, contract=self.contract)
        if terminated_early:
            evaluated.update(passed=False, status="failed")
        trial = {"id": trial_id, "name": trial_id, "kind": kind, "joint_name": joint_name,
                 "setup": setup, "duration_s": duration, "wall_seconds": time.monotonic() - started,
                 "expected_duration_s": duration, "actual_duration_s": trace[-1]["time_s"] - trace[0]["time_s"],
                 "expected_physics_steps": steps + 1, "completed_physics_steps": len(trace),
                 "terminated_early": terminated_early,
                 "recording_mode": "diagnostic_failure_clip" if terminated_early else "full_trial_uniform_fps",
                 "trajectory_path": str(trajectory), "trajectory_sha256": sha256(trajectory),
                 "frames": frames, "camera": camera_config,
                 **({"video_path": video["video_path"], "video_sha256": video["sha256"],
                     "video_probe": video["probe"]} if video else {}),
                 "geometry_failures": geometry_failures,
                 "passed": evaluated["passed"], "metrics": evaluated}
        timer = time.monotonic()
        if deferred:
            from .deferred_capture import save_scene
            save_scene(self, trial)
        else:
            materialize_keyframes(trial, trace, self.contract, ffmpeg=self.args.ffmpeg)
        timing['keyframes'] += time.monotonic() - timer
        trial['wall_seconds'] = time.monotonic() - started
        trial['timing_wall_seconds'] = timing
        trial['capture_policy'] = {'capture_codec': 'lossless_rgb', 'delivery_codec': 'h264_yuv420p', 'fps': fps,
                                   'initial_rt_subframes': 4, 'subsequent_rt_subframes': 1,
                                   'render_delta_time': 0., 'numerical_sample_frequency_changed': False}
        # All samples are retained on disk; the API/report only receives aggregate evidence.
        return trial, trace


def evaluate_staged_trials(runtime, report, contract, report_path):
    """Diagnose the first completed failing trial without running later trials.

    Only measured trials enter `trials`. Blocked entries identify omitted work
    and the measured failed trial(s); they are not evidence or failed trials.
    """
    if getattr(getattr(runtime, 'args', None), 'evaluation_policy', 'first-failure') == 'complete':
        from .complete_evaluation import evaluate
        return evaluate(runtime, report, contract, report_path)
    from .protocol import free_base_required
    from .trial_checkpoint import for_runtime, observe
    checkpoint=for_runtime(runtime)
    needs_free = free_base_required(contract)
    stable = contract["stable_required"]
    names = list(contract["joints"])
    orders = {phase: names for phase in ('fixed_joint', 'free_joint')}
    order_path = getattr(getattr(runtime, 'args', None), 'trial_order', None)
    if order_path is not None:
        from .trial_order import verify_plan, POLICY as ORDER_POLICY
        plan = json.loads(Path(order_path).read_text())
        orders = verify_plan(plan, contract)
        report['trial_order'] = plan
    sampling_path = getattr(getattr(runtime, 'args', None), 'representative_motion', None)
    if sampling_path is not None:
        from .representative_motion import runtime_selection
        selected = runtime_selection(runtime, report, contract)
        orders = {phase: [n for n in order if n in selected[phase]]
                  for phase, order in orders.items()}
        write_report(report_path, report)
    stages = [("ground_settle", [None] if stable else []),
              ("fixed_joint", orders['fixed_joint']), ("free_joint", orders['free_joint'] if needs_free else [])]
    from .trial_order import LOCAL_POLICY, trial_sequence
    local_order = order_path is not None and plan['policy'] == LOCAL_POLICY
    if local_order:
        required = {(kind, n) for kind, joints in stages for n in joints}
        stages = [(kind, [name]) for kind, name in trial_sequence(plan, contract, required)]
    report.update(evaluation_complete=False, acceptance_complete=False,
                  blocked_trials=[], stopped_after_phase=None,
                  trial_evaluation_policy="first_failed_trial.v1")
    if order_path is not None:
        report['trial_evaluation_policy'] = plan['policy']
    if contract.get("motion_scope_policy"):
        report.update(motion_scope_policy=contract["motion_scope_policy"],
                      inapplicable_ground_operations=[])
    current = contract.get("schema_version") == 3
    if current:
        report["acceptance_policy"] = contract["acceptance_policy"]
        report["support_rest_pose"] = {"status": "pending" if stable else "not_applicable", "verified": False}
    if contract.get("free_base_policy") is not None:
        report["free_base_policy"] = copy.deepcopy(contract["free_base_policy"])
    ground_trial, ground_trace, support_pose = None, None, None
    if current and stable and needs_free:
        report['free_motion_initialization_policy'] = 'measured_stable_rest.v1'
    if not stable:
        # Export the free-root candidate before attaching any test fixture.
        if checkpoint is None or not checkpoint.restore_delivery(runtime):
            runtime.create_trial(fixed=False)
            if checkpoint:checkpoint.save_delivery(runtime)
    for index, (kind, joints) in enumerate(stages):
        for joint_index, name in enumerate(joints):
            rest_args = {'support_pose': support_pose} if kind == 'free_joint' and support_pose is not None else {}
            trial, trace = observe(runtime,checkpoint,kind,name,reload=kind != "ground_settle",**rest_args)
            if trial.get("status") == "not_applicable_ground_blocked":
                if kind != "free_joint" or not contract.get("motion_scope_policy"):
                    raise RuntimeError("Unapproved motion applicability change")
                report["inapplicable_ground_operations"].append(trial)
                write_report(report_path, report)
                continue
            if kind == "ground_settle":
                ground_trial, ground_trace = trial, trace
                if current and trial.get('passed') is True:
                    # Freeze this candidate's measured resting pose BEFORE any
                    # actuation. Ground-feasible ranges must use that pose, not
                    # an upright placement which gravity has already changed.
                    support_pose = runtime.make_rest_candidate(ground_trial, ground_trace)
            report["trials"].append(trial)
            write_report(report_path, report)
            if trial["passed"] is not True:
                reason = {"status": "skipped_by_prior_failure", "blocked_by_phase": kind,
                          "failed_trial_ids": [trial["id"]]}
                pending = [(kind, joints[joint_index + 1:]), *stages[index + 1:]]
                report["blocked_trials"] = [
                    {"kind": later_kind, "joint_name": later_name, **reason}
                    for later_kind, later_joints in pending for later_name in later_joints]
                report["reload_check"] = {"passed": False, **reason}
                report.update(evaluation_complete=True, acceptance_complete=False,
                              numerical_passed=False, passed=False, status="physics_failed",
                              stage="evaluation_complete", stopped_after_phase=kind)
                report["issues"].append("First measured failed trial blocks remaining trials and delivery reload: " + trial["id"])
                write_report(report_path, report)
                return
    kwargs = {"support_pose": support_pose} if support_pose is not None else {}
    reload_trial, reload_trace = observe(runtime,checkpoint,"reload" if stable else "reload_supported",reload=True,**kwargs)
    report["reload_check"] = {"passed": False, "trial": reload_trial,
                              "free_root": True, "skipped_ground": not stable,
                              "joint_motion_trial_ids": [trial["id"] for trial in report["trials"]
                                                         if trial["kind"] == "free_joint" and trial["setup"]["reloaded"]]}
    # Reload the delivered asset only after all applicable stages pass. The
    # measured rest pose still receives the complete eight-second Stable trial.
    report["reload_check"]["passed"] = reload_trial["passed"]
    if current and stable:
        report["support_rest_pose"].update(verified=report["reload_check"]["passed"],
            status="verified" if report["reload_check"]["passed"] else "rejected")
    report["numerical_passed"] = True
    report["passed"] = report["reload_check"]["passed"]
    report.update(status="passed" if report["passed"] else "physics_failed", stage="complete",
                  evaluation_complete=True, acceptance_complete=True)
    if not report["passed"]:
        report["issues"].append("Delivery reload verification rejected the asset; inspect measured evidence")


def run_worker(args, *, shared_app=None):
    from urdf_core.kinematics import validate_spec
    previous=args.out_dir/'report.json'
    if previous.exists():
        import shutil
        archive=args.out_dir/'attempt_reports'/('report_'+sha256(previous)+'.json')
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():shutil.copy2(previous,archive)
    report = initial_report(args, args.run_id)
    app, runtime = None, None
    try:
        if args.expected_sources and json.loads(args.expected_sources) != source_hashes():
            raise RuntimeError("Runtime source changed between supervisor and worker startup")
        spec = validate_spec(json.loads(args.spec.read_text()))
        from .protocol import validate_profile, validate_current_contract
        profile = validate_profile(json.loads(args.profile.read_text()), spec)
        contract = json.loads(args.contract.read_text())
        if profile["schema_version"] != 3:
            raise ValueError("Current physics acceptance requires profile schema 3")
        validate_current_contract(contract)
        if profile["stable_required"] != contract.get("stable_required"):
            raise ValueError("Profile and contract stability policy differ")
        manifest = json.loads((args.asset_dir / "asset_manifest.json").read_text())
        original = load_geometry(manifest, args.asset_dir)
        report["stage"] = "simulation_app_launch"
        report["worker_sources"] = report["source_hashes"]
        write_report(args.out_dir / "report.json", report)
        if shared_app is None:
            app = create_application(args)
        else:
            app = shared_app
        runtime = Runtime(args, report, spec, profile, contract, manifest, original)
        from urdf_core.native_progress import write_execution_progress
        write_execution_progress(args.out_dir, args.run_id, {'stage': 'runtime_ready'})
        partition=getattr(args,'trial_selection',None)
        grouping = getattr(args, 'evaluation_policy', None) == 'grouping'
        if grouping:
            from .joint_groups import freeze_runtime_groups
            report.update(grouping=freeze_runtime_groups(runtime), status='grouping_ready',
                          evaluation_complete=False, acceptance_complete=False, stage='grouping_complete')
        elif partition is not None:
            from .parallel_trials import evaluate_partition
            evaluate_partition(runtime,report,contract,args.out_dir/'report.json')
        else:
            evaluate_staged_trials(runtime, report, contract, args.out_dir / "report.json")
        if not grouping and partition is None and getattr(args, "capture_mode", "deferred") == "deferred":
            from .deferred_capture import finalize_evidence
            finalize_evidence(runtime, report)
    except Exception as exc:
        report.update(status="environment_error", passed=False, evaluation_complete=False, acceptance_complete=False)
        from urdf_core.content_errors import AssetContentError
        if isinstance(exc, AssetContentError):
            report.update(status='asset_rejected', asset_error=exc.evidence)
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()
    finally:
        if runtime:
            try:
                runtime.close_trial()
            except Exception as exc:
                report.update(status="environment_error", passed=False, evaluation_complete=False, acceptance_complete=False)
                report["errors"].append(f"Cleanup failed: {exc}")
        write_report(args.out_dir / "report.json", report)
        if app is not None and shared_app is None:
            app.close()
    return 0 if report["status"] in {"passed", "physics_failed", "trial_fragment", "grouping_ready", "asset_rejected"} else 3


def create_application(args):
    from isaacsim import SimulationApp
    portable = args.out_dir / "runtime" / "kit"
    portable.mkdir(parents=True, exist_ok=True)
    extra = ["--portable-root", str(portable), f"--/app/tokens/cache={args.out_dir / 'runtime/cache'}",
             f"--/app/tokens/data={args.out_dir / 'runtime/data'}", f"--/app/tokens/logs={args.out_dir}",
             "--/app/extensions/fsWatcherEnabled=0", "--/app/extensions/registryEnabled=0",
             "--/app/extensions/syncRegistryOnStartup=0", "--/app/extensions/preferLocalVersions=1",
             f"--/log/file={args.out_dir / 'kit.log'}", "--/physics/updateToUsd=true", "--/physics/updateVelocitiesToUsd=true"]
    from urdf_core.isaac_admission import thread_args
    extra.extend(thread_args())
    sys.argv = [sys.argv[0], "--portable-root", str(portable)]
    app = SimulationApp({"headless": True, "create_new_stage": False, "width": 640, "height": 480,
                         "multi_gpu": False, "extra_args": extra})
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.asset.importer.urdf")
    app.update()
    return app
