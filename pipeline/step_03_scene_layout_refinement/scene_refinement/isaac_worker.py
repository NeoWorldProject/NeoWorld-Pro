"""Resident Isaac 5.1 scene validator. Requests/results are files, never stdout JSON.

Each request opens the candidate USD afresh. All trajectory samples are native
PhysX measurements; no active effort or target-position command is issued.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "step_02_object_generation"))
sys.path.insert(0, str(HERE.parents[1] / "step_02_object_generation/kinematics_validation"))
from scene_refinement.worker_client import atomic_json, assert_source_binding, content_hash, sha256, native_runtime_binding
from scene_refinement.protocol import read_json, write_json
from scene_refinement.metrics import evaluate_scene_trace, physics_settings
from urdf_core.content_errors import AssetContentError, require_cooked_geometry


def evaluation_error_report(error):
    if isinstance(error, AssetContentError):
        return {'status': 'asset_rejected', 'passed': False, 'acceptance_complete': False, 'complete': False,
                'images': [], 'violation': None, 'physical_residual': None,
                'asset_error': error.evidence,
                'asset_scope': 'asset_topology' if error.evidence['stage'] == 'scene_asset_topology' else 'candidate_geometry',
                'issues': [{'kind': 'asset_content_rejected', 'object_id': error.evidence['object_id']}]
                          if error.evidence.get('object_id') else [],
                'errors': [str(error)]}
    return {'status': 'environment_error', 'passed': False, 'acceptance_complete': False, 'complete': False,
            'images': [], 'violation': 1., 'physical_residual': 1.,
            'errors': [f'{type(error).__name__}: {error}'], 'traceback': traceback.format_exc()}


def validate_joint_identity(object_id, joints, names, count):
    from pxr import UsdPhysics
    expected = {name for name, prim in joints.items()
                if (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint))
                and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False
                and UsdPhysics.Joint(prim).GetExcludeFromArticulationAttr().Get() is not True}
    if set(names) != expected or len(names) != len(set(names)) or len(names) != count:
        raise AssetContentError(f'Native joint identities differ for {object_id}: expected={sorted(expected)}, observed={list(names)}',
                                stage='scene_asset_topology', object_id=object_id)


def _meshes(manifest, key):
    import numpy as np
    output = {}
    for oid, obj in manifest["objects"].items():
        for item in obj[key]:
            name = f"{oid}::{item['link']}"
            output.setdefault(name, []).append({"vertices": np.asarray(item["vertices"], dtype=float),
                "faces": np.asarray(item["faces"], dtype=int), "name": item.get("name", name)})
    return output


def evaluate_bound_request(runtime, request, startup):
    """An evaluation cannot relabel already imported modules as newer source."""
    if request["session_id"] != startup["session_id"]:
        raise RuntimeError("Stale request/source binding")
    assert_source_binding(startup["source_hashes"], request["source_hashes"])
    if startup.get("runtime_binding") is not None and (
            request.get("runtime_binding") != startup["runtime_binding"] or
            native_runtime_binding(startup["launcher"]) != startup["runtime_binding"]):
        raise RuntimeError("Native runtime differs from request binding")
    for item in request["inputs"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise RuntimeError("Candidate input changed after submission")
    report = runtime.render(request) if request.get("mode") == "render" else runtime.evaluate(request)
    assert_source_binding(startup["source_hashes"], request["source_hashes"])
    if startup.get("runtime_binding") is not None and native_runtime_binding(startup["launcher"]) != startup["runtime_binding"]:
        raise RuntimeError("Native runtime changed during evaluation")
    if any(sha256(item["path"]) != item["sha256"] for item in request["inputs"].values()):
        raise RuntimeError("Candidate input changed during evaluation")
    return report


def native_box_geometry(prim, mesh_to_body):
    import numpy as np
    import trimesh
    from pxr import UsdGeom
    size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
    box = trimesh.creation.box(extents=[size] * 3)
    transform = np.asarray(mesh_to_body)
    return {"vertices": box.vertices @ transform[:3, :3].T + transform[:3, 3],
            "faces": box.faces, "name": str(prim.GetPath())}


def _cooked_geometry(stage, manifest):
    """Read native convex hulls; static triangle surfaces retain their exact USD mesh."""
    import numpy as np
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils
    from physics_feedback.physics_runtime import triangulate
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    body_lookup = {path: f"{oid}::{link}" for oid, obj in manifest["objects"].items()
                   for link, path in obj["body_paths"].items()}
    output, evidence = {key: [] for key in body_lookup.values()}, []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube)):
            raise RuntimeError(f"Unsupported native collision shape: {prim.GetPath()}")
        path = str(prim.GetPath())
        ancestors = [candidate for candidate in body_lookup if path == candidate or path.startswith(candidate + "/")]
        if not ancestors:
            raise RuntimeError(f"Unbound collision mesh: {path}")
        body_path = max(ancestors, key=len)
        body, name = stage.GetPrimAtPath(body_path), body_lookup[body_path]
        mesh_to_body = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T) @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        def transform(points):
            points = np.asarray(points, dtype=float)
            return points @ mesh_to_body[:3, :3].T + mesh_to_body[:3, 3]
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "none"
        received = []
        if prim.IsA(UsdGeom.Cube):
            received.append(native_box_geometry(prim, mesh_to_body))
            representation = "native_box"
            approximation = "box"
        elif approximation == "none" and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            mesh = UsdGeom.Mesh(prim)
            ids, cursor, faces = list(mesh.GetFaceVertexIndicesAttr().Get()), 0, []
            for count in mesh.GetFaceVertexCountsAttr().Get():
                faces.append(ids[cursor:cursor + count])
                cursor += count
            received.append({"vertices": transform(mesh.GetPointsAttr().Get()), "faces": triangulate(faces), "name": path})
            representation = "native_static_triangle_mesh"
        else:
            if approximation not in {"convexDecomposition", "convexHull"}:
                raise RuntimeError(f"Unsupported observable collision approximation: {path}: {approximation}")
            results = []
            def on_result(result, hulls):
                results.append(result)
                if result != PhysxCollisionRepresentationResult.RESULT_VALID:
                    return
                for index, hull in enumerate(hulls):
                    faces = []
                    for polygon in hull.polygons:
                        ids = hull.indices[polygon.index_base:polygon.index_base + polygon.num_vertices]
                        faces.extend([[ids[0], ids[i], ids[i + 1]] for i in range(1, len(ids) - 1)])
                    received.append({"vertices": transform([list(v) for v in hull.vertices]),
                                     "faces": np.asarray(faces, dtype=int), "name": path + f"#hull{index}"})
            get_physx_cooking_interface().request_convex_collision_representation(
                stage_id=stage_id, collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
                run_asynchronously=False, on_result=on_result)
            oid, link = name.split('::', 1)
            require_cooked_geometry(results, len(received), object_id=oid, link=link, part=path)
            representation = "native_cooked_convex_hulls"
        if not received:
            raise RuntimeError(f"No native collision representation: {path}")
        output[name].extend(received)
        evidence.append({"body": name, "prim_path": path, "representation": representation,
                         "approximation": approximation, "components": len(received)})
    output = {name: meshes for name, meshes in output.items() if meshes}
    if not output:
        raise AssetContentError('Scene lacks collision geometry', stage='scene_asset_topology')
    return output, evidence


class SceneRuntime:
    def __init__(self, app, config):
        self.app, self.config = app, physics_settings(config)
        self.sim = self.contacts = self.capture = None
        self.views = []

    def close_trial(self):
        self.views.clear()
        if self.contacts:
            self.contacts.close()
            self.contacts = None
        if self.sim:
            self.sim.stop()
        if self.capture:
            self.capture.close()
            self.capture = None
        if self.sim:
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
            self.sim = None

    def render(self, request):
        """Render the exact saved USD with a stopped timeline; no extra settling."""
        import numpy as np
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Usd, UsdGeom, UsdLux, UsdPhysics
        from verify_urdf_isaac import CaptureResources, capture_rgb
        from isaac_motion import camera_for_points
        from scene_refinement.framing import in_camera

        started = time.monotonic()
        self.close_trial()
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        candidate = request["candidate"]
        manifest = read_json(candidate["scene_manifest_path"])
        for path, expected in manifest.get("derivative_artifacts", {}).items():
            if sha256(path) != expected:
                raise ValueError("Render dependency changed: " + path)
        if not omni.usd.get_context().open_stage(candidate["usd_path"]):
            raise RuntimeError("Cannot load settled render snapshot")
        stage = omni.usd.get_context().get_stage()
        paths = [p for obj in manifest["objects"].values() for p in obj["body_paths"].values()]
        def poses():
            cache = UsdGeom.XformCache()
            return {p: np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p))).copy() for p in paths}
        before = poses()
        # Render-only in-memory overrides freeze the authored link transforms.
        # The candidate USD is never saved and physics acceptance is separate.
        stage.SetEditTarget(stage.GetSessionLayer())
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        class PausedCapture:
            def step(self, **kwargs):
                return rep.orchestrator.step(**{**kwargs, "pause_timeline": True})
        UsdLux.DomeLight.Define(stage, "/SceneRefinementLight").CreateIntensityAttr(1000.)
        bounds = manifest.get("fixed_render_bounds_m", manifest["fixed_camera_bounds_m"])
        points = np.asarray([bounds["min"], bounds["max"]])
        self.capture = CaptureResources()
        folder = Path(request["out_dir"]); folder.mkdir(parents=True, exist_ok=True)
        images, coverage = [], []
        meshes = _meshes(manifest, "geometry_meshes")
        points_by_body = {}
        for oid, obj in manifest["objects"].items():
            for link, path in obj["body_paths"].items():
                pose = before[path].T
                vertices = np.concatenate([m["vertices"] for m in meshes[f"{oid}::{link}"]])
                points_by_body[f"{oid}::{link}"] = vertices @ pose[:3, :3].T + pose[:3, 3]
        for index, direction in enumerate(((1.4, -1.8, 1.2), (-1.6, -1.4, 2.8))):
            config = camera_for_points(points, direction=direction,
                resolution=(int(self.config["render_width"]), int(self.config["render_height"])))
            coverage.append({body: in_camera(vertices, config) for body, vertices in points_by_body.items()})
            camera = rep.create.camera(position=config["position"], look_at=config["look_at"],
                clipping_range=config["clipping_range"], focal_length=config["focal_length"],
                horizontal_aperture=config["horizontal_aperture"])
            product = rep.create.render_product(camera, tuple(config["resolution"]))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach([product])
            self.capture.keep(rgb, product)
            pixels = capture_rgb(PausedCapture(), rgb)
            path = folder / f"view_{index}.png"; Image.fromarray(pixels).save(path)
            images.append({"path": str(path), "sha256": sha256(path), "camera": config,
                           "state": manifest.get("state_kind", "constructed"), "source_usd_sha256": sha256(candidate["usd_path"])})
            self.capture.suspend(product)
        after = poses()
        if timeline.is_playing() or any(not np.allclose(before[p], after[p], rtol=0, atol=1e-9) for p in paths):
            raise RuntimeError("Snapshot rendering changed the authored state")
        return {"status": "rendered", "passed": False, "native_dynamics": False,
                "images": images, "render_seconds": time.monotonic() - started,
                "render_policy": "static_authored_snapshot_no_physics",
                "semantic_frame_complete": all(all(view.values()) for view in coverage),
                "frustum_coverage": coverage,
                "render_state_verified": True, "source_usd_sha256": sha256(candidate["usd_path"])}

    def evaluate(self, request):
        evaluation_started = time.monotonic()
        phase_times = {"render_seconds": 0., "geometry_seconds": 0., "native_step_seconds": 0.}
        import numpy as np
        import omni.usd
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics
        from scipy.spatial.transform import Rotation
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import Articulation, RigidPrim
        from physics_feedback.physics_runtime import Contacts
        from verify_urdf_isaac import CaptureResources, capture_rgb, tensor_transform
        from isaac_motion import camera_for_points, encode_video

        self.close_trial()
        folder = Path(request["out_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        candidate = request["candidate"]
        manifest = read_json(candidate["scene_manifest_path"])
        if not omni.usd.get_context().open_stage(str(candidate["usd_path"])):
            raise RuntimeError("Cannot reload candidate scene USD")
        stage = omni.usd.get_context().get_stage()
        self.app.update()
        objects = manifest["objects"]
        for path, expected_hash in manifest.get("derivative_artifacts", {}).items():
            if sha256(path) != expected_hash:
                raise ValueError("Scaled asset dependency changed: " + path)
        original, reference = _meshes(manifest, "geometry_meshes"), _meshes(manifest, "collision_meshes")
        body_paths = {f"{oid}::{link}": path for oid, obj in objects.items() for link, path in obj["body_paths"].items()}
        if set(original) != set(body_paths) or not set(reference) <= set(body_paths):
            raise ValueError("Candidate geometry/body mapping is incomplete")
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        static_poses, authored_poses, dynamic = {}, {}, []
        for name, path in body_paths.items():
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError(f"Missing native body: {path}")
            pose = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            authored_poses[name] = pose.copy()
            if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4):
                raise ValueError("Scene body frames must be rigid; bake uniform scale into the asset")
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid = UsdPhysics.RigidBodyAPI(prim)
                if rigid.GetKinematicEnabledAttr().Get():
                    raise ValueError("Dynamic object link unexpectedly kinematic")
                physx = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                if physx.GetDisableGravityAttr().Get():
                    raise ValueError("Dynamic object link disables gravity")
                rigid.CreateVelocityAttr(Gf.Vec3f(0))
                rigid.CreateAngularVelocityAttr(Gf.Vec3f(0))
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
                dynamic.append(name)
            else:
                static_poses[name] = pose
        bounds = manifest.get("fixed_camera_bounds_m")
        if not bounds:
            raise ValueError("Candidate lacks camera bounds frozen at initial scene preparation")
        outside = []
        for name, meshes in original.items():
            pose = authored_poses[name]
            points = np.concatenate([mesh["vertices"] for mesh in meshes]) @ pose[:3, :3].T + pose[:3, 3]
            if (np.any(points.min(0) < np.asarray(bounds["min"]) - self.config["geometry_tolerance_m"])
                    or np.any(points.max(0) > np.asarray(bounds["max"]) + self.config["geometry_tolerance_m"])):
                outside.append(name)
        if outside:
            return {"status": "invalid_candidate", "passed": False, "complete": True,
                    "acceptance_complete": True, "violation": 1e6, "physical_residual": 1e6,
                    "images": [], "errors": [], "issues": [{"kind": "outside_frozen_camera_envelope", "bodies": outside}]}
        # Native passive damping remains authored. Active drives are not allowed.
        authored_joint_positions = {}
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                for kind in ("linear", "angular"):
                    state = prim.GetAttribute(f"state:{kind}:physics:position")
                    if state and state.HasAuthoredValueOpinion():
                        value = float(state.Get())
                        authored_joint_positions[str(prim.GetPath())] = math.radians(value) if kind == "angular" else value
                    if prim.HasAPI(UsdPhysics.DriveAPI, kind):
                        drive = UsdPhysics.DriveAPI(prim, kind)
                        if float(drive.GetStiffnessAttr().Get() or 0) != 0 or float(drive.GetTargetVelocityAttr().Get() or 0) != 0:
                            raise ValueError(f"Active drive present in no-actuation scene: {prim.GetPath()}")
        physics_scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
        if len(physics_scenes) > 1:
            raise ValueError("Candidate contains multiple physical scenes")
        authored_physics_scene = bool(physics_scenes)
        if authored_physics_scene:
            scene = UsdPhysics.Scene(physics_scenes[0])
            prim = scene.GetPrim()
            expected = {"physxScene:enableGPUDynamics": False, "physxScene:enableCCD": True,
                        "physxScene:solverType": "TGS", "physxScene:enableExternalForcesEveryIteration": True,
                        "physxScene:broadphaseType": "MBP", "physxScene:timeStepsPerSecond": round(1 / self.config["physics_dt_s"])}
            if (not np.allclose(scene.GetGravityDirectionAttr().Get(), [0, 0, -1], rtol=0., atol=1e-7)
                    or not math.isclose(scene.GetGravityMagnitudeAttr().Get(), self.config["gravity_m_s2"], rel_tol=1e-6)
                    or any(prim.GetAttribute(name).Get() != value for name, value in expected.items())):
                raise ValueError("Candidate authored PhysicsScene differs from the frozen worker settings")
        elif manifest.get("synthetic_fixture"):
            # Legacy direct synthetic fixtures predate self-contained composition.
            scene = UsdPhysics.Scene.Define(stage, "/SceneRefinementPhysics")
            scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
            scene.CreateGravityMagnitudeAttr(self.config["gravity_m_s2"])
        else:
            raise ValueError("Candidate USD lacks its required authored PhysicsScene")
        physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx.CreateEnableGPUDynamicsAttr(False)
        physx.CreateBroadphaseTypeAttr("MBP")
        physx.CreateEnableCCDAttr(True)
        physx.CreateTimeStepsPerSecondAttr(round(1 / self.config["physics_dt_s"]))
        physx.CreateSolverTypeAttr("TGS")
        physx.CreateEnableExternalForcesEveryIterationAttr(True)
        if (physx.GetSolverTypeAttr().Get() != "TGS" or
                physx.GetEnableExternalForcesEveryIterationAttr().Get() is not True):
            raise RuntimeError("Native scene solver policy differs from the authored contract")
        physics_scene_evidence = {"path": str(scene.GetPath()), "authored_in_candidate": authored_physics_scene,
            "gravity_m_s2": self.config["gravity_m_s2"], "physics_dt_s": self.config["physics_dt_s"],
            "gpu_dynamics": False, "broadphase_type": "MBP", "ccd": True,
            "solver_type": physx.GetSolverTypeAttr().Get(),
            "enable_external_forces_every_iteration": physx.GetEnableExternalForcesEveryIterationAttr().Get()}
        UsdLux.DomeLight.Define(stage, "/SceneRefinementLight").CreateIntensityAttr(1000.0)
        cooked, cooking_evidence = _cooked_geometry(stage, manifest)
        cooked_serialized = {name: [{"vertices": mesh["vertices"].tolist(), "faces": mesh["faces"].tolist()}
                                     for mesh in meshes] for name, meshes in cooked.items()}
        cooked_path = folder / ("cooked_collisions.json.gz" if self.config.get("compact_evidence") else "cooked_collisions.json")
        write_json(cooked_path, {"geometry": cooked_serialized, "representations": cooking_evidence})
        fidelity = {"status": "not_required_functional_policy", "acceptance_gate": False}
        dt = self.config["physics_dt_s"]
        self.sim = SimulationContext(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0,
            physics_prim_path=str(scene.GetPath()), set_defaults=False, backend="numpy", device="cpu")
        self.sim.initialize_physics()
        self.sim.play()
        from urdf_core.isaac_admission import initialized
        initialized()
        covered, views, resets, native_articulations = set(), [], [], {}
        for oid, obj in objects.items():
            root = obj.get("articulation_root_path")
            if root:
                art = Articulation(prim_paths_expr=root, name="scene_" + oid, reset_xform_properties=False)
                art.initialize()
                names = list(art._physics_view.shared_metatype.link_names)
                expected = set(obj["body_paths"])
                if set(names) != expected:
                    raise AssetContentError(f'Native articulation links differ for {oid}: {names}',
                                            stage='scene_asset_topology', object_id=oid)
                if art.num_dof:
                    # Initial-state writes only; no joint actuation in the loop.
                    art.set_joint_velocities(np.zeros((1, art.num_dof), dtype=np.float32))
                    art.set_joint_efforts(np.zeros((1, art.num_dof), dtype=np.float32))
                views.append(("articulation", art, [f"{oid}::{name}" for name in names]))
                joints = {prim.GetName(): prim for prim in Usd.PrimRange(stage.GetPrimAtPath(obj["root_prim_path"]))
                          if prim.IsA(UsdPhysics.Joint)}
                validate_joint_identity(oid, joints, list(art.dof_names), int(art.num_dof))
                q_initial = [authored_joint_positions.get(str(joints[name].GetPath()), 0.)
                             for name in art.dof_names]
                gains = art.get_gains() if art.num_dof else (np.zeros((1, 0)), np.zeros((1, 0)))
                # Tensor link zero is the native base, which can differ from the URDF root.
                native_root_link = names[0]
                native_articulations[oid] = {"root_prim_path": root, "link_names": names,
                    "native_root_link": native_root_link,
                    "dof_names": list(art.dof_names), "dof_count": int(art.num_dof),
                    "joint_limits": np.asarray(art.get_dof_limits()).tolist() if art.num_dof else [],
                    "drive_stiffness": np.asarray(gains[0]).tolist(),
                    "passive_damping": np.asarray(gains[1]).tolist(), "initial_joint_positions": q_initial}
                if np.any(np.asarray(gains[0]) != 0):
                    raise ValueError(f"Native active drive detected for {oid}")
                resets.append(("articulation", art, authored_poses[f"{oid}::{native_root_link}"], q_initial))
                covered.update(f"{oid}::{name}" for name in names)
        for name in dynamic:
            if name not in covered:
                view = RigidPrim(prim_paths_expr=body_paths[name], name="scene_rigid_" + name.replace("::", "_"),
                                 reset_xform_properties=False)
                view.initialize()
                views.append(("rigid", view, [name]))
                resets.append(("rigid", view, authored_poses[name], []))
        self.views = views
        self.contacts = Contacts()
        self.capture = CaptureResources()
        render_bounds = manifest.get("fixed_render_bounds_m", bounds)
        points = np.asarray([render_bounds["min"], render_bounds["max"]], dtype=float)
        camera_configs, renderers, images = [], [], []
        directions = () if request.get("mode") == "physics" else ((1.4, -1.8, 1.2), (-1.6, -1.4, 2.8))
        for index, direction in enumerate(directions):
            camera_config = camera_for_points(points, direction=direction,
                resolution=(int(self.config["render_width"]), int(self.config["render_height"])))
            camera_configs.append(camera_config)
            camera = rep.create.camera(position=camera_config["position"], look_at=camera_config["look_at"],
                clipping_range=camera_config["clipping_range"], focal_length=camera_config["focal_length"],
                horizontal_aperture=camera_config["horizontal_aperture"])
            product = rep.create.render_product(camera, tuple(camera_config["resolution"]))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb.attach([product])
            self.capture.keep(rgb, product)
            renderers.append((rgb, product))
        # Build attachment observations from exact authored joint frames.
        attachment_frames, external_joints = [], []
        path_to_key = {path: key for key, path in body_paths.items()}
        for attachment in manifest.get("attachments", []):
            joint = UsdPhysics.Joint(stage.GetPrimAtPath(attachment["joint_path"]))
            excluded = joint.GetExcludeFromArticulationAttr().Get() is True
            if not excluded:
                raise ValueError("External attachment must be excluded from the internal articulation")
            external_joints.append({"joint_path": attachment["joint_path"], "child_id": attachment["child_id"],
                "parent_id": attachment["parent_id"], "exclude_from_articulation": excluded,
                "collision_enabled": joint.GetCollisionEnabledAttr().Get(),
                "body0": [str(p) for p in joint.GetBody0Rel().GetTargets()],
                "body1": [str(p) for p in joint.GetBody1Rel().GetTargets()]})
            frames = []
            for number in (0, 1):
                targets = getattr(joint, f"GetBody{number}Rel")().GetTargets()
                target = str(targets[0]) if targets else None
                quat = getattr(joint, f"GetLocalRot{number}Attr")().Get()
                frame = np.eye(4)
                frame[:3, :3] = Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()
                frame[:3, 3] = list(getattr(joint, f"GetLocalPos{number}Attr")().Get())
                if target and target not in path_to_key:
                    raise ValueError(f"Attachment targets an unknown body: {target}")
                frames.append((path_to_key.get(target), frame))
            attachment_frames.append((attachment["child_id"], frames))
        def measure(t):
            poses = {key: pose.copy() for key, pose in static_poses.items()}
            velocities = {key: np.zeros(6) for key in static_poses}
            for kind, view, names in views:
                if kind == "articulation":
                    transforms = np.asarray(view._physics_view.get_link_transforms(), dtype=float)[0]
                    poses.update({name: tensor_transform(transform) for name, transform in zip(names, transforms)})
                    observed = np.asarray(view._physics_view.get_link_velocities(), dtype=float)[0]
                else:
                    xyz, wxyz = view.get_world_poses()
                    q = np.asarray(wxyz)[0]
                    poses[names[0]] = tensor_transform([*np.asarray(xyz)[0], q[1], q[2], q[3], q[0]])
                    observed = np.asarray(view.get_velocities(), dtype=float)
                if observed.shape != (len(names), 6) or not np.isfinite(observed).all():
                    raise RuntimeError("Missing/nonfinite native rigid-link velocity")
                velocities.update(dict(zip(names, observed)))
            attachments = {}
            for child, frames in attachment_frames:
                world = [(poses[key] if key else np.eye(4)) @ frame for key, frame in frames]
                delta = np.linalg.inv(world[0]) @ world[1]
                attachments[child] = {"observed": True, "position_error_m": float(np.linalg.norm(delta[:3, 3])),
                    "rotation_error_deg": float(np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude()))}
            joint_states = {names[0].split("::")[0]: {
                "joint_names": list(view.dof_names),
                "positions": np.asarray(view.get_joint_positions()).tolist(),
                "velocities": np.asarray(view.get_joint_velocities()).tolist()}
                for kind, view, names in views if kind == "articulation" and view.num_dof}
            return {"time_s": t, "body_poses": {key: value.tolist() for key, value in poses.items()},
                    "body_velocities": {key: {"linear_m_s": value[:3].tolist(), "angular_rad_s": value[3:].tolist()}
                                        for key, value in velocities.items()},
                    "contacts": self.contacts.take(), "attachments": attachments,
                    "native_joint_states": joint_states}
        # Warm capture graphs, then reset authored state before recording.
        for rgb, _ in renderers:
            capture_rgb(rep.orchestrator, rgb)
        # Graph/sensor initialization may advance a playing timeline. Restore
        # authored state once at this explicit reset boundary, before recording.
        self.sim.pause()
        for kind, view, pose, q_initial in resets:
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()
            view.set_world_poses(positions=pose[None, :3, 3], orientations=np.asarray([[q[3], *q[:3]]]))
            view.set_linear_velocities(np.zeros((1, 3), dtype=np.float32))
            view.set_angular_velocities(np.zeros((1, 3), dtype=np.float32))
            if kind == "articulation" and view.num_dof:
                view.set_joint_positions(np.asarray([q_initial], dtype=np.float32))
                view.set_joint_velocities(np.zeros((1, view.num_dof), dtype=np.float32))
                view.set_joint_efforts(np.zeros((1, view.num_dof), dtype=np.float32))
        self.contacts.pairs.clear()
        self.contacts.active_pairs.clear()
        initial = measure(0.0)
        state_errors = {name: float(np.max(np.abs(np.asarray(initial["body_poses"][name]) - pose)))
                        for name, pose in authored_poses.items()}
        atomic_json(folder / "initial_state_receipt.json", {"authored_body_poses": {name: pose.tolist() for name, pose in authored_poses.items()},
            "measured_body_poses": initial["body_poses"], "max_matrix_errors": state_errors,
            "reset_policy": "authored_root_and_joint_positions_zero_velocities_initialization_only"})
        if any(not np.allclose(initial["body_poses"][name], pose, rtol=0.,
                               atol=self.config["geometry_tolerance_m"])
               for name, pose in authored_poses.items()):
            raise RuntimeError("Physics initialization changed the authored candidate state")
        for index, (rgb, product) in enumerate(renderers):
            pixels = capture_rgb(rep.orchestrator, rgb)
            path = folder / f"initial_view_{index}.png"
            Image.fromarray(pixels).save(path)
            images.append({"path": str(path), "sha256": sha256(path), "time_s": 0.0,
                           "state": "settled_reload_initial" if manifest.get("state_kind") == "settled" else "constructed_initial",
                           "camera": camera_configs[index]})
            self.capture.suspend(product)
        for kind, view, names in views:
            if kind == "articulation":
                transforms = np.asarray(view._physics_view.get_link_transforms(), dtype=float)[0]
                current = {name: tensor_transform(value) for name, value in zip(names, transforms)}
            else:
                xyz, q = view.get_world_poses()
                current = {names[0]: tensor_transform([*xyz[0], *q[0][1:], q[0][0]])}
            if any(not np.allclose(current[name], initial["body_poses"][name], atol=1e-7) for name in names):
                raise RuntimeError("Initial rendering advanced native physics state")
        duration = float(request["duration_s"])
        terminal_mode = request.get('mode') == 'terminal_settle'
        if terminal_mode and duration != 10.:
            raise ValueError('Terminal release is capped at ten simulation seconds')
        self.sim.play()
        minimum_steps = int(math.ceil(duration / dt))
        phase_times["load_initialize_seconds"] = time.monotonic() - evaluation_started
        simulation_started = time.monotonic()
        steps = int(math.ceil((duration if terminal_mode else max(duration, self.config["final_duration_s"])) / dt))
        trace, frame_paths = [initial], []
        failure, interim = None, None
        terminal_report = None
        phase_times['measurement_reduction_seconds'] = 0.
        from scene_refinement.physics import PrefixBound, prunable, pruned_report, needs_extension
        witness = candidate.get("pruning_witness")
        prefix_bound = (PrefixBound(manifest, self.config) if request.get("mode") == "physics"
                        and duration < self.config["final_duration_s"] - dt and witness
                        and self.config.get("certified_early_exit") else None)
        if prefix_bound:
            prefix_bound.add(initial)
        compressed = bool(self.config.get("compress_trajectories"))
        trace_path = folder / ("trajectory.jsonl.xz" if compressed and self.config.get("compact_evidence") else
                               "trajectory.jsonl.gz" if compressed else "trajectory.jsonl")
        final_run = terminal_mode or duration >= self.config["final_duration_s"] - dt
        from scene_refinement.evidence_writer import EvidenceWriter
        with EvidenceWriter(trace_path, compressed=compressed,
                            asynchronous=bool(self.config.get("async_evidence"))) as handle:
            handle.write(json.dumps(initial, allow_nan=False) + "\n")
            for index in range(1, steps + 1):
                step_started = time.monotonic()
                self.sim.step(render=False, update_fabric=True)
                phase_times["native_step_seconds"] += time.monotonic() - step_started
                sample = measure(index * dt)
                trace.append(sample)
                handle.write(json.dumps(sample, allow_nan=False, separators=(",", ":")) + "\n")
                if prefix_bound and index <= minimum_steps:
                    bound = prefix_bound.add(sample)
                    if self.config.get("certified_early_exit") and prunable(bound, witness, bool(prefix_bound.issues())):
                        failure = pruned_report(bound, witness, sample['time_s'], duration, prefix_bound.issues(), 'native')
                        break
                if index % 240 == 0:
                    handle.flush()
                    atomic_json(folder / "progress.json", {"time_s": index * dt, "required_s": duration,
                                                           "request_id": request["request_id"]})
                if final_run and renderers and (index == steps or index % max(1, round(1 / dt / self.config["video_fps"])) == 0):
                    render_started = time.monotonic()
                    rgb, product = renderers[0]
                    product.hydra_texture.set_updates_enabled(True)
                    pixels = capture_rgb(rep.orchestrator, rgb)
                    path = folder / "frames" / f"frame_{len(frame_paths):05d}.png"
                    path.parent.mkdir(exist_ok=True)
                    handle.frame(pixels, path)
                    frame_paths.append(path)
                    self.capture.suspend(product)
                    phase_times["render_seconds"] += time.monotonic() - render_started
                if not terminal_mode and index == minimum_steps and minimum_steps < steps:
                    reduction_started = time.monotonic()
                    interim = evaluate_scene_trace(trace, manifest, duration_s=duration, config=self.config,
                                                   native_articulations=native_articulations)
                    phase_times['measurement_reduction_seconds'] += time.monotonic() - reduction_started
                    if interim.get('status') == 'environment_error':
                        raise RuntimeError(str(interim.get('errors')))
                    extend = needs_extension(interim, self.config)
                    if not extend:
                        break
                if terminal_mode and (index == steps or index % max(1, round(.1/dt)) == 0):
                    from scene_refinement.terminal_settle import evaluate_window
                    terminal_report = evaluate_window(trace, manifest, self.config, native_articulations)
                    if terminal_report and terminal_report.get('status') == 'environment_error':
                        raise RuntimeError(str(terminal_report.get('errors')))
                    if terminal_report and terminal_report.get('window_passed'):
                        break
        phase_times["simulation_loop_seconds"] = time.monotonic() - simulation_started
        if terminal_mode:
            if terminal_report is None:
                raise RuntimeError('Terminal release has no complete measured window')
            report = terminal_report
        elif failure:
            report = failure
        elif interim is not None and index == minimum_steps:
            report = interim
        else:
            reduction_started = time.monotonic()
            report = evaluate_scene_trace(trace, manifest, duration_s=trace[-1]['time_s'], config=self.config,
                                          native_articulations=native_articulations)
            phase_times['measurement_reduction_seconds'] += time.monotonic() - reduction_started
        report.update(native_duration_s=trace[-1]['time_s'], requested_duration_s=duration,
                      effective_duration_s=None if failure else trace[-1]['time_s'], speculative_extra_s=0.)
        if not failure:
            terminal_path = folder/'effective_terminal_state.json'
            write_json(terminal_path, trace[-1])
            report['terminal_state'] = {'path':str(terminal_path), 'sha256':sha256(terminal_path)}
        self.sim.pause()
        initial_images = images
        if ((report.get("passed") and manifest.get("state_kind") != "settled") or terminal_mode) and candidate.get("layout_path"):
            from scene_refinement.composition import materialize_state
            settled = materialize_state(candidate, trace[-1], native_articulations, folder / "settled", preserve_velocity=terminal_mode)
            images = []
            for index, (rgb, product) in enumerate(renderers):
                product.hydra_texture.set_updates_enabled(True)
                pixels = capture_rgb(rep.orchestrator, rgb)
                path = folder / f"settled_view_{index}.png"
                Image.fromarray(pixels).save(path)
                images.append({"path": str(path), "sha256": sha256(path), "time_s": trace[-1]["time_s"],
                               "state": "settled", "camera": camera_configs[index]})
                self.capture.suspend(product)
            report["terminal_candidate" if terminal_mode else "settled_candidate"] = settled
        from scene_refinement.framing import in_camera
        frame_poses = trace[-1]["body_poses"] if report.get("passed") or terminal_mode else initial["body_poses"]
        coverage = []
        for camera_config in camera_configs:
            view = {}
            for name, meshes in original.items():
                pose = np.asarray(frame_poses[name])
                vertices = np.concatenate([m["vertices"] for m in meshes]) @ pose[:3, :3].T + pose[:3, 3]
                view[name] = in_camera(vertices, camera_config)
            coverage.append(view)
        report.update(semantic_frame_complete=bool(coverage) and all(all(v.values()) for v in coverage),
            frustum_coverage=coverage,
            images=images, image_paths=[item["path"] for item in images], initial_images=initial_images,
            trace={"path": str(trace_path), "sha256": sha256(trace_path)}, proxy_fidelity=fidelity,
            initial_state_receipt={"path": str(folder / "initial_state_receipt.json"),
                                   "sha256": sha256(folder / "initial_state_receipt.json")},
            cooked_collisions=cooking_evidence,
            cooked_geometry={"path": str(cooked_path), "sha256": sha256(cooked_path)},
            acceptance_complete=report.get("acceptance_complete", report["complete"]),
            native_articulations=native_articulations, external_joints=external_joints,
            physics_scene=physics_scene_evidence,
            native_dynamics=True, actuation=False, stage_reset="independent_USD_reload",
            initial_state="candidate_authored_state", cameras=camera_configs,
            observation_scope="inter_object_collisions_support_major_orientation_stability",
            source_geometry_gate=False, native_self_collision_gate=False,
            rollout_extended=trace[-1]["time_s"] > duration + dt/2)
        if frame_paths:
            report["video"] = encode_video(frame_paths, folder / "simulation.mp4", self.config["video_fps"], root=folder)
            listing = Path(report["video"]["frame_list_path"])
            frame_manifest = json.loads(listing.read_text())
            frame_manifest["frame_sha256"] = [sha256(path) for path in frame_paths]
            atomic_json(listing, frame_manifest)
            report["video"]["frame_list_sha256"] = sha256(listing)
        if manifest.get("diagnostic_only") and not terminal_mode:
            report.update(diagnostic_only=True, passed=False, status="diagnostic_only", acceptance_complete=False)
            report["violation"] = max(float(report["violation"]), 1e6)
            report["issues"].append({"kind": "unresolved_layout_diagnostic_not_eligible_for_selection"})
        phase_times["evaluate_seconds"] = time.monotonic() - evaluation_started
        report["timings"] = phase_times
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--gpu-lock-fd", type=int)
    parser.add_argument("--gpu-lock-path", type=Path, default=Path('/tmp/neoworld_isaac_gpu.lock'))
    args = parser.parse_args(argv)
    queue = args.queue.resolve()
    startup = json.loads((queue / "startup.json").read_text())
    # Native children own separate process groups. If their supervisor crashes,
    # its inherited lease must not leave this idle GPU worker alive indefinitely.
    import threading, signal
    from scene_refinement.process_identity import owner_alive
    owner_stop = threading.Event()
    def watch_owner():
        while not owner_stop.wait(1.):
            if not owner_alive(startup['coordinator_identity']):
                atomic_json(queue/'coordinator_lost.json', startup['coordinator_identity'])
                os.killpg(os.getpgrp(), signal.SIGTERM)
                return
    threading.Thread(target=watch_owner, daemon=True).start()
    app = runtime = None
    from urdf_core.isaac_cache import gpu_worker_slot
    standalone_overlap = args.gpu_lock_fd is None and os.environ.get('NEOWORLD_SCENE_ALLOW_EXTERNAL_CONTENTION') == '1'
    if standalone_overlap:
        args.gpu_lock_path = queue/'private_gpu.lock'
        if not os.environ.get('NEOWORLD_ISAAC_CACHE_ROOT'):
            raise ValueError('Standalone overlap requires an isolated writable cache root')
    if args.gpu_lock_path != Path('/tmp/neoworld_isaac_gpu.lock') and args.gpu_lock_fd is None and not standalone_overlap:
        raise ValueError('A private admission lease must be inherited from its coordinator')
    lock_context = gpu_worker_slot(args.gpu_lock_fd, _lock_path=args.gpu_lock_path)
    lock_entered = False
    try:
        atomic_json(queue / "progress.json", {"stage": "waiting_gpu_lock"})
        lock_fd = lock_context.__enter__()
        lock_entered = True
        atomic_json(queue/'gpu_admission.json', {'lock_path': str(args.gpu_lock_path),
            'allow_external_contention': args.gpu_lock_path != Path('/tmp/neoworld_isaac_gpu.lock')})
        assert_source_binding(startup["source_hashes"])
        if startup["config"].get("shared_shader_cache"):
            from urdf_core.isaac_cache import prepare_runtime_cache
            prepare_runtime_cache(queue, Path(startup["launcher"]), gpu_lock_fd=lock_fd,
                                  _lock_path=args.gpu_lock_path)
        atomic_json(queue / "progress.json", {"stage": "simulation_app_launch"})
        from isaacsim import SimulationApp
        portable = queue / "runtime/kit"
        portable.mkdir(parents=True, exist_ok=True)
        extra = ["--portable-root", str(portable), f"--/app/tokens/cache={queue / 'runtime/cache'}",
                 f"--/app/tokens/data={queue / 'runtime/data'}", f"--/app/tokens/logs={queue}",
                 "--/app/extensions/fsWatcherEnabled=0", "--/app/extensions/registryEnabled=0",
                 "--/app/extensions/syncRegistryOnStartup=0", "--/app/extensions/preferLocalVersions=1",
                 f"--/log/file={queue / 'kit.log'}", "--/physics/updateToUsd=true", "--/physics/updateVelocitiesToUsd=true"]
        from urdf_core.isaac_admission import thread_args
        extra.extend(thread_args())
        sys.argv = [sys.argv[0], "--portable-root", str(portable)]
        app = SimulationApp({"headless": True, "create_new_stage": False, "multi_gpu": False,
                             "width": 640, "height": 480, "extra_args": extra})
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.asset.importer.urdf")
        app.update()
        runtime = SceneRuntime(app, startup["config"])
        atomic_json(queue / "ready.json", {"status": "ready", "session_id": startup["session_id"], "pid": os.getpid()})
        while not (queue / "stop.json").exists():
            requests = sorted(path for path in queue.glob("request_*.json")
                              if not (queue / path.name.replace("request_", "response_")).exists())
            if not requests:
                time.sleep(.05)
                continue
            for path in requests:
                request = json.loads(path.read_text())
                try:
                    report = evaluate_bound_request(runtime, request, startup)
                except Exception as exc:
                    report = evaluation_error_report(exc)
                report.update(schema_version="scene_physics_report.v1", request_id=request["request_id"],
                              request_digest=request["request_digest"], session_id=startup["session_id"],
                              worker_log=str(queue / "worker.log"), source_hashes=startup["source_hashes"],
                              runtime_binding=startup.get("runtime_binding"))
                atomic_json(queue / f"response_{request['request_id']}.json", report)
    except Exception as exc:
        atomic_json(queue / "ready.json", {"status": "environment_error", "session_id": startup["session_id"],
                    "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        return 3
    finally:
        if runtime:
            runtime.close_trial()
        if app:
            app.close()
        if lock_entered:
            lock_context.__exit__(None, None, None)
        owner_stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
