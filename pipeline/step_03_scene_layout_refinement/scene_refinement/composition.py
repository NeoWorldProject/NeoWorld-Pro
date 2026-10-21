"""Prepare published assets and compose native USD candidates."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .assets import (asset_points, bake_scaled_asset, file_hash, fit_uniform_scale,
                     load_published_asset, matrix, scaled_properties, transformed, write_json)
from .support import AssetGeometryIncompatible, SupportUnresolved, resolve_binding, solve_pose, surface_catalog
from .metrics import physics_settings


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _read(value):
    from .protocol import read_json
    return read_json(value) if isinstance(value, (str, Path)) else copy.deepcopy(value)


def _rotation(angles):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("xyz", angles, degrees=True).as_matrix()


def _environment_asset(size):
    import trimesh
    mesh = trimesh.creation.box(extents=size)
    record = {"link": "support", "name": "environment_support", "vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist()}
    return {"geometry_meshes": [record], "collision_meshes": [copy.deepcopy(record)],
            "body_rest_transforms": {"support": np.eye(4).tolist()}, "root_link": "support",
            "link_properties": {}, "source_hashes": {}, "ignored_internal_link_pairs": []}




def prepare_scene(layout, asset_manifest, out_dir, resolve_binding=None, verify_assets=True,
                  diagnostic_only=False, retain_unresolved=False):
    """Validate inputs, fit source geometry, and resolve real parent-local surfaces.

    ``verify_assets=False`` is a Python-only development fixture interface; every
    supplied asset must carry synthetic_fixture=true. Production CLI never offers it.
    """
    stage1 = Path(__file__).resolve().parents[2] / "step_01_scene_parsing"
    if str(stage1) not in sys.path:
        sys.path.insert(0, str(stage1))
    from scene_core.schema import normalize_layout
    layout_input, manifest_input = layout, asset_manifest
    layout, manifest = normalize_layout(_read(layout)), _read(asset_manifest)
    if verify_assets and isinstance(layout_input, (str, Path)):
        from urdf_core.layout_handoff import verify_layout_delivery
        verify_layout_delivery(layout_input)
    records = manifest.get("objects")
    if not isinstance(records, list):
        raise ValueError("Asset manifest objects must be a list")
    records_by_id = {str(r["object_id"]): r for r in records}
    if len(records_by_id) != len(records):
        raise ValueError("Duplicate asset object IDs")
    parents = {child: parent for parent, node in layout["scene_tree"].items() for child in node["children"]}
    by_name = {o["name"]: o["object_id"] for o in layout["objects_data"]}
    objects, source_hashes = {}, {"layout_json": _hash(layout), "asset_manifest_json": _hash(manifest)}
    bounds = []
    for obj in layout["objects_data"]:
        oid, environment = obj["object_id"], obj["is_environment_support"]
        if environment:
            asset = _environment_asset(obj["obb"]["size_m"])
            scale, center, residual = 1.0, [0., 0., 0.], [0., 0., 0.]
        else:
            if oid not in records_by_id:
                raise ValueError("Missing scene asset: " + oid)
            asset = load_published_asset(records_by_id[oid], verify=verify_assets)
            if records_by_id[oid].get('visual_material_package'):
                from .visual_materials import bind
                package = records_by_id[oid]['visual_material_package']
                asset = bind(asset, package['path'], package['sha256'])
            scale, center, residual = fit_uniform_scale(asset_points(asset), obj["obb"]["size_m"])
        points = asset_points(asset)
        transform = np.eye(4); transform[:3, :3] = _rotation(obj["obb"]["rotation_xyz_deg"])
        transform[:3, 3] = np.asarray(obj["obb"]["center_m"])-transform[:3, :3] @ (np.asarray(center)*scale)
        parent_id = by_name.get(parents.get(obj["name"]))
        if obj["support_relation"] == "unknown" and not (diagnostic_only or retain_unresolved):
            raise SupportUnresolved("Unresolved support relation requires layout correction: " + oid)
        objects[oid] = {"object_id": oid, "layout_object": obj, "parent_id": parent_id,
                        "relation": obj["support_relation"], "is_environment_support": environment,
                        "asset": asset, "initial_scale": scale, "local_center": center,
                        "scale_fit_residual_m": residual, "initial_transform": transform.tolist(),
                        "rest_points": points.tolist(), "surface_catalog": surface_catalog(asset["geometry_meshes"])}
        source_hashes.update(asset["source_hashes"])
        bounds.append(transformed(points*scale, transform))
    # Parents appear first even when objects_data is shuffled.
    order, pending = [], set(objects)
    while pending:
        ready = sorted(k for k in pending if objects[k]["parent_id"] is None or objects[k]["parent_id"] in order)
        if not ready:
            raise ValueError("Scene tree has unresolved dependency/cycle")
        order.extend(ready); pending.difference_update(ready)
    bindings = {}
    unresolved = {oid: 'Unknown frozen support relation' for oid, obj in objects.items()
                  if obj['relation'] == 'unknown'}
    source_hashes.update(manifest.get("input_bindings", {}))
    for oid in order:
        obj = objects[oid]
        if obj["parent_id"] is not None and not obj["is_environment_support"]:
            try:
                bindings[oid] = resolve_binding_fn(obj, objects[obj["parent_id"]], resolve_binding)
            except SupportUnresolved as exc:
                if isinstance(exc, AssetGeometryIncompatible) and not retain_unresolved:
                    raise
                if not (diagnostic_only or retain_unresolved):
                    raise
                unresolved[oid] = str(exc)
    # A frozen upstream defect cannot acquire acceptance through a different
    # placement. Retain a loadable diagnostic scene for bounded search/delivery.
    diagnostic_only = diagnostic_only or bool(unresolved)
    points = np.concatenate(bounds)
    scene = {"schema_version": "scene_prepared.v1", "source_layout": layout,
             "objects": objects, "bindings": bindings, "object_order": order,
             "initial_diagonal_m": float(np.linalg.norm(np.ptp(points, axis=0))),
             "fixed_camera_bounds_m": {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()},
             "source_hashes": source_hashes, "synthetic_fixture": not verify_assets,
             "diagnostic_only": diagnostic_only, "unresolved_bindings": unresolved,
             "output_dir": str(Path(out_dir).resolve())}
    if isinstance(layout_input, (str, Path)):
        scene["source_layout_path"] = str(Path(layout_input).resolve())
    if isinstance(manifest_input, (str, Path)):
        scene["source_asset_manifest_path"] = str(Path(manifest_input).resolve())
    write_json(Path(out_dir) / "original_placement.json", {
        "state_kind": "input_before_height_alignment",
        "objects": {oid: {"transform": obj["initial_transform"], "scale": obj["initial_scale"],
                         "bound_initial_placement": obj.get("bound_initial_placement"),
                         "body_rest_transforms": obj["asset"]["body_rest_transforms"]}
                    for oid, obj in objects.items()}})
    write_json(Path(out_dir) / "prepared_scene.json", scene)
    return scene


# Avoid shadowing the callback argument while preserving the public keyword API.
resolve_binding_fn = resolve_binding


def _usd_transform(xform, transform):
    from pxr import Gf
    xform.MakeMatrixXform().Set(Gf.Matrix4d(np.asarray(transform).T.tolist()))


def _define_mesh(stage, path, mesh):
    from pxr import Gf, UsdGeom, UsdPhysics
    geom = UsdGeom.Mesh.Define(stage, path)
    geom.CreatePointsAttr([Gf.Vec3f(*v) for v in mesh["vertices"]])
    geom.CreateFaceVertexCountsAttr([3]*len(mesh["faces"]))
    geom.CreateFaceVertexIndicesAttr(np.asarray(mesh["faces"]).reshape(-1).tolist())
    geom.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim()).CreateCollisionEnabledAttr(True)
    return geom


def build_candidate(scene, parameters, out_dir):
    """Materialize one candidate; input layout, accepted assets and bindings stay immutable."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    if scene.get("schema_version") != "scene_prepared.v1":
        raise ValueError("Unsupported prepared scene")
    physics = physics_settings(scene.get("physics_settings"))
    hz = 1. / physics["physics_dt_s"]
    if not np.isclose(hz, round(hz), rtol=0., atol=1e-7):
        raise ValueError("Native USD physics timestep must be an integer Hz reciprocal")
    parameters = {} if parameters is None else parameters
    if not isinstance(parameters, dict) or set(parameters)-set(scene["objects"]):
        raise ValueError("Candidate parameters must map existing object IDs to four values")
    values = {oid: list(map(float, parameters.get(oid, [0, 0, 0, 0]))) for oid in scene["object_order"]}
    if any(len(value) != 4 or not np.isfinite(value).all() for value in values.values()):
        raise ValueError("Every candidate object needs exactly four finite surface parameters")
    candidate_id = _hash({"parameters": values, "source_hashes": scene["source_hashes"], "bindings": scene["bindings"],
                          "support_placement_policy": scene.get("support_placement_policy", "component_support_alignment.v1"),
                          "diagnostic_only": scene.get("diagnostic_only", False), "physics_settings": physics,
                          "fixed_camera_bounds_m": scene["fixed_camera_bounds_m"],
                          "fixed_render_bounds_m": scene.get("fixed_render_bounds_m")})[:20]
    out_dir = Path(out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    compact = bool(physics.get("compact_evidence"))
    from .protocol import write_json
    path = out_dir / ("scene.usdc" if compact else "scene.usda")
    manifest_path, layout_path = out_dir / ("scene_manifest.json.gz" if compact else "scene_manifest.json"), out_dir / "layout.json"
    if manifest_path.exists():
        previous = _read(manifest_path)
        if previous.get("candidate_id") != candidate_id:
            raise ValueError("Candidate output directory belongs to different parameters")
        for source, digest in scene["source_hashes"].items():
            if source not in {"layout_json", "asset_manifest_json"} and file_hash(source) != digest:
                raise ValueError("Source asset changed during candidate replay")
        if file_hash(path) != previous.get("usd_sha256") or file_hash(layout_path) != previous.get("layout_sha256"):
            raise ValueError("Candidate USD/layout changed; refusing replay")
        for artifact, digest in previous.get("derivative_artifacts", {}).items():
            if file_hash(artifact) != digest:
                raise ValueError("Candidate asset derivative changed; refusing replay")
        return {"candidate_id": candidate_id, "usd_path": str(path), "scene_usd_path": str(path),
                "layout_path": str(layout_path), "scene_manifest_path": str(manifest_path)}
    if path.exists():
        # A previous interrupted build has no publication manifest; rebuild only
        # this candidate's owned USD, retaining other diagnostic artifacts.
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(world.GetPrim())
    native_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    native_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    native_scene.CreateGravityMagnitudeAttr(physics["gravity_m_s2"])
    physics_prim = native_scene.GetPrim()
    physics_prim.AddAppliedSchema("PhysxSceneAPI")
    for name, value, kind in (("enableGPUDynamics", False, Sdf.ValueTypeNames.Bool),
                              ("enableCCD", True, Sdf.ValueTypeNames.Bool),
                              ("solverType", "TGS", Sdf.ValueTypeNames.Token),
                              ("enableExternalForcesEveryIteration", True, Sdf.ValueTypeNames.Bool),
                              ("broadphaseType", "MBP", Sdf.ValueTypeNames.Token),
                              ("timeStepsPerSecond", int(round(hz)), Sdf.ValueTypeNames.UInt)):
        physics_prim.CreateAttribute("physxScene:" + name, kind, custom=False).Set(value)
    UsdGeom.Xform.Define(stage, "/World/Objects"); UsdGeom.Scope.Define(stage, "/World/Attachments")
    results, attachments, poses, scales, pose_receipts = {}, [], {}, {}, {}
    layout = copy.deepcopy(scene["source_layout"])
    layout_by_id = {o["object_id"]: o for o in layout["objects_data"]}
    for oid in scene["object_order"]:
        obj, binding = scene["objects"][oid], scene["bindings"].get(oid)
        parent_id = obj["parent_id"]
        support_resolution = {}
        if obj["is_environment_support"]:
            if any(values[oid]):
                raise ValueError("Environment support geometry is frozen during CEM")
            transform, scale, anchor = np.asarray(obj["initial_transform"]), obj["initial_scale"], None
        else:
            transform, scale, anchor = solve_pose(obj, binding, scene["objects"].get(parent_id),
                poses.get(parent_id), scales.get(parent_id), values[oid],
                placement_policy=scene.get("support_placement_policy", "component_support_alignment.v1"),
                placement_receipt=support_resolution)
        poses[oid], scales[oid] = transform, scale
        from pxr import Tf
        prim_name = oid if Tf.IsValidIdentifier(oid) else Tf.MakeValidIdentifier(oid)+'_'+_hash(oid)[:12]
        root_path = "/World/Objects/" + prim_name
        root = UsdGeom.Xform.Define(stage, root_path); _usd_transform(root, transform)
        asset = obj["asset"]
        if obj["is_environment_support"]:
            body_paths = {"support": root_path + "/support"}
            _define_mesh(stage, body_paths["support"], asset["collision_meshes"][0])
            art_path = None
        else:
            derivative = bake_scaled_asset(asset, scale, out_dir / "assets" / oid,
                                           cache_dir=scene.get("derivative_cache_dir"), compact=compact)
            prim = stage.DefinePrim(root_path + "/Asset")
            prim.GetReferences().AddReference(derivative["usd_path"])
            def mapped(source_path):
                return root_path + "/Asset" + source_path[len(asset["source_default_path"]):]
            body_paths = {link: mapped(p) for link, p in asset["source_body_paths"].items()}
            art_path = mapped(asset["source_articulation_path"]) if asset["source_articulation_path"] else None
        meshes = {}
        for key in ("geometry_meshes", "collision_meshes"):
            meshes[key] = [{**m, "vertices": (np.asarray(m["vertices"])*scale).tolist()} for m in asset[key]]
        result = {"object_id": oid, "root_prim_path": root_path, "prim_path": root_path,
                  "articulation_root_path": art_path, "articulation_root": art_path,
                  "body_paths": body_paths, "root_link": asset["root_link"], "initial_transform": transform.tolist(),
                  "scale": scale, "relation": obj["relation"], "parent_id": parent_id,
                  "orientation_reference_world": (transform[:3, :3] @ np.asarray(asset["body_rest_transforms"][asset["root_link"]])[:3, :3]).tolist(),
                  "initial_diagonal_m": float(np.linalg.norm(np.ptp(np.asarray(obj["rest_points"])*obj["initial_scale"], axis=0))),
                  "is_environment_support": obj["is_environment_support"],
                  "link_properties": scaled_properties(asset["link_properties"], scale),
                  "upstream_quality": asset.get("upstream_quality"),
                  "stable_required": asset.get("stable_required", True),
                  "stability_applicability": asset.get("stability_applicability"),
                  "degradation_policy": asset.get("degradation_policy"),
                  "upstream_degradations": asset.get("upstream_degradations", []),
                  "asset_verification": asset.get("asset_verification"),
                  "ignored_internal_link_pairs": asset["ignored_internal_link_pairs"], **meshes}
        if support_resolution:
            result["support_pose_resolution"] = support_resolution
        results[oid] = result
        if asset.get("support_rest_pose"):
            measured = asset["support_rest_pose"]
            body_cache = UsdGeom.XformCache()
            pose_receipts[oid] = {
                "source_acceptance_policy": asset.get("acceptance_policy"),
                "measured_state": measured, "uniform_scale": scale,
                "parent_id": parent_id, "support_binding": binding,
                "world_support_transform": transform.tolist(),
                "authored_world_body_poses": {link: matrix(body_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p))).tolist()
                                               for link, p in body_paths.items()},
                "support_pose_resolution": support_resolution,
                "application_policy": ("measured_support_local_state_once_with_recorded_clearance"
                    if measured.get("measured_support_pose", True) else "explicit_published_state_with_real_support_placement"),
                "source_hashes": asset["source_hashes"]}
        if binding and binding["relation"] == "attached":
            parent_result = results[parent_id]
            child_body = body_paths[binding["child_link"]]
            parent_body = parent_result["body_paths"][binding["parent_link"]]
            cache = UsdGeom.XformCache()
            ctf = matrix(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(child_body)))
            ptf = matrix(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_body)))
            fixed = parent_result["is_environment_support"]
            joint_path = "/World/Attachments/" + prim_name
            joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            joint.CreateBody1Rel().SetTargets([child_body])
            if not fixed:
                joint.CreateBody0Rel().SetTargets([parent_body])
            local0 = anchor if fixed else transformed([anchor], np.linalg.inv(ptf))[0]
            local1 = transformed([anchor], np.linalg.inv(ctf))[0]
            joint.CreateLocalPos0Attr(Gf.Vec3f(*local0)); joint.CreateLocalPos1Attr(Gf.Vec3f(*local1))
            for attr, rot in ((joint.CreateLocalRot0Attr(), np.eye(3) if fixed else ptf[:3, :3].T),
                              (joint.CreateLocalRot1Attr(), ctf[:3, :3].T)):
                q = Rotation.from_matrix(rot).as_quat(); attr.Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
            joint.CreateCollisionEnabledAttr(True); joint.CreateExcludeFromArticulationAttr(True)
            attachments.append({"parent_id": parent_id, "child_id": oid, "parent_link": binding["parent_link"],
                                "child_link": binding["child_link"], "parent_body_path": parent_body,
                                "child_body_path": child_body, "world_anchor": anchor.tolist(),
                                "parent_anchor_local": transformed([anchor], np.linalg.inv(ptf))[0].tolist(),
                                "child_anchor_local": local1.tolist(), "joint_path": joint_path, "fixed_to_world": fixed})
        local_points = np.asarray(obj["rest_points"])*scale
        world_points = transformed(local_points, transform)
        updated = layout_by_id[oid]
        updated["obb"] = {"center_m": transformed([np.asarray(obj["local_center"])*scale], transform)[0].tolist(),
                          "size_m": np.ptp(local_points, axis=0).tolist(),
                          "rotation_xyz_deg": Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True).tolist()}
        updated["3d_bbox_meters"] = [*((world_points.min(axis=0)+world_points.max(axis=0))/2).tolist(),
                                        *np.ptp(world_points, axis=0).tolist()]
    stage.GetRootLayer().Save()
    write_json(layout_path, layout)
    write_json(out_dir / "assembly_pose_receipt.json", {"candidate_id": candidate_id, "objects": pose_receipts})
    derivatives = {str(p): file_hash(p) for p in sorted((out_dir / "assets").rglob("*")) if p.is_file()}
    manifest = {"schema_version": "scene_candidate.v1", "candidate_id": candidate_id,
                "physics_settings": physics, "physics_scene_path": "/World/PhysicsScene",
                "solver_policy": {"solver_type": "TGS", "enable_external_forces_every_iteration": True},
                "diagnostic_only": scene.get("diagnostic_only", False),
                "unresolved_bindings": scene.get("unresolved_bindings", {}),
                "parameters": values, "objects": results, "instances": list(results.values()),
                "bindings": scene["bindings"], "attachments": attachments,
                "support_placement_policy": scene.get("support_placement_policy", "component_support_alignment.v1"),
                "initial_diagonal_m": scene["initial_diagonal_m"], "source_hashes": scene["source_hashes"],
                "fixed_camera_bounds_m": scene["fixed_camera_bounds_m"],
                "fixed_render_bounds_m": scene.get("fixed_render_bounds_m", scene["fixed_camera_bounds_m"]),
                "synthetic_fixture": scene["synthetic_fixture"],
                "geometry_checksum": _hash({k: v["geometry_meshes"] for k, v in results.items()}),
                "usd_path": str(path), "scene_usd_path": str(path), "usd_sha256": file_hash(path),
                "layout_sha256": file_hash(layout_path), "derivative_artifacts": derivatives}
    manifest["assembly_pose_receipt"] = {"path": str(out_dir / "assembly_pose_receipt.json"),
                                         "sha256": file_hash(out_dir / "assembly_pose_receipt.json")}
    if compact:
        manifest.pop("instances", None)
    write_json(manifest_path, manifest)
    return {"candidate_id": candidate_id, "usd_path": str(path), "scene_usd_path": str(path),
            "layout_path": str(layout_path), "scene_manifest_path": str(manifest_path)}


def materialize_state(candidate, sample, articulations, out_dir, *, preserve_velocity):
    """Save one measured link/joint snapshot, preserving native topology and units."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read(candidate["scene_manifest_path"])
    layout = _read(candidate["layout_path"])
    source = Usd.Stage.Open(candidate["usd_path"])
    stage = Usd.Stage.Open(source.Flatten())
    from .metrics import _pose
    desired = {key: _pose(value) for key, value in sample["body_poses"].items()}
    required = {f"{oid}::{link}" for oid, obj in manifest["objects"].items() for link in obj["body_paths"]}
    if set(desired) != required or preserve_velocity and set(sample.get("body_velocities", {})) != required:
        raise ValueError("Terminal state has an incomplete link inventory")
    def velocity(value):
        vector = np.asarray(value, float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("Invalid measured terminal velocity")
        return vector

    by_id = {o["object_id"]: o for o in layout["objects_data"]}
    joint_receipts = {}
    for oid, obj in manifest["objects"].items():
        cache = UsdGeom.XformCache()
        root_link = obj["root_link"]
        root_body = stage.GetPrimAtPath(obj["body_paths"][root_link])
        old_body = matrix(cache.GetLocalToWorldTransform(root_body))
        outer = stage.GetPrimAtPath(obj["root_prim_path"])
        old_outer = matrix(cache.GetLocalToWorldTransform(outer))
        outer_world = desired[f"{oid}::{root_link}"] @ np.linalg.inv(old_body) @ old_outer
        parent_world = matrix(cache.GetLocalToWorldTransform(outer.GetParent()))
        _usd_transform(UsdGeom.Xformable(outer), np.linalg.inv(parent_world) @ outer_world)
        for link, path in sorted(obj["body_paths"].items(), key=lambda item: item[1].count('/')):
            prim = stage.GetPrimAtPath(path)
            cache = UsdGeom.XformCache()
            parent_world = matrix(cache.GetLocalToWorldTransform(prim.GetParent()))
            _usd_transform(UsdGeom.Xformable(prim), np.linalg.inv(parent_world) @ desired[f"{oid}::{link}"])
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body = UsdPhysics.RigidBodyAPI(prim)
                measured = sample.get("body_velocities", {}).get(f"{oid}::{link}", {})
                linear = velocity(measured["linear_m_s"]) if preserve_velocity else np.zeros(3)
                angular = np.degrees(velocity(measured["angular_rad_s"])) if preserve_velocity else np.zeros(3)
                body.CreateVelocityAttr(Gf.Vec3f(*linear))
                body.CreateAngularVelocityAttr(Gf.Vec3f(*angular))
        art = articulations.get(oid, {})
        names = art.get("dof_names", [])
        if names:
            state = sample.get("native_joint_states", {}).get(oid)
            if not state:
                raise ValueError("Settled snapshot lacks native joint state: " + oid)
            measured_names = state["joint_names"]
            positions = np.asarray(state["positions"], float).reshape(-1)
            speeds = np.asarray(state["velocities"], float).reshape(-1)
            if (len(set(measured_names)) != len(names) or set(measured_names) != set(names)
                    or positions.shape != speeds.shape or len(positions) != len(names)
                    or not np.isfinite(speeds).all()):
                raise ValueError("Terminal joint inventory or velocities differ")
            values = np.asarray([positions[measured_names.index(n)] for n in names])
            speeds = np.asarray([speeds[measured_names.index(n)] for n in names]) if preserve_velocity else np.zeros(len(names))
            if len(values) != len(names) or not np.isfinite(values).all():
                raise ValueError("Settled joint names/positions do not match")
            joints = {prim.GetName(): prim for prim in Usd.PrimRange(outer) if prim.IsA(UsdPhysics.Joint)}
            for name, value, speed in zip(names, values, speeds):
                prim = joints[name]
                kind = "linear" if prim.IsA(UsdPhysics.PrismaticJoint) else "angular"
                authored = float(value) if kind == "linear" else float(np.degrees(value))
                prim.CreateAttribute(f"state:{kind}:physics:position", Sdf.ValueTypeNames.Float, custom=False).Set(authored)
                prim.CreateAttribute(f"state:{kind}:physics:velocity", Sdf.ValueTypeNames.Float, custom=False).Set(float(speed) if kind == "linear" else float(np.degrees(speed)))
            joint_receipts[oid] = {"joint_names": names, "positions_SI": values.tolist(), "velocities_SI": speeds.tolist()}
        if obj.get("is_environment_support"):
            continue
        # The OBB follows the actual rest geometry after passive joint settling.
        rotation = outer_world[:3, :3]
        points = np.concatenate([transformed(m["vertices"], desired[f"{oid}::{m['link']}"])
                                 for m in obj["geometry_meshes"]])
        local = points @ rotation
        center = ((local.min(0) + local.max(0)) / 2) @ rotation.T
        by_id[oid]["obb"] = {"center_m": center.tolist(), "size_m": np.ptp(local, axis=0).tolist(),
                             "rotation_xyz_deg": Rotation.from_matrix(rotation).as_euler('xyz', degrees=True).tolist()}
        by_id[oid]["3d_bbox_meters"] = [*((points.min(0) + points.max(0))/2).tolist(), *np.ptp(points, axis=0).tolist()]
        obj["initial_transform"] = outer_world.tolist()
    from .protocol import write_json
    compact = bool(manifest.get("physics_settings", {}).get("compact_evidence"))
    usd_name = "scene.usdc" if compact else "scene.usda"
    manifest_name = "scene_manifest.json.gz" if compact else "scene_manifest.json"
    stage.GetRootLayer().Export(str(out_dir / usd_name))
    state_receipt = {"schema_version": "scene_settled_state.v1", "source_candidate_id": candidate["candidate_id"],
                     "time_s": sample["time_s"], "body_poses": sample["body_poses"], "joint_states": joint_receipts,
                     "body_velocities": sample.get("body_velocities") if preserve_velocity else None,
                     "velocities_policy": "measured_terminal" if preserve_velocity else "zero_on_reload", "actuation_validated": False}
    write_json(out_dir / "settled_state.json", state_receipt)
    write_json(out_dir / "layout.json", layout)
    path = out_dir / usd_name
    ident = _hash({"source": candidate["candidate_id"], "settled_state": state_receipt})
    state_kind = "measured_terminal" if preserve_velocity else "settled"
    manifest.update(candidate_id=ident, state_kind=state_kind, settled_state=state_receipt,
                    usd_path=str(path), scene_usd_path=str(path), usd_sha256=file_hash(path),
                    layout_sha256=file_hash(out_dir / "layout.json"))
    write_json(out_dir / manifest_name, manifest)
    return {"candidate_id": ident, "usd_path": str(path), "scene_usd_path": str(path),
            "layout_path": str(out_dir / "layout.json"), "scene_manifest_path": str(out_dir / manifest_name),
            "settled_state_path": str(out_dir / "settled_state.json"), "state_kind": state_kind,
            "settled_artifacts": {str(out_dir / name): file_hash(out_dir / name)
                                  for name in (usd_name, "layout.json", manifest_name, "settled_state.json")}}


def materialize_settled(candidate, sample, articulations, out_dir):
    """Measured link/joint pose with zero velocity for independent acceptance."""
    return materialize_state(candidate, sample, articulations, out_dir, preserve_velocity=False)
