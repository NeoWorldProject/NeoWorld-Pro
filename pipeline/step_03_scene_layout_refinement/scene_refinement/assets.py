"""Read-only published asset ingestion and explicit scene-only metric derivatives."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from .hashing import file_hash
from .protocol import write_json

STAGE2 = Path(__file__).resolve().parents[2] / "step_02_object_generation"


from urdf_core.dynamic_asset import SceneAssetUnavailable


def require_physical_file(path, role):
    if not path or not Path(path).is_file():
        raise SceneAssetUnavailable(
            f"scene_asset_unavailable: missing {role}: {path}; "
            "a structurally valid delivery does not imply completed physicalization")
    return Path(path)


def matrix(usd_matrix):
    """USD row-vector matrix to the pipeline's column-vector convention."""
    return np.asarray(usd_matrix, dtype=float).T


def transformed(points, transform):
    transform = np.asarray(transform, float)
    return np.asarray(points, float) @ transform[:3, :3].T + transform[:3, 3]


def asset_publication_paths(record):
    from .frozen_assets import publication_paths
    return publication_paths(record["object_run_dir"])


from urdf_core.dynamic_asset import stability_applicability


def load_published_asset(record, *, rest_state_reader=None):
    import os
    from .initial_assembly import initial_state_transforms
    rest_state_reader = rest_state_reader or initial_state_transforms
    if os.environ.get("NEOWORLD_SCENE_STATIC_CACHE"):
        from .static_preparation import shared_asset
        return shared_asset(record, rest_state_reader)
    return _load_published_asset(record, rest_state_reader=rest_state_reader)


def _load_published_asset(record, *, rest_state_reader=None, checked=None):
    from pxr import Usd, UsdGeom, UsdPhysics
    root = Path(record["object_run_dir"]).resolve()
    from .frozen_assets import verify_record
    checked = checked or verify_record(record)
    usd_path = Path(checked["usd_path"]).resolve()
    if Path(record["usd_path"]).resolve() != usd_path:
        raise ValueError("Batch manifest differs from verified dynamic USD")
    paths = checked
    applicability = stability_applicability(paths)
    asset_manifest_path = require_physical_file(paths.get("asset_manifest_path"), "physical asset_manifest.json")
    kinematics_path = require_physical_file(paths.get("kinematics_path"), "kinematics.json")
    from urdf_core.dynamic_asset import inspect_dynamic, validate_joint_profile
    dynamic = inspect_dynamic(usd_path, asset_manifest_path, kinematics_path)
    stage, manifest, spec = (dynamic[k] for k in ('stage', 'manifest', 'spec'))
    bodies, rest, arts, rigid = (dynamic[k] for k in ('bodies', 'rest', 'arts', 'rigid'))
    hashes, geometry = dynamic['hashes'], dynamic['geometry']
    default_path, roots, axis_repairs = (dynamic[k] for k in ('default_path', 'roots', 'axis_repairs'))
    hashes.update({item['path']: item['sha256'] for item in applicability['evidence']})
    from urdf_core.asset_state import select_state
    rest, rest_pose = select_state(dynamic, paths, reader=rest_state_reader)
    if record.get("verification_receipt_path"):
        hashes[record["verification_receipt_path"]] = record["verification_receipt_sha256"]
    for name in ("physics_test_contract.json", "physics_profile.json", "kinematics.json", "run_manifest.json"):
        path = root / name
        if path.exists():
            hashes[str(path)] = file_hash(path)
    for name in ("kinematics_path", "physics_profile_path", "physics_contract_path",
                 "delivery_manifest_path", "physics_report_path"):
        path = paths.get(name)
        if path and Path(path).is_file():
            hashes[str(path)] = file_hash(path)
    profile = json.loads(Path(paths['physics_profile_path']).read_text()) if paths.get('physics_profile_path') else {}
    groups = {name: {name} for name in bodies}
    for j in spec["joints"]:
        if j["type"] == "fixed":
            merged = groups[j["parent"]] | groups[j["child"]]
            for name in merged:
                groups[name] = merged
    return {"source_usd_path": str(usd_path), "source_default_path": default_path,
            "source_articulation_path": None if rigid else arts[0], "source_body_paths": bodies,
            "runtime_topology": "rigid_body" if rigid else "articulation",
            "removed_articulation_markers": arts if rigid else [],
            "body_rest_transforms": rest, "root_link": next(iter(roots)),
            "joint_axis_repairs": axis_repairs,
            "support_rest_pose": rest_pose,
            "scene_scale": manifest.get("scene_scale"),
            "stable_required": applicability['stable_required'],
            "stability_applicability": applicability,
            "physics_contract_path": paths.get("physics_contract_path"),
            "joint_friction": validate_joint_profile(dynamic, profile),
            "upstream_quality": checked.get("upstream_quality", {}) if checked else {},
            "asset_verification": checked,
            "acceptance_policy": checked.get("acceptance_policy"),
            "link_properties": manifest["link_properties"], "source_hashes": hashes,
            "degradation_policy": manifest.get("degradation_policy"),
            "upstream_degradations": manifest.get("upstream_degradations", []),
            "ignored_internal_link_pairs": [[a, b] for a in bodies for b in bodies if a < b and b in groups[a]],
            **geometry}


def asset_points(asset):
    points = np.concatenate([transformed(m["vertices"], asset["body_rest_transforms"][m["link"]])
                             for m in asset["geometry_meshes"]])
    if asset.get('degradation_policy') and np.any(np.ptp(points, axis=0) <= 0):
        points = np.concatenate([points, *[transformed(m['vertices'], asset['body_rest_transforms'][m['link']])
                                          for m in asset['collision_meshes']]])
    return points


def fit_uniform_scale(points, target_size):
    points, target = np.asarray(points, float), np.asarray(target_size, float)
    dimensions = np.ptp(points, axis=0)
    if np.any(dimensions <= 0) or target.shape != (3,) or np.any(target <= 0):
        raise ValueError("Scale fitting requires positive three-dimensional bounds")
    scale = float(dimensions @ target / (dimensions @ dimensions))
    return scale, ((points.max(axis=0) + points.min(axis=0)) / 2).tolist(), (scale * dimensions - target).tolist()


def scaled_properties(properties, scale):
    result = copy.deepcopy(properties)
    for info in result.values():
        for key, power in (("mass_kg", 3), ("volume_m3", 3), ("center_of_mass_m", 1),
                           ("inertia_kg_m2", 5), ("principal_inertias_kg_m2", 5)):
            if key in info and info[key] is not None:
                value = np.asarray(info[key]) * scale ** power
                info[key] = value.tolist()
    return result


def bake_scaled_asset(asset, scale, out_dir, cache_dir=None, compact=False):
    """Bake lengths/mass/inertia once; never edit the Stage2 source or contract."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Uniform scale must be finite and positive")
    for path, digest in asset["source_hashes"].items():
        if file_hash(path) != digest:
            raise ValueError("Accepted asset source changed: " + path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        from .protocol import digest
        key = digest({"source_hashes": asset["source_hashes"], "scale": scale, "compact": compact,
                      "usd_runtime": list(Usd.GetVersion()),
                      "rest": asset["body_rest_transforms"], "implementation": file_hash(__file__),
                      "state_implementation": file_hash(Path(__file__).with_name("rest_state.py")),
                      "hierarchy_implementation": file_hash(STAGE2 / 'urdf_core/usd_hierarchy.py')})
        cached = Path(cache_dir) / key
        import fcntl
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with (Path(cache_dir) / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (cached / "derivative.json").is_file():
                report = json.loads((cached / "derivative.json").read_text())
                if (report["source_hashes"] != asset["source_hashes"] or report["scale"] != scale
                        or file_hash(report["usd_path"]) != report["usd_sha256"]):
                    raise ValueError("Immutable derived asset cache changed")
            else:
                report = bake_scaled_asset(asset, scale, cached, compact=compact)
        alias = out_dir / ("asset.usdc" if compact else "asset.usda")
        if not alias.exists():
            alias.symlink_to(Path(report["usd_path"]).resolve())
        elif file_hash(alias) != report["usd_sha256"]:
            raise ValueError("Candidate derivative alias changed")
        write_json(out_dir / "derivative.json", report)
        return report
    source = Usd.Stage.Open(asset["source_usd_path"])
    stage = Usd.Stage.Open(source.Flatten())
    for prim in stage.Traverse():
        if prim.IsInstance():
            prim.SetInstanceable(False)
    # Instance deactivation exposes referenced points for independent baking.
    stage = Usd.Stage.Open(stage.Flatten())
    for path in asset.get("removed_articulation_markers", []):
        stage.GetPrimAtPath(path).RemoveAPI(UsdPhysics.ArticulationRootAPI)
    from .rest_state import apply_joint_axis_repairs
    apply_joint_axis_repairs(stage, asset.get('joint_axis_repairs', {}))
    if any(p.IsA(UsdPhysics.Joint) and not p.IsA(UsdPhysics.FixedJoint)
           and p.GetAttribute('physics:axis').Get() not in ('X','Y','Z') for p in stage.Traverse()):
        raise ValueError('Cannot derive a dynamic asset with an unbound joint axis')
    from urdf_core.usd_hierarchy import type_mesh_ancestors
    typed_mesh_ancestors = type_mesh_ancestors(stage)
    visited = set()
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetNumTimeSamples():
                raise ValueError("Scene asset derivatives require static rest-state USD")
        xf = UsdGeom.Xformable(prim)
        if xf:
            for op in xf.GetOrderedXformOps():
                attr = op.GetAttr()
                if str(attr.GetPath()) in visited:
                    continue
                visited.add(str(attr.GetPath()))
                value = op.Get()
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    attr.Set(type(value)(*(np.asarray(value, dtype=float) * scale).tolist()))
                elif op.GetOpType() == UsdGeom.XformOp.TypeTransform:
                    value = Gf.Matrix4d(value)
                    value.SetTranslateOnly(value.ExtractTranslation() * scale)
                    attr.Set(value)
                elif op.GetOpType() in (UsdGeom.XformOp.TypeTranslateX, UsdGeom.XformOp.TypeTranslateY, UsdGeom.XformOp.TypeTranslateZ):
                    attr.Set(float(value) * scale)
        if prim.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(prim).GetPointsAttr()
            points.Set(Vt.Vec3fArray([Gf.Vec3f(*(np.asarray(p, dtype=float) * scale).tolist()) for p in points.Get()]))
        if prim.IsA(UsdGeom.Boundable):
            extent = UsdGeom.Boundable(prim).GetExtentAttr()
            if extent.HasAuthoredValueOpinion() and extent.Get():
                extent.Set(Vt.Vec3fArray([Gf.Vec3f(*(np.asarray(p, dtype=float) * scale).tolist()) for p in extent.Get()]))
        for name in ("radius", "height", "size", "widths"):
            attr = prim.GetAttribute(name)
            if attr and attr.HasAuthoredValueOpinion():
                value = attr.Get()
                attr.Set(float(value) * scale if np.isscalar(value) else type(value)([float(v) * scale for v in value]))
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(prim)
            for attr, power in ((mass.GetMassAttr(), 3), (mass.GetCenterOfMassAttr(), 1), (mass.GetDiagonalInertiaAttr(), 5)):
                if attr.HasAuthoredValueOpinion():
                    value = attr.Get()
                    attr.Set(float(value) * scale ** power if np.isscalar(value) else type(value)(*(np.asarray(value, dtype=float) * scale ** power).tolist()))
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            for attr in (joint.GetLocalPos0Attr(), joint.GetLocalPos1Attr()):
                if attr.HasAuthoredValueOpinion():
                    value = attr.Get()
                    attr.Set(type(value)(*(np.asarray(value, dtype=float) * scale).tolist()))
            if prim.IsA(UsdPhysics.PrismaticJoint):
                for name in ("physics:lowerLimit", "physics:upperLimit", "drive:linear:physics:targetPosition",
                             "state:linear:physics:position"):
                    attr = prim.GetAttribute(name)
                    if attr and attr.HasAuthoredValueOpinion():
                        attr.Set(float(attr.Get()) * scale)
        for name in ("physxCollision:restOffset", "physxCollision:contactOffset", "physxConvexDecompositionCollision:minThickness"):
            attr = prim.GetAttribute(name)
            if attr and attr.HasAuthoredValueOpinion():
                attr.Set(float(attr.Get()) * scale)
        for name in ("physics:velocity", "physics:angularVelocity", "drive:linear:physics:targetVelocity", "drive:angular:physics:targetVelocity"):
            attr = prim.GetAttribute(name)
            if attr and attr.HasAuthoredValueOpinion() and np.any(np.asarray(attr.Get()) != 0):
                raise ValueError("A scene source cannot contain nonzero velocity targets or initial velocity")
    if asset.get("support_rest_pose"):
        from .rest_state import author_measured_state
        author_measured_state(stage, asset, scale)
    path = out_dir / ("asset.usdc" if compact else "asset.usda")
    stage.GetRootLayer().Export(str(path))
    report = {"schema_version": "scene_asset_derivative.v1", "source_hashes": asset["source_hashes"],
              "scale": float(scale), "usd_path": str(path.resolve()), "usd_sha256": file_hash(path),
              "typed_mesh_ancestors": typed_mesh_ancestors,
              "joint_axis_repairs": asset.get('joint_axis_repairs', {}),
              "runtime_topology": asset.get("runtime_topology"),
              "removed_articulation_markers": asset.get("removed_articulation_markers", []),
              "mass_policy": "constant_density", "passive_damping_policy": "preserve_source_SI",
              "acceptance_scope": "requires_scene_release_penetration_stability_validation",
              "link_properties": scaled_properties(asset["link_properties"], scale)}
    write_json(out_dir / "derivative.json", report)
    return report
