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


class SceneAssetUnavailable(ValueError):
    """A published delivery lacks the physical inputs required by a scene."""


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


def stability_applicability(paths):
    """Use the published object contract to classify independent ground tests."""
    observed = []
    for key in ('physics_contract_path', 'physics_profile_path'):
        path = paths.get(key)
        if not path or not Path(path).is_file():
            continue
        value = json.loads(Path(path).read_text())
        if 'stable_required' not in value:
            continue
        if type(value['stable_required']) is not bool:
            raise ValueError('Published stable_required must be a boolean')
        observed.append({'stable_required': value['stable_required'],
                         'path': str(Path(path).resolve()), 'sha256': file_hash(path)})
    if observed and any(item['stable_required'] != observed[0]['stable_required'] for item in observed):
        raise ValueError('Published stability applicability differs between contract and profile')
    return {'stable_required': observed[0]['stable_required'] if observed else True,
            'source': 'published_object' if observed else 'conservative_default', 'evidence': observed}


def load_published_asset(record, *, verify=True, rest_state_reader=None):
    import os
    from .initial_assembly import initial_state_transforms
    rest_state_reader = rest_state_reader or initial_state_transforms
    if verify and os.environ.get("NEOWORLD_SCENE_STATIC_CACHE"):
        from .static_preparation import shared_asset
        return shared_asset(record, rest_state_reader)
    return _load_published_asset(record, verify=verify, rest_state_reader=rest_state_reader)


def _load_published_asset(record, *, verify=True, rest_state_reader=None, checked=None):
    from pxr import Usd, UsdGeom, UsdPhysics
    root = Path(record["object_run_dir"]).resolve()
    if verify:
        from .frozen_assets import verify_record
        checked = checked or verify_record(record)
        usd_path = Path(checked["usd_path"]).resolve()
        if Path(record["usd_path"]).resolve() != usd_path:
            raise ValueError("Batch manifest differs from verified dynamic USD")
    else:
        if record.get("synthetic_fixture") is not True:
            raise ValueError("Unverified assets are restricted to explicit synthetic fixtures")
        usd_path = Path(record["usd_path"]).resolve()
    paths = (checked if verify else {
        'asset_manifest_path': str(root/'urdf_export/asset_manifest.json'),
        'kinematics_path': str(root/'kinematics.json'),
        'support_rest_pose_path': str(root/'urdf_export/support_rest_pose.json')})
    if not verify and (root/'final/delivery_manifest.json').is_file():
        from .frozen_assets import publication_paths
        paths = publication_paths(root)
    applicability = stability_applicability(paths)
    asset_manifest_path = require_physical_file(paths.get("asset_manifest_path"), "physical asset_manifest.json")
    kinematics_path = require_physical_file(paths.get("kinematics_path"), "kinematics.json")
    manifest = json.loads(asset_manifest_path.read_text())
    missing = {"link_properties", "geometry_meshes", "collision_meshes"} - manifest.keys()
    if missing:
        raise SceneAssetUnavailable("scene_asset_unavailable: physical asset manifest lacks " + ", ".join(sorted(missing)))
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None or not stage.GetDefaultPrim():
        raise ValueError("Dynamic USD must have a default prim")
    if UsdGeom.GetStageUpAxis(stage) != "Z" or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1):
        raise ValueError("Only metre-scale Z-up dynamic USD is supported")
    default_path = str(stage.GetDefaultPrim().GetPath())
    cache = UsdGeom.XformCache()
    bodies, rest = {}, {}
    arts = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            arts.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            name = prim.GetName()
            if name in bodies or not str(prim.GetPath()).startswith(default_path + "/") and str(prim.GetPath()) != default_path:
                raise ValueError("Asset body names must be unique and inside its default prim")
            bodies[name] = str(prim.GetPath())
            transform = matrix(cache.GetLocalToWorldTransform(prim))
            if (not np.isfinite(transform).all() or
                    not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5) or
                    not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-5)):
                raise ValueError("Native body frames must be rigid before scene scaling")
            rest[name] = transform.tolist()
    spec = json.loads(kinematics_path.read_text())
    if {link["name"] for link in spec["links"]} != set(bodies):
        raise SceneAssetUnavailable("scene_asset_unavailable: kinematics and native body inventory differ")
    rigid = len(bodies) == 1 and not spec["joints"] and not any(
        p.IsA(UsdPhysics.Joint) for p in stage.Traverse())
    if (not rigid and len(arts) != 1) or set(bodies) != set(manifest["link_properties"]):
        raise SceneAssetUnavailable("scene_asset_unavailable: asset needs a rigid body or one native articulation and all manifest bodies")
    import trimesh
    geometry = {}
    hashes = {str(usd_path): file_hash(usd_path), str(asset_manifest_path): file_hash(asset_manifest_path)}
    hashes.update({item['path']: item['sha256'] for item in applicability['evidence']})
    for key in ("geometry_meshes", "collision_meshes"):
        geometry[key] = []
        for item in manifest[key]:
            path = Path(item["path"])
            if not path.is_absolute():
                path = asset_manifest_path.parent / path
            if file_hash(path) != item["sha256"]:
                raise ValueError("Geometry source hash changed: " + str(path))
            mesh = trimesh.load(str(path), force="mesh", process=False)
            if item["link"] not in bodies or not len(mesh.faces):
                raise ValueError("Invalid mesh/body binding")
            geometry[key].append({"link": item["link"], "name": item.get("object", path.stem),
                                  "role": item.get("role", "structural"), "collision_enabled": item.get("collision_enabled", True),
                                  "vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist()})
            hashes[str(path.resolve())] = item["sha256"]
    spec = json.loads(kinematics_path.read_text())
    axis_repairs = {}
    if any(p.IsA(UsdPhysics.Joint) and not p.IsA(UsdPhysics.FixedJoint)
           and p.GetAttribute('physics:axis').Get() not in ('X', 'Y', 'Z') for p in stage.Traverse()):
        raise SceneAssetUnavailable('Current publications require explicit native joint axes')
    children = {j["child"] for j in spec["joints"]}
    roots = set(bodies) - children
    if len(roots) != 1:
        raise ValueError("Cannot identify one native root link")
    rest_pose = None
    rest_path = Path(paths["support_rest_pose_path"]) if paths.get("support_rest_pose_path") else None
    if rest_path and rest_path.is_file():
        from .initial_assembly import initial_state_transforms
        rest_pose = json.loads(rest_path.read_text())
        rest = (rest_state_reader or initial_state_transforms)(stage, bodies, next(iter(roots)), rest_pose)
        hashes[str(rest_path)] = file_hash(rest_path)
    if rest_pose is None:
        from .rest_state import published_state, apply_joint_axis_repairs
        state_stage = Usd.Stage.Open(stage.Flatten()) if axis_repairs else stage
        apply_joint_axis_repairs(state_stage, axis_repairs)
        rest, rest_pose = published_state(state_stage, bodies, next(iter(roots)), paths, rest)
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
            "stable_required": applicability['stable_required'],
            "stability_applicability": applicability,
            "physics_contract_path": paths.get("physics_contract_path"),
            "upstream_quality": checked.get("upstream_quality", {}) if checked else {},
            "asset_verification": checked if checked else {"synthetic_fixture": True},
            "acceptance_policy": checked.get("acceptance_policy") if verify else "synthetic_fixture",
            "link_properties": manifest["link_properties"], "source_hashes": hashes,
            "degradation_policy": manifest.get("degradation_policy"),
            "upstream_degradations": manifest.get("upstream_degradations", []),
            "ignored_internal_link_pairs": [[a, b] for a in bodies for b in bodies if a < b and b in groups[a]],
            "synthetic_fixture": not verify, **geometry}


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
                      "state_implementation": file_hash(Path(__file__).with_name("rest_state.py"))})
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
              "joint_axis_repairs": asset.get('joint_axis_repairs', {}),
              "runtime_topology": asset.get("runtime_topology"),
              "removed_articulation_markers": asset.get("removed_articulation_markers", []),
              "mass_policy": "constant_density", "passive_damping_policy": "preserve_source_SI",
              "acceptance_scope": "requires_scene_release_penetration_stability_validation",
              "link_properties": scaled_properties(asset["link_properties"], scale)}
    write_json(out_dir / "derivative.json", report)
    return report
