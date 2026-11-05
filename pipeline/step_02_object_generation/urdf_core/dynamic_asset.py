"""Shared static contract for dynamically published and imported object assets."""
import json
from pathlib import Path
import numpy as np
from .run_state import file_hash

class AssetBindingError(ValueError):
    """Recorded source content changed; never reinterpret as a quality failure."""


class SceneAssetUnavailable(ValueError):
    """An asset cannot satisfy the dynamic handoff contract."""


def matrix(value):
    return np.asarray(value, dtype=float).T


def joint_axis_repairs(stage):
    from pxr import UsdPhysics
    repairs = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.FixedJoint):
            continue
        attr = prim.GetAttribute('physics:axis')
        axis = attr.Get() if attr else None
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            raise ValueError('Unsupported necessary joint topology: ' + str(prim.GetPath()))
        if axis in (None, '') or not attr.HasAuthoredValueOpinion():
            repairs[str(prim.GetPath())] = {'original_axis': axis, 'axis': 'X',
                                          'policy': 'empty_native_axis_defaults_to_x.v1'}
        elif axis not in ('X', 'Y', 'Z'):
            raise ValueError('Invalid native joint axis: ' + str(prim.GetPath()))
    return repairs


def apply_joint_axis_repairs(stage, repairs):
    from pxr import Sdf
    for path, repair in repairs.items():
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError('Joint-axis repair source changed: ' + path)
        attr = prim.GetAttribute('physics:axis')
        if (attr.Get() if attr else None) != repair['original_axis']:
            raise ValueError('Joint-axis repair source changed: ' + path)
        if not attr:
            attr = prim.CreateAttribute('physics:axis', Sdf.ValueTypeNames.Token, custom=False)
        attr.Set(repair['axis'])


def inspect_dynamic(usd_path, asset_manifest_path, kinematics_path):
    from pxr import Usd, UsdGeom, UsdPhysics
    usd_path, asset_manifest_path, kinematics_path = map(Path, (usd_path, asset_manifest_path, kinematics_path))
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
    for key in ("geometry_meshes", "collision_meshes"):
        geometry[key] = []
        for item in manifest[key]:
            path = Path(item["path"])
            if not path.is_absolute():
                path = asset_manifest_path.parent / path
            if file_hash(path) != item["sha256"]:
                raise AssetBindingError("Geometry source hash changed: " + str(path))
            mesh = trimesh.load(str(path), force="mesh", process=False)
            if item["link"] not in bodies or not len(mesh.faces):
                raise ValueError("Invalid mesh/body binding")
            geometry[key].append({"link": item["link"], "name": item.get("object", path.stem),
                                  "role": item.get("role", "structural"), "collision_enabled": item.get("collision_enabled", True),
                                  "vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist()})
            hashes[str(path.resolve())] = item["sha256"]
    spec = json.loads(kinematics_path.read_text())
    axis_repairs = joint_axis_repairs(stage)
    if axis_repairs:
        stage = Usd.Stage.Open(stage.Flatten())
        apply_joint_axis_repairs(stage, axis_repairs)
    children = {j["child"] for j in spec["joints"]}
    roots = set(bodies) - children
    if len(roots) != 1:
        raise ValueError("Cannot identify one native root link")
    native_joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    if {p.GetName() for p in native_joints} != {j['name'] for j in spec['joints']}:
        raise SceneAssetUnavailable('Native joint inventory differs from kinematics')
    for prim in native_joints:
        joint = UsdPhysics.Joint(prim)
        for index in (0, 1):
            targets = getattr(joint, f'GetBody{index}Rel')().GetTargets()
            if len(targets) != 1 or str(targets[0]) not in bodies.values():
                raise SceneAssetUnavailable('Joint references an unavailable body: ' + str(prim.GetPath()))
            position = getattr(joint, f'GetLocalPos{index}Attr')().Get()
            rotation = getattr(joint, f'GetLocalRot{index}Attr')().Get()
            if position is None or rotation is None or not np.isfinite([*position, *rotation.GetImaginary(), rotation.GetReal()]).all():
                raise SceneAssetUnavailable('Invalid local joint frame: ' + str(prim.GetPath()))
    for key in ('geometry_meshes', 'collision_meshes'):
        if not geometry[key]:
            raise SceneAssetUnavailable('Dynamic asset has no ' + key)
    return dict(stage=stage, manifest=manifest, spec=spec, bodies=bodies, rest=rest,
                arts=arts, rigid=rigid, hashes=hashes, geometry=geometry,
                default_path=default_path, roots=roots, axis_repairs=axis_repairs)


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



def validate_joint_profile(dynamic, profile):
    names = {j['name'] for j in dynamic['spec']['joints'] if j['type'] != 'fixed'}
    values = profile.get('joints', {})
    if set(values) != names:
        raise SceneAssetUnavailable('Dynamic joint friction inventory differs from kinematics')
    for name in names:
        for key in ('static_friction_effort', 'dynamic_friction_effort', 'damping'):
            value = values[name].get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise SceneAssetUnavailable('Invalid joint friction parameter: ' + name + '::' + key)
    return values
