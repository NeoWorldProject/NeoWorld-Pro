"""Forward kinematics for the measured support-local publication state."""
from __future__ import annotations

import json
from .run_state import file_hash
from pathlib import Path

def read_json(path):
    return json.loads(Path(path).read_text())

import numpy as np
from scipy.spatial.transform import Rotation


def joint_frame(joint, index):
    position = getattr(joint, f"GetLocalPos{index}Attr")().Get()
    quaternion = getattr(joint, f"GetLocalRot{index}Attr")().Get()
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix()
    return result




from .dynamic_asset import joint_axis_repairs, apply_joint_axis_repairs


def measured_body_transforms(stage, bodies, root_link, pose):
    from pxr import UsdPhysics
    if pose.get("schema_version") not in {"support_rest_pose.v1", "support_rest_pose.v2"}:
        raise ValueError("Unsupported measured rest pose")
    root = np.asarray(pose["T_support_root"], float).copy()
    if (root.shape != (4, 4) or not np.isfinite(root).all()
            or not np.allclose(root[3], [0, 0, 0, 1])
            or not np.allclose(root[:3, :3].T @ root[:3, :3], np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(root[:3, :3]), 1.)):
        raise ValueError("Measured rest transform must be rigid")
    # v1 stores the ground trial's horizontal drift. The scene chooses the
    # support-plane origin; height and the complete measured rotation survive.
    if pose.get("measured_support_pose", True):
        root[:2, 3] = 0.
    poses = {bodies[root_link]: root}
    pending = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    used = set()
    while pending:
        progress = False
        for prim in pending[:]:
            joint = UsdPhysics.Joint(prim)
            parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if len(parents) != 1 or len(children) != 1:
                raise ValueError("Measured scene asset cannot contain a world-fixed joint")
            parent, child = str(parents[0]), str(children[0])
            if parent not in poses:
                continue
            if child in poses:
                raise ValueError("Asset joint graph is not a rooted tree")
            motion = np.eye(4)
            if not prim.IsA(UsdPhysics.FixedJoint):
                name = prim.GetName()
                q = float(pose["joint_positions"][name])
                if not np.isfinite(q):
                    raise ValueError("Nonfinite measured joint coordinate")
                axis_name = prim.GetAttribute("physics:axis").Get()
                axis = np.eye(3)[{"X": 0, "Y": 1, "Z": 2}[axis_name]]
                if prim.IsA(UsdPhysics.PrismaticJoint):
                    motion[:3, 3] = axis * q
                elif prim.IsA(UsdPhysics.RevoluteJoint):
                    motion[:3, :3] = Rotation.from_rotvec(axis * q).as_matrix()
                else:
                    raise ValueError("Unsupported measured joint type: " + str(prim.GetPath()))
                used.add(name)
            poses[child] = poses[parent] @ joint_frame(joint, 0) @ motion @ np.linalg.inv(joint_frame(joint, 1))
            pending.remove(prim)
            progress = True
        if not progress:
            raise ValueError("Cannot resolve measured articulation topology")
    if used != set(pose["joint_positions"]) or set(poses) != set(bodies.values()):
        raise ValueError("Measured joint/link coverage differs from the USD")
    return {name: poses[path].tolist() for name, path in bodies.items()}


def author_measured_state(stage, asset, scale):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    for link, path in sorted(asset["source_body_paths"].items(), key=lambda item: item[1].count("/")):
        prim = stage.GetPrimAtPath(path)
        desired = np.asarray(asset["body_rest_transforms"][link]).copy()
        desired[:3, 3] *= scale
        parent = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())).T
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d((np.linalg.inv(parent) @ desired).T.tolist()))
        body = UsdPhysics.RigidBodyAPI(prim)
        velocity = asset['support_rest_pose'].get('link_velocities', {}).get(link, {})
        body.CreateVelocityAttr(Gf.Vec3f(*(np.asarray(velocity.get('linear_m_s', [0, 0, 0])) * scale)))
        body.CreateAngularVelocityAttr(Gf.Vec3f(*np.degrees(velocity.get('angular_rad_s', [0, 0, 0]))))
    values = asset["support_rest_pose"]["joint_positions"]
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint) and prim.GetName() in values:
            linear = prim.IsA(UsdPhysics.PrismaticJoint)
            kind = "linear" if linear else "angular"
            value = values[prim.GetName()] * scale if linear else np.degrees(values[prim.GetName()])
            prim.CreateAttribute(f"state:{kind}:physics:position", Sdf.ValueTypeNames.Float, custom=False).Set(float(value))
            speed = asset['support_rest_pose'].get('joint_velocities', {}).get(prim.GetName(), 0.)
            speed = speed * scale if linear else np.degrees(speed)
            prim.CreateAttribute(f"state:{kind}:physics:velocity", Sdf.ValueTypeNames.Float, custom=False).Set(float(speed))


def terminal_state(paths, bodies, root_link, joint_names, hashes):
    """Use a bound free-ground observation without claiming it is a stable rest pose."""
    from pathlib import Path
    from .run_state import file_hash
    from physics_feedback.trajectory_retention import load_trace
    from physics_feedback.support_rest_pose import _rigid_matrix
    report_path = paths.get('physics_report_path')
    if not report_path:
        return None, 'no_bound_native_report'
    report = read_json(report_path)
    if report.get('delivered_usd_sha256') != file_hash(paths['usd_path']):
        raise ValueError('Terminal observation belongs to another asset')
    trials = [t for t in report.get('trials', []) if t.get('kind') == 'ground_settle']
    for trial in reversed(trials):
        path = Path(trial['trajectory_path'])
        if file_hash(path) != trial['trajectory_sha256']:
            raise ValueError('Terminal trajectory changed')
        hashes[str(path)] = trial['trajectory_sha256']
        for ref in trial.get('trajectory_storage', {}).get('dependencies', []):
            if file_hash(ref['path']) != ref['sha256']:
                raise ValueError('Terminal trajectory dependency changed')
            hashes[ref['path']] = ref['sha256']
        trace = load_trace(path)
        if not len(trace):
            continue
        sample = trace[-1]
        try:
            transforms = sample['link_transforms']
            positions, velocities = sample['joint_positions'], sample['joint_velocities']
            link_velocities = sample['link_velocities']
            if set(transforms) != set(bodies) or set(link_velocities) != set(bodies):
                raise ValueError('Incomplete link state')
            if set(positions) != set(joint_names) or set(velocities) != set(joint_names):
                raise ValueError('Incomplete joint state')
            transforms = {name: _rigid_matrix(value).tolist() for name, value in transforms.items()}
            if not np.isfinite([*positions.values(), *velocities.values(), sample['time_s']]).all():
                raise ValueError('Nonfinite terminal state')
            for value in link_velocities.values():
                for key in ('linear_m_s', 'angular_rad_s'):
                    if np.asarray(value[key]).shape != (3,) or not np.isfinite(value[key]).all():
                        raise ValueError('Invalid terminal velocity')
        except (KeyError, ValueError, TypeError) as error:
            return None, 'invalid_terminal_state: ' + str(error)
        pose = {'schema_version': 'support_rest_pose.v2',
                'T_support_root': transforms[root_link], 'joint_positions': positions,
                'joint_velocities': velocities, 'link_velocities': link_velocities,
                'measured_support_pose': False, 'stability_verified': False,
                'state_provenance': 'measured_object_terminal_state',
                'source_time_s': sample['time_s'], 'source_trial_id': trial['id'],
                'source_trial_passed': trial.get('passed'),
                'source_terminated_early': trial.get('terminated_early'),
                'source_trajectory': {'path': str(path), 'sha256': trial['trajectory_sha256']}}
        return (transforms, pose), None
    return None, 'no_free_ground_terminal_observation'


def published_state(stage, bodies, root_link, paths, authored_transforms):
    """Explicit USD joint states take precedence over explicit contract/URDF states."""
    from pxr import UsdPhysics
    from pathlib import Path
    contract = read_json(paths['physics_contract_path']) if paths.get('physics_contract_path') and Path(paths['physics_contract_path']).is_file() else {}
    positions, sources = {}, {}
    missing_authored = False
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.FixedJoint):
            continue
        if not (prim.IsA(UsdPhysics.PrismaticJoint) or prim.IsA(UsdPhysics.RevoluteJoint)):
            raise ValueError('Unsupported necessary joint topology: ' + str(prim.GetPath()))
        name = prim.GetName()
        linear = prim.IsA(UsdPhysics.PrismaticJoint)
        attr = prim.GetAttribute(f'state:{"linear" if linear else "angular"}:physics:position')
        if attr and attr.HasAuthoredValueOpinion() and attr.Get() is not None:
            positions[name] = float(attr.Get()) if linear else float(np.radians(attr.Get()))
            sources[name] = 'published_usd_explicit_state'
        elif name in contract.get('joints', {}) and 'initial_position' in contract['joints'][name]:
            positions[name] = float(contract['joints'][name]['initial_position'])
            sources[name] = 'explicit_contract_initial_state'
            missing_authored = True
        else:
            raise ValueError('No explicit initial position for joint: ' + name)
        if not np.isfinite(positions[name]):
            raise ValueError('Nonfinite joint state: ' + name)
    root = _pose(authored_transforms[root_link])
    pose = {'schema_version': 'support_rest_pose.v2', 'T_support_root': root.tolist(),
            'joint_positions': positions, 'joint_state_sources': sources,
            'state_provenance': 'explicit_published_state', 'measured_support_pose': False}
    if missing_authored:
        transforms = measured_body_transforms(stage, bodies, root_link, pose)
    else:
        transforms = {name: _pose(value).tolist() for name, value in authored_transforms.items()}
    return transforms, pose


def _pose(value):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
            or np.linalg.det(pose[:3, :3]) < .999):
        raise ValueError("Missing/nonfinite/nonrigid body transform")
    return pose



def select_state(dynamic, paths, *, reader=None):
    stage, bodies, roots = (dynamic[k] for k in ('stage', 'bodies', 'roots'))
    root_link = next(iter(roots))
    hashes = dynamic['hashes']
    path = paths.get('support_rest_pose_path')
    if path:
        pose = read_json(path)
        rest = (reader or measured_body_transforms)(stage, bodies, root_link, pose)
        hashes[str(path)] = file_hash(path)
        return rest, pose
    state, reason = terminal_state(paths, bodies, root_link,
        {j['name'] for j in dynamic['spec']['joints'] if j['type'] != 'fixed'}, hashes)
    if state is not None:
        return state
    rest, pose = published_state(stage, bodies, root_link, paths, dynamic['rest'])
    pose.update(terminal_state_unavailable_reason=reason, stability_verified=False)
    return rest, pose
