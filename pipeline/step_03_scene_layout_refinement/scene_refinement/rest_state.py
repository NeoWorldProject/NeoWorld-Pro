"""Forward kinematics for the measured support-local publication state."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def joint_frame(joint, index):
    position = getattr(joint, f"GetLocalPos{index}Attr")().Get()
    quaternion = getattr(joint, f"GetLocalRot{index}Attr")().Get()
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix()
    return result




def apply_joint_axis_repairs(stage, repairs):
    for path, repair in repairs.items():
        attr = stage.GetPrimAtPath(path).GetAttribute('physics:axis')
        if not attr or attr.Get() != repair['original_axis']:
            raise ValueError('Joint-axis repair source changed: ' + path)
        attr.Set(repair['axis'])


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
        body.CreateVelocityAttr(Gf.Vec3f(0))
        body.CreateAngularVelocityAttr(Gf.Vec3f(0))
    values = asset["support_rest_pose"]["joint_positions"]
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint) and prim.GetName() in values:
            linear = prim.IsA(UsdPhysics.PrismaticJoint)
            kind = "linear" if linear else "angular"
            value = values[prim.GetName()] * scale if linear else np.degrees(values[prim.GetName()])
            prim.CreateAttribute(f"state:{kind}:physics:position", Sdf.ValueTypeNames.Float, custom=False).Set(float(value))
            prim.CreateAttribute(f"state:{kind}:physics:velocity", Sdf.ValueTypeNames.Float, custom=False).Set(0.)


def published_state(stage, bodies, root_link, paths, authored_transforms):
    """Explicit USD joint states take precedence over explicit contract/URDF states."""
    from pxr import UsdPhysics
    from .protocol import read_json
    from .metrics import _pose
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
