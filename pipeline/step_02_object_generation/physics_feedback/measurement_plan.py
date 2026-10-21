"""Precomputed URDF geometry for every native sample; no temporal decimation."""
from __future__ import annotations

import math
import numpy as np


class MeasurementPlan:
    def __init__(self, parsed, links, root, cooked):
        from urdf_core.urdf_validation import _transform
        self.links = tuple(links)
        self.indices = {n: i for i, n in enumerate(self.links)}
        self.root_index = self.indices[root]
        pending = list(parsed['joints'])
        visited = {parsed['root']}
        self.joints = []
        self.joint_names = {j['name'] for j in pending}
        while pending:
            ready = [j for j in pending if j['parent'] in visited]
            if not ready:
                raise ValueError('URDF joints do not form a rooted tree')
            for joint in ready:
                axis = np.asarray(joint.get('axis', [1., 0., 0.]), dtype=float)
                if joint['type'] in ('continuous', 'revolute'):
                    axis = axis / np.linalg.norm(axis)
                self.joints.append((joint, self.indices[joint['parent']],
                    self.indices[joint['child']], _transform(joint['origin']), axis))
                visited.add(joint['child']); pending.remove(joint)
        self.vertices = [(self.indices[n], np.concatenate([m['vertices'] for m in meshes]))
                         for n, meshes in cooked.items()]

    def fk(self, positions):
        if set(positions) - self.joint_names:
            raise ValueError('unknown URDF joint coordinate')
        frames = np.repeat(np.eye(4)[None], len(self.links), axis=0)
        for j, parent, child, origin, axis in self.joints:
            q = positions.get(j['name'], 0.)
            if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q):
                raise ValueError('joint coordinate must be finite')
            kind = j['type']
            if kind == 'fixed' and q != 0:
                raise ValueError('fixed joint cannot move')
            if kind in ('prismatic', 'revolute') and not j['limit']['lower']-1e-10 <= q <= j['limit']['upper']+1e-10:
                raise ValueError('joint coordinate outside URDF limits')
            motion = np.eye(4)
            if kind == 'prismatic':
                motion[:3, 3] = axis * q
            elif kind in ('revolute', 'continuous'):
                x, y, z = axis * math.sin(q/2); w = math.cos(q/2)
                motion[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
            frames[child] = frames[parent] @ origin @ motion
        return frames

    def geometry(self, transforms, positions):
        from scipy.spatial.transform import Rotation
        values = np.asarray(transforms, dtype=float)
        if values.shape != (len(self.links), 7) or not np.isfinite(values).all():
            raise ValueError('Invalid Isaac link transforms')
        x, y, z, w = values[:, 3:].T
        norm = np.sqrt(x*x+y*y+z*z+w*w)
        if np.any(norm < 1e-12):
            raise ValueError('Zero quaternion from Isaac')
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        poses = np.repeat(np.eye(4)[None], len(self.links), axis=0)
        poses[:, :3, :3] = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]).transpose(2, 0, 1)
        poses[:, :3, 3] = values[:, :3]
        expected = self.fk(positions)
        relative = np.linalg.inv(poses[self.root_index]) @ poses
        error = float(np.linalg.norm(relative[:, :3, 3]-expected[:, :3, 3], axis=1).max())
        rotation = expected[:, :3, :3].transpose(0, 2, 1) @ relative[:, :3, :3]
        angular = float(Rotation.from_matrix(rotation).magnitude().max())
        bottom = min(float((vertices @ poses[i, :3, :3].T + poses[i, :3, 3])[:, 2].min())
                     for i, vertices in self.vertices)
        return dict(zip(self.links, poses)), bottom, error, angular
