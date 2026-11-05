"""Explicit v9 motion-only acceptance; historical and ground rules are unchanged."""
from __future__ import annotations

import copy
import math

POLICY = 'object_functional_physics.v11'
from .control import MOTION_V9_RULES as RULES


def enabled(contract, kind=None):
    if not contract or contract.get('acceptance_policy') not in {POLICY}:
        return False
    if contract.get('motion_acceptance') != RULES:
        raise ValueError('v9 requires its exact frozen motion acceptance rules')
    return kind is None or kind in RULES['trial_kinds']


def freeze(contract, spec):
    contract['motion_acceptance'] = copy.deepcopy(RULES)
    contract['structural_joints'] = copy.deepcopy(spec['joints'])


def hold_tolerance(joint, contract):
    if joint['type'] == 'prismatic':
        return max(1e-5, (joint['upper'] - joint['lower']) * .2)
    return math.radians(10.)


class StructurePlan:
    """Vectorized local constraint checks, with no accumulation along a chain."""
    def __init__(self, contract):
        import numpy as np
        from urdf_core.urdf_validation import _transform
        if not enabled(contract):
            raise ValueError('Per-joint structure plan requires v9')
        self.names = tuple(contract['required_links'])
        indices = {name: i for i, name in enumerate(self.names)}
        self.joints = contract['structural_joints']
        self.parents = [indices[j['parent']] for j in self.joints]
        self.children = [indices[j['child']] for j in self.joints]
        self.origins = np.asarray([_transform(j['origin']) for j in self.joints])
        axes = np.asarray([j.get('axis', [1., 0., 0.]) for j in self.joints], dtype=float)
        self.axes = axes / np.linalg.norm(axes, axis=1)[:, None]
        self.prismatic = np.asarray([j['type'] == 'prismatic' for j in self.joints])
        self.rotating = np.asarray([j['type'] in ('revolute', 'continuous') for j in self.joints])
        self.tolerances = np.asarray([max(1e-5, (contract['joints'][j['name']]['upper'] -
            contract['joints'][j['name']]['lower']) * .2) if j['type'] == 'prismatic'
            else contract['config']['position_tolerance_m'] for j in self.joints])

    def measure(self, poses, positions):
        import numpy as np
        transforms = np.asarray([poses[n] for n in self.names], dtype=float)
        q = np.asarray([positions.get(j['name'], 0.) for j in self.joints], dtype=float)
        return self.arrays(transforms, q)

    def arrays(self, transforms, q):
        import numpy as np
        # Leading dimensions may be absent (one step) or a full trace batch.
        parent = transforms[..., self.parents, :, :]
        child = transforms[..., self.children, :, :]
        rotation = np.swapaxes(parent[..., :3, :3], -1, -2)
        actual_t = np.einsum('...nij,...nj->...ni', rotation, child[..., :3, 3] - parent[..., :3, 3])
        actual_r = rotation @ child[..., :3, :3]
        shift = self.axes * (q * self.prismatic)[..., None]
        expected_t = self.origins[:, :3, 3] + np.einsum('nij,...nj->...ni', self.origins[:, :3, :3], shift)
        angles = q * self.rotating
        x, y, z = self.axes.T
        cross = np.zeros((len(self.joints), 3, 3))
        cross[:, 0, 1], cross[:, 0, 2] = -z, y
        cross[:, 1, 0], cross[:, 1, 2] = z, -x
        cross[:, 2, 0], cross[:, 2, 1] = -y, x
        outer = self.axes[:, :, None] * self.axes[:, None, :]
        c, s = np.cos(angles)[..., None, None], np.sin(angles)[..., None, None]
        motion = c * np.eye(3) + (1 - c) * outer + s * cross
        expected_r = self.origins[:, :3, :3] @ motion
        relative_r = np.swapaxes(expected_r, -1, -2) @ actual_r
        distances = np.linalg.norm(actual_t - expected_t, axis=-1)
        angles_deg = np.degrees(np.arccos(np.clip((np.trace(relative_r, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
        if not np.isfinite(distances).all() or not np.isfinite(angles_deg).all():
            raise ValueError('Nonfinite measured joint structure')
        return distances, angles_deg

    def passed(self, distances, angles):
        import numpy as np
        return bool(np.all(distances <= self.tolerances + 1e-8) and np.all(angles <= 10. + 1e-6))
