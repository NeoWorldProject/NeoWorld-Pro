"""Carry a scene target through object ground tests and measured publication."""
import copy

import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = 'object_scene_orientation.v1'
FRAME = 'generated_asset:+Z_up,-Y_front;scene:Z_up;RzRyRx'


def validate(value):
    if value is None:
        return None
    keys = {'schema_version', 'object_id', 'layout_sha256', 'target_rotation_xyz_deg',
            'initial_rotation_matrix', 'frame_convention'}
    if (not isinstance(value, dict) or set(value) != keys or value['schema_version'] != SCHEMA
            or value['frame_convention'] != FRAME or not isinstance(value['object_id'], str)
            or not value['object_id'] or not isinstance(value['layout_sha256'], str)
            or len(value['layout_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in value['layout_sha256'])):
        raise ValueError('Invalid scene orientation initialization binding')
    angles = np.asarray(value['target_rotation_xyz_deg'], float)
    matrix = np.asarray(value['initial_rotation_matrix'], float)
    if (angles.shape != (3,) or not np.isfinite(angles).all() or matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix, Rotation.from_euler('xyz', angles, degrees=True).as_matrix(),
                               rtol=0, atol=1e-12)):
        raise ValueError('Scene target rotation differs from its initialization matrix')
    return copy.deepcopy(value)


def from_context(context):
    if context is None:
        return None
    target = context['target']
    angles = target['obb']['rotation_xyz_deg']
    return validate({'schema_version': SCHEMA, 'object_id': target['object_id'],
                     'layout_sha256': context['layout']['sha256'],
                     'target_rotation_xyz_deg': list(angles),
                     'initial_rotation_matrix': Rotation.from_euler('xyz', angles, degrees=True).as_matrix().tolist(),
                     'frame_convention': FRAME})


def trial_rotation(contract, *, fixed=False):
    value = validate(contract.get('scene_orientation'))
    return np.eye(3) if fixed or value is None else np.asarray(value['initial_rotation_matrix'])


def ground_height(points, rotation):
    """Align the rotated native collider minimum, in metres, with world Z=0."""
    return -float((np.asarray(points) @ np.asarray(rotation).T)[:, 2].min())


def quaternion(contract, *, fixed=False):
    value = Rotation.from_matrix(trial_rotation(contract, fixed=fixed)).as_quat()
    return value[[3, 0, 1, 2]].tolist()
