"""Place a measured object state without applying its scene target twice."""
import numpy as np
from scipy.spatial.transform import Rotation

from .assets import asset_points, fit_uniform_scale

POLICY = 'object_state_scene_orientation.v2'


def initial_scale(asset, obj, points):
    initialized = asset.get('scene_scale')
    if initialized is None:
        return fit_uniform_scale(points, obj['obb']['size_m'])
    binding = initialized['target']
    if (binding['policy'] != 'object_scene_scale.v1' or binding['object_id'] != obj['object_id']
            or not np.array_equal(binding['target_size_m'], obj['obb']['size_m'])):
        raise ValueError('Scene size changed after object physics; repeat object validation')
    center = ((points.max(axis=0) + points.min(axis=0)) / 2).tolist()
    return 1., center, initialized['size_residual_m']


def placement(asset, obj):
    target = Rotation.from_euler('xyz', obj['obb']['rotation_xyz_deg'], degrees=True).as_matrix()
    points = asset_points(asset)
    rest = asset.get('support_rest_pose') or {}
    initialization = rest.get('scene_orientation')
    receipt = {'policy': POLICY, 'target_rotation_xyz_deg': list(obj['obb']['rotation_xyz_deg']),
               'target_applied_in_object_test': False, 'scale_initialization': asset.get('scene_scale')}
    if initialization is None or rest.get('measured_support_pose', True) is not True:
        root = np.asarray(asset['body_rest_transforms'][asset['root_link']])[:3, :3]
        scale, _, residual = initial_scale(asset, obj, points @ root)
        center = ((points.max(axis=0) + points.min(axis=0)) / 2).tolist()
        receipt.update(state_source=rest.get('state_provenance', 'explicit_published_state'),
                       stability_verified=False, applied_rotation_matrix=np.eye(3).tolist(),
                       measured_root_rotation_matrix=root.tolist())
        return np.eye(3), scale, center, residual, receipt
    from physics_feedback.scene_orientation import validate
    initialization = validate(initialization)
    if initialization['object_id'] != obj['object_id']:
        raise ValueError('Measured orientation belongs to a different scene object')
    initial = np.asarray(initialization['initial_rotation_matrix'])
    delta = target @ initial.T
    # A horizontal heading change preserves the measured support orientation.
    if not np.allclose(delta @ [0., 0., 1.], [0., 0., 1.], rtol=0, atol=1e-10):
        raise ValueError('Scene target tilt changed after object validation; repeat object physics')
    root = np.asarray(asset['body_rest_transforms'][asset['root_link']])[:3, :3]
    scale, _, residual = initial_scale(asset, obj, points @ root)
    center = ((points.max(axis=0) + points.min(axis=0)) / 2).tolist()
    receipt.update(target_applied_in_object_test=True, state_source='measured_scene_target',
                   object_initialization=initialization, applied_rotation_matrix=delta.tolist(),
                   measured_root_rotation_matrix=root.tolist(),
                   target_deviation_deg=float(np.degrees(Rotation.from_matrix(target.T @ delta @ root).magnitude())))
    return delta, scale, center, residual, receipt
