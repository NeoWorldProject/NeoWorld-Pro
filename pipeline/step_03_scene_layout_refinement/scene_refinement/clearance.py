"""Bounded, translation-only initialization from measured collision geometry.

This proposes a layout; it never certifies physics. Only siblings on the same
real support plane are separated. Inside/attached relations are not unpacked.
"""
from itertools import combinations
import math

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.spatial import ConvexHull, QhullError

from .protocol import read_json
from .trajectory import read_trace


def separate_footprints(footprints, bases, specs, parameters, clearance_m):
    """Smallest normalized translation that separates overlapping convex envelopes.

    Envelopes are conservative proposal geometry, not a collision acceptance
    test. Bounds and original major ordering are preserved; no resizing/yaw.
    """
    ids = sorted(footprints)
    variable = sorted(set(ids) & set(specs) & set(bases))
    if not variable:
        return None, {'status': 'no_movable_siblings'}
    sigmas = np.array([specs[o]['sigma'][axis] for o in variable for axis in (0, 1)])
    low = np.array([specs[o]['bounds'][axis][0]-parameters[o][axis] for o in variable for axis in (0, 1)]) / sigmas
    high = np.array([specs[o]['bounds'][axis][1]-parameters[o][axis] for o in variable for axis in (0, 1)]) / sigmas
    rows, rhs, pairs = [], [], []
    for a, b in combinations(ids, 2):
        pa, pb = np.asarray(footprints[a]), np.asarray(footprints[b])
        axes = []
        for points in (pa, pb):
            try:
                hull = points[ConvexHull(points).vertices]
            except QhullError:
                return None, {'status': 'degenerate_footprint', 'object_ids': [a, b]}
            edges = np.roll(hull, -1, axis=0)-hull
            axes.extend(np.column_stack([-edges[:, 1], edges[:, 0]]) / np.linalg.norm(edges, axis=1)[:, None])
        choices = []
        for axis in axes:
            if (pb.mean(0)-pa.mean(0)) @ axis < 0:
                axis = -axis
            gap = float((pb @ axis).min()-(pa @ axis).max())
            row = np.zeros(len(sigmas))
            for oid, sign in ((a, -1), (b, 1)):
                if oid in variable:
                    offset = 2*variable.index(oid)
                    row[offset:offset+2] = sign * (axis @ np.asarray(bases[oid])) * sigmas[offset:offset+2]
            required = clearance_m-gap
            capacity = float(np.where(row >= 0, row*high, row*low).sum())
            norm = float(row @ row)
            if norm > 1e-20 and required <= capacity + 1e-10:
                choices.append((max(0., required)**2/norm, -gap, row, required, axis))
        if not choices:
            return None, {'status': 'no_feasible_separation_axis', 'object_ids': [a, b]}
        _, _, row, required, axis = min(choices, key=lambda v: v[:2])
        # Also retain a separating axis for already-clear siblings, so fixing
        # one collision cannot create another on that same support plane.
        rows.append(row); rhs.append(required)
        pairs.append({'object_ids': [a, b], 'axis': axis.tolist(), 'required_motion_m': required})
    if not rows or max(rhs) <= 0:
        return None, {'status': 'already_clear'}
    matrix = np.asarray(rows); target = np.asarray(rhs)
    result = minimize(lambda x: .5*float(x @ x), np.zeros(len(sigmas)), jac=lambda x: x,
                      method='SLSQP', bounds=Bounds(low, high),
                      constraints=[LinearConstraint(matrix, target, np.inf)],
                      options={'ftol': 1e-12, 'maxiter': 200})
    if not result.success or (matrix @ result.x < target-1e-8).any():
        return None, {'status': 'bounded_projection_infeasible', 'reason': str(result.message), 'pairs': pairs}
    repaired = {oid: list(values) for oid, values in parameters.items()}
    for index, oid in enumerate(variable):
        repaired[oid][:2] = (np.asarray(parameters[oid][:2])+result.x[index*2:index*2+2]*sigmas[index*2:index*2+2]).tolist()
    return repaired, {'status': 'proposed', 'pairs': pairs, 'normalized_squared_motion': float(result.x @ result.x),
                      'clearance_m': clearance_m, 'changed_axes': ['support_u', 'support_v']}


def propose(candidate, scene, specs, config):
    """Use actual cooked initial colliders and preserve the original pose receipt."""
    physics = candidate['physics']
    if physics.get('passed') or not physics.get('trace') or not physics.get('cooked_geometry'):
        return {'parameters': None, 'status': 'not_applicable'}
    trace = read_trace(physics['trace']['path'])
    geometry = read_json(physics['cooked_geometry']['path'])['geometry']
    spec = {s['object_id']: s for s in specs}
    parameters = candidate['parameters']
    groups = {}
    for oid, obj in scene['objects'].items():
        if oid in spec and obj['relation'] == 'on' and obj.get('parent_id'):
            groups.setdefault(obj['parent_id'], []).append(oid)
    proposals, details = [], []
    for parent_id, ids in sorted(groups.items()):
        if len(ids) < 2:
            continue
        # Only repair a confirmed inter-sibling violation; ordinary close but
        # non-intersecting assets do not trigger an envelope-based relocation.
        if not any(set(i.get('object_ids', [])) <= set(ids) and len(i.get('object_ids', [])) == 2
                   and i['kind'] in ('native_contact_penetration', 'penetration') for i in physics.get('issues', [])):
            continue
        parent = scene['objects'][parent_id]
        plane = None; footprints = {}; bases = {}
        for oid in ids:
            binding = scene['bindings'].get(oid)
            if not binding:
                break
            rest = np.asarray(parent['asset']['body_rest_transforms'][binding['parent_link']])
            frame = np.asarray(parent['initial_transform'])[:3, :3] @ rest[:3, :3] @ np.asarray(binding['support_frame_local']['basis'])
            if plane is None:
                plane = frame
            if not np.allclose(frame[:, 2], plane[:, 2], atol=1e-8):
                break
            points = []
            for link, meshes in geometry.items():
                if link.split('::')[0] != oid:
                    continue
                transform = np.asarray(trace[0]['body_poses'][link])
                points.extend(np.asarray(m['vertices']) @ transform[:3, :3].T+transform[:3, 3] for m in meshes)
            if not points:
                break
            footprints[oid] = np.concatenate(points) @ plane[:, :2]
            bases[oid] = plane[:, :2].T @ frame[:, :2]
        if len(footprints) != len(ids):
            details.append({'parent_id': parent_id, 'status': 'incompatible_support_frames'})
            continue
        result, detail = separate_footprints(footprints, bases, spec, parameters,
                                            max(.002, 4*config['penetration_tolerance_m']))
        details.append({'parent_id': parent_id, **detail})
        if result is not None:
            proposals.append((ids, result))
    result = {oid: list(v) for oid, v in parameters.items()}
    for ids, proposal in proposals:
        for oid in ids:
            result[oid] = proposal[oid]
    return {'parameters': result if proposals else None, 'status': 'proposed' if proposals else 'not_applicable',
            'groups': details, 'policy': 'measured_sibling_clearance_initialization.v1',
            'acceptance': 'Requires unchanged full physics, semantic review and independent reload.'}
