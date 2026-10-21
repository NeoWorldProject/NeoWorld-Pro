"""Deterministic, explicitly unaccepted layouts after content repair exhaustion."""
from copy import deepcopy
import hashlib
import math
import re

from .schema import FRAME_CONSTANTS, normalize_layout, _number

POLICY = 'layout_degraded_delivery.v1'


def _serializable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'invalid_numeric_value': repr(value)}
    if isinstance(value, dict): return {k: _serializable(v) for k, v in value.items()}
    if isinstance(value, list): return [_serializable(v) for v in value]
    return value


def recover_layout(responses):
    proposals = [r for r in responses if isinstance(r, dict) and isinstance(r.get('objects_data'), list)]
    def usable(row):
        return isinstance(row, dict) and any(isinstance(row.get(k), str) and row[k].strip()
                                             for k in ('name', 'description'))
    source = max(proposals, key=lambda r: sum(usable(o) for o in r['objects_data']), default={})
    original = deepcopy(source)
    rows, changes, ids, names = [], [], set(), set()
    reserved_ids = {r['object_id'] for r in source.get('objects_data', []) if isinstance(r, dict)
                    and isinstance(r.get('object_id'), str) and re.fullmatch(r'[A-Za-z0-9_-]+', r['object_id'])}
    def vector(value, default, positive=False):
        return list(value) if (isinstance(value, list) and len(value) == 3 and
            all(_number(v) and abs(v) < 1e6 and
                (not positive or v > 0) for v in value)) else list(default)
    for index, raw in enumerate(source.get('objects_data', [])):
        if not usable(raw):
            changes.append({'source_index': index, 'reason': 'No recoverable object identity', 'action': 'omit_unidentified_row'})
            continue
        row = deepcopy(raw)
        name = row['name'].strip() if isinstance(row.get('name'), str) and row['name'].strip() else 'RecoveredObject'
        if name in names:
            name += '_' + str(index + 1)
        while name in names:
            name += '_'
        ident = row.get('object_id')
        if not isinstance(ident, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', ident) or ident in ids:
            ident = 'recovered_' + str(index + 1)
            while ident in ids or ident in reserved_ids:
                ident += '_'
        ids.add(ident); names.add(name)
        env = row.get('is_environment_support') is True
        box = row.get('2d_bbox_normalized')
        if not (isinstance(box, list) and len(box) == 4 and
                all(_number(v) for v in box) and
                0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            box = None if env else [0., 0., 1., 1.]
        obb = row.get('obb') if isinstance(row.get('obb'), dict) else {}
        size = vector(obb.get('size_m'), [.2, .2, .2], True)
        description = row.get('description')
        occlusion = row.get('occlusion_reasoning')
        row = dict(name=name, object_id=ident, description=description if isinstance(description, str) and description.strip() else name,
            is_environment_support=env, is_occluded=row.get('is_occluded') is True,
            occlusion_reasoning=occlusion if isinstance(occlusion, str) and occlusion.strip() else 'Unknown after parsing repairs',
            support_relation='root', **{'2d_bbox_normalized': box},
            geometry_reasoning='Recovered fields; explicit defaults are recorded in the degradation receipt.',
            uncertainty='Unaccepted parsing content; reconstruction quality is not established.',
            obb={'size_m': size, 'center_m': vector(obb.get('center_m'), [index * .3, 0, size[2]/2]),
                 'rotation_xyz_deg': vector(obb.get('rotation_xyz_deg'), [0, 0, 0])})
        row.pop('3d_bbox_meters', None)
        rows.append(row)
        changes.append({'source_index': index, 'object_id': ident, 'original': raw, 'recovered': deepcopy(row)})
    ground_name, ground_id = 'FallbackGround', 'fallback_ground'
    while ground_name in names: ground_name += '_'
    while ground_id in ids: ground_id += '_'
    rows.append({'object_id': ground_id, 'name': ground_name, 'description': 'Explicit fallback ground',
        'is_environment_support': True, 'is_occluded': False, 'occlusion_reasoning': '',
        'support_relation': 'root', '2d_bbox_normalized': None,
        'obb': {'center_m': [0, 0, -.05], 'size_m': [4, 4, .1], 'rotation_xyz_deg': [0, 0, 0]},
        'geometry_reasoning': 'Deterministic fallback environment', 'uncertainty': 'Not inferred from RGB'})
    tree = {o['name']: {'children': []} for o in rows}
    by_name = {o['name']: o for o in rows}
    parents = {}
    for parent, node in (source.get('scene_tree') or {}).items() if isinstance(source.get('scene_tree'), dict) else []:
        if parent not in by_name or not isinstance(node, dict) or not isinstance(node.get('children'), list): continue
        for child in node['children']:
            if not isinstance(child, str) or child not in by_name or child in parents or child == parent: continue
            ancestor = parent
            while ancestor in parents and ancestor != child: ancestor = parents[ancestor]
            if ancestor == child: continue
            parents[child] = parent
    original_rows = {r['name']: r for r in source.get('objects_data', []) if usable(r) and isinstance(r.get('name'), str)}
    for row in rows:
        if row['is_environment_support']: continue
        name = row['name']; parent = parents.get(name, ground_name)
        relation = original_rows.get(name, {}).get('support_relation')
        row['support_relation'] = relation if name in parents and relation in ('on', 'inside', 'attached') else 'on'
        tree[parent]['children'].append(name)
    result = normalize_layout({'objects_data': rows, 'scene_tree': tree,
        'scene_frame': {**FRAME_CONSTANTS, 'origin_description': 'Recovered scene or explicit fallback ground',
                        'scale_reasoning': 'Preserved finite dimensions; otherwise explicit 0.2 m fallback'},
        'inventory_changes': []})
    mode = 'degraded_layout' if any(not o['is_environment_support'] for o in rows) else 'environment_only'
    receipt = {'policy': POLICY, 'delivery_mode': mode, 'source_response': original,
               'changes': changes, 'foreground_inventory_complete': False, 'quality_passed': False}
    return result, _serializable(receipt)


def identity_stem(object_id, name):
    """Readable filename with an identity-preserving digest, independent of punctuation."""
    safe = re.sub(r'[^A-Za-z0-9_-]', '_', str(object_id))[:60] or 'object'
    digest = hashlib.sha256((str(object_id) + '\0' + str(name)).encode()).hexdigest()[:16]
    return safe + '_' + digest
