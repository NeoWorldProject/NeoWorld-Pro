"""One explicitly degraded release after exhausted search; never normal acceptance."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from .protocol import read_json, write_json, file_hash, digest

POLICY = 'scene_terminal_settle.v2_pose_stability'
MAX_SECONDS = 10.
WINDOW_SECONDS = 1.


def evaluate_window(trace, manifest, config, native_articulations=None):
    """Use only measured, post-step observations in the last continuous second."""
    from .physics import evaluate
    from scipy.spatial.transform import Rotation
    dt = config['physics_dt_s']
    tail = [s for s in trace if s['time_s'] > 0 and s['time_s'] >= trace[-1]['time_s'] - WINDOW_SECONDS - 1e-8]
    if len(tail) < 2 or tail[-1]['time_s'] - tail[0]['time_s'] < WINDOW_SECONDS - 1e-8:
        return None
    local = deepcopy(manifest)
    local['state_kind'] = 'terminal_release'
    from .metrics import _contact_ids
    owners = {path: oid for oid, obj in manifest['objects'].items() for path in obj['body_paths'].values()}
    contacts = [_contact_ids(s, owners) for s in tail]
    deviations = {}
    for oid, obj in local['objects'].items():
        if obj.get('is_environment_support'): continue
        key = f"{oid}::{obj['root_link']}"
        first, last = np.asarray(tail[0]['body_poses'][key]), np.asarray(tail[-1]['body_poses'][key])
        reference = np.asarray(obj.get('orientation_reference_world', first[:3, :3]))
        original = obj.get('pre_fallback_relation', obj)
        deviations[oid] = {'orientation_deg': float(np.degrees(Rotation.from_matrix(reference.T @ last[:3, :3]).magnitude())),
            'original_parent_id': original.get('parent_id'), 'original_relation': original.get('relation'),
            'observed_contact_objects': sorted({other for pairs in contacts for pair in pairs if oid in pair
                                                 for other in pair if other != oid}),
            'original_parent_contact_fraction': (sum(frozenset((oid, original['parent_id'])) in pairs for pairs in contacts)/len(contacts)
                                                  if original.get('parent_id') else None)}
        obj['orientation_reference_world'] = first[:3, :3].tolist()
        if obj.get('relation') != 'attached':
            obj['parent_id'], obj['relation'] = None, 'root'
    rebased = [{**s, 'time_s': i * dt} for i, s in enumerate(tail)]
    # Preserve and check the original sampling cadence before rebasing.
    if not np.allclose(np.diff([s['time_s'] for s in tail]), dt, rtol=0, atol=max(1e-8, dt*.001)):
        raise ValueError('Terminal settling trace has missing physics steps')
    report = evaluate(rebased, local, duration_s=rebased[-1]['time_s'],
                      config={**config, 'settle_window_s': WINDOW_SECONDS}, native_articulations=native_articulations)
    if report.get('status') == 'environment_error':
        return report
    passed = report['passed']
    report.update(policy=POLICY, passed=False, window_passed=passed, acceptance_complete=False,
                  status='settled_window_passed' if passed else 'terminal_not_settled',
                  window_start_s=tail[0]['time_s'], window_end_s=tail[-1]['time_s'],
                  window_duration_s=tail[-1]['time_s']-tail[0]['time_s'],
                  duration_s=tail[-1]['time_s'], independent_reload_performed=False,
                  semantic_evaluation_performed=False, original_relation_deviations=deviations)
    return report


def _bound_evidence(candidate):
    from .physics import observation_ready
    report = candidate.get('physics', {})
    if not observation_ready(report): return False
    refs = [report.get('trace'), report.get('terminal_state')]
    hashes = candidate.get('settled_artifacts', {})
    return (bool(hashes) and all(file_hash(p) == h for p, h in hashes.items()) and
            all(r and file_hash(r['path']) == r['sha256'] for r in refs))


def arrange(candidate, output, config):
    """Keep observed safe groups; put unsupported groups in a separate ground area."""
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    from .assets import matrix, transformed
    from .composition import _usd_transform, _environment_asset, _define_mesh
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    source_manifest = read_json(candidate['scene_manifest_path'])
    manifest, layout = deepcopy(source_manifest), read_json(candidate['layout_path'])
    stage = Usd.Stage.Open(Usd.Stage.Open(candidate['usd_path']).Flatten())
    objects = manifest['objects']; cache = UsdGeom.XformCache()
    for obj in objects.values():
        obj['pre_fallback_relation'] = {key: obj.get(key) for key in ('parent_id', 'relation')}
    bounds = {}
    for oid, obj in objects.items():
        points = np.concatenate([transformed(m['vertices'], matrix(cache.GetLocalToWorldTransform(
            stage.GetPrimAtPath(obj['body_paths'][m['link']])))) for m in obj['collision_meshes']])
        if not np.isfinite(points).all(): raise ValueError('Nonfinite collision placement geometry')
        bounds[oid] = np.array([points.min(0), points.max(0)])
    physics = candidate.get('physics', {})
    observed = _bound_evidence(candidate)
    bad_pairs = {frozenset(r['object_ids']) for r in physics.get('native_pair_maxima', [])
                 if r['depth_m'] > config['penetration_tolerance_m'] + 1e-12}
    edges = {}
    for oid, obj in objects.items():
        parent = obj.get('parent_id'); detail = physics.get('objects', {}).get(oid, {})
        if (observed and parent in objects and not obj.get('is_environment_support')
                and detail.get('tail_support_fraction') == 1. and frozenset((oid, parent)) not in bad_pairs
                and not any(oid in pair for pair in bad_pairs)
                and (obj.get('relation') != 'attached' or detail.get('passed'))):
            edges[oid] = parent
    def groups():
        result = {oid: {oid} for oid in objects}
        for child, parent in edges.items():
            merged = result[child] | result[parent]
            for oid in merged: result[oid] = merged
        return {tuple(sorted(v)) for v in result.values()}
    # A known conflict anywhere inside a group invalidates its retained support edges.
    for group in groups():
        if any(pair <= set(group) for pair in bad_pairs):
            edges = {c: p for c, p in edges.items() if c not in group}
    retained, movable = [], []
    for group in sorted(groups()):
        (retained if any(objects[o].get('is_environment_support') for o in group) else movable).append(group)
    removed = []
    for oid, obj in objects.items():
        if not obj.get('is_environment_support') and oid not in edges:
            removed.append({'object_id': oid, 'parent_id': obj.get('parent_id'), 'relation': obj.get('relation'),
                            'reason': 'No bound, conflict-free support observation' if not observed else 'Unsupported or conflicting relation'})
            obj.update(parent_id=None, relation='root')
            manifest.get('bindings', {}).pop(oid, None)
    attachments = []
    for row in manifest.get('attachments', []):
        if row['child_id'] in edges:
            attachments.append(row)
        else:
            stage.RemovePrim(row['joint_path'])
    manifest['attachments'] = attachments
    gap = .02
    cursor = max(b[1, 0] for b in bounds.values()) + gap
    moves, ground_bounds = {}, []
    for group in sorted(movable):
        box = np.array([np.min([bounds[o][0] for o in group], axis=0), np.max([bounds[o][1] for o in group], axis=0)])
        delta = np.array([cursor-box[0, 0], -(box[0, 1]+box[1, 1])/2, .002-box[0, 2]])
        for oid in group:
            obj = objects[oid]; prim = stage.GetPrimAtPath(obj['root_prim_path'])
            tf = matrix(UsdGeom.XformCache().GetLocalToWorldTransform(prim)); tf[:3, 3] += delta
            parent = matrix(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent()))
            _usd_transform(UsdGeom.Xformable(prim), np.linalg.inv(parent) @ tf)
            obj['initial_transform'] = tf.tolist(); bounds[oid] += delta
            moves[oid] = delta.tolist()
        ground_bounds.append(box+delta); cursor = box[1, 0]+delta[0]+gap
    by_id = {o['object_id']: o for o in layout['objects_data']}
    for oid, delta in moves.items():
        row = by_id[oid]
        row['obb']['center_m'] = (np.asarray(row['obb']['center_m'])+delta).tolist()
        if row.get('3d_bbox_meters'):
            row['3d_bbox_meters'][:3] = (np.asarray(row['3d_bbox_meters'][:3])+delta).tolist()
    floor_id = 'terminal_ground'
    while floor_id in objects: floor_id += '_'
    if movable:
        lo, hi = np.min([b[0] for b in ground_bounds], axis=0), np.max([b[1] for b in ground_bounds], axis=0)
        size = [hi[0]-lo[0]+.4, hi[1]-lo[1]+.4, .1]
        center = [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, -.05]
        root_path = '/World/Objects/'+floor_id; body_path = root_path+'/support'
        tf = np.eye(4); tf[:3, 3] = center
        root = UsdGeom.Xform.Define(stage, root_path); _usd_transform(root, tf)
        asset = _environment_asset(size); _define_mesh(stage, body_path, asset['collision_meshes'][0])
        objects[floor_id] = {'object_id': floor_id, 'is_environment_support': True, 'parent_id': None,
            'relation': 'root', 'root_link': 'support', 'body_paths': {'support': body_path},
            'root_prim_path': root_path, 'prim_path': root_path, 'articulation_root_path': None,
            'articulation_root': None, 'initial_transform': tf.tolist(), 'scale': 1.,
            'initial_diagonal_m': float(np.linalg.norm(size)), 'orientation_reference_world': np.eye(3).tolist(),
            **{k: asset[k] for k in ('geometry_meshes', 'collision_meshes', 'link_properties', 'ignored_internal_link_pairs')}}
        floor_name = 'TerminalGround'
        while any(o['name'] == floor_name for o in layout['objects_data']): floor_name += '_'
        floor = {'object_id': floor_id, 'name': floor_name, 'description': 'Explicit degraded-delivery ground',
            'is_environment_support': True, 'is_occluded': False, 'occlusion_reasoning': '',
            'support_relation': 'root', '2d_bbox_normalized': None,
            'obb': {'center_m': center, 'size_m': size, 'rotation_xyz_deg': [0, 0, 0]},
            'geometry_reasoning': 'Ground covering the separate release area', 'uncertainty': 'Added environment'}
        layout['objects_data'].append(floor); by_id[floor_id] = floor
        bounds[floor_id] = np.array([np.asarray(center)-np.asarray(size)/2, np.asarray(center)+np.asarray(size)/2])
        for group in movable:
            for oid in group:
                if oid not in edges: objects[oid].update(parent_id=floor_id, relation='on')
    layout['scene_tree'] = {o['name']: {'children': []} for o in layout['objects_data']}
    for oid, obj in objects.items():
        by_id[oid]['support_relation'] = obj['relation']
        if obj.get('parent_id'):
            layout['scene_tree'][by_id[obj['parent_id']]['name']]['children'].append(by_id[oid]['name'])
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body = UsdPhysics.RigidBodyAPI(prim); body.CreateVelocityAttr(Gf.Vec3f(0)); body.CreateAngularVelocityAttr(Gf.Vec3f(0))
        for kind in ('angular', 'linear'):
            attr = prim.GetAttribute(f'state:{kind}:physics:velocity')
            if attr: attr.Set(0.)
    receipt = {'policy': POLICY, 'source_candidate_id': candidate['candidate_id'], 'support_evidence_bound': observed,
        'retained_support_edges': edges, 'removed_relations': removed, 'translations_m': moves,
        'initial_clearance_m': .002, 'horizontal_gap_m': gap, 'release_velocity': 'zero',
        'new_ground_id': floor_id if movable else None, 'retained_group_nonpenetration_basis': 'bound_native_observations' if observed else None,
        'source_hashes': {p: file_hash(p) for p in [candidate['usd_path'], candidate['scene_manifest_path'], candidate['layout_path']]}}
    identity = digest(receipt); usd_path = output/'scene.usdc'; stage.Export(str(usd_path))
    # Placement uses collision bounds; camera coverage also includes visual geometry.
    camera_cache = UsdGeom.XformCache()
    camera_points = np.concatenate([
        transformed(mesh['vertices'], matrix(camera_cache.GetLocalToWorldTransform(
            stage.GetPrimAtPath(obj['body_paths'][mesh['link']]))))
        for obj in objects.values()
        for kind in ('geometry_meshes', 'collision_meshes')
        for mesh in obj[kind]])
    envelope = {'min': camera_points.min(0).tolist(), 'max': camera_points.max(0).tolist()}
    write_json(output/'layout.json', layout); write_json(output/'placement.json', receipt)
    manifest.update(candidate_id=identity, state_kind='terminal_release', diagnostic_only=False, unresolved_bindings={},
        geometry_checksum=digest({k: v['geometry_meshes'] for k, v in objects.items()}),
        objects=objects, instances=list(objects.values()), fixed_camera_bounds_m=envelope, fixed_render_bounds_m=envelope,
        usd_path=str(usd_path), scene_usd_path=str(usd_path), usd_sha256=file_hash(usd_path),
        layout_sha256=file_hash(output/'layout.json'), terminal_placement={'path': str(output/'placement.json'), 'sha256': file_hash(output/'placement.json')})
    manifest.pop('settled_state', None)
    write_json(output/'scene_manifest.json.gz', manifest)
    return {'candidate_id': identity, 'usd_path': str(usd_path), 'scene_usd_path': str(usd_path),
            'layout_path': str(output/'layout.json'), 'scene_manifest_path': str(output/'scene_manifest.json.gz'),
            'state_kind': 'terminal_release', 'parameters': candidate.get('parameters', {})}


def legal_initial(scene, output):
    from .composition import build_candidate
    prepared = deepcopy(scene)
    prepared.update(bindings={}, diagnostic_only=True)
    for obj in prepared['objects'].values():
        if not obj['is_environment_support']: obj.update(parent_id=None, relation='root')
    return build_candidate(prepared, {}, output)


def valid_result(result):
    from .worker_client import valid_asset_rejection, valid_terminal_evidence
    physical = result.get('physics', {})
    if valid_terminal_evidence(physical):
        return True
    if not valid_asset_rejection(physical):
        return False
    try:
        ref = result['native_rejection_receipt']
        observed = read_json(ref['path'])
        recorded = deepcopy(physical)
        recorded.pop('cache_hit', None)
        recorded.pop('pool_evidence', None)
        if 'timings' in recorded:
            recorded['timings'].pop('pool_queue_seconds', None)
            if not recorded['timings'] and 'timings' not in observed:
                recorded.pop('timings')
        return (file_hash(ref['path']) == ref['sha256'] and recorded == observed
                and result.get('used_terminal_state') is False and result.get('normal_accepted') is False)
    except (OSError, KeyError, TypeError, ValueError):
        return False


def run(worker, checkpoint, initial, output, config, scene=None):
    from .delivery import select_fallback, terminal_candidate
    from .worker_client import valid_terminal_evidence, valid_asset_rejection
    from .workflow import InfrastructureError
    output = Path(output); journal = output/'result.json'
    binding = digest({'checkpoint': checkpoint, 'initial': initial, 'policy': POLICY, 'config': config})
    if journal.exists():
        result = read_json(journal)
        if result['binding'] != binding or not valid_result(result):
            raise InfrastructureError('Terminal fallback evidence or binding changed')
        return result
    selected = select_fallback(checkpoint)
    excluded, ineligible = set(), []
    while selected:
        try:
            selected = terminal_candidate(selected, output/'prior_best'/selected['candidate_id'])
            break
        except (ValueError, KeyError, FileNotFoundError) as error:
            ineligible.append({'candidate_id': selected['candidate_id'], 'reason': str(error)})
            excluded.add(selected['selection_key']); selected = select_fallback(checkpoint, excluded)
    if not initial or not initial.get('usd_path'):
        if scene is None: raise ValueError('Missing prepared assets for terminal release')
        initial = legal_initial(scene, output/'legal_initial')
    source = selected or initial
    if not source: raise ValueError('No legal state for terminal fallback')
    candidate = arrange(source, output/'release', config)
    physical = worker.evaluate(candidate, MAX_SECONDS, out_dir=output/'simulation', mode='terminal_settle')
    if valid_asset_rejection(physical):
        rejection = output/'simulation/native_operation_result.json'
        result = {'binding': binding, 'policy': POLICY, 'selected': selected, 'initial': initial,
            'physics': physical, 'normal_accepted': False, 'delivery_ineligible': ineligible,
            'used_terminal_state': False, 'stop_reason': 'native_asset_rejected',
            'native_rejection_receipt': {'path': str(rejection), 'sha256': file_hash(rejection)}}
        if not valid_result(result):
            raise InfrastructureError('Terminal asset rejection evidence changed')
        write_json(journal, result)
        return result
    if physical.get('status') not in ('settled_window_passed', 'terminal_not_settled') or not valid_terminal_evidence(physical):
        raise InfrastructureError(str(physical.get('errors', 'Incomplete terminal fallback evidence')))
    chosen = ({**physical['terminal_candidate'], 'physics': physical, 'images': physical.get('images', []),
               'semantic': {'score': None, 'reliable': False}, 'final_validation_status': 'not_completed'}
              if physical.get('window_passed') else selected)
    result = {'binding': binding, 'policy': POLICY, 'selected': chosen, 'initial': initial,
              'physics': physical, 'normal_accepted': False, 'delivery_ineligible': ineligible,
              'used_terminal_state': bool(physical.get('window_passed'))}
    write_json(journal, result)
    return result
