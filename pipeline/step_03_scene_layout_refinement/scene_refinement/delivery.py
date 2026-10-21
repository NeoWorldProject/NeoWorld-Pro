"""Publish existing scene evidence, including unaccepted measured terminal states."""
from copy import deepcopy
import math
from pathlib import Path
import shutil

from .protocol import read_json, write_json, file_hash
from .physics import acceptance_ready, observation_ready
from .scoring import _semantic, _key


def complete_candidate(candidate):
    p = candidate.get('physics', {})
    return (p.get('status') in ('passed', 'physics_failed') and observation_ready(p)
            and math.isfinite(p.get('violation', float('nan')))
            and all(candidate.get(k) for k in ('usd_path', 'layout_path', 'scene_manifest_path')))


def select_fallback(checkpoint, excluded=()):
    """Known final failures have already replaced search conclusions in the checkpoint."""
    ranked = []
    specs = checkpoint.get('identity', {}).get('specs', {})
    order_keys = [checkpoint.get('initial_key'), (checkpoint.get('clearance_initialization') or {}).get('candidate_key')]
    for generation in checkpoint.get('history', []):
        order_keys.extend(generation.get('population_keys', []))
    order_keys.extend(_key(p) for p in (checkpoint.get('pending_round') or {}).get('population', []))
    sample_order = {key: index for index, key in reversed(list(enumerate(order_keys))) if key}
    for order, (key, original) in enumerate(checkpoint.get('candidates', {}).items()):
        if key in excluded:
            continue
        c = deepcopy(original)
        final_record = checkpoint.get('final_validations', {}).get(key, {})
        final = final_record.get('physical')
        if final and final.get('status') == 'asset_rejected':
            continue
        c['final_validation_status'] = ('passed' if final_record.get('accepted') else
            'failed' if final and final.get('known_infeasible') and observation_ready(final) else 'not_completed')
        if final and final.get('known_infeasible'):
            if not complete_candidate({**c, 'physics': final}):
                continue
            search = c.get('search_physics', c['physics'])
            c['physics'] = deepcopy(final)
            if search['violation'] > final['violation']:
                c['physics'].update(violation=search['violation'], reward_terms=search.get('reward_terms', {}))
        if not complete_candidate(c):
            continue
        records = checkpoint.get('scores', {}).get(key, [])
        c.update(semantic=_semantic(records), semantic_samples=records)
        p = c['physics']
        penetrating = any(i.get('kind') in ('penetration', 'native_contact_penetration') for i in p.get('issues', []))
        penetrating |= p.get('reward_terms', {}).get('penetration', 0) < 0
        normalized = [(v / max(abs(b) for b in specs[oid]['bounds'][axis])) ** 2
                      for oid, values in c.get('parameters', {}).items() if oid in specs
                      for axis, v in enumerate(values)]
        deviation = min(1., math.sqrt(sum(normalized)/len(normalized))) if normalized else 0.
        semantic = c['semantic']['score']
        rank = (not acceptance_ready(p), penetrating, p['violation'],
                -(semantic if semantic is not None else -1), deviation, sample_order.get(key, len(order_keys)), key)
        ranked.append((rank, {**c, 'selection_key': key}))
    return min(ranked, key=lambda row: row[0])[1] if ranked else None


def terminal_candidate(candidate, output):
    from .trajectory import read_trace
    from .composition import materialize_state
    p = candidate['physics']
    ref = p.get('terminal_state')
    if ref:
        if file_hash(ref['path']) != ref['sha256']:
            raise ValueError('Effective terminal state changed')
        sample = read_json(ref['path'])
    else:
        ref = p['trace']
        if file_hash(ref['path']) != ref['sha256']:
            raise ValueError('Native trajectory changed')
        duration = p.get('effective_duration_s', p.get('duration_s'))
        if duration is None:
            raise ValueError('No effective scoring duration')
        samples = [s for s in read_trace(ref['path']) if abs(s['time_s']-duration) < 1e-8]
        if len(samples) != 1:
            raise ValueError('Effective terminal sample is missing or ambiguous')
        sample = samples[0]
    state = materialize_state(candidate, sample, p['native_articulations'], output, preserve_velocity=True)
    return {**candidate, **state, 'source_candidate_id': candidate['candidate_id']}


def _copy(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    if expected and file_hash(source) != expected:
        raise ValueError('Evidence changed: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if file_hash(destination) != file_hash(source):
        raise ValueError('Evidence copy mismatch')


def publish(run, selected, initial, *, accepted, stop_reason):
    """Static packaging only. This module never starts Isaac or invokes a model."""
    from .terminal_settle import POLICY as TERMINAL_POLICY
    from .material_bundle import package
    run = Path(run)
    target, pending = run/'delivery', run/'delivery.pending'
    if target.exists():
        receipt = read_json(target/'receipt.json')
        if receipt['accepted'] != accepted or any(file_hash(target/p) != h for p, h in receipt['artifacts'].items()):
            raise ValueError('Existing delivery does not match this result')
        return receipt
    if pending.exists():
        # Only an uncommitted output of this run may be replaced on resume.
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    p = (selected or {}).get('physics', {})
    optional_failures = []
    def optional(action, description):
        existing = set(pending.rglob('*'))
        try:
            action()
            return True
        except (ValueError, FileNotFoundError) as error:
            if accepted: raise
            for path in sorted(set(pending.rglob('*'))-existing, key=lambda p: len(p.parts), reverse=True):
                path.rmdir() if path.is_dir() else path.unlink()
            optional_failures.append({'artifact': description, 'reason': str(error)})
            return False
    if accepted:
        semantic = selected.get('semantic', {})
        if not (acceptance_ready(p) and p.get('native_dynamics') and p.get('duration_s', 0) >= 8.
                and p.get('stage_reset') == 'independent_USD_reload'
                and semantic.get('reliable') and semantic.get('score') == 1 and semantic.get('sample_count', 0) >= 2):
            raise ValueError('Missing independent acceptance evidence')
        from .worker_client import valid_cached_evidence
        if not valid_cached_evidence(p, 8., p['thresholds']):
            raise ValueError('Incomplete final evidence')
        output = selected
    elif selected:
        artifacts = selected.get('settled_artifacts', {})
        if selected.get('state_kind') == 'measured_terminal' and artifacts:
            if any(file_hash(path) != checksum for path, checksum in artifacts.items()):
                raise ValueError('Measured terminal export changed')
            output = selected
        else:
            output = terminal_candidate(selected, pending/'state')
    elif initial and initial.get('usd_path'):
        output = initial.get('constructed_candidate', initial)
    else:
        pending.rmdir()
        return {'schema_version': 'scene_delivery.v3', 'delivered': False, 'accepted': False,
                'stop_reason': stop_reason, 'reason': 'No structurally valid initial scene'}
    try:
        package(output['usd_path'], pending/'final', allow_missing_textures=not accepted)
    except (ValueError, FileNotFoundError) as exc:
        if selected:
            raise
        shutil.rmtree(pending)
        return {'schema_version': 'scene_delivery.v3', 'delivered': False, 'accepted': False,
                'stop_reason': stop_reason, 'reason': 'Initial scene is technically unavailable: ' + str(exc)}
    _copy(output['layout_path'], pending/'layout_final.json')
    if output.get('settled_state_path'):
        _copy(output['settled_state_path'], pending/'pose_receipt.json')
    if initial and initial.get('usd_path'):
        original = initial.get('constructed_candidate', initial)
        optional(lambda: package(original['usd_path'], pending/'initial'), 'initial_scene')
        optional(lambda: _copy(original['layout_path'], pending/'layout_initial.json'), 'initial_layout')
        def copy_assembly():
            assembly = read_json(original['scene_manifest_path']).get('assembly_pose_receipt')
            if assembly:
                _copy(assembly['path'], pending/'assembly_pose_receipt.json', assembly['sha256'])
        optional(copy_assembly, 'assembly_pose')
    objects = read_json(output['scene_manifest_path'])['objects']
    if (pending/'state').exists():
        # The published bundle contains the state; retain just its portable receipt.
        shutil.rmtree(pending/'state')
    write_json(pending/'physics.json', p or {'status': 'not_evaluated'})
    write_json(pending/'semantic_reviews.json', (selected or {}).get('semantic_samples', []))
    available = {'final_validation_video': False, 'images': 0, 'pose': (pending/'pose_receipt.json').exists()}
    video = p.get('video')
    if video:
        terminal_video = p.get('policy') == TERMINAL_POLICY
        saved = optional(lambda: _copy(video['video_path'], pending/('terminal_release.mp4' if terminal_video else 'final_validation.mp4'), video['sha256']), 'video')
        available['terminal_release_video' if terminal_video else 'final_validation_video'] = saved
    for index, image in enumerate((selected or {}).get('images', p.get('images', []))):
        available['images'] += int(optional(lambda: _copy(image['path'], pending/f'view_{index}.png', image['sha256']), 'image'))
    if p.get('initial_state_receipt'):
        ref = p['initial_state_receipt']
        optional(lambda: _copy(ref['path'], pending/('release_pose_receipt.json' if p.get('policy') == TERMINAL_POLICY else 'reload_pose_receipt.json'), ref['sha256']), 'initial_native_pose')
    semantic = (selected or {}).get('semantic', {})
    degradations = {oid: {'policy': obj.get('degradation_policy'), 'upstream': obj.get('upstream_degradations', [])}
                    for oid, obj in objects.items() if obj.get('degradation_policy') or obj.get('upstream_degradations')}
    receipt = {'schema_version': 'scene_delivery.v3', 'delivered': True, 'accepted': accepted,
        'stop_reason': stop_reason, 'source_candidate_id': (selected or {}).get('candidate_id'),
        'physical_status': p.get('status', 'not_evaluated'),
        'terminal_window_passed': p.get('window_passed', False),
        'degraded_delivery': bool(degradations) or p.get('policy') == TERMINAL_POLICY or stop_reason == 'environment_only',
        'foreground_reconstruction_complete': stop_reason != 'environment_only' and not degradations,
        'object_degradations': degradations,
        'foreground_asset_count': sum(not obj.get('is_environment_support') for obj in objects.values()),
        'semantic_status': 'passed' if semantic.get('score') == 1 and semantic.get('reliable') else 'not_passed',
        'final_validation_status': 'passed' if accepted else (selected or {}).get('final_validation_status', 'not_completed'),
        'state_kind': output.get('state_kind', 'initial_unassessed'), 'evidence': available, 'missing_optional_evidence': optional_failures,
        'static_checks': {'loadable': True, 'dependencies_complete': True, 'physical_validation': False},
        'scene': 'final/scene.usdc', 'layout': 'layout_final.json',
        'artifacts': {str(f.relative_to(pending)): file_hash(f) for f in pending.rglob('*') if f.is_file()}}
    write_json(pending/'receipt.json', receipt)
    pending.rename(target)
    return receipt


def publish_fallback(run, selected, initial, attempt_dir, *, stop_reason):
    """Try intact recorded states, then a newly assembled legal initial state."""
    run, attempt_dir = Path(run), Path(attempt_dir)
    path = attempt_dir/'search_checkpoint.json'
    checkpoint = read_json(path) if path.is_file() else {}
    excluded, failures = set(), []
    candidate = selected
    while candidate:
        try:
            receipt = publish(run, candidate, initial, accepted=False, stop_reason=stop_reason)
            break
        except (ValueError, KeyError, FileNotFoundError) as error:
            failures.append({'candidate_id': candidate.get('candidate_id'), 'reason': str(error)})
            excluded.update(k for k, row in checkpoint.get('candidates', {}).items()
                            if row.get('candidate_id') == candidate.get('candidate_id'))
            if candidate.get('selection_key'): excluded.add(candidate['selection_key'])
            candidate = select_fallback(checkpoint, excluded)
    else:
        receipt = publish(run, None, initial, accepted=False, stop_reason=stop_reason)
        prepared = attempt_dir/'prepared/prepared_scene.json'
        if not receipt['delivered'] and prepared.is_file():
            from .terminal_settle import legal_initial
            initial = legal_initial(read_json(prepared), attempt_dir/'delivery_initial')
            receipt = publish(run, None, initial, accepted=False, stop_reason=stop_reason)
    write_json(attempt_dir/'packaging_recovery.json', {'failures': failures,
               'delivered': receipt['delivered'], 'source_candidate_id': receipt.get('source_candidate_id')})
    return receipt
