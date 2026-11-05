"""Publish existing scene evidence, including unaccepted measured terminal states."""
from copy import deepcopy
import math
from pathlib import Path
import shutil
import time

from .protocol import read_json, write_json, file_hash
from .physics import acceptance_ready, observation_ready
from .scoring import _semantic, _key


def complete_candidate(candidate):
    if not isinstance(candidate, dict): return False
    p = candidate.get('physics', {})
    if not isinstance(p, dict): return False
    violation = p.get('violation')
    return (p.get('status') in ('passed', 'physics_failed') and observation_ready(p)
            and isinstance(violation, (int, float)) and math.isfinite(violation)
            and all(candidate.get(k) for k in ('usd_path', 'layout_path', 'scene_manifest_path')))


def ranked_fallbacks(checkpoint, excluded=()):
    """Known final failures have already replaced search conclusions in the checkpoint."""
    ranked = []
    specs = checkpoint.get('identity', {}).get('specs', {})
    order_keys = [checkpoint.get('initial_key')]
    for generation in checkpoint.get('history', []):
        order_keys.extend(generation.get('population_keys', []))
    order_keys.extend(_key(p) for p in (checkpoint.get('pending_round') or {}).get('population', []))
    sample_order = {key: index for index, key in reversed(list(enumerate(order_keys))) if key}
    for order, (key, original) in enumerate(checkpoint.get('candidates', {}).items()):
        if key in excluded or not isinstance(original, dict):
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
                      for oid, values in c.get('placement_parameters', c.get('parameters', {})).items() if oid in specs
                      for axis, v in enumerate(values)]
        deviation = min(1., math.sqrt(sum(normalized)/len(normalized))) if normalized else 0.
        semantic = c['semantic']['score']
        rank = (not acceptance_ready(p), penetrating, p['violation'],
                -(semantic if semantic is not None else -1), deviation, sample_order.get(key, len(order_keys)), key)
        ranked.append((rank, {**c, 'selection_key': key}))
    return [row[1] for row in sorted(ranked, key=lambda row: row[0])]


def select_fallback(checkpoint, excluded=()):
    ranked = ranked_fallbacks(checkpoint, excluded)
    return ranked[0] if ranked else None


def _bound_file(path, expected, run=None, *, shared=False):
    from urdf_core.error_recovery import VerifiedExternalError
    path = Path(path)
    if not path.exists() and run and path.resolve().is_relative_to(Path(run).resolve()):
        from urdf_core.storage import copy_asset
        for parent in path.parents:
            blob = parent/'.asset_blobs'/expected
            if blob.is_file() and file_hash(blob) == expected:
                copy_asset(blob, path, digest=expected)
                break
    if not path.is_file() or file_hash(path) != expected:
        message = 'Bound asset missing or changed: ' + str(path)
        if shared:
            raise VerifiedExternalError(message, kind='asset_integrity', evidence={'path': str(path), 'sha256': expected})
        raise ValueError(message)


def _state_assets(candidate):
    """Separate pose evidence from the files needed to load the saved state."""
    required = {candidate.get(key) for key in ('usd_path', 'layout_path', 'scene_manifest_path')}
    pose = candidate.get('settled_state_path')
    return {path: checksum for path, checksum in candidate.get('settled_artifacts', {}).items()
            if path != pose or path in required}


def candidate_metadata(candidate, run=None):
    from .candidate_storage import metadata
    manifest = metadata(candidate['scene_manifest_path'])
    for path, expected in manifest.get('source_hashes', {}).items():
        if path not in ('layout_json', 'asset_manifest_json'):
            _bound_file(path, expected, shared=True)
    refs = dict(manifest.get('derivative_artifacts', {}))
    refs.update(_state_assets(candidate))
    for key, hash_key in (('usd_path', 'usd_sha256'), ('layout_path', 'layout_sha256')):
        if manifest.get(hash_key): refs[candidate[key]] = manifest[hash_key]
        elif not Path(candidate[key]).is_file(): raise ValueError('Missing candidate '+key)
    for path, expected in refs.items():
        _bound_file(path, expected, run)
    return manifest


def terminal_sample_for(candidate):
    from .trajectory import read_trace, terminal_sample
    from physics_feedback.compact_trajectory import is_compact_path
    p = candidate['physics']
    ref = p.get('terminal_state') or p.get('trace')
    if not ref: raise ValueError('No effective terminal state')
    _bound_file(ref['path'], ref['sha256'])
    duration = p.get('effective_duration_s', p.get('duration_s'))
    if duration is None: raise ValueError('No effective scoring duration')
    if p.get('terminal_state'):
        samples = [read_json(ref['path'])]
    else:
        samples = ([terminal_sample(ref['path'])] if is_compact_path(ref['path']) else read_trace(ref['path']))
    samples = [s for s in samples if abs(s['time_s']-duration) < 1e-8]
    if len(samples) != 1: raise ValueError('Effective terminal sample is missing or ambiguous')
    return samples[0]


def choose_delivery_candidate(checkpoint, run, failures):
    from urdf_core.error_recovery import external_failure, cancelled
    from .metrics import _pose
    import numpy as np
    for candidate in ranked_fallbacks(checkpoint):
        try:
            manifest = candidate_metadata(candidate, run)
            if candidate.get('state_kind') in ('settled', 'measured_terminal') and candidate.get('settled_artifacts'):
                return candidate
            sample = terminal_sample_for(candidate)
            required = {f'{oid}::{link}' for oid, obj in manifest['objects'].items() for link in obj['body_paths']}
            if set(sample['body_poses']) != required or set(sample.get('body_velocities', {})) != required:
                raise ValueError('Incomplete terminal body state')
            for pose in sample['body_poses'].values(): _pose(pose)
            for velocity in sample['body_velocities'].values():
                for vector in velocity.values():
                    if np.asarray(vector).shape != (3,) or not np.isfinite(vector).all():
                        raise ValueError('Invalid terminal velocity')
            if 'native_articulations' not in candidate['physics']: raise ValueError('Missing native articulation state')
            for oid, articulation in candidate['physics']['native_articulations'].items():
                names = articulation.get('dof_names', [])
                if not names: continue
                joint = sample.get('native_joint_states', {}).get(oid, {})
                if (len(joint.get('joint_names', [])) != len(names) or set(joint['joint_names']) != set(names)
                        or any(np.asarray(joint.get(key)).shape != (len(names),) or not np.isfinite(joint[key]).all()
                               for key in ('positions', 'velocities'))):
                    raise ValueError('Incomplete terminal joint state: '+oid)
            return candidate
        except Exception as error:
            if external_failure(error) or cancelled(error): raise
            failures.append({'candidate_id': candidate.get('candidate_id'), 'phase': 'selection', 'reason': str(error)})
    return None


def terminal_candidate(candidate, output):
    from .composition import materialize_state
    p = candidate['physics']
    sample = terminal_sample_for(candidate)
    state = materialize_state(candidate, sample, p['native_articulations'], output, preserve_velocity=True)
    return {**candidate, **state, 'source_candidate_id': candidate['candidate_id']}


def _copy(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    if expected and file_hash(source) != expected:
        raise ValueError('Evidence changed: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    from urdf_core.storage import copy_asset
    copy_asset(source, destination, digest=expected)
    if file_hash(destination) != file_hash(source):
        raise ValueError('Evidence copy mismatch')


def publish(run, selected, initial, *, accepted, stop_reason):
    """Static packaging only. This module never starts Isaac or invokes a model."""
    from .terminal_settle import POLICY as TERMINAL_POLICY
    from .material_bundle import package
    from .candidate_storage import metadata
    run = Path(run)
    started = time.monotonic()
    requested_acceptance = accepted
    required_failures = []
    target, pending = run/'delivery', run/'delivery.pending'
    if target.exists():
        receipt = read_json(target/'receipt.json')
        if receipt.get('delivered'):
            if (receipt.get('requested_acceptance', receipt['accepted']) != requested_acceptance
                    or receipt.get('source_candidate_id') != (selected or {}).get('candidate_id')
                    or any(file_hash(target/p) != h for p, h in receipt['artifacts'].items())):
                raise ValueError('Existing delivery does not match this result')
            return receipt
        history = run/'prior_deliveries'/str(time.time_ns())
        history.parent.mkdir(exist_ok=True)
        target.rename(history)
    request = {'selected': (selected or {}).get('usd_path'),
               'initial': (initial or {}).get('usd_path'), 'accepted': requested_acceptance}
    request_path = pending/'request.json'
    if pending.exists() and (not request_path.is_file() or read_json(request_path) != request):
        shutil.rmtree(pending)
    pending.mkdir(parents=True, exist_ok=True)
    write_json(request_path, request)
    p = (selected or {}).get('physics', {})
    optional_failures = []
    def optional(action, description, *, required=False):
        existing = set(pending.rglob('*'))
        try:
            action()
            return True
        except Exception as error:
            from urdf_core.error_recovery import external_failure, cancelled
            if external_failure(error) or cancelled(error): raise
            for path in sorted(set(pending.rglob('*'))-existing, key=lambda p: len(p.parts), reverse=True):
                path.rmdir() if path.is_dir() else path.unlink()
            optional_failures.append({'artifact': description, 'reason': str(error)})
            if required: required_failures.append(description)
            return False
    if accepted:
        semantic = selected.get('semantic', {})
        if not (acceptance_ready(p) and p.get('native_dynamics') and p.get('evaluation_phase') == 'full'
                and p.get('stage_reset') == 'independent_USD_reload'
                and semantic.get('reliable') and semantic.get('score') == 1 and semantic.get('sample_count', 0) >= 2):
            required_failures.append('independent_acceptance')
        from .worker_client import valid_cached_evidence
        config = p.get('thresholds', {})
        if not config or not valid_cached_evidence(p, config.get('rollout_max_duration_s'), config):
            required_failures.append('final_evidence')
        output = selected
    elif selected:
        artifacts = selected.get('settled_artifacts', {})
        if selected.get('state_kind') in ('settled', 'measured_terminal') and artifacts:
            output = selected
        else:
            state_record = pending/'state_ready.json'
            state_binding = {'manifest': file_hash(selected['scene_manifest_path']),
                             'physics': selected['physics']}
            record = read_json(state_record) if state_record.is_file() else None
            if record and record['binding'] == state_binding and all(
                    Path(path).is_file() and file_hash(path) == checksum
                    for path, checksum in _state_assets(record['output']).items()):
                output = record['output']
            else:
                output = terminal_candidate(selected, pending/'state')
                write_json(state_record, {'binding': state_binding, 'output': output})
    elif initial and initial.get('usd_path'):
        output = initial.get('constructed_candidate', initial)
    else:
        return failure_receipt(run, stop_reason, 'No structurally valid initial scene')
    objects = candidate_metadata(output, run)['objects']
    binding = {'usd': file_hash(output['usd_path']), 'manifest': file_hash(output['scene_manifest_path'])}
    saved = pending/'bundle_ready.json'
    bundle = None
    if saved.is_file():
        record = read_json(saved)
        if record['binding'] == binding and all((pending/name).is_file() and file_hash(pending/name) == checksum
                                               for name, checksum in record['artifacts'].items()):
            bundle = record['bundle']
    package_started = time.monotonic()
    if bundle is None:
        if (pending/'final').exists(): shutil.rmtree(pending/'final')
        try:
            bundle = package(output['usd_path'], pending/'final', allow_missing_textures=True, objects=objects)
        except (ValueError, FileNotFoundError) as exc:
            if selected: raise
            return failure_receipt(run, stop_reason, 'Initial scene is technically unavailable: ' + str(exc))
        write_json(saved, {'binding': binding, 'bundle': bundle,
                          'artifacts': {str(f.relative_to(pending)): file_hash(f) for f in (pending/'final').rglob('*') if f.is_file()}})
    package_seconds = time.monotonic()-package_started
    if bundle.get('material_degradation'): required_failures.append('material_degradation')
    _copy(output['layout_path'], pending/'layout_final.json')
    if output.get('settled_state_path'):
        def copy_pose():
            path = output['settled_state_path']
            checksum = output.get('settled_artifacts', {}).get(path)
            if not checksum: raise ValueError('Unbound settled pose evidence')
            _bound_file(path, checksum, run)
            _copy(path, pending/'pose_receipt.json', checksum)
        optional(copy_pose, 'pose', required=requested_acceptance)
    if initial and initial.get('usd_path'):
        original = initial.get('constructed_candidate', initial)
        optional(lambda: write_json(pending/'initial_source.json',
            {'usd_path': original['usd_path'], 'sha256': file_hash(original['usd_path']),
             'scope': 'Retained run artifact; not a second portable scene bundle'}), 'initial_scene')
        optional(lambda: _copy(original['layout_path'], pending/'layout_initial.json'), 'initial_layout')
        def copy_assembly():
            assembly = metadata(original['scene_manifest_path']).get('assembly_pose_receipt')
            if assembly:
                _copy(assembly['path'], pending/'assembly_pose_receipt.json', assembly['sha256'])
        optional(copy_assembly, 'assembly_pose')
    write_json(pending/'physics.json', p or {'status': 'not_evaluated'})
    write_json(pending/'semantic_reviews.json', (selected or {}).get('semantic_samples', []))
    available = {'final_validation_video': False, 'images': 0, 'pose': (pending/'pose_receipt.json').exists()}
    video = p.get('video')
    if video:
        terminal_video = p.get('policy') == TERMINAL_POLICY
        video_saved = optional(lambda: _copy(video['video_path'], pending/('terminal_release.mp4' if terminal_video else 'final_validation.mp4'), video['sha256']), 'video', required=requested_acceptance)
        available['terminal_release_video' if terminal_video else 'final_validation_video'] = video_saved
    for index, image in enumerate((selected or {}).get('images', p.get('images', []))):
        available['images'] += int(optional(lambda: _copy(image['path'], pending/f'view_{index}.png', image['sha256']), 'image', required=requested_acceptance))
    if p.get('initial_state_receipt'):
        ref = p['initial_state_receipt']
        optional(lambda: _copy(ref['path'], pending/('release_pose_receipt.json' if p.get('policy') == TERMINAL_POLICY else 'reload_pose_receipt.json'), ref['sha256']), 'initial_native_pose', required=requested_acceptance)
    semantic = (selected or {}).get('semantic', {})
    degradations = {oid: {'policy': obj.get('degradation_policy'), 'upstream': obj.get('upstream_degradations', [])}
                    for oid, obj in objects.items() if obj.get('degradation_policy') or obj.get('upstream_degradations')}
    accepted = bool(requested_acceptance and not required_failures)
    receipt = {'schema_version': 'scene_delivery.v3', 'delivered': True, 'accepted': accepted,
        'requested_acceptance': requested_acceptance, 'incomplete_acceptance': required_failures,
        'delivery_timings': {'package_seconds': package_seconds, 'total_seconds': time.monotonic()-started},
        'stop_reason': stop_reason, 'source_candidate_id': (selected or {}).get('candidate_id'),
        'physical_status': p.get('status', 'not_evaluated'),
        'terminal_window_passed': p.get('window_passed', False),
        'degraded_delivery': bool(degradations) or bool(bundle.get('material_degradation')) or p.get('policy') == TERMINAL_POLICY or stop_reason == 'environment_only',
        'foreground_reconstruction_complete': stop_reason != 'environment_only' and not degradations,
        'object_degradations': degradations,
        'object_quality': {oid: {'quality': obj.get('upstream_quality', {}),
                                'degradation': degradations.get(oid),
                                'verification': {key: obj.get('asset_verification', {}).get(key)
                                                 for key in ('acceptance_policy', 'delivery_manifest_path', 'physics_report_path', 'source_fingerprint')
                                                 if obj.get('asset_verification', {}).get(key) is not None}}
                           for oid, obj in objects.items() if not obj.get('is_environment_support')},
        'foreground_asset_count': sum(not obj.get('is_environment_support') for obj in objects.values()),
        'semantic_status': 'passed' if semantic.get('score') == 1 and semantic.get('reliable') else 'not_passed',
        'final_validation_status': 'passed' if accepted else (selected or {}).get('final_validation_status', 'not_completed'),
        'state_kind': output.get('state_kind', 'initial_unassessed'), 'evidence': available, 'missing_optional_evidence': optional_failures,
        'static_checks': bundle['static_checks'],
        'material_degradation': bundle.get('material_degradation', []),
        'scene': 'final/scene.usdc', 'layout': 'layout_final.json',
        'artifacts': {str(f.relative_to(pending)): file_hash(f) for f in pending.rglob('*') if f.is_file()
                      and f.relative_to(pending).parts[0] != 'state'
                      and f.name not in ('bundle_ready.json', 'state_ready.json', 'request.json')}}
    write_json(pending/'receipt.json', receipt)
    pending.rename(target)
    try:
        if (target/'state').exists(): shutil.rmtree(target/'state')
        for name in ('bundle_ready.json', 'state_ready.json', 'request.json'):
            (target/name).unlink(missing_ok=True)
    except OSError:
        pass  # A committed asset remains valid if temporary-file cleanup is delayed.
    return receipt


def failure_receipt(run, stop_reason, reason):
    receipt = {'schema_version': 'scene_delivery.v3', 'delivered': False, 'accepted': False,
               'stop_reason': stop_reason, 'reason': reason, 'artifacts': {}}
    path = Path(run)/'delivery/receipt.json'
    if path.is_file() and read_json(path).get('delivered'):
        path = Path(run)/'delivery_failure.json'
    write_json(path, receipt)
    return receipt


def publish_fallback(run, selected, initial, attempt_dir, *, stop_reason, accepted=False, selection_checked=False):
    """One selected state, one retained initial, and one deterministic reconstruction."""
    from urdf_core.error_recovery import external_failure, cancelled
    run, attempt_dir = Path(run), Path(attempt_dir)
    failures, tried = [], set()
    if selected is None and not selection_checked:
        path = attempt_dir/'search_checkpoint.json'
        selected = choose_delivery_candidate(read_json(path) if path.is_file() else {}, run, failures)
    receipt = None
    for phase in ('selected', 'initial', 'reconstructed_initial'):
        try:
            if phase == 'selected':
                if selected is None: continue
                candidate, origin = selected, initial
            elif phase == 'initial':
                if initial is None: continue
                candidate, origin = None, initial
            else:
                prepared = attempt_dir/'prepared/prepared_scene.json'
                if not prepared.is_file(): continue
                from .terminal_settle import legal_initial
                candidate, origin = None, legal_initial(read_json(prepared), attempt_dir/'delivery_initial')
            source = candidate or origin.get('constructed_candidate', origin)
            identity = (file_hash(source['usd_path']), file_hash(source['layout_path']))
            if candidate and not accepted and not candidate.get('settled_artifacts'):
                physical = candidate.get('physics', {})
                endpoint = physical.get('terminal_state') or physical.get('trace') or {}
                identity += (endpoint.get('sha256'), physical.get('effective_duration_s', physical.get('duration_s')), 'terminal')
            if identity in tried: continue
            tried.add(identity)
            receipt = publish(run, candidate, origin, accepted=accepted and phase == 'selected', stop_reason=stop_reason)
            if receipt['delivered']: break
            failures.append({'phase': phase, 'reason': receipt.get('reason')})
        except Exception as error:
            if external_failure(error) or cancelled(error): raise
            failures.append({'phase': phase, 'reason': str(error)})
    if not receipt or not receipt['delivered']:
        receipt = failure_receipt(run, stop_reason, 'No deliverable bound scene: '+str(failures))
    write_json(attempt_dir/'packaging_recovery.json', {'failures': failures, 'sources_attempted': len(tried),
               'delivered': receipt['delivered'], 'source_candidate_id': receipt.get('source_candidate_id')})
    return receipt
