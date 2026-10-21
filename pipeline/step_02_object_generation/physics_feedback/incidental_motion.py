"""Per-trial API adjudication of non-target hold deviations; never a repair.

The numerical threshold remains observable. Only its specific hold errors can
be adjudicated, and a decision is bound to this exact trial and physical input.
The Isaac process pauses without stepping while its owner makes the API call.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
import time

from urdf_core.run_state import file_hash, fingerprint, write_json

POLICY = 'incidental_motion.per_trial_api.v1'
INPUT_FILE = 'incidental_motion_input.json'
PREFIX = 'non-tested joint exceeded its initial hold tolerance: '
from urdf_core.paths import RESOURCE_ROOT
PROMPT = (RESOURCE_ROOT / 'prompts/isaac/physics_incidental_motion.md').read_text(encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_hash(path)}


def checked(record, root=None):
    path = Path(record['path']).resolve()
    if ((root is not None and not path.is_relative_to(Path(root).resolve()))
            or not path.is_file() or file_hash(path) != record['sha256']):
        raise ValueError('Incidental-motion evidence changed or escaped its run')
    return path


def pending_names(result, kind):
    if kind not in {'fixed_joint', 'free_joint'}:
        return []
    return sorted({s[len(PREFIX):] for s in result.get('errors', []) if s.startswith(PREFIX)})


def same_raw(left, right):
    from .verification import _same_metrics
    return _same_metrics(dict(left, errors=sorted(left.get('errors', []))),
                         dict(right, errors=sorted(right.get('errors', []))))


def validate_decision(value, names, refs):
    if not isinstance(value, dict) or set(value) != {'items'} or not isinstance(value['items'], list):
        raise ValueError('Expected only items, with one decision per non-target joint')
    items = value['items']
    if (len(items) != len(names) or any(not isinstance(x, dict) for x in items)
            or {x.get('joint_name') for x in items} != set(names)):
        raise ValueError('Every supplied non-target joint must occur exactly once')
    for x in items:
        if (set(x) != {'joint_name', 'verdict', 'reason', 'evidence'}
                or x['verdict'] not in {'acceptable', 'unacceptable', 'uncertain'}
                or not isinstance(x['reason'], str) or not x['reason'].strip()
                or not isinstance(x['evidence'], list) or not x['evidence']
                or any(not isinstance(r, str) or r not in refs for r in x['evidence'])):
            raise ValueError('Decision requires a known verdict, reason and supplied evidence references')
    return {'items': sorted(copy.deepcopy(items), key=lambda x: x['joint_name'])}


def apply_decision(raw, kind, decision):
    names = pending_names(raw, kind)
    if not names or {x['joint_name'] for x in decision['items']} != set(names):
        raise ValueError('Decision does not cover this trial hold-deviation set')
    result = copy.deepcopy(raw)
    acceptable = {x['joint_name'] for x in decision['items'] if x['verdict'] == 'acceptable'}
    uncertain = {x['joint_name'] for x in decision['items'] if x['verdict'] == 'uncertain'}
    result['errors'] = [s for s in raw['errors'] if s not in {PREFIX + n for n in acceptable}]
    result['errors'] = [('unresolved non-tested joint motion review: ' + s[len(PREFIX):])
                        if s in {PREFIX + n for n in uncertain} else s for s in result['errors']]
    result.update(passed=not result['errors'], status='needs_evidence' if uncertain else
                  'failed' if result['errors'] else 'passed')
    result['incidental_motion'] = {'policy': POLICY, **copy.deepcopy(decision)}
    return result


def prepare_input(path, contract_path, *, model, reasoning_effort, images):
    path = Path(path)
    local_images = []
    for i, source in enumerate(images):
        source = Path(source)
        target = path.parent / 'incidental_source_images' / (str(i) + source.suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_hash(target) != file_hash(source):
            raise ValueError('Frozen incidental-motion source image changed')
        if not target.exists():
            shutil.copy2(source, target)
        local_images.append(binding(target))
    value = {'policy': POLICY, 'prompt': PROMPT, 'contract': binding(contract_path),
             'model': model, 'reasoning_effort': reasoning_effort,
             'source_images': local_images, 'source_image_origins': [binding(p) for p in images]}
    if path.exists() and read(path) != value:
        raise ValueError('Frozen incidental-motion input changed')
    if not path.exists():
        write_json(path, value)
    return path


def summarize_events(trace, raw, contract, spec, target):
    from .api_evidence import contact_summary
    definitions = {j['name']: j for j in spec['joints']}
    result = []
    for name in pending_names(raw, 'fixed_joint'):
        joint = contract['joints'][name]
        tol = raw['metrics']['joints'][name]['hold_tolerance']
        initial = joint['initial_position']
        deviation = [abs(s['joint_positions'][name] - initial) for s in trace]
        first = next(i for i, d in enumerate(deviation) if d > tol + 1e-8)
        peak = max(range(len(trace)), key=deviation.__getitem__)
        times = [trace[i]['time_s'] for i in (first, peak)]
        indices = {0, len(trace)-1, first, peak}
        # Bounded, deterministic +/- 0.1 s windows, native samples without interpolation.
        for t in times:
            near = [i for i, s in enumerate(trace) if abs(s['time_s']-t) <= .1]
            if near:
                indices.update(near[::max(1, len(near)//4)])
        tail = [s['joint_positions'][name] for s in trace if s['time_s'] >= trace[-1]['time_s']-1]
        samples = [{'time_s': trace[i]['time_s'], 'q': trace[i]['joint_positions'][name],
                    'q_target': trace[i]['joint_targets'][name],
                    'velocity': trace[i]['joint_velocities'][name],
                    'effort': trace[i]['joint_efforts'][name],
                    'operated_joint_q': trace[i]['joint_positions'][target],
                    'contacts': contact_summary(trace[i].get('contacts', {}))} for i in sorted(indices)]
        result.append({'evidence_ref': 'joint:' + name, 'joint_name': name,
            'parent': definitions[name].get('parent'), 'child': definitions[name].get('child'),
            'type': joint['type'], 'unit': 'm' if joint['type'] == 'prismatic' else 'rad',
            'initial': initial, 'tolerance': tol, 'max_deviation': deviation[peak],
            'max_deviation_deg': math.degrees(deviation[peak]) if joint['type'] != 'prismatic' else None,
            'first_exceeded_time_s': times[0], 'maximum_time_s': times[1],
            'final_q': trace[-1]['joint_positions'][name], 'last_second_q_span': max(tail)-min(tail),
            'samples': samples})
    return result


def await_review(runtime, trial, trace):
    """Called before publishing a final trial verdict or running the next trial."""
    input_path = getattr(getattr(runtime, 'args', None), 'incidental_motion', None)
    names = pending_names(trial['metrics'], trial['kind'])
    if input_path is None or not names:
        return
    from .deferred_capture import render_trial
    if trial.get('recording_mode') == 'numeric_pending_media':
        render_trial(runtime, trial, full_video=False)
    directory = runtime.args.out_dir / 'incidental_motion' / trial['id']
    directory.mkdir(parents=True, exist_ok=True)
    # These API images stay immutable when delivery media is rendered later.
    frames = []
    for i, frame in enumerate(trial['frames']):
        source = Path(frame['path']); dest = directory / f'frame_{i:04d}.png'
        shutil.copy2(source, dest)
        frames.append({**frame, **binding(dest), 'evidence_ref': f'frame:{i}'})
    render = None
    if trial.get('recorded_pose_render'):
        saved = read(trial['recorded_pose_render']['path'])
        saved['retained_frames'] = [{k: v for k, v in f.items() if k != 'evidence_ref'} for f in frames]
        write_json(directory / 'evidence_render.json', saved)
        render = binding(directory / 'evidence_render.json')
    request = {'policy': POLICY, 'input': binding(input_path), 'trial_id': trial['id'],
        'kind': trial['kind'], 'joint_name': trial['joint_name'],
        'inputs': copy.deepcopy(runtime.report['inputs']), 'sources': copy.deepcopy(runtime.report['source_hashes']),
        'trajectory': binding(trial['trajectory_path']), 'raw_metrics': copy.deepcopy(trial['metrics']),
        'events': summarize_events(trace, trial['metrics'], runtime.contract, runtime.spec, trial['joint_name']),
        'frames': frames, 'evidence_render': render, 'replay_scene': trial.get('replay_scene')}
    if render:
        request['media_trial'] = {k: copy.deepcopy(trial[k]) for k in ('trajectory_path', 'trajectory_sha256',
            'media_policy', 'replay_scene', 'camera', 'joint_name', 'kind', 'id', 'recording_mode')}
        request['media_trial'].update(passed=False, recorded_pose_render=render,
            frames=[{k: v for k, v in f.items() if k != 'evidence_ref'} for f in frames])
    path = directory / 'request.json'; write_json(path, request)
    runtime.report.update(stage='incidental_motion_review', active_trial=trial['id'],
                          pending_incidental_motion=binding(path))
    write_json(runtime.args.out_dir / 'report.json', runtime.report)
    response = directory / 'decision.json'
    started = time.monotonic()
    while not response.is_file():
        if time.monotonic() - started > 1900:
            raise TimeoutError('Incidental-motion API review did not finish; no physical verdict inferred')
        time.sleep(.2)
    receipt = read(response)
    if receipt.get('policy') != POLICY or receipt.get('request') != binding(path):
        raise ValueError('Owner returned a decision for different incidental-motion evidence')
    refs = {'joint:' + n for n in names} | {f['evidence_ref'] for f in frames}
    decision = validate_decision(receipt['decision'], names, refs)
    trial['raw_numerical_passed'] = trial['metrics']['passed']
    trial['metrics'] = apply_decision(trial['metrics'], trial['kind'], decision)
    trial['passed'] = trial['metrics']['passed']
    trial['incidental_motion_review'] = binding(response)
    runtime.report.pop('pending_incidental_motion', None)


def payload(request, plan, *, supplemental=False):
    m = request['raw_metrics']['metrics']
    value = {'trial_id': request['trial_id'], 'operation': {'kind': request['kind'], 'joint': request['joint_name']},
        'events': request['events'], 'other_numerical_errors': [e for e in request['raw_metrics']['errors'] if not e.startswith(PREFIX)],
        'operation_metrics': {k: m[k] for k in ('coverage', 'endpoint_errors', 'endpoint_actual_holds_s',
            'max_tilt_deg', 'horizontal_slip_m', 'max_horizontal_displacement_m', 'free_slip_tolerance_m', 'passive_tail') if k in m},
        'joint_definitions': read(checked(request['inputs']['spec']))['joints'],
        'physical_parameters': read(checked(request['inputs']['profile'])),
        'frozen_joint_contracts': read(checked(plan['contract']))['joints']}
    if supplemental:
        value['supplemental'] = 'Additional recorded views and full per-joint native metrics; use uncertain if still insufficient.'
        value['all_joint_metrics'] = m['joints']
    return value


def content_for(api, request, plan, *, supplemental=False):
    frames = request['frames']
    # Keep first/final and the largest measured deviation; bound API images.
    count = 12 if supplemental else 6
    indices = {0, len(frames)-1}
    events = sorted(request['events'], key=lambda e: (-e['max_deviation']/e['tolerance'], e['joint_name']))
    for e in events:
        for t in (e['maximum_time_s'], e['first_exceeded_time_s']):
            if len(indices) < count:
                indices.add(min(range(len(frames)), key=lambda i: abs(frames[i]['time_seconds']-t)))
    for i in range(count):
        if len(indices) < count:
            indices.add(round(i*(len(frames)-1)/(count-1)))
    chosen = [frames[i] for i in sorted(indices)]
    content = [api.make_text_input(plan['prompt']), api.make_text_input(json.dumps(payload(request, plan, supplemental=supplemental), ensure_ascii=False))]
    for item in plan['source_images']:
        content.extend([api.make_text_input('Original object crop'), api.make_image_input(checked(item))])
    for item in chosen:
        content.extend([api.make_text_input(json.dumps({k: item[k] for k in ('evidence_ref', 'time_seconds', 'positions')})),
                        api.make_image_input(checked(item))])
    refs = {'joint:' + e['joint_name'] for e in request['events']} | {f['evidence_ref'] for f in chosen}
    return content, refs


def service_request(path, args, api):
    """The host owns networking; the paused Isaac process owns no API client."""
    from .loop import _audited_ask, _audit_content
    from .critic import ask_initial_with_budget
    from urdf_core.delivery_policy import SharedBudget
    from .trajectory_io import iter_trajectory_samples
    from .metrics import evaluate_trial
    import os
    request = read(path); plan = read(checked(request['input']))
    if (request['policy'] != POLICY or plan['policy'] != POLICY or plan['prompt'] != PROMPT
            or plan['model'] != args.model or plan['reasoning_effort'] != os.environ.get('URDF_REASONING_EFFORT', '')):
        raise ValueError('Unexpected incidental-motion review configuration')
    contract = read(checked(plan['contract']))
    trace = list(iter_trajectory_samples(checked(request['trajectory'])))
    raw = evaluate_trial(trace, kind=request['kind'], joint_name=request['joint_name'], contract=contract)
    from .verification import _same_metrics
    if not same_raw(raw, request['raw_metrics']):
        raise ValueError('Incidental-motion request differs from native measurements')
    expected_events = summarize_events(trace, raw, contract, read(checked(request['inputs']['spec'])), request['joint_name'])
    if not _same_metrics(expected_events, request['events']):
        raise ValueError('Incidental-motion event summary differs from native measurements')
    if request.get('media_trial'):
        from .deferred_capture import verify_media, POLICY as MEDIA_POLICY
        verify_media(request['media_trial'], {'media_policy': MEDIA_POLICY, 'passed': False}, contract, Path(path).parents[2])
    names = pending_names(raw, request['kind'])
    directory = Path(path).parent
    ask = _audited_ask(args, api, directory)
    decisions = []
    budget = SharedBudget(args.out_dir, 'physics', args.max_physics_repairs)
    for supplemental in (False, True):
        content, refs = content_for(api, request, plan, supplemental=supplemental)
        stage = 'incidental_motion_supplement' if supplemental else 'incidental_motion'
        # The per-trial stage prefix also makes shared budget reservations unique.
        stage += '_' + fingerprint(str(Path(path).resolve()))[:16]
        result = ask_initial_with_budget(ask, api, stage, content,
            lambda v: validate_decision(v, names, refs), budget)
        decisions.append({'supplemental': supplemental, 'result': result,
                          'content_sha256': fingerprint(_audit_content(content))})
        if all(x['verdict'] != 'uncertain' for x in result['items']):
            break
    calls = sorted((directory / 'model_calls').glob('*'))
    write_json(directory / 'decision.json', {'policy': POLICY, 'request': binding(path),
        'rounds': decisions, 'decision': result, 'api_records': [binding(p) for p in calls if p.is_file()]})


def verify_receipt(record, *, raw, trial, report, root):
    """Replay saved API decisions and their exact inputs; no network during audit."""
    from .verification import _same_metrics
    from .loop import _audit_content
    import generate_object_urdf as api
    from urdf_core.prompt_payload import prepare_content
    path = checked(record, root); receipt = read(path)
    request_path = checked(receipt['request'], root); request = read(request_path)
    plan_path = checked(request['input'], Path(root).parent); plan = read(plan_path)
    if (receipt['policy'] != POLICY or plan['policy'] != POLICY or plan['prompt'] != PROMPT
            or request['input'] != report['inputs'].get('incidental_motion')
            or plan['contract'] != report['inputs'].get('contract')
            or request['inputs'] != report['inputs'] or request['sources'] != report['source_hashes']
            or request['kind'] != trial['kind'] or request['trial_id'] != trial['id']
            or request['joint_name'] != trial['joint_name']
            or request['trajectory'] != {'path': trial['trajectory_path'], 'sha256': trial['trajectory_sha256']}
            or not same_raw(request['raw_metrics'], raw)):
        raise ValueError('Incidental-motion decision is not bound to this exact candidate and trial')
    files = {path, request_path, plan_path, checked(request['trajectory'], root)}
    if request.get('media_trial'):
        from .deferred_capture import verify_media, POLICY as MEDIA_POLICY
        files.update(verify_media(request['media_trial'], {'media_policy': MEDIA_POLICY, 'passed': False},
                                 read(checked(plan['contract'])), root))
    for item in request['inputs'].values():
        files.add(checked(item))
    for item in plan['source_images'] + request['frames'] + receipt['api_records']:
        files.add(checked(item))
    rounds = receipt['rounds']
    if not 1 <= len(rounds) <= 2 or [r['supplemental'] for r in rounds] != [False, True][:len(rounds)]:
        raise ValueError('Invalid incidental-motion evidence review sequence')
    names = pending_names(raw, trial['kind'])
    previous = None
    for row in rounds:
        if previous and not any(x['verdict'] == 'uncertain' for x in previous['items']):
            raise ValueError('Cannot request another semantic opinion after a conclusive decision')
        content, refs = content_for(api, request, plan, supplemental=row['supplemental'])
        audited = _audit_content(content)
        if row['content_sha256'] != fingerprint(audited):
            raise ValueError('Incidental-motion API content changed')
        request_content = _audit_content(prepare_content(content))
        stage = 'incidental_motion_supplement' if row['supplemental'] else 'incidental_motion'
        found = False
        calls = sorted(path.parent.glob(f'model_calls/*_{stage}*_try_*.request.json'))
        for call in calls:
            reply = call.with_name(call.name.removesuffix('.request.json') + '.response.txt')
            if not reply.is_file():
                continue
            if call.resolve() not in files or reply.resolve() not in files:
                raise ValueError('Unbound incidental-motion API receipt')
            saved = read(call)
            if saved['content'][:len(request_content)] != request_content:
                continue  # Another supplemental/initial review of this same trial.
            if saved['model'] != plan['model'] or saved['reasoning_effort'] != plan['reasoning_effort']:
                raise ValueError('Incidental-motion response received different evidence')
            try:
                parsed = validate_decision(api.extract_json(reply.read_text()), names, refs)
            except (ValueError, KeyError, TypeError):
                continue
            if parsed != row['result']:
                raise ValueError('Recorded incidental-motion verdict differs from the API response')
            found = True
            break
        if not found:
            raise ValueError('No received API decision supports incidental-motion verdict')
        previous = row['result']
    if receipt['decision'] != previous:
        raise ValueError('Incidental-motion final decision differs from received response')
    return previous, files
