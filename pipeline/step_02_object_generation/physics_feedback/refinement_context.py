"""Bound diagnostic API context independently of retained local evidence.

This is a transport projection, never a physical critic or parameter proposal.
All trial verdicts, failed metrics, permissions and tested patch outcomes remain.
Only automatically selected diagnostic samples/images enter a finite request.
"""
from __future__ import annotations

import copy
from collections import defaultdict

from urdf_core.run_state import fingerprint
from .api_evidence import (DEFAULT_MAX_BYTES, EvidenceProjectionTooLarge, _compact_evidence,
    _sample, _prepare_refinement_payload,
    serialize_refinement_payload)

# Engineering headroom, not a claim about the configured model's context size.
# o200k_base is an explicit local proxy; images and instructions need headroom.
MAX_TEXT_TOKENS = 128000
MAX_IMAGES = 8


def text_tokens(value):
    import tiktoken
    return len(tiktoken.get_encoding("o200k_base").encode(
        serialize_refinement_payload(value), disallowed_special=()))


def _without_audit(value):
    """Remove local file identities, never native values or body/link names."""
    if isinstance(value, dict):
        return {key: _without_audit(child) for key, child in value.items()
                if key not in {'path', 'source_report', 'source_evidence_content_sha256'}
                and not key.endswith(('_sha256', '_path')) and key != 'sha256'}
    if isinstance(value, list):
        return [_without_audit(child) for child in value]
    return copy.deepcopy(value)


def _checks(values):
    # These generated hash labels duplicate the human-readable error list.
    # Keep every real normalized measurement; retain ALL errors separately.
    return {key: value for key, value in values.items() if '/unresolved/' not in key}


def _history_summary(rows):
    """One deterministic result per tested candidate; no operator event text.

    Exact API patch values and every measured failure survive. Repeated error
    strings use a small dictionary; full hashes/receipts remain on disk and the
    duplicate-candidate validator still consumes the ORIGINAL complete rows.
    """
    errors, groups, result = [], {}, []
    for row in rows:
        identity = (row.get('implementation'), row.get('pipeline_version'))
        group = groups.setdefault(identity, 'implementation_' + str(len(groups)))
        measured = row['result']
        failures = []
        for failure in measured.get('observed_failures', []):
            entry = {key: copy.deepcopy(failure[key]) for key in
                     ('failure_id', 'phase', 'trial_id', 'joint_name') if key in failure}
            indices = []
            for error in failure.get('errors', []):
                if error not in errors:
                    errors.append(error)
                indices.append(errors.index(error))
            entry['error_ids'] = indices
            failures.append(entry)
        patches = []
        for patch in row.get('patches', []):
            # Validated numeric parameter patches only, never arbitrary nested
            # historical annotations or operator suggestions.
            if (not isinstance(patch.get('path'), str)
                    or type(patch.get('value')) not in (float, int)):
                raise ValueError('Historical API patch must contain a numeric path/value')
            patches.append({'path': patch['path'], 'value': patch['value']})
        result.append({'candidate_id': row['candidate_id'], 'parent': row.get('parent'),
            **({'parameter_origin': 'saved_random_representative_copy'}
               if row.get('parameter_origin') == 'saved_random_representative_copy' else {}),
            'implementation_group': group,
            'same_implementation_and_policy': row.get('same_implementation_and_policy', False),
            'patches': patches,
            'result': {**{key: copy.deepcopy(measured[key]) for key in
                         ('passed', 'stage_progress', 'passed_trials', 'normalized_violation') if key in measured},
                       'failing_checks': _checks(measured.get('failing_checks', {})),
                       'observed_failures': failures}})
    return {'candidate_count': len(result), 'errors': errors, 'trials': result,
        'interpretation': 'Every tested candidate is listed. Patches are relative to its named parent, '
            'not the previous row. error_ids index errors. Distinct implementation_group values '
            'denote different code/policy versions, so outcomes are not directly comparable. '
            'Historical outcomes are not new acceptance. Exact duplicate proposals are checked '
            'against complete local evidence by code; hashes and audit events are omitted here.'}


def _control_linearization_context(value, kind):
    """Quadratic diagnostic arrays stay local; never truncate actual failed metrics."""
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    dense = {'effective_joint_mass_matrix', 'native_generalized_mass_matrix',
             'dt_inverse_mass_kd', 'dt_squared_inverse_mass_kp'}
    if kind == 'ground_settle':
        # Stable has zero active effort. A hypothetical PD stability model is not
        # evidence of passive holding and should not dominate its diagnosis.
        return {**{k: copy.deepcopy(value[k]) for k in ('schema_version', 'available',
                   'acceptance_gate', 'parameters_modified', 'physics_dt_s', 'mass_source',
                   'native_matrix_shape', 'assumptions') if k in value},
                'active_control_in_this_trial': False,
                'detail_retained_locally': True,
                'omission_reason': 'Zero-effort Stable trial; active-controller linearization is inapplicable.'}
    result = {k: copy.deepcopy(v) for k, v in value.items() if k not in dense}
    summaries = {}
    for key in sorted(dense & value.keys()):
        matrix = value[key]
        if not isinstance(matrix, list) or not matrix:
            summaries[key] = {'detail_retained_locally': True}
            continue
        widths = {len(row) for row in matrix if isinstance(row, list)}
        if len(widths) != 1 or len(matrix) != len([r for r in matrix if isinstance(r, list)]):
            raise ValueError('Invalid diagnostic matrix shape')
        width = next(iter(widths))
        summaries[key] = {'shape': [len(matrix), width],
            'diagonal': [matrix[i][i] for i in range(min(len(matrix), width))],
            'row_absolute_off_diagonal_sum': [sum(abs(x) for j, x in enumerate(row) if i != j)
                                             for i, row in enumerate(matrix)],
            'full_matrix_retained_locally': True,
            'omitted_entries_are_not_zero': True}
    result['matrix_summaries'] = summaries
    return result


def _focused_projection(payload, radius):
    # Do not construct/send the append-only bookkeeping ledger at all. It stays
    # hash-bound locally. The older projector still validates the current/best
    # parent relationship and acceptance policy before we select diagnostics.
    transport_input = {**payload, 'history': [], 'candidates': []}
    transport_input.pop('tested_candidate_trials', None)
    result = _prepare_refinement_payload(transport_input)
    from .diagnosis import refinement_parameter_rules
    result['parameter_rules'] = refinement_parameter_rules()
    result.pop('history', None)
    result.pop('candidates', None)
    result['observed_failures'] = _without_audit(result['observed_failures'])
    result['experiment_history'] = _history_summary(payload.get('tested_candidate_trials', []))
    raw = {value['candidate']['id']: value['evidence'] for value in (payload['current'], payload['best'])}
    diagnostic = payload.get('diagnostic_source_candidate_id', payload['current']['candidate']['id'])
    for identity, context in result['candidate_contexts'].items():
        context['candidate'] = {key: copy.deepcopy(context['candidate'][key]) for key in
            ('id', 'index', 'status', 'parent', 'cycle_id', 'budget_family', 'diagnostic_source_candidate_id')
            if key in context['candidate']}
        context.pop('parameters_content_sha256', None)
        evidence = context['evidence'] = _without_audit(context['evidence'])
        evidence['numeric_summary']['failing_checks'] = _checks(evidence['numeric_summary']['failing_checks'])
        for name, trial in evidence['trials'].items():
            setup = trial.get('setup', {})
            if 'control_discrete_linearization' in setup:
                setup['control_discrete_linearization'] = _control_linearization_context(
                    setup['control_discrete_linearization'], trial.get('kind'))
            if trial.get('passed') is True or identity != diagnostic:
                evidence['trials'][name] = {key: trial[key] for key in
                    ('id', 'kind', 'joint_name', 'passed', 'status') if key in trial}
                evidence['trials'][name]['detail_retained_locally'] = True
        if identity != diagnostic:
            evidence['trajectory_windows'] = {}
            evidence['temporal_detail'] = 'Comparison candidate: outcome and normalized failures retained; detailed metrics/time series remain local.'
            continue
        for name, window in evidence['trajectory_windows'].items():
            if not window.get('available'):
                continue
            source = raw[identity]['trajectory_windows'][name]
            target = evidence['trials'][name].get('joint_name')
            anchors = source['anchors']
            # Always retain actual crossed acceptance signals and operated-joint
            # tracking/effort context. Speed/depth maxima remain diagnostics.
            joint_names = set(payload['contract'].get('joints', {}))
            for native_sample in source['samples'].values():
                joint_names.update(native_sample.get('joint_positions', {}))
            measured_errors = evidence['trials'][name].get('metrics', {}).get('errors', [])
            failed_joints = {joint for joint in joint_names
                             if any(joint in error for error in measured_errors)}
            failed_joints.update(f['joint_name'] for f in evidence['observed_failures']
                                 if f.get('trial_id') == name and f.get('joint_name'))
            selected = {key: value for key, value in anchors.items()
                if key.split('/')[0] in failed_joints | ({target} if target else set())
                or (key.startswith('root/') and value.get('first') is not None
                    and value.get('acceptance_gate') is not False)}
            if not selected:
                # Unknown failure formats get conservative crossed-signal
                # evidence, never an empty hand-picked explanation.
                selected = {key: value for key, value in anchors.items()
                            if value.get('first') is not None and value.get('acceptance_gate') is not False}
            centers = {int(value[k]) for value in selected.values()
                       for k in ('first', 'max') if value.get(k) is not None}
            if any(str(i) not in source['samples'] for i in centers):
                raise ValueError('Selected first/max anchor lacks its native sample')
            indices = sorted({i for center in centers for i in range(center-radius, center+radius+1)
                              if str(i) in source['samples']})
            joints = set(payload['contract'].get('joints', {}))
            # Legacy contracts can lack the joint map; native sample names are
            # still observations, not a reason to discard a failed joint.
            for sample in source['samples'].values():
                joints.update(sample.get('joint_positions', {}))
            relevant = {key.split('/')[0] for key in selected if key.split('/')[0] in joints}
            if target is not None:
                relevant.add(target)
            errors = evidence['trials'][name].get('metrics', {}).get('errors', [])
            relevant.update(joint for joint in joints if any(joint in error for error in errors))
            samples = {str(i): _without_audit(_sample(source['samples'][str(i)])) for i in indices}
            for sample in samples.values():
                for key in ('joint_positions', 'joint_velocities', 'joint_targets', 'joint_efforts'):
                    if key in sample:
                        sample[key] = {joint: value for joint, value in sample[key].items() if joint in relevant}
            window.update(samples=samples, api_selected_sample_count=len(indices), api_radius_steps=radius,
                sample_selection={'policy': 'operated_joint_and_crossed_signals.v1',
                    'selected_anchor_names': sorted(selected), 'sampled_joint_names': sorted(relevant),
                    'joint_series_omitted': sorted(joints-relevant),
                    'all_anchor_metadata_preserved': True, 'failed_trial_metrics_preserved': True,
                    'omitted_series_are_not_evidence_of_no_motion': True})
    result['projection'] = {'policy': 'failure_focused_diagnostics.v1',
        'source_payload_content_sha256': fingerprint(payload),
        'full_evidence_is_retained_locally': True, 'acceptance_and_permissions_unchanged': True,
        'current_failed_metrics_exact': True, 'historical_patch_values_exact': True,
        'audit_events_and_indices_sent': False, 'passed_trial_details_sent': False,
        'notes': ['Passed/blocked trial identities remain; omitted details do not imply zero motion.',
                  'First/max failure samples retain exact native numbers; no rounding is used.',
                  'Speed and penetration diagnostics do not create acceptance gates.']}
    return result




def project_context(payload, *, max_text_tokens=MAX_TEXT_TOKENS, max_bytes=DEFAULT_MAX_BYTES):
    """Default to focused evidence; retain full evidence for local verification."""
    if type(max_text_tokens) is not int or max_text_tokens <= 0:
        raise ValueError('Diagnostic text budget must be a positive integer')
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('Diagnostic byte budget must be a positive integer')
    for radius in (1, 0):
        result = _focused_projection(payload, radius)
        result['projection']['diagnostic_context_budget'] = {
            'tokenizer_proxy': 'o200k_base', 'max_payload_text_tokens': max_text_tokens,
            'max_payload_utf8_bytes': max_bytes,
            'instructions_and_images_additional': True}
        readable = result
        readable_tokens = text_tokens(result)
        # Exact dictionary/table sharing precedes reduction to anchor-only
        # samples. Authority fields and every historical experiment stay exact.
        result = _compact_evidence(result)
        result['evidence_encoding']['instructions'][3] = (
            'Window sample_selection declares the retained joints/anchors and radius. '
            'Failed trial metrics and all anchor metadata remain exact; series for other '
            'joints and historical candidates stay local. Omission proves no absence of motion.')
        compact_tokens = text_tokens(result)
        fitting = [(tokens, value) for tokens, value in
                   ((readable_tokens, readable), (compact_tokens, result))
                   if tokens <= max_text_tokens
                   and len(serialize_refinement_payload(value).encode('utf-8')) <= max_bytes]
        if fitting:
            return min(fitting, key=lambda item: item[0])[1]
    raise EvidenceProjectionTooLarge('Failed metrics, parameter history and frozen permissions '
        f'cannot fit diagnostic payload allowance {max_text_tokens} tokens/{max_bytes} bytes; '
        'no API request was sent')


def select_refinement_frames(frames, *, max_images=MAX_IMAGES, source_image_count=0, failed_trial_ids=None):
    """Balanced, identity-independent initial/end/extreme/event selection.

    Retention means a file stays on disk. It does not mean send every retained
    frame. Each supplied trial remains represented; no trial is silently lost.
    """
    if type(max_images) is not int or type(source_image_count) is not int or source_image_count < 0:
        raise ValueError('Invalid diagnostic image allowance')
    original_frames = frames
    if failed_trial_ids is not None:
        failed_trial_ids = set(failed_trial_ids)
        frames = [frame for frame in frames if frame['trial_id'] in failed_trial_ids]
        if failed_trial_ids - {frame['trial_id'] for frame in frames}:
            raise ValueError('Failed trial is missing diagnostic images')
    limit = max_images-source_image_count
    groups = defaultdict(list)
    for frame in frames:
        groups[frame['trial_id']].append(frame)
    if not groups or limit < sum(min(2, len(rows)) for rows in groups.values()):
        raise EvidenceProjectionTooLarge('Image allowance cannot represent initial/end for every trial')
    selected, priorities = {}, {}
    for trial, rows in groups.items():
        rows.sort(key=lambda f: (float(f['time_seconds']), f['frame_index']))
        selected[trial] = {0, len(rows)-1}
        events = [i for i, f in enumerate(rows) if f.get('failure_event') is True]
        order = [*events[:1], *events[-1:]]
        names = sorted(set().union(*(set(f.get('positions', {})) for f in rows)))
        # The largest observed per-joint extent has a stable numerical ranking;
        # it does not select a physical hypothesis or adjust any parameter.
        extrema = []
        for name in names:
            qs = [f.get('positions', {}).get(name) for f in rows]
            if all(type(q) in (float, int) for q in qs):
                lo, hi = min(range(len(qs)), key=qs.__getitem__), max(range(len(qs)), key=qs.__getitem__)
                extrema.append((max(qs)-min(qs), name, lo, hi))
        for _, _, lo, hi in sorted(extrema, key=lambda x: (-x[0], x[1])):
            order.extend((lo, hi))
        order.extend(round(i*(len(rows)-1)/6) for i in range(7))
        priorities[trial] = list(dict.fromkeys(i for i in order if i not in selected[trial]))
    count = sum(map(len, selected.values()))
    while count < limit:
        changed = False
        for trial in sorted(groups):
            if priorities[trial] and count < limit:
                selected[trial].add(priorities[trial].pop(0)); count += 1; changed = True
        if not changed:
            break
    result = [groups[trial][i] for trial in sorted(groups) for i in sorted(selected[trial])]
    manifest = {'policy': 'failure_focused_native_frames.v2', 'max_images': max_images,
        'source_image_count': source_image_count, 'available_frame_count': len(frames),
        'selected_frame_count': len(result), 'available_frames_content_sha256': fingerprint(frames),
        'trials': {trial: {'available': len(groups[trial]),
            'selected_frame_refs': [groups[trial][i]['frame_ref'] for i in sorted(selected[trial])]} for trial in sorted(groups)},
        'omitted_trial_ids': sorted({f['trial_id'] for f in original_frames} - set(groups)),
        'local_retention_unchanged': True}
    return result, manifest
