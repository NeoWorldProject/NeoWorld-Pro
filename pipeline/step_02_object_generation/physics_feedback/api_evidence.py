"""Bounded API-only diagnostic projection; original evidence stays immutable.

This module does not judge or alter physical acceptance, parameters or budgets.
Every omission is backed by local source hashes. A context that cannot preserve
mandatory evidence inside the byte budget fails locally before an API request.
"""
from __future__ import annotations

import copy
from collections import Counter
import json
import math

from urdf_core.run_state import fingerprint

# Fixed transport ceiling, not an acceptance threshold. A completed multi-DOF
# motion phase requires more context than a single failed trial. Its native
# first/max windows reached 1.22 MB even after lossless encoding. Keep those
# observations, permissions and history intact; never remove joints or round
# measurements to make a request fit. Explicit smaller ceilings still fail
# locally before any API call.
DEFAULT_MAX_BYTES = 2000000


class EvidenceProjectionTooLarge(ValueError):
    pass


def serialize_refinement_payload(value):
    """Use the same lossless UTF-8 JSON for byte accounting and model input."""
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(',', ':'))


def _size(value):
    return len(serialize_refinement_payload(value).encode('utf-8'))


def _pick(value, keys):
    return {key: copy.deepcopy(value[key]) for key in keys if key in value}


def _norm(value):
    return math.sqrt(sum(float(x) ** 2 for x in value)) if isinstance(value, list) else abs(float(value))


def contact_summary(contacts):
    """Aggregate active pairs once; events may repeat those same contact points."""
    pairs, events = contacts.get('pairs', []), contacts.get('events', [])
    groups = {}
    for pair in pairs:
        bodies = tuple(str(pair.get(key, pair.get('actor' + str(i), 'unknown')))
                       for i, key in enumerate(('body0', 'body1')))
        group = groups.setdefault(bodies, {'bodies': list(bodies), 'collider_pair_count': 0,
                                           'point_count': 0, 'min_separation_m': None, 'max_impulse_Ns': 0.,
                                           'net_impulse_Ns': [0., 0., 0.]})
        group['collider_pair_count'] += 1
        for point in pair.get('points', []):
            group['point_count'] += 1
            separation = point.get('separation_m')
            if isinstance(separation, (float, int)) and math.isfinite(separation):
                previous = group['min_separation_m']
                group['min_separation_m'] = separation if previous is None else min(previous, separation)
                if previous is None or separation < previous:
                    group['deepest_contact'] = _pick(point, ('position_m', 'normal', 'normal_world', 'separation_m', 'impulse_Ns'))
            impulse = point.get('impulse_Ns')
            if impulse is not None:
                magnitude = _norm(impulse)
                if 'strongest_impulse_contact' not in group or magnitude > group['max_impulse_Ns']:
                    group['strongest_impulse_contact'] = _pick(point, ('position_m', 'normal', 'normal_world', 'separation_m', 'impulse_Ns'))
                group['max_impulse_Ns'] = max(group['max_impulse_Ns'], magnitude)
                if isinstance(impulse, list) and len(impulse) == 3:
                    group['net_impulse_Ns'] = [a + float(b) for a, b in zip(group['net_impulse_Ns'], impulse)]
    ordered = sorted(groups.values(), key=lambda g: (g['min_separation_m'] if g['min_separation_m'] is not None else 0., g['bodies']))
    return {'observed': contacts.get('observed'), 'state_semantics': contacts.get('state_semantics'),
            'active_pair_count': len(pairs), 'body_pair_count': len(groups),
            'point_count': sum(g['point_count'] for g in ordered),
            'min_separation_m': min((g['min_separation_m'] for g in ordered if g['min_separation_m'] is not None), default=None),
            'max_impulse_Ns': max((g['max_impulse_Ns'] for g in ordered), default=None),
            'event_count': len(events), 'event_type_counts': dict(Counter(str(e.get('event_type', 'unknown')) for e in events)),
            'active_body_pairs': [list(pair) for pair in sorted(groups)],
            'body_pairs': ordered[:8], 'omitted_body_pair_count': max(0, len(ordered) - 8),
            'source_content_sha256': fingerprint(contacts),
            'point_arrays_omitted': True, 'events_not_double_counted_as_active_points': True}


def geometry_summary(geometry):
    diagnostics = geometry.get('diagnostics', [])
    depths = [max(float(item.get('depth_m', 0.) or 0.), -float(item.get('separation_m', 0.) or 0.), 0.)
              for item in diagnostics if isinstance(item, dict)]
    return {**_pick(geometry, ('passed', 'complete', 'status')),
            'issue_count': len(geometry.get('issues', [])),
            'issue_type_counts': dict(Counter(str(item.get('type', item.get('kind', 'unknown'))) if isinstance(item, dict)
                                             else str(item) for item in geometry.get('issues', []))),
            'diagnostic_type_counts': dict(Counter(str(item.get('type', 'unknown')) for item in diagnostics if isinstance(item, dict))),
            'max_diagnostic_depth_m': max(depths, default=None), 'depth_acceptance_gate': False,
            'source_content_sha256': fingerprint(geometry), 'full_geometry_omitted': True}


def _sample(step):
    result = _pick(step, ('time_s', 'root_position', 'root_quaternion_wxyz', 'joint_positions', 'joint_velocities',
                         'joint_targets', 'joint_efforts', 'derived_trajectory_phase'))
    result['contacts_summary'] = contact_summary(step.get('contacts', {}))
    result['geometry_summary'] = geometry_summary(step.get('geometry', {}))
    velocities = step.get('link_velocities', {})
    result['link_velocity_extrema'] = {
        'max_linear_m_s': max((_norm(v.get('linear_m_s', [0., 0., 0.])) for v in velocities.values()), default=None),
        'max_angular_rad_s': max((_norm(v.get('angular_rad_s', [0., 0., 0.])) for v in velocities.values()), default=None),
        'link_count': len(velocities)}
    return result


def _windows(window, radius):
    if window.get('available') is not True:
        return copy.deepcopy(window)
    anchors, samples = window.get('anchors', {}), window.get('samples', {})
    centers = {int(item[key]) for item in anchors.values() for key in ('first', 'max') if item.get(key) is not None}
    if any(str(index) not in samples for index in centers):
        raise ValueError('Bound diagnostic window lacks an explicitly recorded first/max anchor')
    indices = sorted({index for center in centers for index in range(center-radius, center+radius+1) if str(index) in samples})
    return {**_pick(window, ('available', 'path', 'sha256', 'sample_count', 'anchors')),
            'original_radius_steps': window.get('radius_steps'), 'api_radius_steps': radius,
            'original_selected_sample_count': len(samples), 'api_selected_sample_count': len(indices),
            'samples': {str(index): _sample(samples[str(index)]) for index in indices},
            'source_window_content_sha256': fingerprint(window)}


def _trial(trial):
    row = _pick(trial, ('id', 'name', 'kind', 'joint_name', 'passed', 'status', 'duration_s', 'actual_duration_s',
                       'expected_duration_s', 'completed_physics_steps', 'expected_physics_steps', 'terminated_early',
                       'trajectory_path', 'trajectory_sha256'))
    # Numerical aggregate dictionaries contain no raw contact or mesh arrays.
    row['metrics'] = copy.deepcopy(trial.get('metrics', {}))
    row['setup'] = _pick(trial.get('setup', {}), ('fixed_root', 'reloaded', 'measured_properties', 'native_joint_friction',
        'control_discrete_linearization',
        'native_articulation', 'gravity_enabled', 'self_collision_enabled', 'ground_material', 'solver_policy',
        'joint_position_writes', 'active_drive_stiffness', 'target_speed_fraction', 'target_speed_limits', 'passive_damping'))
    setup = trial.get('setup', {})
    plan = setup.get('motion_scope_plan')
    if isinstance(plan, dict):
        # Source validation belongs to the native verifier. Preserve its bound
        # receipt identities and declared scope without resending collider curves
        # or deriving a new applicability/acceptance claim here.
        row['setup']['motion_scope_plan_summary'] = {
            **_pick(plan, ('schema_version', 'planning_only', 'scope', 'joint_name', 'joint_type',
                'plan_sha256', 'source_input_sha256', 'full_lower', 'full_upper', 'initial_position',
                'free_lower', 'free_upper', 'free_status', 'free_range_fraction', 'blocked_lower',
                'blocked_upper', 'range_resolution', 'root_transform', 'fixed_min_z_m', 'fixed_clearance_lift_m')),
            'source_content_sha256': fingerprint(plan)}
        row['setup'].update(_pick(setup, ('fixture_clearance_m', 'cooked_geometry_sha256')))
        if isinstance(setup.get('motion_scope_geometry'), dict):
            row['setup']['motion_scope_geometry'] = _pick(setup['motion_scope_geometry'], ('path', 'sha256'))
        row['source_trial_content_sha256'] = fingerprint(trial)
    return row


def _failures(values, prefix):
    return [{**_pick(value, ('failure_id', 'phase', 'trial_id', 'joint_name', 'errors', 'terminated_early')),
             'metrics_ref': prefix + '/' + str(value.get('trial_id')),
             'source_failure_content_sha256': fingerprint(value)} for value in values]


def _parameters_diff(current, best, path=()):
    if current == best:
        return []
    if isinstance(current, dict) and isinstance(best, dict):
        return [item for key in sorted(set(current) | set(best))
                for item in _parameters_diff(current.get(key), best.get(key), (*path, key))]
    return [{'path': list(path), 'best': copy.deepcopy(best), 'current': copy.deepcopy(current)}]


def _candidate_metadata(candidate):
    return {**_pick(candidate, ('id', 'index', 'status', 'cycle_id', 'parent', 'budget_family', 'snapshot_path',
        'snapshot_sha256', 'evidence_path', 'evidence_sha256', 'report_path', 'diagnostic_source_candidate_id')),
        'source_candidate_content_sha256': fingerprint(candidate)}


def _project_evidence(evidence, candidate_id, radius):
    report = evidence['full_report']
    trials = list(report.get('trials', []))
    if isinstance(report.get('reload_check', {}).get('trial'), dict):
        trials.append(report['reload_check']['trial'])
    trial_rows = {str(t.get('id', t.get('kind'))): _trial(t) for t in trials}
    failures = evidence.get('observed_failures', [])
    failed_ids = {str(f['trial_id']) for f in failures if f.get('trial_id') is not None}
    windows = evidence.get('trajectory_windows', {})
    numbers = evidence.get('numeric_summary', {})
    result = {'source_report': copy.deepcopy(evidence['report']), 'source_evidence_content_sha256': fingerprint(evidence),
        'report_verdict': _pick(report, ('status', 'passed', 'numerical_passed', 'evaluation_complete',
            'acceptance_complete', 'acceptance_policy', 'stopped_after_phase', 'blocked_trials', 'errors', 'issues',
            'motion_scope_policy', 'orientation_policy', 'trial_evaluation_policy')),
        'trials': trial_rows,
        'observed_failures': _failures(failures, 'candidate_contexts/' + candidate_id + '/evidence/trials'),
        'numeric_summary': {**_pick(numbers, ('passed', 'stage_progress', 'passed_trials', 'normalized_violation')),
            'failing_checks': {k: v for k, v in numbers.get('checks', {}).items() if v > 1. + 1e-6},
            'check_count': len(numbers.get('checks', {})), 'source_content_sha256': fingerprint(numbers)},
        'trajectory_windows': {name: _windows(value, radius) for name, value in windows.items() if name in failed_ids},
        'windows_omitted_for_nonfailed_trials': sorted((set(windows) | {
            str(t.get('id', t.get('kind'))) for t in trials if t.get('passed') is True}) - failed_ids)}
    if 'collision_geometry' in evidence:
        result['collision_geometry'] = copy.deepcopy(evidence['collision_geometry'])
    if 'controller_selections' in report:
        # Native execution facts must survive the compact API projection. A
        # contract's gains alone do not describe an inertia-based calculation.
        # Admit only measured trials, known phases and numerical-policy fields;
        # arbitrary operator annotations never become model instructions.
        from .control import INERTIA_POLICIES, INTEGRAL_POLICIES
        selections = report['controller_selections']
        if not isinstance(selections, dict):
            raise ValueError('Malformed native controller execution evidence')
        execution = {}
        for trial_id, phases in selections.items():
            if trial_id not in trial_rows:
                continue
            if not isinstance(phases, dict):
                raise ValueError('Malformed native controller execution phase')
            observed = {}
            for phase in ('pre_motion_hold', 'motion'):
                if phase not in phases:
                    continue
                value = phases[phase]
                if (not isinstance(value, dict) or value.get('method') not in {'explicit', 'implicit'}
                        or value.get('selection_policy') not in INERTIA_POLICIES
                        or type(value.get('spectral_radius')) not in (int, float)
                        or not math.isfinite(value['spectral_radius']) or value['spectral_radius'] < 0):
                    raise ValueError('Invalid native controller execution observation')
                observed[phase] = _pick(value, ('method', 'spectral_radius', 'selection_policy',
                    'numeric_comparison_epsilon', 'selection_frozen_before_motion',
                    'parameters_modified', 'acceptance_gate'))
                if value['selection_policy'] in INTEGRAL_POLICIES:
                    constants = {'integral_time_constant_s': 1.0,
                        'integral_applicability': 'implicit_path_only',
                        'integral_antiwindup': 'freeze_outward_integration_at_saturation',
                        'integral_effort_bound': 'frozen_joint_effort_cap'}
                    if any(value.get(key) != expected for key, expected in constants.items()):
                        raise ValueError('Invalid bounded integral controller observation')
                    observed[phase].update(constants)
            execution[trial_id] = observed
        result['controller_execution'] = execution
    if 'post_operation_observation' in report:
        observations = {}
        for trial_id, value in report['post_operation_observation'].items():
            if trial_id not in failed_ids:
                continue
            active_v9 = report.get('acceptance_policy') in {'object_functional_physics.v11'}
            if active_v9:
                if (value.get('policy') != 'active_control_until_trial_end.v1'
                        or value.get('zero_effort_duration_s') != 0.
                        or value.get('stability_window_s') != 0.
                        or value.get('active_control_duration_unchanged') is not False):
                    raise ValueError('Invalid v9 active-control end evidence')
            elif (value.get('policy') != 'passive_settle_before_final_window.v1'
                    or value.get('active_control_duration_unchanged') is not True
                    or any(type(value.get(key)) not in (int, float)
                           or not math.isfinite(value[key]) or value[key] <= 0
                           for key in ('zero_effort_duration_s', 'stability_window_s'))
                    or value['zero_effort_duration_s'] < value['stability_window_s']):
                raise ValueError('Invalid passive observation duration evidence')
            observations[trial_id] = _pick(value, ('policy', 'zero_effort_duration_s',
                'stability_window_s', 'active_control_duration_unchanged'))
        result['post_operation_observation'] = observations
    if 'representative_motion' in report:
        plan = report['representative_motion']['plan']
        result['representative_motion_scope'] = _pick(plan, ('policy', 'selected', 'coverage',
            'all_joint_ids', 'observed_and_controlled_joint_ids', 'exhaustive_individual_motion_claimed'))
    if 'inapplicable_ground_operations' in report:
        # Keep these outside measured trials: null motion success and zero full
        # ground coverage must not become a successful/full-range free trial.
        result['inapplicable_ground_operations'] = [
            {**_trial(trial), **_pick(trial, ('motion_passed', 'ground_coverage_of_full_range')),
             'source_trial_content_sha256': fingerprint(trial)}
            for trial in report['inapplicable_ground_operations']]
    return result


def _history_payload(payload):
    # Match ResearchState.event payloads, rather than guessed candidate_id names.
    result = _pick(payload, ('id', 'candidate', 'candidate_id', 'parent', 'path', 'sha256',
        'decision', 'cycle_id', 'used_total', 'used_cycle', 'implementation', 'checkpoint_id', 'receipt_path', 'receipt_sha256',
        'archived_research_path', 'archived_manifest_path', 'remeasurement_source_candidate_id'))
    if isinstance(payload.get('receipt'), dict):
        receipt = payload['receipt']
        # Only bookkeeping crosses this boundary. No operator explanation,
        # diagnosis, proposed experiment, or arbitrary nested receipt survives.
        result['operation'] = _pick(receipt, ('schema_version', 'checkpoint_id', 'checkpoint_sha256',
            'decision', 'pipeline_version', 'from_policy', 'to_policy'))
        result['receipt_content_sha256'] = fingerprint(receipt)
    return result


def _compact_evidence(result, *, share_parameter_history=False):
    """Losslessly encode the already selected evidence, never the authority fields.

    All anchors and their neighbors are retained. Column labels replace repeated
    sample/joint keys; a shared JSON dictionary replaces repeated values. The
    explicit decoding instructions are also part of the model's context.
    """
    result = copy.deepcopy(result)
    for candidate in result['candidate_contexts'].values():
        for window in candidate['evidence']['trajectory_windows'].values():
            samples = window.get('samples', {})
            if not samples:
                continue
            columns = list(next(iter(samples.values())))
            if any(set(sample) != set(columns) for sample in samples.values()):
                continue  # Heterogeneous historical rows retain their explicit keys.
            dictionaries = {}
            for key in ('joint_positions', 'joint_velocities', 'joint_targets', 'joint_efforts'):
                if key not in columns:
                    continue
                names = list(next(iter(samples.values()))[key])
                if all(isinstance(sample[key], dict) and set(sample[key]) == set(names) for sample in samples.values()):
                    dictionaries[key] = names
            window['sample_table'] = {'columns': columns, 'dictionary_columns': dictionaries,
                'rows': {index: [[sample[key][name] for name in dictionaries[key]] if key in dictionaries
                                 else sample[key] for key in columns] for index, sample in samples.items()}}
            del window['samples']

    def canonical(value):
        return json.dumps(value, sort_keys=True, allow_nan=False)

    counts = Counter()
    def count(value):
        if isinstance(value, (dict, list, str)):
            encoded = canonical(value)
            # The profitability check below still accounts for reference and
            # pool overhead. Repeated diagnostic messages between 40 and 63
            # bytes also save space in large articulated-object reports.
            if len(encoded.encode('utf-8')) >= 40:
                counts[encoded] += 1
        if isinstance(value, (dict, list)):
            for child in (value.values() if isinstance(value, dict) else value):
                count(child)
    for candidate in result['candidate_contexts'].values():
        count(candidate['evidence'])
        if share_parameter_history:
            count(candidate.get('parameter_changes_from_best', []))
    if share_parameter_history:
        count(result.get('tested_candidate_trials', []))
    values, identities = {}, {}
    def encode(value, *, allow_reference=True):
        if not isinstance(value, (dict, list, str)):
            return value
        encoded = canonical(value)
        repetitions = counts[encoded]
        # Include repeated root poses and paths only if the reference/index
        # overhead is smaller; short unique numbers/labels stay directly visible.
        saves_bytes = (repetitions-1)*len(encoded.encode('utf-8')) > repetitions*40 + 25
        if allow_reference and repetitions > 1 and saves_bytes:
            if encoded not in identities:
                identity = 'evidence_' + str(len(identities))
                identities[encoded] = identity
                values[identity] = encode(value, allow_reference=False)
            return {'$evidence_ref': identities[encoded]}
        if isinstance(value, dict):
            return {key: encode(child) for key, child in value.items()}
        return [encode(child) for child in value] if isinstance(value, list) else value
    for candidate in result['candidate_contexts'].values():
        candidate['evidence'] = encode(candidate['evidence'])
        if share_parameter_history:
            candidate['parameter_changes_from_best'] = encode(candidate.get('parameter_changes_from_best', []))
    if share_parameter_history and 'tested_candidate_trials' in result:
        result['tested_candidate_trials'] = encode(result['tested_candidate_trials'])
    result['shared_evidence_values'] = values
    result['api_evidence_schema'] = 'physics_api_evidence.v2'
    result['evidence_encoding'] = {
        'kind': 'lossless_shared_json_and_labeled_sample_tables',
        'instructions': [
            'Within candidate evidence, replace every object containing only $evidence_ref with the value named in shared_evidence_values; repeat recursively. References reuse identical JSON values, not different measurements.',
            'In each trajectory window, sample_table.rows maps the original sample index to a row. Zip sample_table.columns with that row to recover the named sample fields.',
            'For each field listed in sample_table.dictionary_columns, zip its listed joint names with that field\'s row array to recover the joint-name-to-native-value dictionary. Joint order is explicit and may differ between fields.',
            'All first/max anchors and their available immediately preceding/following samples remain. Time, q, qd, target, applied effort, root pose, contact body identities and summarized impulse vectors retain their original numeric values.',
            'Contact net_impulse_Ns is a vector sum of native active point impulses, not force and not event double-counting; strongest/deepest contacts preserve representative positions, normals and impulse direction. Omitted pairs remain counted; complete contacts are retained locally.',
            'Top-level profile, initial_profile, controller, contract, allowed_paths, remaining, used_total and review strategy remain explicit and unchanged. Decode the tables before comparing candidates; absent or omitted observations do not prove no contact.'],
        'raw_contact_arrays_summarized_before_encoding': True,
        'additional_observations_discarded_by_encoding': False}
    if share_parameter_history:
        result['api_evidence_schema'] = 'physics_api_evidence.v3'
        result['evidence_encoding']['instructions'][0] = (
            'Within candidate_contexts evidence and parameter_changes_from_best and tested_candidate_trials, '
            'replace every object containing only $evidence_ref with the named shared_evidence_values '
            'value; repeat recursively. These references reuse identical JSON values, not different '
            'measurements or parameter versions. All candidates, rejected parameter values and '
            'measured failures remain after decoding; no historical trial is dropped.')
        result['evidence_encoding']['parameter_history_losslessly_shared'] = True
    return result


def expand_evidence_projection(result):
    """Expand v2 tables/references for offline equality audits, without file IO."""
    result = copy.deepcopy(result)
    pool = result.get('shared_evidence_values', {})
    def expand(value, stack=()):
        if isinstance(value, dict):
            if set(value) == {'$evidence_ref'}:
                identity = value['$evidence_ref']
                if identity not in pool or identity in stack:
                    raise ValueError('Invalid or cyclic shared API evidence reference')
                return expand(pool[identity], (*stack, identity))
            return {key: expand(child, stack) for key, child in value.items()}
        return [expand(child, stack) for child in value] if isinstance(value, list) else value
    for candidate in result['candidate_contexts'].values():
        if result.get('api_evidence_schema') == 'physics_api_evidence.v3':
            candidate['parameter_changes_from_best'] = expand(candidate.get('parameter_changes_from_best', []))
        evidence = candidate['evidence'] = expand(candidate['evidence'])
        for window in evidence['trajectory_windows'].values():
            table = window.pop('sample_table', None)
            if table is None:
                continue
            columns, dictionaries = table['columns'], table['dictionary_columns']
            samples = {}
            for index, row in table['rows'].items():
                if len(row) != len(columns):
                    raise ValueError('API evidence table has inconsistent row width')
                sample = dict(zip(columns, row))
                for key, names in dictionaries.items():
                    if len(sample[key]) != len(names):
                        raise ValueError('API evidence table has inconsistent joint width')
                    sample[key] = dict(zip(names, sample[key]))
                samples[index] = sample
            window['samples'] = samples
    if (result.get('api_evidence_schema') == 'physics_api_evidence.v3'
            and 'tested_candidate_trials' in result):
        result['tested_candidate_trials'] = expand(result['tested_candidate_trials'])
    result.pop('shared_evidence_values', None)
    result.pop('evidence_encoding', None)
    return result


def _prepare_refinement_payload(payload, *, max_bytes=DEFAULT_MAX_BYTES):
    """Validate authority and build intermediate evidence; never send this directly.

    The focused projector still removes passed/comparison trial details. Its
    byte limit must apply after that projection, not to this larger intermediate.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('API evidence byte budget must be a positive integer')
    contract = payload.get('contract', {})
    policy = contract.get('acceptance_policy')
    if (policy != 'object_functional_physics.v11'
            or contract.get('joint_peak_velocity_policy') != 'diagnostic_only'
            or contract.get('motion_scope_policy') != 'fixed_full_ground_operable.v1'
            or contract.get('passive_tail_velocity_policy') != 'diagnostic_only'):
        raise ValueError('API evidence requires the current v11 diagnostic and motion-scope policies')
    from .protocol import validate_orientation_policy, validate_free_base_policy
    orientation = validate_orientation_policy(contract.get('orientation_policy'))
    validate_free_base_policy(contract.get('free_base_policy'))
    current, best = payload['current'], payload['best']
    identities = [current['candidate']['id'], best['candidate']['id']]
    if identities[0] == identities[1] and current != best:
        raise ValueError('The same current/best candidate ID has conflicting evidence or parameters')
    for key in ('profile', 'controller', 'contract'):
        if payload[key] != best['parameters'][key]:
            raise ValueError('Top-level repair context differs from the immutable best parent: ' + key)
    base = _pick(payload, ('phase', 'profile', 'initial_profile', 'controller', 'contract', 'allowed_paths', 'remaining',
                           'used_total', 'cycle_id', 'review_after_tested_patches', 'diagnostic_source_candidate_id'))
    if 'framework_gate' in payload:
        gate = payload['framework_gate']
        keys = ('proposal_allowed', 'completed_retests_since_review', 'checkpoint_after_retests', 'applied_review_count')
        if (not isinstance(gate, dict) or gate.get('proposal_allowed') is not True
                or type(gate.get('completed_retests_since_review')) is not int
                or not 0 <= gate['completed_retests_since_review'] < 3
                or type(gate.get('checkpoint_after_retests')) is not int or gate['checkpoint_after_retests'] != 3
                or type(gate.get('applied_review_count')) is not int or gate['applied_review_count'] < 0):
            raise ValueError('Invalid code-owned framework gate state')
        base['framework_gate'] = _pick(gate, keys)
    if payload.get('joint_groups') is not None:
        from .joint_groups import public_context, SUPPORTED_POLICIES
        group_policy = payload['joint_groups'].get('policy')
        if group_policy not in SUPPORTED_POLICIES:
            raise ValueError('Unknown shared joint grouping policy in diagnostic context')
        base['joint_groups'] = public_context({'policy': group_policy,
                                             'groups': payload['joint_groups']['groups']})
    base['api_evidence_schema'] = 'physics_api_evidence.v1'
    base['candidate_refs'] = dict(current=identities[0], best=identities[1])
    diagnostic_id = payload.get('diagnostic_source_candidate_id', identities[1])
    if diagnostic_id not in identities:
        raise ValueError('Diagnostic source must be an explicitly supplied measured candidate')
    diagnostic_evidence = current['evidence'] if diagnostic_id == identities[0] else best['evidence']
    if ('diagnostic_source_candidate_id' in payload
            and payload['observed_failures'] != diagnostic_evidence['observed_failures']):
        raise ValueError('Primary diagnostic failures differ from the measured candidate evidence')
    base['observed_failures'] = _failures(payload['observed_failures'], 'candidate_contexts/' + diagnostic_id + '/evidence/trials')
    base['history'] = {'event_count': len(payload.get('history', [])),
        'source_content_sha256': fingerprint(payload.get('history', [])),
        'events': [{**_pick(e, ('sequence', 'kind', 'cycle_id', 'pipeline_version', 'sha256', 'previous_sha256')),
            'payload': _history_payload(e.get('payload', {})),
            'payload_content_sha256': fingerprint(e.get('payload', {}))} for e in payload.get('history', [])]}
    base['candidates'] = [_candidate_metadata(c) for c in payload.get('candidates', [])]
    if 'tested_candidate_trials' in payload:
        # Already hash-bound by ResearchState. Full native verification is lazy
        # and only runs when a proposal matches a provisional key. Preserve old rejected
        # patch values and measured failures inline, not only unreadable paths.
        # The ordinary total byte bound below applies; never truncate away a
        # tested parameter or failure to make a repeated experiment look new.
        base['tested_candidate_trials'] = copy.deepcopy(payload['tested_candidate_trials'])
        base['tested_candidate_interpretation'] = (
            'These are hash-bound historical measurements, not acceptance under a new policy. '
            'same_implementation_and_policy is only a version prefilter, not independent current verification; '
            'verification_status=not_checked and a provisional physical_input_sha256 do not establish asset/source equality. '
            'An exact repeated physical input is invalid only after matching source, asset and native evidence are '
            'independently verified. Use the measured rejected '
            'patches and failures when choosing a distinct permitted experiment; if evidence is insufficient, '
            'return uncertain with empty patches. A controlled repeat requires explicit review. No historical '
            'trial is removed or refunded, and no old result is published as a new pass.')
    for radius in (1,):
        result = copy.deepcopy(base)
        result['candidate_contexts'] = {}
        for value in (current, best):
            identity = value['candidate']['id']
            if identity in result['candidate_contexts']:
                continue
            result['candidate_contexts'][identity] = {'candidate': _candidate_metadata(value['candidate']),
                'parameters_content_sha256': fingerprint(value['parameters']),
                'parameter_changes_from_best': _parameters_diff(
                    _pick(value['parameters'], ('profile', 'controller', 'contract')),
                    _pick(best['parameters'], ('profile', 'controller', 'contract'))),
                'parameters_note': 'Best parameters are the exact top-level profile/controller/contract; current differences listed above.',
                'evidence': _project_evidence(value['evidence'], identity, radius)}
        result['projection'] = {'source_payload_content_sha256': fingerprint(payload), 'source_serialized_bytes': _size(payload),
            'max_serialized_bytes': max_bytes, 'serialized_bytes': 0, 'window_radius_steps': radius,
            'serialization': 'compact_utf8_json.v1',
            'full_evidence_is_retained_locally': True, 'acceptance_and_permissions_unchanged': True,
            'notes': ['Contact points and duplicate event arrays are summarized; all local evidence stays hash-bound.',
                      'Instantaneous speed and penetration-depth diagnostics do not create acceptance gates.',
                      'Local complete evidence is authoritative; missing or truncated data is not evidence of absence.']}
        if orientation is not None:
            result['orientation_interpretation'] = {
                'orientation_policy': copy.deepcopy(orientation),
                'tilt_acceptance_gate': orientation['upright_required'],
                'scope': 'Only initial/reference-relative root tilt; passive-tail orientation stability, q0, '
                         'ground contact and all applicable joint operations remain required.'}
            result['projection']['notes'].append(
                'The frozen orientation decision requires upright placement; the reference-relative tilt gate remains active.'
                if orientation['upright_required'] else
                'The frozen orientation decision allows natural rolling. Initial/reference-relative tilt is diagnostic '
                'and is not a failure or repair target; final-window pose stability and joint initial-state rules remain active.')
        return result
