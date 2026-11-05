"""Select complete failure groups when the full diagnostic exceeds its budget."""
import copy

from .api_evidence import EvidenceProjectionTooLarge, serialize_refinement_payload
from .diagnostic_summary import mentions
from .diagnostic_transport import canonical, decode, encode


def _units(projected):
    identity = projected['diagnostic_source_candidate_id']
    trials = projected['candidate_contexts'][identity]['evidence']['trials']
    membership = {}
    for group in (projected.get('joint_groups') or {}).get('groups', []):
        names = set(group['members']) | set(group.get('child_links', {}).values())
        for name in names:
            membership.setdefault(name, set()).add(group['id'])
    units = []
    for trial_id, trial in trials.items():
        if trial.get('passed') is not False:
            continue
        wrapper = trial.get('metrics', {})
        metrics = copy.deepcopy(wrapper.get('metrics', {}))
        buckets = {}
        # Split only explicitly named entity maps. Unknown measurements remain
        # intact as shared evidence; no numerical field becomes optional.
        for path in (('joints',), ('links',), ('initial_pose_links',), ('passive_tail', 'links')):
            parent = metrics
            for key in path[:-1]:
                parent = parent.get(key, {})
            rows = parent.pop(path[-1], None)
            if not isinstance(rows, dict):
                if rows is not None:
                    parent[path[-1]] = rows
                continue
            for name, values in rows.items():
                groups = membership.get(name, set())
                key = next(iter(groups)) if len(groups) == 1 else 'entity:' + name
                bucket = buckets.setdefault(key, {'names': set(), 'records': []})
                bucket['names'].add(name)
                bucket['records'].append(((*path, name), values))
        if not buckets:
            buckets['whole_trial'] = {'names': set(), 'records': []}
        all_names = set().union(*(b['names'] for b in buckets.values()))
        for group, bucket in sorted(buckets.items()):
            errors = [e for e in wrapper.get('errors', []) if
                not any(mentions(e, n) for n in all_names) or any(mentions(e, n) for n in bucket['names'])]
            units.append({'id': trial_id + '/' + group, 'trial_id': trial_id,
                          'names': sorted(bucket['names']), 'errors': errors,
                          'shared_metrics': metrics, 'records': bucket['records']})
    return units


def select_failure_groups(projected, *, max_text_tokens, max_bytes, count_tokens,
                          cursor=0, max_trials=None):
    core = copy.deepcopy(projected)
    identity = core['diagnostic_source_candidate_id']
    originals = core['candidate_contexts'][identity]['evidence']['trials']
    units = _units(core)
    if not units:
        raise EvidenceProjectionTooLarge('No complete failed measurement group is available for diagnosis')
    offset = cursor % len(units)
    units = units[offset:] + units[:offset]
    for context in core['candidate_contexts'].values():
        evidence = context['evidence']
        evidence['trials'] = {}
        evidence['trajectory_windows'] = {}
        evidence['observed_failures'] = []
        evidence.pop('collision_geometry', None)
        evidence.pop('native_control_execution', None)
        evidence.pop('post_operation_observation', None)
        evidence.pop('inapplicable_ground_operations', None)
        evidence.pop('representative_motion_scope', None)
        numbers = evidence.get('numeric_summary', {})
        numbers.pop('failing_checks', None)
    for key in ('experiment_history', 'refinement_attempts', 'joint_test_observations'):
        core.pop(key, None)
    core['projection'].update(failed_entity_metrics_exact=True,
        failed_metrics_scope='selected_failure_groups', historical_patch_values_exact=False,
        historical_details_retained_locally=True)
    core['projection']['notes'].append(
        'Diagnose the selected failure groups. Deferred groups are unhandled, not passed. '
        'All selected measurements and the complete parameter rules and thresholds are exact. '
        'Other measurements, setup details, history and trajectories remain local. '
        'Related untested groups may receive permitted parameter changes by reasoning; '
        'their acceptance still requires native evaluation.')

    def candidate(selected):
        value = copy.deepcopy(core)
        evidence = value['candidate_contexts'][identity]['evidence']
        rows = evidence['trials']
        for unit in selected:
            tid = unit['trial_id']
            if tid not in rows:
                original = originals[tid]
                rows[tid] = {k: copy.deepcopy(v) for k, v in original.items()
                             if k not in {'metrics', 'setup'}}
                rows[tid]['metrics'] = {k: copy.deepcopy(v) for k, v in original.get('metrics', {}).items()
                                       if k not in {'metrics', 'errors', 'passed_entity_summaries'}}
                rows[tid]['metrics'].update(metrics=copy.deepcopy(unit['shared_metrics']), errors=[])
                rows[tid]['omitted_details_retained_locally'] = True
            wrapper = rows[tid]['metrics']
            for error in unit['errors']:
                if error not in wrapper['errors']:
                    wrapper['errors'].append(error)
            for path, record in unit['records']:
                parent = wrapper['metrics']
                for key in path[:-1]:
                    parent = parent.setdefault(key, {})
                parent[path[-1]] = copy.deepcopy(record)
        failures = []
        for failure in projected.get('observed_failures', []):
            if failure.get('trial_id') not in rows:
                continue
            row = copy.deepcopy(failure)
            errors = rows[row['trial_id']]['metrics']['errors']
            row['errors'] = [e for e in row.get('errors', []) if e in errors]
            # Explicit identity remains available for proposal validation.
            failures.append(row)
        value['observed_failures'] = failures
        evidence['observed_failures'] = copy.deepcopy(failures)
        selected_ids = {u['id'] for u in selected}
        value['diagnostic_scope'] = {'policy': 'complete_failure_groups.v1',
            'selected': [u['id'] for u in selected],
            'deferred': [u['id'] for u in units if u['id'] not in selected_ids],
            'trial_ids': sorted(rows), 'selection_cursor': cursor,
            'same_shared_repair_budget': True, 'deferred_groups_are_unhandled': True}
        for wire in (value, encode(value)):
            if (len(serialize_refinement_payload(wire).encode('utf-8')) <= max_bytes
                    and count_tokens(wire) <= max_text_tokens):
                if wire is not value and canonical(decode(wire)) != canonical(value):
                    raise ValueError('Diagnostic transport changed selected failure measurements')
                return wire
        return None

    selected, wire = [], None
    for unit in units:
        trial_ids = {u['trial_id'] for u in [*selected, unit]}
        if max_trials is not None and len(trial_ids) > max_trials:
            continue
        attempt = candidate([*selected, unit])
        if attempt is not None:
            selected.append(unit)
            wire = attempt
    if wire is None:
        raise EvidenceProjectionTooLarge(
            'No complete failure group fits with its measurements, thresholds and repair authority; '
            'no API request was sent')
    return wire
