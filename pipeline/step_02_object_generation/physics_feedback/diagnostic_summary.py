"""API-only summaries of measured failures; native evidence remains unchanged."""
import copy
import re
from collections import defaultdict


def mentions(error, name):
    return re.search(r'(?<![\w])' + re.escape(name) + r'(?![\w])', error) is not None


def representatives(failed, errors, groups, target=None):
    membership = {}
    for group in (groups or {}).get('groups', []):
        for name in group['members']:
            if name in membership:
                raise ValueError('Ambiguous diagnostic group membership')
            membership[name] = group
    buckets = defaultdict(list)
    for name in sorted(failed):
        kinds = tuple(sorted(re.sub(r'(?<![\w])' + re.escape(name) + r'(?![\w])', '{joint}', e)
                             for e in errors if mentions(e, name)))
        group = membership.get(name)
        # Unknown error formats retain individual trajectories.
        key = (group['id'] if group and kinds else 'individual:' + name, kinds)
        buckets[key].append(name)
    selected, rows = set(), []
    for (identity, kinds), members in buckets.items():
        group = membership.get(members[0], {})
        preferred = target if target in members else group.get('representative')
        representative = preferred if preferred in members else members[0]
        selected.add(representative)
        rows.append({'group_id': identity, 'failure_kinds': list(kinds), 'failed_members': members,
                     'trajectory_representative': representative})
    return selected, rows


def _summary(rows):
    maxima = {}
    for name, row in rows.items():
        for key, value in row.items() if isinstance(row, dict) else []:
            if type(value) in (int, float) and (key not in maxima or value > maxima[key]['value']):
                maxima[key] = {'name': name, 'value': value}
    return {'count': len(rows), 'names': sorted(rows), 'measured_maxima': maxima,
            'detail_retained_locally': True}


def summarize_trial(trial, groups, contract):
    result = copy.deepcopy(trial)
    wrapper = result.get('metrics', {})
    metrics = wrapper.get('metrics', {})
    errors = wrapper.get('errors', [])
    names = set(contract.get('joints', {})) | set(contract.get('required_links', []))
    names.update(metrics.get('joints', {}))
    for key in ('links', 'initial_pose_links'):
        names.update(metrics.get(key, {}))
    names.update(metrics.get('passive_tail', {}).get('links', {}))
    # Global/unknown failures cannot establish that any particular entity passed.
    if not errors or any(not any(mentions(error, name) for name in names) for error in errors):
        return result
    failed = {name for name in names if any(mentions(error, name) for error in errors)}
    summaries = {}
    for key in ('joints', 'links', 'initial_pose_links'):
        if isinstance(metrics.get(key), dict):
            rows = metrics[key]
            passed = {name: row for name, row in rows.items() if name not in failed}
            metrics[key] = {name: row for name, row in rows.items() if name in failed}
            summaries[key] = _summary(passed)
    tail = metrics.get('passive_tail', {})
    if isinstance(tail.get('links'), dict):
        rows = tail['links']
        tail['links'] = {name: row for name, row in rows.items() if name in failed}
        summaries['passive_tail_links'] = _summary({name: row for name, row in rows.items() if name not in failed})
    wrapper['passed_entity_summaries'] = summaries
    related_links = set(failed & set(contract.get('required_links', [])))
    for group in (groups or {}).get('groups', []):
        related_links.update(link for name, link in group.get('child_links', {}).items() if name in failed)
    structural = contract.get('structural_joints', [])
    structural = structural.items() if isinstance(structural, dict) else ((j['name'], j) for j in structural)
    for name, joint in structural:
        if name in failed and isinstance(joint, dict):
            related_links.update(joint[k] for k in ('parent', 'child') if k in joint)
    setup = result.get('setup', {})
    setup_summaries = {}
    for key, relevant in (('measured_properties', related_links), ('native_joint_friction', failed),
                          ('passive_damping', failed), ('target_speed_fraction', failed), ('target_speed_limits', failed)):
        rows = setup.get(key)
        if isinstance(rows, dict) and relevant:
            setup[key] = {name: row for name, row in rows.items() if name in relevant}
            setup_summaries[key] = {'count': len(rows)-len(setup[key]), 'detail_retained_locally': True}
    setup['unrelated_entity_summaries'] = setup_summaries
    return result


def summarize_history(history):
    result = copy.deepcopy(history)
    by_id = {row['candidate_id']: row for row in result['trials']}
    def failures(row):
        return {i for f in row['result']['observed_failures'] for i in f['error_ids']}
    for row in result['trials']:
        changes = defaultdict(list)
        for patch in row.pop('patches'):
            # Preserve integer/float identity and exact values.
            changes[(type(patch['value']).__name__, repr(patch['value']))].append(patch)
        row['changes'] = [{'paths': [p['path'] for p in patches], 'value': patches[0]['value']}
                          for patches in changes.values()]
        parent = by_id.get(row.get('parent'))
        comparable = parent is not None and parent['implementation_group'] == row['implementation_group']
        observed_trials = {f.get('trial_id') for f in row['result']['observed_failures']}
        absent = failures(parent)-failures(row) if comparable else set()
        rechecked = {i for f in parent['result']['observed_failures']
                     if row['result'].get('passed') is True or f.get('trial_id') in observed_trials
                     for i in f['error_ids']} if comparable else set()
        row['resolved_error_ids'] = sorted(absent & rechecked) if comparable else None
        row['not_reobserved_error_ids'] = sorted(absent-rechecked) if comparable else None
        row['remaining_error_ids'] = sorted(failures(row))
        row['resolution_scope'] = 'Resolved only within reobserved trials or a fully passed candidate; skipped tests remain unverified.'
    result['interpretation'] += (' changes contains exact values for each listed path, relative to parent. '
        'resolved_error_ids only describes the measured failure difference, not a pass for skipped tests.')
    return result
