"""Measured failures order trials within phases, never alter the required set."""
from __future__ import annotations

import json
from pathlib import Path

from urdf_core.run_state import file_hash

POLICY = 'first_failed_trial.failed_first.v1'
SCHEMA = 'physics_trial_order.v1'
LOCAL_POLICY = 'first_failed_trial.prerequisites_first.v2'
PHASES = ('fixed_joint', 'free_joint')


def make_plan(contract, previous_report=None, *, local_retest=False):
    names = list(contract['joints'])
    orders = {phase: list(names) for phase in PHASES}
    source = None
    if previous_report is not None:
        path = Path(previous_report).resolve()
        report = json.loads(path.read_text())
        if (report.get('evaluation_complete') is not True or report.get('errors')
                or report.get('status') not in ('passed', 'physics_failed')):
            raise ValueError('Trial ordering requires complete, valid previous measurements')
        old_contract = report['inputs']['contract']
        old_path = Path(old_contract['path'])
        if file_hash(old_path) != old_contract['sha256']:
            raise ValueError('Previous trial contract changed')
        for phase in PHASES:
            failed = [t['joint_name'] for t in report['trials']
                      if t.get('kind') == phase and t.get('passed') is False]
            if len(failed) != len(set(failed)) or not set(failed) <= set(names):
                raise ValueError('Invalid measured failure identities')
            orders[phase] = failed + [name for name in names if name not in failed]
        source = {'path': str(path), 'sha256': file_hash(path)}
    result = {'schema_version': SCHEMA, 'policy': POLICY,
            'spec_sha256': contract['spec_sha256'], 'source_report': source,
            'orders': orders, 'phase_order': ['ground_settle', *PHASES],
            'required_trials_changed': False}
    if local_retest:
        failures = ([] if previous_report is None else [
            {'kind': t['kind'], 'joint_name': t.get('joint_name')}
            for t in report['trials'] if t.get('passed') is False])
        result.update(policy=LOCAL_POLICY, failed_trials=failures,
                      phase_order=None)
    return result


def verify_plan(plan, contract):
    source = plan.get('source_report')
    if source is not None:
        if set(source) != {'path', 'sha256'} or file_hash(source['path']) != source['sha256']:
            raise ValueError('Prior failure evidence changed')
    expected = make_plan(contract, source['path'] if source else None,
                         local_retest=plan.get('policy') == LOCAL_POLICY)
    if plan != expected:
        raise ValueError('Trial order differs from measured failure priority')
    return plan['orders']


def trial_sequence(plan, contract, required):
    """Same-candidate prerequisites, failed operation, then ALL remaining work.

    Stable always comes first. A free operation must first pass its own fixed
    full-range test. No predecessor's pass is treated as this candidate's pass.
    """
    orders = verify_plan(plan, contract)
    normal = ([('ground_settle', None)] if ('ground_settle', None) in required else [])
    normal += [(p, n) for p in PHASES for n in orders[p] if (p, n) in required]
    if set(normal) != set(required):
        raise ValueError('Unknown or uncovered required trial')
    result = normal[:1] if normal and normal[0][0] == 'ground_settle' else []
    for failure in plan.get('failed_trials', []):
        kind, name = failure['kind'], failure['joint_name']
        if kind == 'free_joint':
            if ('fixed_joint', name) not in required:
                continue  # Legacy unequal fixed/free sampling: retain full order.
            result.append(('fixed_joint', name))
        if (kind, name) in required:
            result.append((kind, name))
    return list(dict.fromkeys([*result, *normal]))


def validate_local_coverage(report, contract, required):
    sequence = trial_sequence(report['trial_order'], contract, required)
    trials = report['trials']
    identities = [(t['kind'], t.get('joint_name')) for t in trials]
    if identities != sequence[:len(trials)]:
        raise ValueError('Local retest omitted a prerequisite or changed trial order')
    failures = [t for t in trials if t.get('passed') is not True]
    if not failures:
        if (identities != sequence or report.get('acceptance_complete') is not True
                or report.get('blocked_trials') != [] or report.get('stopped_after_phase') is not None):
            raise ValueError('Local retest success is not complete acceptance')
        return False
    if len(failures) != 1 or failures[0] is not trials[-1]:
        raise ValueError('Local retest continued after failure')
    failure = failures[0]
    reason = {'status': 'skipped_by_prior_failure', 'blocked_by_phase': failure['kind'],
              'failed_trial_ids': [failure['id']]}
    blocked = [{'kind': k, 'joint_name': n, **reason} for k, n in sequence[len(trials):]]
    if (report.get('blocked_trials') != blocked or report.get('reload_check') != {'passed': False, **reason}
            or report.get('acceptance_complete') is not False or report.get('passed') is not False
            or report.get('numerical_passed') is not False or report.get('status') != 'physics_failed'
            or report.get('stopped_after_phase') != failure['kind']):
        raise ValueError('Local retest rejection cannot claim omitted tests or reload passed')
    return True
