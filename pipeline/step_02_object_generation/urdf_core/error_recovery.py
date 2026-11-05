"""Stage-owned, bounded diagnosis of errors outside existing content handlers."""
from __future__ import annotations

import errno
import hashlib
import json
import math
from pathlib import Path
import re
import time
import traceback

from .api_requests import _lock, _write

POLICY = 'error_recovery.v1'


class RecoveryExhausted(RuntimeError):
    """A local operation could not recover; route to the stage's legal fallback."""


def exhausted(error):
    raise RecoveryExhausted(str(error)) from error


class VerifiedExternalError(RuntimeError):
    def __init__(self, message, *, kind, evidence):
        super().__init__(message)
        self.external_evidence = {'kind': kind, 'evidence': safe_evidence(evidence)}


class ReportedError(RuntimeError):
    def __init__(self, report):
        super().__init__('; '.join(map(str, report.get('errors', []))) or report.get('status', 'unknown error'))
        self.diagnostic_evidence = safe_evidence(report.get('error_context', report))


def cancelled(error):
    from .execution import ExecutionCancelled
    from concurrent.futures import CancelledError
    return isinstance(error, (KeyboardInterrupt, SystemExit, ExecutionCancelled, CancelledError)) or (
        type(error).__name__ == 'CandidateCancelled' and type(error).__module__.endswith('candidate_workers'))


def external_failure(error):
    """Only typed transport faults, errno, or evidence emitted at a verified boundary."""
    from .api_requests import APIAttemptsExhausted, ServiceIssue
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (APIAttemptsExhausted, ServiceIssue, VerifiedExternalError)):
            return True
        if getattr(error, 'verified_external_fault', None):
            return True
        if type(error).__module__.split('.')[0] == 'openai' and api_transport_failure(error):
            return True
        if type(error).__name__ == 'APITechnicalFailure' and type(error).__module__.endswith('scene_refinement.provider'):
            return True
        if isinstance(error, OSError) and error.errno in {
            errno.ENOSPC, errno.EDQUOT, errno.EACCES, errno.EPERM, errno.EROFS,
            errno.ENOMEM, errno.EIO, errno.ENETDOWN, errno.ENETUNREACH,
            errno.ECONNRESET, errno.ECONNREFUSED, errno.ETIMEDOUT, errno.EHOSTUNREACH}:
            return True
        error = error.__cause__
    return False


def api_transport_failure(error):
    from openai import APIConnectionError, APIError, APIStatusError
    from .api_requests import _fatal
    # Streaming service errors can omit an HTTP status while retaining SDK evidence.
    return isinstance(error, (APIConnectionError, APIStatusError)) or (
        isinstance(error, APIError) and _fatal(error))


def safe_evidence(value, *, depth=0):
    if depth > 10:
        return '<nested evidence omitted>'
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else ('NaN' if math.isnan(value) else '+Infinity' if value > 0 else '-Infinity')
    if isinstance(value, str):
        value = re.sub(r'(?i)(Bearer\s+|(?:api[_-]?key|token|password)\s*[=:]\s*)[^\s,;]+', r'\1<redacted>', value)
        return re.sub(r'\bsk-[A-Za-z0-9_-]+', '<redacted>', value)[:12000]
    if isinstance(value, dict):
        return {str(k): ('<redacted>' if re.search(r'(?i)api.?key|password|authorization|secret', str(k))
                         else safe_evidence(v, depth=depth+1)) for k, v in list(value.items())[:120]}
    if isinstance(value, (list, tuple)):
        return [safe_evidence(v, depth=depth+1) for v in value[:120]]
    if hasattr(value, 'tolist'):
        return safe_evidence(value.tolist(), depth=depth+1)
    return safe_evidence(str(value), depth=depth+1)


def error_evidence(error):
    return safe_evidence({'type': type(error).__name__, 'message': str(error),
        'context': getattr(error, 'diagnostic_evidence', {}),
        'traceback': ''.join(traceback.format_exception(type(error), error, error.__traceback__))[-6000:]})


def checked_native_transform(name, value, time_s, previous, convert):
    try:
        return convert(value)
    except Exception as error:
        error.diagnostic_evidence = safe_evidence({
            'object_id': name.split('::')[0], 'link': name, 'transform': value,
            'simulation_time_s': time_s, 'last_valid_pose': previous.get('body_poses', {}).get(name),
            'last_valid_time_s': previous.get('time_s'), 'contacts': previous.get('contacts', [])})
        raise


def report_error(report):
    error = ReportedError(report)
    # Only locally produced boundary evidence is authoritative, never an MLLM field.
    proof = report.get('verified_external_fault')
    if proof and proof.get('kind') in ('runtime_startup', 'worker_exit', 'watchdog', 'resource_unavailable'):
        raise VerifiedExternalError(str(error), kind=proof['kind'], evidence=proof)
    raise error


class Recovery:
    def __init__(self, root, stage, ask, *, limit=5):
        if type(limit) is not int or not 0 <= limit <= 5:
            raise ValueError('max_error_diagnoses must be an integer between 0 and 5')
        self.root = Path(root) / 'error_recovery'
        self.stage, self.ask, self.limit = stage, ask, limit
        self.path = self.root / 'ledger.json'

    def _load(self):
        data = json.loads(self.path.read_text()) if self.path.exists() else {
            'policy': POLICY, 'stage': self.stage, 'limit': self.limit, 'reserved': 0, 'errors': {}}
        if (data['policy'], data['stage'], data['limit']) != (POLICY, self.stage, self.limit):
            raise ValueError('Error diagnosis ledger binding changed')
        return data

    def decide(self, operation, error, *, allowed, evidence=None, fallback='fallback'):
        if cancelled(error) or external_failure(error):
            raise error
        detail = error_evidence(error)
        binding = safe_evidence({'operation': operation, 'type': detail['type'], 'message': detail['message'],
                                 'context': detail['context'], 'evidence': evidence or {}})
        key = hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()
        allowed = tuple(allowed)
        if fallback not in allowed:
            raise ValueError('Recovery requires an allowed deterministic fallback')
        with _lock(self.root / 'ledger.lock'):
            data = self._load()
            record = data['errors'].get(key)
            if record and record.get('decision'):
                decision = record['decision']
                if decision['action'] not in allowed:
                    decision = {'action': fallback, 'reason': 'cached action no longer available'}
                return {**decision, 'error_id': key}
            if record is None:
                record = {'binding': binding, 'error': detail, 'allowed': allowed,
                          'status': 'reserved', 'actions': [], 'created_at': time.time()}
                data['errors'][key] = record
                if data['reserved'] >= self.limit:
                    record['decision'] = {'action': fallback, 'reason': 'diagnosis budget exhausted'}
                else:
                    data['reserved'] += 1
                    record['reservation'] = data['reserved']
                _write(self.path, data)
            if not record.get('decision'):
                started = time.monotonic()
                try:
                    prompt = (Path(__file__).resolve().parents[3] / 'prompts/urdf/error_diagnosis.md').read_text()
                    payload = {'stage': self.stage, **binding, 'error': detail, 'allowed_actions': allowed,
                               'default_action': fallback}
                    raw = self.ask('error_diagnosis_' + key, prompt, json.dumps(payload, ensure_ascii=False, allow_nan=False))
                    if isinstance(raw, str):
                        raw = json.loads(raw)
                    if not isinstance(raw, dict) or raw.get('action') not in allowed:
                        raise ValueError('Invalid diagnosis or action outside stage permissions')
                    record['decision'] = {'action': raw['action'], 'reason': str(raw.get('reason', ''))[:2000],
                                          'object_ids': safe_evidence(raw.get('object_ids', []))}
                except BaseException as exc:
                    record['diagnosis_error'] = error_evidence(exc)
                    if cancelled(exc) or external_failure(exc):
                        record['status'] = 'diagnosis_pending'
                        _write(self.path, data)
                        raise
                    record['decision'] = {'action': fallback, 'reason': 'diagnosis unavailable or invalid'}
                finally:
                    record['diagnosis_seconds'] = time.monotonic() - started
                record['status'] = 'decided'
            _write(self.path, data)
            return {**record['decision'], 'error_id': key}

    def disposition(self, operation, error, *, evidence=None):
        try:
            return self.decide(operation, error, allowed=('fallback',), evidence=evidence)
        except BaseException as failure:
            if cancelled(failure) or external_failure(failure):
                raise
            _write(self.root / 'diagnosis_unavailable.json', error_evidence(failure))
            return {'action': 'fallback', 'reason': 'diagnosis mechanism unavailable'}

    def _event(self, key, action, state, **details):
        with _lock(self.root / 'ledger.lock'):
            data = self._load()
            data['errors'][key]['actions'].append({'action': action, 'state': state,
                                                   'time': time.time(), **safe_evidence(details)})
            data['errors'][key]['status'] = state
            _write(self.path, data)

    def run(self, operation, fn, *, fallback, evidence=None, refine=None, reset=None,
            attempts=2, fallback_action='fallback', known=(), validate_result=None):
        """Callbacks use their existing durable step receipts; never replay a complete repair."""
        binding = json.dumps(safe_evidence({'operation': operation, 'evidence': evidence or {}}), sort_keys=True)
        action_key = hashlib.sha256(binding.encode()).hexdigest()
        with _lock(self.root / 'actions' / (action_key + '.lock')):
            return self._run(operation, fn, fallback=fallback, evidence=evidence, refine=refine,
                             reset=reset, attempts=attempts, fallback_action=fallback_action,
                             known=known, validate_result=validate_result)

    def _run(self, operation, fn, *, fallback, evidence, refine, reset, attempts,
             fallback_action, known, validate_result):
        if validate_result and self.path.exists():
            try:
                with _lock(self.root / 'ledger.lock'):
                    prior = [r for r in self._load()['errors'].values()
                             if r['binding']['operation'] == operation
                             and r['binding']['evidence'] == safe_evidence(evidence or {})
                             and 'result' in r and r['status'] == 'resolved']
                for record in reversed(prior):
                    if validate_result(record['result']):
                        return record['result']
            except BaseException as error:
                if cancelled(error) or external_failure(error):
                    raise
                _write(self.root / 'diagnosis_unavailable.json', error_evidence(error))
        decision = None
        for attempt in range(attempts):
            try:
                result = fn()
                if decision:
                    self._event(decision['error_id'], 'retry', 'validated')
                return result
            except BaseException as error:
                if cancelled(error) or external_failure(error) or isinstance(error, known):
                    raise
                allowed = (['retry'] if attempts > 1 else []) + [fallback_action] + (['refine'] if refine else [])
                if decision is None:
                    try:
                        decision = self.decide(operation, error, allowed=allowed,
                                               evidence=evidence, fallback=fallback_action)
                    except BaseException as diagnostic_error:
                        if cancelled(diagnostic_error) or external_failure(diagnostic_error):
                            raise
                        # Preserve a damaged ledger; do not reset its quota or diagnose it recursively.
                        _write(self.root / 'diagnosis_unavailable.json', error_evidence(diagnostic_error))
                        return fallback(error)
                key = decision['error_id']
                with _lock(self.root / 'ledger.lock'):
                    actions = self._load()['errors'][key]['actions']
                    retries = sum(a['action'] == 'retry' and a['state'] == 'started' for a in actions)
                    repair_started = any(a['action'] == 'refine' and a['state'] == 'started' for a in actions)
                if decision['action'] == 'retry' and attempt+1 < attempts and retries < attempts-1:
                    self._event(key, 'retry', 'started')
                    if reset:
                        try:
                            reset()
                        except BaseException as reset_error:
                            if cancelled(reset_error) or external_failure(reset_error):
                                raise
                            self._event(key, 'retry', 'failed', error=error_evidence(reset_error))
                            decision = {**decision, 'action': fallback_action}
                        else:
                            continue
                    else:
                        continue
                if decision['action'] == 'refine' and refine and not repair_started:
                    self._event(key, 'refine', 'started')
                    # The repair callback is an existing budgeted stage transition.
                    return refine(error, decision)
                self._event(key, fallback_action, 'started')
                try:
                    result = fallback(error)
                except BaseException as failure:
                    self._event(key, fallback_action, 'fallback_failed', error=error_evidence(failure))
                    raise
                self._event(key, fallback_action, 'resolved', verification='stage callback completed')
                if validate_result and validate_result(result):
                    json.dumps(result, allow_nan=False)
                    with _lock(self.root / 'ledger.lock'):
                        data = self._load()
                        data['errors'][key]['result'] = result
                        _write(self.path, data)
                return result


def object_recovery(args, api, root, ask):
    owner = Path(getattr(args, 'error_recovery_root', None) or root)
    def diagnose(stage, prompt, payload):
        return ask(stage, [api.make_text_input(prompt), api.make_text_input(payload)])
    return Recovery(owner, 'object', diagnose, limit=getattr(args, 'max_error_diagnoses', 5))
