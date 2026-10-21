"""Persistent API attempts and one process-shared adaptive admission queue."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import random
import re
import shutil
import time
import uuid

from .run_state import file_hash, fingerprint

POLICY = 'object_api_requests.v2'
MAX_ATTEMPTS = 5


class APIAttemptsExhausted(RuntimeError):
    pass


class ServiceIssue(RuntimeError):
    """The provider requires account or request configuration changes."""


class RejectedContent(str):
    """An exhausted/filtered output is received evidence, never a usable answer."""
    def __new__(cls, raw_text, reason):
        result = super().__new__(cls, '')
        result.evidence = {'provider_status': 'incomplete', 'reason': reason, 'raw_text': raw_text}
        return result


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _owner():
    pid = os.getpid()
    try:
        started = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except OSError:
        started = ''
    return {'pid': pid, 'started': started}


def _alive(owner):
    try:
        fields = Path(f'/proc/{owner["pid"]}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == owner['started']
    except (OSError, KeyError, IndexError):
        return False


class APISlots:
    """FIFO admission; rate limits halve capacity, sustained success restores it."""
    def __init__(self, root=None, limit=None, *, sleep=time.sleep, now=time.time):
        self.root = Path(root or os.environ.get('NEOWORLD_API_QUEUE_DIR',
                         str(Path('/tmp') / f'neoworld_api_{os.getuid()}')))
        self.limit = int(limit if limit is not None else os.environ.get('NEOWORLD_API_CONCURRENCY', '64'))
        if self.limit < 1:
            raise ValueError('API concurrency must be positive')
        self.now, self.sleep = now, sleep

    @contextmanager
    def state(self):
        with _lock(self.root / 'queue.lock'):
            path = self.root / 'queue.json'
            data = json.loads(path.read_text()) if path.exists() else {
                'policy': POLICY, 'limit': self.limit, 'capacity': self.limit,
                'active': {}, 'waiting': [], 'cooldown_until': 0., 'successes': 0}
            data['active'] = {k: v for k, v in data['active'].items() if _alive(v)}
            data['waiting'] = [v for v in data['waiting'] if _alive(v['owner'])]
            if data['limit'] != self.limit:
                if data['active'] or data['waiting']:
                    raise ValueError('Active API workers must share one concurrency setting')
                data.update(limit=self.limit, capacity=self.limit, successes=0)
            try:
                yield data
            finally:
                _write(path, data)

    @contextmanager
    def admission(self):
        ticket = uuid.uuid4().hex
        with self.state() as data:
            data['waiting'].append({'ticket': ticket, 'owner': _owner()})
        try:
            while True:
                with self.state() as data:
                    ready = (data['waiting'] and data['waiting'][0]['ticket'] == ticket
                             and len(data['active']) < data['capacity']
                             and self.now() >= data['cooldown_until'])
                    if ready:
                        data['waiting'].pop(0)
                        data['active'][ticket] = _owner()
                if ready:
                    break
                self.sleep(.1)
            yield
        finally:
            with self.state() as data:
                data['active'].pop(ticket, None)
                data['waiting'] = [v for v in data['waiting'] if v['ticket'] != ticket]

    def outcome(self, error=None):
        with self.state() as data:
            if getattr(error, 'status_code', None) == 429:
                data['capacity'] = max(1, data['capacity'] // 2)
                headers = getattr(getattr(error, 'response', None), 'headers', {}) or {}
                try:
                    delay = max(2., float(headers.get('retry-after', 2.)))
                except (ValueError, TypeError):
                    delay = 2.
                data['cooldown_until'] = max(data['cooldown_until'], self.now() + delay)
                data['successes'] = 0
            elif error is None:
                data['successes'] += 1
                if data['successes'] >= 20 and self.now() >= data['cooldown_until']:
                    data['capacity'] = min(data['limit'], data['capacity'] + 1)
                    data['successes'] = 0


def logical_stage(stage):
    while re.search(r'(?:_json_repair_\d+|_try_\d+)$', stage):
        stage = re.sub(r'(?:_json_repair_\d+|_try_\d+)$', '', stage)
    return stage


def _error_details(error):
    def service_body(value):
        if isinstance(value, dict):
            return {key: service_body(item) for key, item in value.items()
                    if key in {'error', 'type', 'code', 'message', 'param'}}
        if isinstance(value, list):
            return [service_body(item) for item in value]
        return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
    return {'error_type': type(error).__name__, 'error_message': str(error),
            'error_code': getattr(error, 'code', None),
            'error_body': service_body(getattr(error, 'body', None)),
            'status_code': getattr(error, 'status_code', None)}


def _fatal(error):
    """Streaming failures may carry only a message, without an HTTP status."""
    status = getattr(error, 'status_code', None)
    details = _error_details(error)
    text = ' '.join(str(details[key]) for key in ('error_code', 'error_body', 'error_message')).lower()
    markers = ('insufficient_quota', 'insufficient_credits', 'credits_exhausted',
               'billing_hard_limit_reached', 'billing_not_active', 'account_deactivated',
               'organization_deactivated', 'invalid_api_key', 'no credits remaining',
               'no credits', 'out of credits', 'not enough credits', 'insufficient credits',
               'credit balance is too low', 'credit balance exhausted',
               'account balance is insufficient', 'insufficient account balance',
               'billing limit', 'account has been deactivated')
    return any(marker in text for marker in markers) or (
        isinstance(status, int) and 400 <= status < 500 and status not in {408, 409, 429})


def _retryable(error):
    return not _fatal(error)


def _reuse_received(source_root, directory, key, logical, request_key):
    from .api_metadata import receipt_metadata
    source = Path(source_root).resolve() / 'model_calls'
    ledger = source / f'request_{key}.json'
    if not ledger.is_file():
        return None
    digest = file_hash(ledger)
    data = json.loads(ledger.read_text())
    if data.get('policy') != POLICY or data.get('stage') != logical:
        raise ValueError('API reuse ledger has an incompatible stage or policy')
    saved = data.get('responses', {}).get(request_key)
    if saved is None:
        return None
    for field in ('request', 'response'):
        name = saved[field]
        if Path(name).name != name:
            raise ValueError('API reuse path escapes its source')
        path = source / name
        if file_hash(path) != saved[field + '_sha256']:
            raise ValueError('API reuse source evidence changed')
    if fingerprint(json.loads((source / saved['request']).read_text())) != request_key:
        raise ValueError('API reuse request does not match current input')
    attempts = [a for a in data['attempts'] if a.get('request_key') == request_key
                and a.get('status') == 'received' and a.get('response') == saved['response']]
    if len(attempts) != 1:
        raise ValueError('API reuse needs exactly one bound received response')
    attempts = [{**row, 'api_metadata': receipt_metadata(row['api_metadata'])}
                if 'api_metadata' in row else dict(row) for row in attempts]
    if not (source / saved['response']).read_text().strip():
        return None
    for field in ('request', 'response'):
        shutil.copy2(source / saved[field], directory / saved[field])
        if file_hash(directory / saved[field]) != saved[field + '_sha256']:
            raise ValueError('API reuse copy differs')
    if file_hash(ledger) != digest:
        raise ValueError('API reuse ledger changed during import')
    return {'policy': POLICY, 'stage': logical, 'attempts': attempts,
            'responses': {request_key: saved},
            'reused_from': {'ledger': str(ledger), 'sha256': digest,
                            'request_key': request_key, 'api_call_made': False}}


def audited_call(root, stage, request, produce, *, prepare=None, slots=None, sleep=time.sleep,
                 service_restored=False, reuse_root=None):
    """Bound transport attempts per resumed process; retain the cumulative ledger."""
    from .api_metadata import receipt_metadata
    directory = Path(root) / 'model_calls'
    directory.mkdir(parents=True, exist_ok=True)
    logical = logical_stage(stage)
    key = fingerprint({'stage': logical})
    ledger = directory / f'request_{key}.json'
    request_key = fingerprint(request)
    scheduler = slots or APISlots()
    with _lock(directory / f'request_{key}.lock'):
        if not ledger.exists() and reuse_root is not None:
            reused = _reuse_received(reuse_root, directory, key, logical, request_key)
            if reused is not None:
                _write(ledger, reused)
        data = json.loads(ledger.read_text()) if ledger.exists() else {
            'policy': POLICY, 'stage': logical, 'attempts': [], 'responses': {}}
        if data['policy'] != POLICY:
            raise ValueError('API request policy changed')
        for attempt in data['attempts']:
            response_path = directory / attempt.get('response', '__missing__')
            if (attempt.get('request_key') == request_key and attempt['status'] == 'dispatched'
                    and (response_path.is_file() or 'content_evidence' in attempt)):
                request_path = directory / attempt['request']
                if file_hash(request_path) != attempt['request_sha256']:
                    raise ValueError('Interrupted API request changed')
                evidence = attempt.get('content_evidence')
                if evidence is not None:
                    if (not isinstance(evidence, dict) or evidence.get('provider_status') != 'incomplete'
                            or evidence.get('reason') not in {'max_output_tokens', 'content_filter'}
                            or not isinstance(evidence.get('raw_text'), str)):
                        raise ValueError('Interrupted rejected-content evidence is invalid')
                    if response_path.is_file() and response_path.read_text() != '':
                        raise ValueError('Rejected-content response changed')
                    if not response_path.is_file():
                        response_path.write_text('')
                attempt.update(status='received', finished_epoch=time.time(),
                    content_status='empty_or_rejected' if not response_path.read_text().strip() else 'unvalidated')
                data['responses'][request_key] = {'request': request_path.name, 'response': response_path.name,
                    'request_sha256': file_hash(request_path), 'response_sha256': file_hash(response_path)}
                _write(ledger, data)
        saved = data['responses'].get(request_key)
        if saved:
            response_path = directory / saved['response']
            request_path = directory / saved['request']
            if (file_hash(response_path) != saved['response_sha256']
                    or file_hash(request_path) != saved['request_sha256']):
                raise ValueError('Saved API request/reply changed')
            return response_path.read_text(), [request_path, response_path, ledger]
        if data.get('initial_stage') == stage and data.get('initial_request') != request_key:
            raise ValueError('A logical API request changed its original evidence')
        data.setdefault('initial_stage', stage)
        data.setdefault('initial_request', request_key)
        if data.get('terminal_error') and data.get('service_issue') and service_restored:
            prior = data['attempts'][-1]
            if prior.get('status') != 'error' or prior.get('request_key') != request_key:
                raise ValueError('Service recovery must resume the exact failed request')
            data.setdefault('service_recoveries', []).append({
                'confirmed_epoch': time.time(), 'owner': _owner(), 'request_key': request_key,
                'previous_service_issue': data['service_issue'], 'prior_attempts': len(data['attempts'])})
            data['terminal_error'] = False
            data.pop('service_issue')
            _write(ledger, data)
        if data.get('terminal_error'):
            if data.get('service_issue'):
                raise ServiceIssue(f'{logical}: {data["service_issue"]["error_message"]}')
            raise APIAttemptsExhausted(f'{logical}: previous request was not retryable')
        owner = _owner()
        session_id = fingerprint({'request_key': request_key, 'owner': owner})
        sessions = data.setdefault('transport_sessions', {})
        sessions.setdefault(session_id, {'owner': owner, 'request_key': request_key,
            'started_epoch': time.time(), 'prior_attempts': len(data['attempts']),
            'limit': MAX_ATTEMPTS})
        last_error = None
        def request_attempts():
            return sum(item.get('transport_session') == session_id for item in data['attempts'])
        while request_attempts() < MAX_ATTEMPTS:
            with scheduler.admission():
                if prepare is not None:
                    prepare()
                number = len(data['attempts']) + 1
                stem = f'{key}_{number:02d}_{re.sub(r"[^a-zA-Z0-9_-]", "_", stage)}'
                request_path = directory / f'{stem}.request.json'
                response_path = directory / f'{stem}.response.txt'
                _write(request_path, request)
                attempt = {'number': number, 'request': request_path.name,
                           'request_sha256': file_hash(request_path), 'request_key': request_key,
                           'transport_session': session_id,
                           'response': response_path.name, 'status': 'dispatched',
                           'started_epoch': time.time()}
                data['attempts'].append(attempt)
                _write(ledger, data)
                try:
                    response = produce()
                    if not isinstance(response, str):
                        raise TypeError('API adapter must return response text')
                except Exception as error:
                    if getattr(error, 'api_metadata', None) is not None:
                        attempt['api_metadata'] = receipt_metadata(error.api_metadata)
                    fatal = _fatal(error)
                    if not fatal:
                        scheduler.outcome(error)
                    attempt.update(status='error', **_error_details(error),
                                   retryable=_retryable(error), finished_epoch=time.time())
                    _write(directory / f'{stem}.error.json', attempt)
                    data['terminal_error'] = not attempt['retryable']
                    if fatal:
                        data['service_issue'] = _error_details(error)
                    _write(ledger, data)
                    if data['terminal_error']:
                        raise ServiceIssue(f'{logical}: {error}') from error
                    last_error = error
                else:
                    if hasattr(response, 'api_metadata'):
                        attempt['api_metadata'] = receipt_metadata(response.api_metadata)
                        _write(ledger, data)
                    if isinstance(response, RejectedContent):
                        attempt['content_evidence'] = response.evidence
                        _write(ledger, data)
                    temporary = response_path.with_suffix('.pending')
                    temporary.write_text(response)
                    temporary.replace(response_path)
                    scheduler.outcome()
                    attempt.update(status='received', finished_epoch=time.time(),
                                   content_status='empty_or_rejected' if not response.strip() else 'unvalidated')
                    saved = {'request': request_path.name, 'response': response_path.name,
                             'request_sha256': file_hash(request_path),
                             'response_sha256': file_hash(response_path)}
                    data['responses'][request_key] = saved
                    _write(ledger, data)
                    return response, [request_path, response_path, ledger]
            if request_attempts() < MAX_ATTEMPTS:
                sleep(min(30., 2. ** request_attempts()) + random.random())
        raise APIAttemptsExhausted(
            f'{logical}: five technical API attempts exhausted in this process; '
            'resume after the service recovers; refinement budget is unchanged') from last_error
