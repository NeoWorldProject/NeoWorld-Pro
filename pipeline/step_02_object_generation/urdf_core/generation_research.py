"""Generation trial checkpoints backed by the object's shared repair ledger."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import json
from pathlib import Path
import threading

from .run_state import file_hash, fingerprint, write_json
from .delivery_policy import SharedBudget, BudgetExhausted


class GenerationResearch:
    def __init__(self, root, stage, *, baseline, implementation, limit=3):
        if stage not in {'geometry', 'articulation'}:
            raise ValueError('Unknown generation stage')
        self.root, self.stage = Path(root).resolve(), stage
        self.path = self.root / 'generation_research' / stage / 'state.json'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.implementation = implementation
        self._mutex = threading.RLock()
        manifest = self.root / 'run_manifest.json'
        config = json.loads(manifest.read_text()).get('config', {}) if manifest.exists() else {}
        limit = config.get(stage + '_repairs', limit)
        if type(limit) is not int or not 0 <= limit <= 3:
            raise ValueError('Geometry and articulation allow at most three refinements; use frozen code for older runs')
        self.shared = SharedBudget(self.root, stage, limit)
        with self._transaction():
            if not self.data:
                self.data = {'schema_version': 1, 'stage': stage, 'baseline_hash': fingerprint(baseline),
                    'baseline': copy.deepcopy(baseline), 'implementation': implementation,
                    'initial_limit': limit, 'allowance': limit, 'used_repairs': 0,
                    'trials': {}, 'phases': {}, 'history': [], 'pending_review': None, 'status': 'running'}
                self.event('initialized', {'baseline_hash': self.data['baseline_hash'], 'limit': limit})
            elif (self.data['baseline_hash'] != fingerprint(baseline)
                    or self.data['initial_limit'] != limit or self.data['implementation'] != implementation):
                raise ValueError('Generation inputs, implementation or budget changed; use a new run')
            self._verify()

    @contextmanager
    def _transaction(self):
        with self._mutex, self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def snapshot(self):
        with self._transaction():
            return copy.deepcopy(self.data)

    def event(self, kind, payload):
        event = {'sequence': len(self.data['history']), 'kind': kind,
                 'previous_sha256': self.data['history'][-1]['sha256'] if self.data['history'] else None,
                 'payload': copy.deepcopy(payload)}
        event['sha256'] = fingerprint(event)
        self.data['history'].append(event)
        write_json(self.path, self.data)

    def _files(self, paths):
        files = {}
        for value in paths:
            path = Path(value).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError('Generation evidence must belong to this run')
            files[str(path.relative_to(self.root))] = file_hash(path)
        return files

    def _verify_files(self, files):
        for relative, digest in files.items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root) or not path.is_file() or file_hash(path) != digest:
                raise ValueError('Generation evidence changed: ' + relative)

    def _verify(self):
        previous, trials, phases = None, {}, {}
        for event in self.data['history']:
            body = {k: v for k, v in event.items() if k != 'sha256'}
            if body['previous_sha256'] != previous or fingerprint(body) != event['sha256']:
                raise ValueError('Generation history changed')
            previous = event['sha256']
            if event['kind'] in {'trial_started', 'trial_checked'}:
                trials[event['payload']['token']] = event['payload']
            elif event['kind'] == 'phase_completed':
                phases[event['payload']['identity']] = event['payload']
        if trials != self.data['trials'] or phases != self.data['phases']:
            raise ValueError('Generation checkpoints differ from their history')
        if sum(t['is_repair'] for t in trials.values() if t['status'] != 'pending') != self.data['used_repairs']:
            raise ValueError('Generation counters differ from their history')
        for value in [*trials.values(), *phases.values()]:
            self._verify_files(value.get('artifacts', {}))

    def start(self, token, *, scope='object', context=None):
        with self._transaction():
            if token in self.data['trials']:
                return copy.deepcopy(self.data['trials'][token])
            prior = [t for t in self.data['trials'].values() if t['scope'] == scope]
            if any(t['status'] == 'pending' for t in prior):
                raise ValueError('Resume the pending candidate in this scope first')
            repair = bool(prior)
            if repair:
                self.shared.reserve(self.stage + ':' + token, self.stage, {'scope': scope, 'context': context})
            trial = {'token': token, 'scope': scope, 'is_repair': repair, 'status': 'pending',
                     'context': copy.deepcopy(context), 'implementation': self.implementation}
            self.data['trials'][token] = trial
            self.event('trial_started', trial)
            return copy.deepcopy(trial)

    def checked(self, token, *, passed, reason, paths=(), result=None):
        with self._transaction():
            trial = self.data['trials'][token]
            status = 'passed' if passed else 'failed'
            if trial['status'] != 'pending':
                if trial['status'] != status or trial['reason'] != reason:
                    raise ValueError('Recorded generation result changed')
                self._verify_files(trial['artifacts'])
                return
            trial.update(status=status, reason=reason, artifacts=self._files(paths), result=copy.deepcopy(result))
            if trial['is_repair']:
                self.shared.complete(self.stage + ':' + token)
                self.data['used_repairs'] += 1
            self.event('trial_checked', trial)
            if trial['is_repair'] and self.data['used_repairs'] % 3 == 0:
                self.event('internal_strategy_review', {'remaining': self.shared.remaining,
                    'decision': 'use_measured_history_for_next_proposal', 'acceptance_changed': False})

    def phase(self, token, name, produce, *, key=None):
        identity, digest = token + ':' + name, fingerprint(key)
        phase_lock = self.path.parent / ('phase_' + fingerprint(identity) + '.lock')
        with phase_lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self._transaction():
                saved = self.data['phases'].get(identity)
                if saved:
                    if saved['key'] != digest:
                        raise ValueError('Generation phase inputs changed')
                    self._verify_files(saved['artifacts'])
                    return copy.deepcopy(saved['result'])
            result, paths = produce()
            with self._transaction():
                value = {'identity': identity, 'key': digest, 'implementation': self.implementation,
                         'result': copy.deepcopy(result), 'artifacts': self._files(paths)}
                self.data['phases'][identity] = value
                self.event('phase_completed', value)
            return result

    def refinement_context(self, token):
        with self._transaction():
            saved = self.data.get('refinement_contexts', {}).get(token)
            if saved is not None:
                self._verify_files({saved['path']: saved['sha256']})
                return json.loads((self.root / saved['path']).read_text())
            trial = self.data['trials'][token]
            history = []
            for prior in self.data['trials'].values():
                if prior['token'] == token:
                    break
                if prior['status'] != 'pending':
                    history.append({'token': prior['token'], 'reason': prior['reason'], 'result': prior['result']})
            value = {'stage': self.stage, 'current': token, 'is_repair': trial['is_repair'],
                     'remaining_repairs': self.shared.remaining, 'measured_history': history}
            path = self.path.parent / 'refinement_contexts' / (fingerprint(token) + '.json')
            write_json(path, value)
            binding = {'token': token, 'path': str(path.relative_to(self.root)), 'sha256': file_hash(path)}
            self.data.setdefault('refinement_contexts', {})[token] = binding
            self.event('refinement_context_bound', binding)
            return copy.deepcopy(value)


def cached_response(root, stage, request, produce, *, prepare=None):
    from .api_requests import audited_call
    from .api_autonomy import assert_autonomous_context
    assert_autonomous_context(request.get("content", []))
    return audited_call(root, stage, request, produce, prepare=prepare)
