"""Iteration budgets and delivery policy for the current object pipeline."""
from __future__ import annotations
import json
import time
import fcntl
from contextlib import contextmanager
from pathlib import Path
from .run_state import fingerprint, write_json

POLICY = 'object_pipeline.v1'
REPAIR_LIMITS = {'geometry': 3, 'articulation': 3, 'physics': 8}


def enabled(args):
    if getattr(args, 'delivery_policy', 'best-effort') != 'best-effort':
        raise ValueError('Current object pipeline always delivers its best valid asset')
    return True


def resolve(args):
    enabled(args) if getattr(args, 'delivery_policy', None) is not None else None
    path = Path(args.out_dir) / 'run_manifest.json'
    if path.is_file():
        previous = json.loads(path.read_text()).get('config', {})
        if previous.get('harness_policy') != POLICY:
            raise ValueError('Historical run uses a frozen harness; use a new output directory')
    args.delivery_policy = 'best-effort'
    args.harness_policy = POLICY
    return args.delivery_policy


class SharedBudget:
    """Persistent atomic repair reservations shared by concurrent candidates."""
    def __init__(self, root, domain, limit=None):
        if domain not in REPAIR_LIMITS:
            raise ValueError('Unknown current repair domain; historical budgets require frozen code')
        maximum = REPAIR_LIMITS[domain]
        limit = maximum if limit is None else limit
        if type(limit) is not int or not 0 <= limit <= maximum:
            raise ValueError(f'{domain} repair budget must be an integer in 0..{maximum}')
        self.path = Path(root) / 'delivery_budgets' / (domain + '.json')
        self.domain, self.limit = domain, limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as data:
            self._validate(data)

    def _validate(self, data):
        if data.get('schema_version') != POLICY:
            raise ValueError('Historical budget belongs to its frozen harness')
        if data['limit'] != self.limit or data['domain'] != self.domain:
            raise ValueError('Shared budget is frozen')

    @contextmanager
    def transaction(self):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else {
                'schema_version': POLICY, 'domain': self.domain, 'limit': self.limit, 'entries': {}}
            self._validate(data)
            yield data
            write_json(self.path, data)

    @property
    def data(self):
        with self.transaction() as data:
            return data

    @property
    def remaining(self):
        data = self.data
        return data['limit'] - len(data['entries'])

    def reserve(self, identity, category, binding):
        value = {'category': category, 'binding': fingerprint(binding), 'status': 'pending'}
        with self.transaction() as data:
            old = data['entries'].get(identity)
            if old is not None:
                if any(old[k] != value[k] for k in ('category', 'binding')):
                    raise ValueError('Repair reservation changed')
                return
            if len(data['entries']) >= data['limit']:
                raise BudgetExhausted('Shared repair budget exhausted')
            data['entries'][identity] = value

    def complete(self, identity, *, status='measured', **details):
        with self.transaction() as data:
            data['entries'][identity].update(status=status, **details)


class BudgetExhausted(RuntimeError):
    pass


class RunTiming:
    """Record elapsed time without imposing a stopping condition."""
    def __init__(self, root):
        self.path = Path(root) / 'run_timing.json'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.data = json.loads(self.path.read_text()) if self.path.exists() else {
                'schema_version': POLICY, 'started_epoch': time.time()}
            if self.data.get('schema_version') != POLICY:
                raise ValueError('Historical timing record belongs to its frozen harness')
            write_json(self.path, self.data)

    @property
    def elapsed(self):
        return max(0., time.time() - self.data['started_epoch'])
