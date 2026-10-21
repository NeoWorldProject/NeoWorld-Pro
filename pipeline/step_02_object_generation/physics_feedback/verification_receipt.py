"""Reuse independently computed full-trace metrics, never a report's verdict.

The report verifier still checks input/evidence hashes, coverage, all reported
metrics, applicability, reload and delivery on every call. Only the expensive
JSONL parsing and numerical recomputation are memoized across processes.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

from urdf_core.run_state import file_hash

POLICY = 'independent_full_trace_metrics.v1'


def receipt_folder(root):
    root = Path(root).resolve()
    for parent in (root, *root.parents):
        manifest = parent / 'run_manifest.json'
        if manifest.is_file() and json.loads(manifest.read_text()).get('config', {}).get('harness_policy') == 'object_pipeline.v1':
            root = parent
            break
    return root / 'verification_receipts'


def evidence_binding(binding):
    return {**binding, 'verifier': {'files': binding['verifier']['files']}}


def find_verified_metrics(path, *, root, kind, joint_name, contract, inputs, include_initial_sample=False):
    """Read the owner's completed review; native resume never becomes a reviewer."""
    path = Path(path).resolve()
    expected = {'policy': POLICY, 'trajectory': {'path': str(path), 'sha256': file_hash(path)},
                'kind': kind, 'joint_name': joint_name, 'contract': contract, 'inputs': inputs,
                'verifier': {'files': verifier_identity()['files']},
                'include_initial_sample': include_initial_sample}
    folder = receipt_folder(root)
    index = folder / ('evidence_' + digest(expected) + '.json')
    if not index.exists():
        return None
    record = json.loads(index.read_text())
    target = folder / record['receipt']
    if target.parent != folder or file_hash(target) != record['sha256']:
        raise ValueError('Independent review receipt index changed')
    saved = json.loads(target.read_text())
    payload = saved.get('payload')
    if (not isinstance(payload, dict) or saved.get('payload_sha256') != digest(payload)
            or evidence_binding(payload['binding']) != expected):
        raise ValueError('Independent review does not bind this completed trial')
    result = payload['result']
    if include_initial_sample:
        result = tuple(result)
    return result, {'policy': POLICY, 'reused': True, 'receipt': str(target),
                    'review_environment': {key: value for key, value in payload['binding']['verifier'].items() if key != 'files'}}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def verifier_identity():
    """Bind calculation, parsing and acceptance dependencies, including this cache."""
    import numpy
    import scipy
    base = Path(__file__).resolve().parent.parent
    paths = sorted(p for name in ('physics_feedback', 'urdf_core', 'kinematics_validation')
                   for p in (base / name).glob('*.py'))
    return {'files': {str(p.relative_to(base)): file_hash(p) for p in paths},
            'python': list(sys.version_info[:3]), 'numpy': numpy.__version__,
            'scipy': scipy.__version__}


def verified_metrics(path, *, root, kind, joint_name, contract, inputs, evaluator,
                     include_initial_sample=False):
    """Compute once, or reuse exact content with the same inputs and verifier.

An unsuccessful/partial computation cannot create a receipt. A computed failure
is reusable evidence of failure, not acceptance. Read-only historical storage
falls back to a full calculation rather than skipping verification.
"""
    started = time.monotonic()
    path = Path(path).resolve()
    source_digest = file_hash(path)
    identity = verifier_identity()
    binding = {'policy': POLICY, 'trajectory': {'path': str(path), 'sha256': source_digest},
               'kind': kind, 'joint_name': joint_name, 'contract': contract,
               'inputs': inputs, 'verifier': identity,
               'include_initial_sample': include_initial_sample}
    key = digest(binding)
    folder = receipt_folder(root)
    target = folder / (key + '.json')

    def compute():
        options = {'include_initial_sample': True} if include_initial_sample else {}
        value = evaluator(path, kind=kind, joint_name=joint_name, contract=contract, **options)
        if file_hash(path) != source_digest or verifier_identity() != identity or digest(binding) != key:
            raise ValueError('Verification inputs changed during full trajectory review')
        return value

    def detail(reused, stored):
        return {'policy': POLICY, 'reused': reused, 'receipt': str(target) if stored else None,
                'wall_seconds': time.monotonic() - started}

    try:
        folder.mkdir(parents=True, exist_ok=True)
        lock = (folder / (key + '.lock')).open('a')
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EROFS):
            raise
        return compute(), detail(False, False)
    with lock:
        # Different publication/collection processes cannot both do the first review.
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.exists():
            saved = json.loads(target.read_text())
            payload = saved.get('payload')
            if (not isinstance(payload, dict) or saved.get('payload_sha256') != digest(payload)
                    or payload.get('binding') != binding):
                raise ValueError('Independent trajectory verification receipt changed')
            # Recheck after acquiring the lock; another process may have waited here.
            if file_hash(path) != source_digest or verifier_identity() != identity:
                raise ValueError('Verification inputs changed while waiting for the receipt')
            value = payload['result']
            if include_initial_sample:
                value = tuple(value)
            return value, detail(True, True)
        value = compute()
        payload = {'binding': binding, 'result': value,
                   'full_review_wall_seconds': time.monotonic() - started,
                   'created_epoch': time.time()}
        pending = target.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        try:
            pending.write_text(json.dumps({'payload': payload, 'payload_sha256': digest(payload)},
                                          allow_nan=False, ensure_ascii=False) + '\n')
            os.replace(pending, target)
            index = folder / ('evidence_' + digest(evidence_binding(binding)) + '.json')
            index_pending = index.with_suffix('.' + uuid.uuid4().hex + '.tmp')
            index_pending.write_text(json.dumps({'receipt': target.name, 'sha256': file_hash(target)}) + '\n')
            os.replace(index_pending, index)
        finally:
            pending.unlink(missing_ok=True)
        return value, detail(False, True)
