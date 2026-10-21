"""Reuse exact per-part collision geometry, independently of physical parameters."""
from __future__ import annotations

import copy
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

import numpy as np
import trimesh

from urdf_core.run_state import file_hash, fingerprint, write_json
from . import lightweight_collision, coacd_worker

POLICY = 'exact_part_collision_inputs.v1'


def _mesh_identity(mesh):
    result = {}
    for name, dtype in (('vertices', '<f8'), ('faces', '<i8')):
        values = np.ascontiguousarray(getattr(mesh, name), dtype=dtype)
        result[name] = {'shape': list(values.shape), 'sha256': hashlib.sha256(values.tobytes()).hexdigest()}
    return result


def _implementation():
    return {'source': {Path(p).name: file_hash(p) for p in (
                __file__, lightweight_collision.__file__, coacd_worker.__file__)},
            'versions': {name: version(name) for name in
                         ('numpy', 'scipy', 'trimesh', 'coacd', 'fast-simplification')}}


def _read(folder, binding, key):
    try:
        envelope = json.loads((folder / 'receipt.json').read_text())
        record = envelope['record']
        if envelope['sha256'] != fingerprint(record) or record['inputs'] != binding:
            raise ValueError('receipt does not match its content or inputs')
        arrays_path = folder / 'meshes.npz'
        if file_hash(arrays_path) != record['meshes_sha256']:
            raise ValueError('mesh content changed')
        proxies = []
        with np.load(arrays_path, allow_pickle=False) as arrays:
            for index, metadata in enumerate(record['proxies']):
                mesh = trimesh.Trimesh(vertices=arrays[f'v{index}'].copy(),
                    faces=arrays[f'f{index}'].copy(), process=False)
                proxies.append({**metadata, 'mesh': mesh})
        audit = copy.deepcopy(record['audit'])
    except (ValueError, KeyError, TypeError, EOFError) as error:
        raise OSError(f'Invalid collision cache {folder}: {error}') from error
    audit['cache'] = {'policy': POLICY, 'key': key, 'reused': True}
    return [p['mesh'] for p in proxies], proxies, audit


def plan_cached_collisions(parts, *, config, cache_root, identity):
    """Hash exact link-local source meshes and collision options; never cache tests."""
    parts = list(parts)
    binding = {'policy': POLICY, 'identity': identity,
        'meshes': [_mesh_identity(mesh) for mesh in parts],
        'collision_policy': lightweight_collision.effective_policy(config),
        'implementation': _implementation()}
    key = fingerprint(binding)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    folder = cache_root / key
    with (cache_root / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if folder.exists():
            return _read(folder, binding, key)
        refs, proxies, audit = lightweight_collision.plan_lightweight_collisions(parts, config=config)
        with tempfile.TemporaryDirectory(prefix='.building_', dir=cache_root) as temporary:
            staging = Path(temporary) / key
            staging.mkdir()
            arrays, metadata = {}, []
            for index, proxy in enumerate(proxies):
                arrays[f'v{index}'] = np.asarray(proxy['mesh'].vertices, dtype=np.float64)
                arrays[f'f{index}'] = np.asarray(proxy['mesh'].faces, dtype=np.int64)
                metadata.append({k: copy.deepcopy(v) for k, v in proxy.items() if k != 'mesh'})
            np.savez_compressed(staging / 'meshes.npz', **arrays)
            record = {'inputs': binding, 'proxies': metadata, 'audit': audit,
                      'meshes_sha256': file_hash(staging / 'meshes.npz')}
            write_json(staging / 'receipt.json', {'record': record, 'sha256': fingerprint(record)})
            staging.rename(folder)
        audit = {**audit, 'cache': {'policy': POLICY, 'key': key, 'reused': False}}
        return refs, proxies, audit
