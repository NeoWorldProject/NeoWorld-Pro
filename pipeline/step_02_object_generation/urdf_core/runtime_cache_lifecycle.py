"""Reclaim only a native worker's private caches after its complete process tree exits."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import stat
import threading
import time
import uuid

from .run_state import write_json

SCHEMA = 'isaac_cache_lifecycle.v1'
RECEIPT_DIR = '.runtime_cache_lifecycle'
CACHE_DIRS = ('runtime/cache', 'runtime/cuda', 'runtime/shader', 'runtime/kit/cache')
MARKER_ENV = 'NEOWORLD_NATIVE_CACHE_LEASE'


def _source_hash():
    import hashlib
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _identity(path):
    value = Path(path).lstat()
    return [value.st_dev, value.st_ino]


def _processes():
    rows = {}
    uncertain = []
    for item in Path('/proc').glob('[0-9]*'):
        try:
            fields = (item / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            row = dict(pid=int(item.name), parent=int(fields[1]), group=int(fields[2]),
                       session=int(fields[3]), start_tick=fields[19], marker=None)
            if item.stat().st_uid == os.getuid():
                try:
                    prefix = (MARKER_ENV + '=').encode()
                    row['marker'] = next((v[len(prefix):].decode() for v in (item / 'environ').read_bytes().split(b'\0')
                                          if v.startswith(prefix)), None)
                except PermissionError:
                    uncertain.append(row['pid'])
            rows[row['pid']] = row
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError):
            uncertain.append(int(item.name))
    return rows, uncertain


def _only_empty_directories_or_links(path):
    for item in path.iterdir():
        mode = item.lstat().st_mode
        if stat.S_ISLNK(mode):
            continue
        if not stat.S_ISDIR(mode) or not _only_empty_directories_or_links(item):
            return False
    return True


def _remove_private_tree(path, device, receipt):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        receipt['retained_shared_links'].append(str(path))
        return
    if info.st_dev != device:
        receipt['retained_other_filesystems'].append(str(path))
        return
    if stat.S_ISDIR(info.st_mode):
        for child in path.iterdir():
            _remove_private_tree(child, device, receipt)
        try:
            path.rmdir()
        except OSError:
            if any(path.iterdir()):
                return
            raise
    elif stat.S_ISREG(info.st_mode):
        path.unlink()
        receipt['removed_files'] += 1
        receipt['removed_logical_bytes'] += info.st_size
        if info.st_nlink == 1:
            receipt['released_allocated_bytes'] += info.st_blocks * 512
    else:
        receipt['retained_special_files'].append(str(path))


class RuntimeCacheLease:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.directory = self.output / RECEIPT_DIR
        self.directory.mkdir(exist_ok=True)
        self.token = uuid.uuid4().hex
        self.path = self.directory / (self.token + '.json')
        self.mutex = threading.RLock()
        self.record = dict(schema_version=SCHEMA, token=self.token, output=str(self.output),
                           output_identity=_identity(self.output), owner_pid=os.getpid(),
                           owner_start_tick=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                           pid_namespace=os.readlink('/proc/self/ns/pid'),
                           created_at=time.time(), creator_source_sha256=_source_hash(), status='prepared', roots=[], retained_preexisting=[],
                           process=None, observed_processes={}, removed_files=0, removed_logical_bytes=0,
                           released_allocated_bytes=0, retained_shared_links=[], retained_other_filesystems=[],
                           retained_special_files=[])
        for relative in CACHE_DIRS:
            target = self.output / relative
            ancestors = [target, *target.parents]
            if any(p.is_symlink() for p in ancestors[:ancestors.index(self.output)]):
                self.record['retained_shared_links'].append(str(target))
                continue
            target.mkdir(parents=True, exist_ok=True)
            if not _only_empty_directories_or_links(target):
                self.record['retained_preexisting'].append(str(target))
                continue
            self.record['roots'].append(dict(relative=relative, identity=_identity(target), ancestors={str(p.relative_to(self.output)): _identity(p) for p in target.parents if p != self.output and p.is_relative_to(self.output)}))
        self._save()

    @classmethod
    def recover(cls, receipt):
        receipt = Path(receipt)
        record = json.loads(receipt.read_text())
        output = Path(record['output'])
        if (record.get('schema_version') != SCHEMA or not output.is_absolute()
                or receipt.is_symlink() or receipt.parent.is_symlink()
                or receipt != output / RECEIPT_DIR / (record['token'] + '.json')
                or record.get('creator_source_sha256') != _source_hash()
                or any(root.get('relative') not in CACHE_DIRS for root in record.get('roots', []))):
            raise ValueError('Unrecognized private cache ownership receipt')
        for root in record['roots']:
            expected = {str(p) for p in Path(root['relative']).parents if str(p) != '.'}
            if set(root.get('ancestors', {})) != expected:
                raise ValueError('Invalid cache ancestor binding')
        instance = cls.__new__(cls)
        instance.output, instance.directory, instance.path = output, receipt.parent, receipt
        instance.token, instance.record = record['token'], record
        instance.mutex = threading.RLock()
        return instance

    def _save(self):
        write_json(self.path, self.record)

    def environment(self, environment):
        env = dict(environment)
        env[MARKER_ENV] = self.token
        # Keep process-global NVIDIA caches private too, including scene workers.
        for key, folder in {'XDG_CACHE_HOME': 'cache', 'CUDA_CACHE_PATH': 'cuda',
                            '__GL_SHADER_DISK_CACHE_PATH': 'shader'}.items():
            env[key] = str(self.output / 'runtime' / folder)
        return env

    def bind(self, pid):
        rows, _ = _processes()
        row = rows.get(pid)
        self.record['process'] = row or dict(pid=pid, group=pid, session=pid, start_tick=None)
        self.record['status'] = 'running'
        self.observe(rows=rows)
        self._save()

    def observe(self, *, rows=None):
        with self.mutex:
            before = dict(self.record['observed_processes'])
            self._observe(rows=rows)
            if before != self.record['observed_processes']:
                self._save()

    def _observe(self, *, rows=None):
        rows = _processes()[0] if rows is None else rows
        leader = self.record['process']
        if not leader:
            return
        known = self.record['observed_processes']
        selected = {pid for pid, row in rows.items() if row['marker'] == self.token
                    or row['group'] == leader['group'] or row['session'] == leader['session']
                    or known.get(str(pid)) == row['start_tick']}
        while True:
            expanded = selected | {pid for pid, row in rows.items() if row['parent'] in selected}
            if expanded == selected:
                break
            selected = expanded
        for pid in selected:
            known[str(pid)] = rows[pid]['start_tick']

    def cleanup(self, *, returncode):
        with self.mutex, (self.directory / 'cleanup.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.record['status'] == 'cleaned':
                return True
            if self.record['pid_namespace'] != os.readlink('/proc/self/ns/pid'):
                return False
            rows, uncertain = _processes()
            self.observe(rows=rows)
            worker_start = (self.record.get('process') or {}).get('start_tick')
            preexisting = [pid for pid in uncertain if worker_start and pid in rows
                           and int(rows[pid]['start_tick']) < int(worker_start)]
            uncertain = [pid for pid in uncertain if pid not in preexisting]
            self.record['unrelated_preexisting_processes'] = preexisting
            live = [pid for pid, row in rows.items()
                    if self.record['observed_processes'].get(str(pid)) == row['start_tick']]
            self.record.update(returncode=returncode, live_processes=live, uncertain_processes=uncertain)
            if returncode is None or live or uncertain:
                self.record['status'] = 'waiting_for_complete_exit'
                self._save()
                return False
            if _identity(self.output) != self.record['output_identity']:
                self.record['status'] = 'retained_identity_changed'
                self._save()
                return True
            for other in self.directory.glob('*.json'):
                if other == self.path:
                    continue
                sibling = json.loads(other.read_text())
                if sibling.get('schema_version') == SCHEMA and any(
                        row['marker'] == sibling.get('token') for row in rows.values()):
                    self.record['status'] = 'waiting_for_other_owner'
                    self._save()
                    return False
            for root in self.record['roots']:
                target = self.output / root['relative']
                if not target.exists() and not target.is_symlink():
                    continue
                changed_ancestor = any((self.output / name).is_symlink() or not (self.output / name).exists()
                                       or _identity(self.output / name) != expected
                                       for name, expected in root['ancestors'].items())
                if changed_ancestor or target.is_symlink() or _identity(target) != root['identity']:
                    self.record.setdefault('retained_identity_changed', []).append(str(target))
                    continue
                _remove_private_tree(target, root['identity'][0], self.record)
            self.record.update(status='cleaned', completed_at=time.time())
            self._save()
            return True
