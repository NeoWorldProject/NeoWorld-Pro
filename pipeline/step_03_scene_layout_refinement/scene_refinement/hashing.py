"""Content hashes bound to exact file versions, with a persistent shared index."""
import atexit
import hashlib
from functools import wraps
import json
import os
from pathlib import Path
import sqlite3
import time
import threading

_CACHE = {}
_PENDING = {}
_DB = None
_DB_NAME = None
_DB_PID = None
_MEMORY_LIMIT = 131072
_LOCK = threading.RLock()


def _locked(function):
    @wraps(function)
    def call(*args, **kwargs):
        with _LOCK:
            return function(*args, **kwargs)
    return call


def _version(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _checksum(path, version, value):
    return hashlib.sha256(json.dumps([path, *version, value], separators=(',', ':')).encode()).hexdigest()


def _connection():
    global _DB, _DB_NAME, _DB_PID
    name = os.environ.get('NEOWORLD_SCENE_HASH_CACHE')
    if not name:
        return None
    path = Path(name).resolve()
    if path.suffix == '.json':
        path = path.with_suffix('.sqlite3')
    if _DB is not None and (_DB_NAME != str(path) or _DB_PID != os.getpid()):
        if _DB_PID == os.getpid():
            flush_cache()
        _DB.close(); _DB = None; _PENDING.clear()
    if _DB is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _DB = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
        _DB.execute('PRAGMA journal_mode=WAL')
        _DB.execute('PRAGMA synchronous=FULL')
        version = _DB.execute('PRAGMA user_version').fetchone()[0]
        if version not in (0, 2):
            raise ValueError('Unsupported scene hash index version')
        _DB.execute('''CREATE TABLE IF NOT EXISTS file_versions (
            path TEXT PRIMARY KEY, device INTEGER, inode INTEGER, size INTEGER,
            mtime_ns INTEGER, ctime_ns INTEGER, digest TEXT, checksum TEXT)''')
        _DB.execute('PRAGMA user_version=2'); _DB.commit()
        _DB_NAME, _DB_PID = str(path), os.getpid()
    return _DB


def _insert(connection, rows):
    with connection:
        connection.executemany('''INSERT INTO file_versions VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(path) DO UPDATE SET device=excluded.device, inode=excluded.inode,
            size=excluded.size, mtime_ns=excluded.mtime_ns, ctime_ns=excluded.ctime_ns,
            digest=excluded.digest, checksum=excluded.checksum
            WHERE excluded.ctime_ns >= file_versions.ctime_ns''', rows)


@_locked
def flush_cache():
    if _DB is not None and _DB_PID == os.getpid() and _PENDING:
        rows = list(_PENDING.values())
        _insert(_DB, rows)
        _PENDING.clear()


atexit.register(flush_cache)



def _remember(key, value, connection, persist):
    if key not in _CACHE and len(_CACHE) >= _MEMORY_LIMIT:
        for oldest in list(_CACHE)[:len(_CACHE)-_MEMORY_LIMIT+1]:
            del _CACHE[oldest]
    _CACHE[key] = value
    if persist and connection is not None:
        path, version = key
        _PENDING[path] = [path, *version, value, _checksum(path, version, value)]
        if len(_PENDING) >= 256:
            flush_cache()


@_locked
def file_hash(path):
    connection = _connection()
    path = Path(path).resolve()
    before = _version(path.stat())
    key = str(path), before
    mature = time.time_ns() - before[-1] >= 2_000_000_000
    if mature:
        value = _CACHE.get(key)
        if value is None and connection is not None:
            row = connection.execute('SELECT device,inode,size,mtime_ns,ctime_ns,digest,checksum FROM file_versions WHERE path=?', (str(path),)).fetchone()
            if row and tuple(row[:5]) == before and row[6] == _checksum(str(path), before, row[5]):
                value = row[5]
        if value is not None and _version(path.stat()) == before:
            _remember(key, value, connection, False)
            return value
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        opened = _version(os.fstat(stream.fileno()))
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        if opened != _version(os.fstat(stream.fileno())) or opened != _version(path.stat()):
            raise ValueError('File changed while hashing: ' + str(path))
    value = digest.hexdigest()
    if time.time_ns() - opened[-1] >= 2_000_000_000:
        _remember((str(path), opened), value, connection, True)
    return value
