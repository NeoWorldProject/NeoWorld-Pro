"""Private CoACD subprocess; native failures never abort the object controller."""
from __future__ import annotations

import atexit
import hashlib
import json
from multiprocessing.connection import Connection
import os
import pickle
import signal
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import numpy as np

MAX_SYSTEM_RETRIES = 10


def resource_failure(message):
    return any(marker in str(message).lower() for marker in (
        'bad_alloc', 'out of memory', 'memoryerror', 'cannot allocate', 'failed to allocate',
        'no space left', 'disk quota', 'too many open files'))


def repeated_native_crash(diagnostics):
    attempts = diagnostics.get('attempts', [])
    signals = {-signal.SIGABRT, -signal.SIGSEGV, -signal.SIGBUS, -signal.SIGFPE}
    return (isinstance(attempts, list) and len(attempts) == MAX_SYSTEM_RETRIES + 1
        and len({a.get('returncode') for a in attempts}) == 1
        and all(a.get('native_started') is True and a.get('returncode') in signals
                and a.get('resource_failure') is False for a in attempts))


class CoacdSystemError(OSError):
    """Technical failure, deliberately distinct from invalid collision geometry."""
    def __init__(self, message, diagnostics=None):
        self.diagnostics = diagnostics or {}
        super().__init__(message)


def _command(fd):
    return [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(fd)]


class _Worker:
    def __init__(self):
        from urdf_core.execution import native_owner_environment
        self.owner = os.getpid()
        self.log = tempfile.TemporaryFile()
        parent, child = socket.socketpair()
        try:
            self.process = subprocess.Popen(_command(child.fileno()), pass_fds=(child.fileno(),),
                stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, **native_owner_environment()))
        except BaseException:
            parent.close()
            self.log.close()
            raise
        finally:
            child.close()
        self.connection = Connection(parent.detach())

    def call(self, request):
        self.native_started = False
        self.connection.send(request)
        reply = self.connection.recv()
        if isinstance(reply, dict) and reply.get('id') == request['id'] and reply.get('status') == 'native_started':
            self.native_started = True
            reply = self.connection.recv()
        if not isinstance(reply, dict) or reply.get('id') != request['id']:
            raise OSError('CoACD worker response does not match its request')
        if reply.get('status') not in {'ok', 'content_error', 'system_error'}:
            raise OSError('CoACD worker returned an invalid status')
        if reply['status'] == 'ok' and 'hulls' not in reply:
            raise OSError('CoACD worker omitted its result')
        return reply

    def close(self):
        self.connection.close()
        if os.getpid() == self.owner:
            try:
                # A crashed worker may close the pipe just before waitpid sees its exit.
                self.process.wait(timeout=.1)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.log.seek(0)
        text = self.log.read().decode(errors='replace')
        self.log.close()
        return self.process.returncode, text


_lock = threading.Lock()
_worker = None


def _after_fork():
    global _lock, _worker
    if _worker is not None:
        # Drop only the inherited descriptors; the original process owns its worker.
        _worker.connection.close()
        _worker.log.close()
    _worker = None
    _lock = threading.Lock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


def close_worker():
    global _worker
    with _lock:
        if _worker is not None:
            try:
                _worker.close()
            finally:
                _worker = None


atexit.register(close_worker)


def run_coacd(vertices, faces, **options):
    """Keep input/seed unchanged across one execution plus ten system retries."""
    global _worker
    request = {'id': uuid.uuid4().hex,
               'vertices': np.array(vertices, dtype=np.float64, copy=True),
               'faces': np.array(faces, dtype=np.int32, copy=True), 'options': dict(options)}
    attempts, evidence = [], None

    def record(error, returncode, log, started, native_started):
        nonlocal evidence
        if evidence is None:
            evidence = Path(tempfile.mkdtemp(prefix='neoworld_coacd_failure_'))
            np.savez(evidence / 'input.npz', vertices=request['vertices'], faces=request['faces'])
            (evidence / 'request.json').write_text(json.dumps({
                'id': request['id'], 'options': options,
                'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'input_sha256': hashlib.sha256((evidence / 'input.npz').read_bytes()).hexdigest()}, indent=2) + '\n')
        number = len(attempts) + 1
        (evidence / f'attempt_{number:02d}.log').write_text(log)
        attempts.append({'attempt': number, 'returncode': returncode, 'error': str(error),
                         'native_started': native_started,
                         'resource_failure': isinstance(error, MemoryError) or resource_failure(str(error) + '\n' + log),
                         'error_type': type(error).__name__, 'seconds': time.monotonic() - started})
        save('retrying' if number <= MAX_SYSTEM_RETRIES else 'system_retries_exhausted')

    def save(status):
        receipt = {'policy': 'coacd_subprocess.v1', 'request_id': request['id'], 'status': status,
                   'max_system_retries': MAX_SYSTEM_RETRIES, 'attempts': attempts,
                   'content_refinements_used': 0,
                   'component_fallback_eligible': repeated_native_crash({'attempts': attempts})}
        temporary = evidence / 'receipt.pending'
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(evidence / 'receipt.json')

    with _lock:
        for _ in range(MAX_SYSTEM_RETRIES + 1):
            started = time.monotonic()
            try:
                if _worker is None:
                    _worker = _Worker()
                reply = _worker.call(request)
                if reply['status'] == 'system_error':
                    raise OSError(f"{reply['error_type']}: {reply['error']}")
            except (OSError, EOFError, ValueError, MemoryError, pickle.UnpicklingError) as error:
                returncode, log = None, ''
                native_started = bool(_worker is not None and getattr(_worker, 'native_started', False))
                if _worker is not None:
                    returncode, log = _worker.close()
                    _worker = None
                record(error, returncode, log, started, native_started)
                print(f'CoACD system attempt {len(attempts)}/{MAX_SYSTEM_RETRIES + 1} failed; '
                      f'evidence: {evidence}', file=sys.stderr, flush=True)
                continue
            except BaseException:
                if _worker is not None:
                    _worker.close()
                    _worker = None
                raise
            if reply['status'] == 'content_error':
                if evidence is not None:
                    attempts.append({'attempt': len(attempts) + 1, 'status': 'content_error',
                                     'error_type': reply['error_type'], 'error': reply['error'],
                                     'seconds': time.monotonic() - started})
                    save('content_error_after_system_recovery')
                kind = TypeError if reply['error_type'] == 'TypeError' else ValueError
                raise kind(reply['error'])
            if evidence is not None:
                attempts.append({'attempt': len(attempts) + 1, 'returncode': 0,
                                 'seconds': time.monotonic() - started})
                save('recovered')
            return reply['hulls']
    raise CoacdSystemError(f'CoACD failed after 1 execution and {MAX_SYSTEM_RETRIES} system retries; '
                           f'evidence: {evidence}', {'evidence_dir': str(evidence), 'attempts': attempts,
                           'max_system_retries': MAX_SYSTEM_RETRIES, 'content_refinements_used': 0})


def _serve(fd, decompose=None):
    import ctypes
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if sys.platform == 'linux':
        ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # Suppress pipe-based crash collectors too.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from urdf_core.execution import start_owner_monitor
    start_owner_monitor()
    connection = Connection(fd)
    if decompose is None:
        import coacd
        coacd.set_log_level('error')
        def decompose(vertices, faces, options):
            return coacd.run_coacd(coacd.Mesh(vertices, faces), **options)
    while True:
        try:
            request = connection.recv()
        except EOFError:
            break
        try:
            connection.send({'id': request['id'], 'status': 'native_started'})
            hulls = decompose(request['vertices'], request['faces'], request['options'])
            reply = {'id': request['id'], 'status': 'ok', 'hulls': hulls}
        except (ValueError, TypeError) as error:
            reply = {'id': request['id'], 'status': 'content_error',
                     'error_type': type(error).__name__, 'error': str(error)}
        except Exception as error:
            reply = {'id': request['id'], 'status': 'system_error',
                     'error_type': type(error).__name__, 'error': str(error)}
        connection.send(reply)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--worker':
        raise SystemExit('Internal CoACD worker requires an inherited connection')
    _serve(int(sys.argv[2]))
