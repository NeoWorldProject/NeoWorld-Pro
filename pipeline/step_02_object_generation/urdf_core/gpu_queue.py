"""FIFO GPU admission with a task-local 5 GiB OOM retry staircase."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid

from .run_state import write_json

POLICY = 'free_memory_5gib_retry_steps.v2'
STEP_MIB = 5 * 1024
QUEUE_ROOT = Path('/tmp/neoworld_gpu_queue')
LEGACY_LOCK = Path('/tmp/neoworld_isaac_gpu.lock')
LAUNCH_GAP_S = 5.


class GpuResourceError(RuntimeError):
    pass


class GpuOutOfMemory(GpuResourceError):
    pass


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (OSError, IndexError):
        return None


def gpu_memory():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.total',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise GpuResourceError(f'GPU memory query unavailable: {error}') from error
    if result.returncode:
        raise GpuResourceError('GPU memory query failed: ' + result.stderr.strip())
    try:
        devices = [tuple(int(v.strip()) for v in line.split(','))
                   for line in result.stdout.splitlines() if line.strip()]
        if not devices or any(len(v) != 2 or not 0 <= v[0] <= v[1] for v in devices):
            raise ValueError('Invalid GPU memory sample')
    except ValueError as error:
        raise GpuResourceError('GPU memory query returned invalid data') from error
    # Admission covers the GPUs visible to the renderer, including RTX allocations.
    return {'free_mib': min(v[0] for v in devices), 'total_mib': min(v[1] for v in devices)}


def oom_evidence(*paths, text=''):
    messages = [text]
    for path in paths:
        try:
            with Path(path).open('rb') as stream:
                stream.seek(max(0, os.fstat(stream.fileno()).st_size - 2 * 1024 * 1024))
                messages.append(stream.read().decode(errors='replace'))
        except FileNotFoundError:
            pass
    pattern = r'cuda(?: error)?[^\n]{0,80}out of memory|cuda_error_out_of_memory|cudaerrormemoryallocation|(?:vk_)?error_out_of_device_memory|gpu[^\n]{0,80}out of memory|out of (?:gpu|device) memory'
    return bool(re.search(pattern, '\n'.join(messages), re.IGNORECASE))


class GpuTask:
    def __init__(self, path, *, identity, query=None, queue_root=None,
                 legacy_lock=None, now=None, sleep=None, launch_gap=LAUNCH_GAP_S,
                 cancelled=None, release_idle=None):
        self.path = Path(path)
        self.query = query or gpu_memory
        self.queue_root = Path(queue_root or os.environ.get('NEOWORLD_GPU_QUEUE_DIR', QUEUE_ROOT))
        self.legacy_lock = Path(legacy_lock or os.environ.get('NEOWORLD_GPU_LOCK', LEGACY_LOCK))
        self.now, self.sleep = now or time.time, sleep or time.sleep
        self.launch_gap = launch_gap
        self.cancelled = cancelled or (lambda: False)
        self.release_idle = release_idle
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.init.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                self.load()
            else:
                self.state = {'policy': POLICY, 'identity': self.identity, 'threshold_mib': STEP_MIB,
                              'oom_count': 0, 'technical_failures': 0, 'attempts': []}
                self.save()

    def load(self):
        self.state = json.loads(self.path.read_text())
        if self.state['policy'] != POLICY or self.state['identity'] != self.identity:
            raise GpuResourceError('GPU task identity changed; use a new task directory')

    def save(self):
        write_json(self.path, self.state)

    def bind_process(self, process):
        pid = process.pid
        identity = process_identity(pid)
        if identity is not None:
            try:
                if os.getpgid(pid) != pid:
                    raise GpuResourceError('GPU subprocess must own its isolated process group')
            except ProcessLookupError:
                return
            self.state['native_process'] = {'pid': pid, 'start_tick': identity}
            self.save()

    def stop_interrupted_process(self):
        owner = self.state.get('native_process')
        if not owner or process_identity(owner['pid']) != owner['start_tick']:
            return
        try:
            if os.getpgid(owner['pid']) != owner['pid']:
                raise GpuResourceError('Interrupted GPU process group binding changed')
            os.killpg(owner['pid'], signal.SIGKILL)
        except ProcessLookupError:
            return
        started = time.monotonic()
        while process_identity(owner['pid']) == owner['start_tick']:
            if time.monotonic() - started > 5:
                raise GpuResourceError('Interrupted GPU process did not stop')
            time.sleep(.05)

    def check_cancelled(self):
        if self.cancelled():
            raise GpuResourceError('GPU task cancelled')

    @contextmanager
    def registry(self):
        self.queue_root.mkdir(parents=True, exist_ok=True)
        path = self.queue_root / 'queue.json'
        with (self.queue_root / 'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = json.loads(path.read_text()) if path.exists() else {'waiting': [], 'next_start_epoch': 0.}
            value['waiting'] = [v for v in value['waiting']
                if process_identity(v['pid']) == v['start_tick']]
            try:
                yield value
            finally:
                write_json(path, value)

    @contextmanager
    def admission(self):
        ticket = {'id': uuid.uuid4().hex, 'pid': os.getpid(),
                  'start_tick': process_identity(os.getpid())}
        with self.registry() as queue:
            queue['waiting'].append(ticket)
        legacy = None
        released_idle = False
        try:
            while True:
                self.check_cancelled()
                memory_wait = True
                with self.registry() as queue:
                    if queue['waiting'] and queue['waiting'][0]['id'] == ticket['id']:
                        memory = self.query()
                        threshold = self.state['threshold_mib']
                        if threshold > memory['total_mib']:
                            raise GpuResourceError(f'GPU admission requires {threshold / 1024:g} GiB, exceeding device capacity')
                        memory_wait = memory['free_mib'] < threshold
                        if memory['free_mib'] >= threshold and self.now() >= queue['next_start_epoch']:
                            legacy = self.legacy_lock.open('a')
                            try:
                                fcntl.flock(legacy, fcntl.LOCK_SH | fcntl.LOCK_NB)
                            except BlockingIOError:
                                legacy.close(); legacy = None
                            else:
                                queue['waiting'].pop(0)
                                queue['next_start_epoch'] = self.now() + self.launch_gap
                if legacy is not None:
                    break
                if memory_wait and self.release_idle is not None and not released_idle:
                    self.release_idle()
                    released_idle = True
                self.sleep(.5)
            self.state['status'] = 'running'
            self.state['attempts'].append({'started_epoch': self.now(),
                'threshold_mib': self.state['threshold_mib'], 'free_mib': memory['free_mib']})
            self.save()
            yield legacy.fileno()
        finally:
            if legacy is not None:
                legacy.close()
            with self.registry() as queue:
                queue['waiting'] = [v for v in queue['waiting'] if v['id'] != ticket['id']]

    def run(self, operation):
        with self.path.with_suffix('.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise GpuResourceError('This GPU task already has a live owner') from error
            self.load()
            try:
                return self._run(operation)
            except BaseException as error:
                status = ('resource_error' if isinstance(error, GpuResourceError) else
                          'operation_error' if isinstance(error, Exception) else 'interrupted')
                self.state.update(status=status, last_error=f'{type(error).__name__}: {error}')
                self.save()
                raise

    def _run(self, operation):
        from .execution import TechnicalExecutionError
        if self.state.get('pending_technical_attempt', False):
            self.stop_interrupted_process()
            self.state['pending_technical_attempt'] = False
            self.state['technical_failures'] = self.state.get('technical_failures', 0) + 1
            self.save()
        if self.state.get('technical_failures', 0) >= 5:
            raise GpuResourceError('Technical execution attempts exhausted; resume does not reset the limit')
        while True:
            try:
                with self.admission() as fd:
                    self.state['pending_technical_attempt'] = True
                    self.save()
                    result = operation(fd)
                    self.state['pending_technical_attempt'] = False
                self.state['status'] = 'completed'
                self.save()
                return result
            except GpuOutOfMemory as error:
                self.state['pending_technical_attempt'] = False
                self.state['oom_count'] += 1
                self.state['technical_failures'] = self.state.get('technical_failures', 0) + 1
                self.state['threshold_mib'] += STEP_MIB
                self.state.update(status='waiting_after_oom', last_error=str(error))
                self.save()
                if self.state['technical_failures'] >= 5:
                    raise GpuResourceError('Five technical execution attempts exhausted after GPU OOM') from error
                self.sleep(self.launch_gap)
            except TechnicalExecutionError as error:
                self.state['pending_technical_attempt'] = False
                self.state['technical_failures'] = self.state.get('technical_failures', 0) + 1
                self.state.update(status='waiting_after_technical_failure', last_error=str(error))
                self.save()
                if self.state['technical_failures'] >= 5:
                    raise GpuResourceError('Technical execution attempts exhausted: ' + str(error)) from error
                self.check_cancelled()
                self.sleep(self.launch_gap)


def private_runtime_cache(out_dir, isaac_python, *, environ=None):
    from .isaac_cache import CACHE_ENV, prepare_runtime_cache
    environment = dict(os.environ if environ is None else environ)
    environment.pop(CACHE_ENV, None)
    return prepare_runtime_cache(out_dir, isaac_python, gpu_lock_fd=None, environ=environment)


def archive_attempt(output, attempt, names=('report.json', 'isaac_stdout.log', 'progress.json'), *, prefix='gpu_attempts'):
    import shutil
    target = Path(output) / prefix / f'{attempt:03d}'
    for name in names:
        path = Path(output) / name
        if path.is_file():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / name)


def record_resource_error(output, error):
    path = Path(output) / 'report.json'
    report = json.loads(path.read_text()) if path.is_file() else {}
    report.update(status='environment_error', passed=False, evaluation_complete=False,
                  acceptance_complete=False, resource_error=str(error))
    report.setdefault('errors', []).append(str(error))
    write_json(path, report)
