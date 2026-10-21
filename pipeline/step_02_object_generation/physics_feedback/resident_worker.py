"""Owned Isaac sessions with GPU admission and a fresh stage for each job."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import copy
import threading
import time
from types import SimpleNamespace
import uuid

from urdf_core.run_state import write_json, file_hash

FIELDS = ('asset_dir', 'spec', 'profile', 'contract', 'out_dir', 'isaac_python',
          'ffmpeg', 'trial_order', 'representative_motion', 'capture_mode', 'incidental_motion', 'measurement_mode', 'trial_cache_root', 'trial_selection', 'evaluation_policy')
PATH_FIELDS = {'asset_dir', 'spec', 'profile', 'contract', 'out_dir', 'isaac_python',
               'trial_order', 'representative_motion', 'incidental_motion', 'trial_cache_root', 'trial_selection'}
POLICY = 'object_owned_resident_isaac.complete_stage_reset.v1'


def process_identity(pid):
    try:
        # Parent launcher may be a shell. Bind owner PID + kernel start tick,
        # not getppid(), and never confuse PID reuse with a living owner.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except OSError:
        return None


class ResidentPhysicsSession:
    def __init__(self, root, *, review_handler=None, gpu_lease_fd=None):
        self.root = Path(root).resolve() / uuid.uuid4().hex
        self.process = None
        self.log = None
        self.count = 0
        self.review_handler = review_handler
        self.gpu_lease_fd = gpu_lease_fd
        self.queue_lease_fd = None
        self.cancelled = threading.Event()

    def __enter__(self):
        return self

    def _start(self, args):
        from .verify_physics_isaac import HERE
        self.root.mkdir(parents=True)
        write_json(self.root / 'owner.json', {'pid': os.getpid(), 'start_tick': process_identity(os.getpid()),
                   'policy': POLICY, 'gpu_lease_fd': self.gpu_lease_fd,
                   'gpu_queue': self.queue_lease_fd is not None})
        command = [str(args.isaac_python), str(HERE / 'verify_physics_isaac.py'),
                   '--resident-server', str(self.root)]
        for name in ('asset_dir', 'spec', 'profile', 'contract', 'out_dir'):
            command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
        command.extend(['--isaac-python', str(args.isaac_python)])
        env = os.environ.copy()
        for key, folder in {'XDG_CACHE_HOME': 'cache', 'XDG_CONFIG_HOME': 'config', 'XDG_DATA_HOME': 'data',
                            'CUDA_CACHE_PATH': 'cuda', '__GL_SHADER_DISK_CACHE_PATH': 'shader', 'TMPDIR': 'tmp'}.items():
            path = self.root / 'runtime' / folder; path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
        env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        from urdf_core.execution import native_owner_environment
        env.update(native_owner_environment())
        self.log = (self.root / 'resident_stdout.log').open('w')
        inherited = self.gpu_lease_fd if self.gpu_lease_fd is not None else self.queue_lease_fd
        from urdf_core.isaac_admission import IsaacProcess, resource_key
        self.process = IsaacProcess(command, resource_key=resource_key(Path(args.asset_dir) / 'object.urdf', args.isaac_python),
                                        resource_output=self.root, cancelled=self.cancelled.is_set,
                                        cwd=self.root, env=env, stdout=self.log,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=() if inherited is None else (inherited,))

    def run(self, args):
        if self.gpu_lease_fd is not None:
            return self._run_once(args)
        from urdf_core.gpu_queue import (GpuTask, GpuOutOfMemory, GpuResourceError,
                                        oom_evidence, archive_attempt, record_resource_error)
        from .verify_physics_isaac import initial_report
        identity = initial_report(args, 'gpu_admission')
        task = GpuTask(Path(args.out_dir) / 'gpu_admission.json',
            identity={'inputs': identity['inputs'], 'sources': identity['source_hashes'],
                      'partition': str(getattr(args, 'trial_selection', None))},
            cancelled=self.cancelled.is_set, release_idle=self._release_idle)

        def execute(fd):
            child = copy.copy(args)
            child._gpu_task = task
            evidence_paths = [Path(args.out_dir) / 'isaac_stdout.log', Path(args.out_dir) / 'report.json']
            def version(path):
                try:
                    stat = path.stat()
                    return (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                except FileNotFoundError:
                    return None
            before = {path: version(path) for path in evidence_paths}
            try:
                return self._run_once(child, queue_fd=fd)
            except Exception as error:
                from urdf_core.execution import TechnicalExecutionError
                changed = [path for path in evidence_paths if version(path) != before[path]]
                oom = oom_evidence(*changed, self.root / 'resident_stdout.log', text=str(error))
                if oom or isinstance(error, TechnicalExecutionError):
                    archive_attempt(args.out_dir, task.state['oom_count'] if oom else task.state.get('technical_failures', 0),
                                    prefix='gpu_attempts' if oom else 'technical_attempts')
                    self._release_idle()
                if oom:
                    raise GpuOutOfMemory('Isaac GPU OOM; restart and reuse completed trials') from error
                raise
        try:
            return task.run(execute)
        except GpuResourceError as error:
            record_resource_error(args.out_dir, error)
            raise

    def _release_idle(self):
        self.close(force=True, cancel=False)
        self.root = self.root.parent / uuid.uuid4().hex
        self.process = None; self.log = None; self.count = 0

    def _run_once(self, args, *, queue_fd=None):
        from .verify_physics_isaac import initial_report, source_hashes, worker_workload
        if self.process is not None and hasattr(self.process, 'set_idle') and not self.process.set_idle(False):
            self._release_idle()
        if self.process is not None and self.process.poll() is not None:
            from urdf_core.execution import TechnicalExecutionError
            raise TechnicalExecutionError('Resident Isaac exited; restart the same job')
        if self.process is None:
            if self.queue_lease_fd is None and queue_fd is not None:
                self.queue_lease_fd = os.dup(queue_fd)
            self._start(args)
        if getattr(args, '_gpu_task', None) is not None:
            args._gpu_task.bind_process(self.process)
        self.count += 1
        job_id = uuid.uuid4().hex
        output = Path(args.out_dir); output.mkdir(parents=True, exist_ok=True)
        report = initial_report(args, job_id)
        if any(not item['sha256'] for item in report['inputs'].values()):
            raise ValueError('Resident job has missing inputs')
        values = {n: str(getattr(args, n)) if n in PATH_FIELDS and getattr(args, n, None) is not None
                  else getattr(args, n, None) for n in FIELDS}
        workload = worker_workload(json.loads(Path(args.contract).read_text()))
        request = {'policy': POLICY, 'id': job_id, 'args': values, 'inputs': report['inputs'],
                   'source_hashes': source_hashes(), 'workload': workload}
        request_path = self.root / f'{self.count:04d}_{job_id}.request.json'
        done = request_path.with_suffix('.done.json')
        write_json(request_path, request)
        started = time.monotonic()
        from urdf_core.execution import ProgressWatchdog, TechnicalExecutionError, ExecutionCancelled
        from urdf_core.native_progress import read_execution_progress
        watchdog = ProgressWatchdog()
        try:
            while not done.is_file():
                if self.cancelled.is_set():
                    raise ExecutionCancelled('GPU task cancelled')
                if self.process.poll() is not None:
                    raise TechnicalExecutionError(f'Resident Isaac crashed; see {self.root / "resident_stdout.log"}')
                watchdog.check(read_execution_progress(output, job_id))
                for review in sorted(output.glob('incidental_motion/*/request.json')):
                    if not (review.parent / 'decision.json').exists():
                        if self.review_handler is None:
                            raise RuntimeError('Incidental-motion review requires its API owner')
                        with watchdog.suspended():
                            self.review_handler(review)
                time.sleep(.2)
            response = json.loads(done.read_text())
            path = output / 'report.json'
            if (response.get('id') != job_id or response.get('returncode') != 0
                    or response.get('report_sha256') != file_hash(path)):
                raise TechnicalExecutionError(f'Resident Isaac job did not complete; see {path}')
            saved = json.loads(path.read_text())
            if (saved.get('status') == 'asset_rejected' and saved.get('asset_error')
                    and saved.get('run_id') == job_id and saved.get('source_hashes') == request['source_hashes']
                    and saved.get('inputs') == request['inputs']):
                from urdf_core.content_errors import AssetContentError
                raise AssetContentError(**saved['asset_error'])
            if (saved.get('run_id') != job_id or saved.get('source_hashes') != request['source_hashes']
                    or saved.get('inputs') != request['inputs']
                    or saved.get('status') not in (('trial_fragment',) if getattr(args,'trial_selection',None)
                                                  else ('passed', 'physics_failed', 'grouping_ready'))):
                raise RuntimeError('Resident report has stale or inconsistent job bindings')
            saved.update(worker_returncode=0, elapsed_seconds=round(time.monotonic()-started, 3),
                         worker_workload=workload)
            write_json(path, saved)
            if hasattr(self.process, 'set_idle'):
                self.process.set_idle(True)
            return path
        except BaseException:
            self.close(force=True, cancel=False)
            raise

    def close(self, *, force=False, cancel=True):
        if cancel:
            self.cancelled.set()
        if self.process is not None and self.process.poll() is None:
            if not force:
                write_json(self.root / 'shutdown.json', {'owner': os.getpid()})
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    force = True
            if force and self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self.log is not None:
            self.log.close()
        if self.queue_lease_fd is not None:
            os.close(self.queue_lease_fd)
            self.queue_lease_fd = None

    def __exit__(self, *exc):
        self.close(force=exc[0] is not None)


@contextmanager
def job_stdout(path):
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    old = (os.dup(1), os.dup(2))
    with Path(path).open('w') as log:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(old[0], 1); os.dup2(old[1], 2)
            os.close(old[0]); os.close(old[1])


@contextmanager
def gpu_lock(owner):
    """A pool inherits the actual locked file description; unrelated jobs wait."""
    if owner.get('gpu_queue'):
        if process_identity(owner['pid']) != owner['start_tick']:
            raise RuntimeError('GPU queue owner exited')
        yield None
        return
    fd=owner.get('gpu_lease_fd')
    if fd is not None:
        if process_identity(owner['pid'])!=owner['start_tick']:
            raise RuntimeError('Parallel GPU lease owner exited')
        from urdf_core.isaac_cache import gpu_worker_slot
        with gpu_worker_slot(fd):yield fd
    else:
        with open('/tmp/neoworld_isaac_gpu.lock','a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            yield lock.fileno()


def serve(bootstrap):
    from urdf_core.execution import start_owner_monitor
    start_owner_monitor()
    from .verify_physics_isaac import source_hashes
    from urdf_core.isaac_cache import prepare_runtime_cache
    root = bootstrap.resident_server.resolve()
    owner = json.loads((root / 'owner.json').read_text())
    frozen = source_hashes()
    app = None
    completed = set()
    count = 0
    try:
        while not (root / 'shutdown.json').exists():
            if process_identity(owner['pid']) != owner['start_tick']:
                break
            pending = sorted(p for p in root.glob('*.request.json') if p.name not in completed)
            if not pending:
                time.sleep(.2)
                continue
            request_path = pending[0]
            request = json.loads(request_path.read_text())
            if request.get('policy') != POLICY or set(request['args']) != set(FIELDS):
                raise ValueError('Invalid resident job schema')
            args = SimpleNamespace(**{n: Path(v).resolve() if n in PATH_FIELDS and v is not None else v
                                      for n, v in request['args'].items()})
            args.run_id = request['id']; args.expected_sources = json.dumps(frozen)
            with gpu_lock(owner) as lock_fd:
                write_json(request_path.with_suffix('.started.json'), {'id': args.run_id})
                if request['source_hashes'] != frozen or source_hashes() != frozen:
                    raise RuntimeError('Resident implementation changed; restart from frozen source')
                for value in request['inputs'].values():
                    if file_hash(value['path']) != value['sha256']:
                        raise RuntimeError('Queued resident job input changed')
                with job_stdout(args.out_dir / 'isaac_stdout.log'):
                    from .physics_runtime import create_application, run_worker
                    if app is None:
                        app_args = SimpleNamespace(out_dir=root)
                        if owner.get('gpu_queue'):
                            from urdf_core.gpu_queue import private_runtime_cache
                            cache_policy = private_runtime_cache(root, args.isaac_python)
                        else:
                            cache_policy = prepare_runtime_cache(root, args.isaac_python, gpu_lock_fd=lock_fd)
                        app = create_application(app_args)
                    count += 1
                    started = time.monotonic()
                    code = run_worker(args, shared_app=app)
                    # Stop/reset callbacks and stage were already cleared by Runtime.
                    # Close the USD context as well before releasing the GPU lock.
                    import omni.usd
                    omni.usd.get_context().close_stage()
                    report_path = args.out_dir / 'report.json'
                    report = json.loads(report_path.read_text())
                    report.update(resident_worker={'policy': POLICY, 'pid': os.getpid(), 'job_number': count,
                        'application_starts': 1, 'complete_stage_reset': True,
                        'lock_scope': ('shared_queue_session' if owner.get('gpu_queue') else
                                       'isolated_pool' if owner.get('gpu_lease_fd') is not None else 'each_job'),
                        'execution_wall_seconds': time.monotonic()-started}, runtime_cache_policy=cache_policy)
                    write_json(report_path, report)
                    write_json(request_path.with_suffix('.done.json'), {'id': args.run_id, 'returncode': code,
                                                                      'report_sha256': file_hash(report_path)})
            completed.add(request_path.name)
            if code != 0:
                break  # Do not reuse an application after an environment/cleanup error.
    finally:
        if app is not None:
            with gpu_lock(owner):
                app.close()
    return 0
