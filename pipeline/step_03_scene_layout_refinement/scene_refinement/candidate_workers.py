"""Resident native workers with host admission and exclusive per-worker requests."""
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import threading
import time

from .worker_client import SceneWorker, atomic_json


class CandidateCancelled(Exception):
    """Undispatched search work, never a physical failure."""


def worker_limit(config):
    requested = config.get('candidate_workers', 0)
    population = config.get('population_size', 16)
    return min(population, requested or population)


class CandidateWorkers:
    def __init__(self, config, work_dir, isaac_python):
        self.config, self.work_dir, self.launcher = config, Path(work_dir), isaac_python
        self.max_concurrency = worker_limit(config)
        self.condition = threading.Condition()
        self.workers, self.busy = [], set()
        self.paused = False
        self.closed = False
        self.lock_context = None
        self.last_probe = 0.
        self.peak_active = 0
        self.completed = 0

    @property
    def dispatch_capacity(self):
        # Keep one request available to grow the pool, without freezing every
        # candidate's pruning proof before the first result has arrived.
        return min(self.max_concurrency, max(1, len(self.workers)) + 1)

    def __enter__(self):
        from urdf_core.isaac_cache import GPU_LOCK, gpu_worker_slot
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = (self.work_dir/'batch_gpu.lock' if
            os.environ.get('NEOWORLD_SCENE_ALLOW_EXTERNAL_CONTENTION') == '1' else GPU_LOCK)
        self.lock_context = gpu_worker_slot(_lock_path=self.lock_path)
        self.lock_fd = self.lock_context.__enter__()
        return self

    def _can_expand(self, candidate):
        # Probe without joining the admission queue: waiting for another slot
        # must not evict this batch's resident workers whenever they become idle.
        if time.monotonic()-self.last_probe < 2:
            return False
        self.last_probe = time.monotonic()
        from urdf_core.isaac_admission import Registry, evaluate, resource_key, resource_estimates, MIB
        from urdf_core.gpu_queue import gpu_memory
        registry = Registry()
        with registry.locked() as state:
            registry.refresh(state)
            history = state['history'].get(resource_key(candidate['usd_path'], Path(self.launcher).resolve()))
            estimates = resource_estimates(history)
            reasons = evaluate(state['samples'], list(state['leases'].values()),
                gpu_memory()['free_mib'], estimates, shutil.disk_usage(self.work_dir).free/MIB)
            if state['waiting']:
                reasons.append('other_admission_waiters')
        atomic_json(self.work_dir/'capacity.json', {'time_unix':time.time(), 'reasons':reasons,
            'resident_slots':len(self.workers), 'active_slots':len(self.busy),
            'worker_ceiling':self.max_concurrency, 'expected_ram_mib':estimates[0],
            'expected_gpu_mib':estimates[1], 'peak_active_slots':self.peak_active})
        return not reasons

    @contextmanager
    def _acquire(self, candidate, mode, cancelled):
        with self.condition:
            while True:
                if self.closed or cancelled():
                    raise CandidateCancelled('Search request cancelled before native dispatch')
                if mode != 'physics' or not self.paused:
                    index = next((i for i in range(len(self.workers)) if i not in self.busy), None)
                    if index is None and len(self.workers) < self.max_concurrency and (
                            not self.workers or self._can_expand(candidate)):
                        index = len(self.workers)
                        worker = SceneWorker(self.config, self.work_dir/f'worker_{index}', self.launcher,
                            gpu_lock_fd=self.lock_fd, gpu_lock_path=self.lock_path,
                            cache_root=self.work_dir/f'private_cache_{index}')
                        self.workers.append(worker)
                    if index is not None:
                        self.busy.add(index)
                        self.peak_active = max(self.peak_active, len(self.busy))
                        worker = self.workers[index]
                        worker.cancelled = cancelled
                        break
                self.condition.wait(.1)
        try:
            yield worker, index
        finally:
            with self.condition:
                self.busy.remove(index)
                self.completed += 1
                atomic_json(self.work_dir/'parallel_summary.json', {
                    'worker_ceiling':self.max_concurrency, 'created_slots':len(self.workers),
                    'peak_active_slots':self.peak_active, 'completed_requests':self.completed})
                self.condition.notify_all()

    def evaluate_cancellable(self, candidate, duration_s, out_dir=None, output_dir=None,
                             mode='full', cancelled=lambda: False):
        started = time.monotonic()
        with self._acquire(candidate, mode, cancelled) as (worker, index):
            queued = time.monotonic()-started
            try:
                report = worker.evaluate(candidate, duration_s, out_dir=out_dir,
                                         output_dir=output_dir, mode=mode)
            except RuntimeError as exc:
                if cancelled() and str(exc) == 'Isaac resource wait cancelled':
                    raise CandidateCancelled(str(exc)) from exc
                raise
            report.setdefault('timings', {})['candidate_queue_seconds'] = queued
            report['candidate_worker'] = {'index':index, 'session':worker.session,
                'evidence_session':report.get('session_id'),
                'native_request_executed':report.get('cache_hit') is not True}
            if out_dir is not None or output_dir is not None:
                atomic_json(Path(out_dir or output_dir)/'physics_report.json', report)
            return report

    def evaluate(self, *args, **kwargs):
        return self.evaluate_cancellable(*args, **kwargs)

    def render(self, candidate, out_dir):
        return self.evaluate(candidate, 0., out_dir=out_dir, mode='render')

    def pause_search(self):
        with self.condition:
            self.paused = True

    def resume_search(self):
        with self.condition:
            self.paused = False
            self.condition.notify_all()

    def __exit__(self, *args):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        try:
            for worker in self.workers:
                worker.close()
        finally:
            if self.lock_context:
                self.lock_context.__exit__(*args)
