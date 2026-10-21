"""Batch-owned resident workers; independent controllers exchange bound requests.

The coordinator alone owns the global GPU lock. Children borrow its descriptor,
and each has a private writable shader cache. A client never owns/terminates a
native process. Final validation has priority over queued search work.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import uuid

from .metrics import physics_settings
from .worker_client import (SceneWorker, atomic_json, content_hash, source_hashes,
                            sha256, native_runtime_binding)

POOL_ENV = "NEOWORLD_SCENE_WORKER_POOL"
API_ENV = "NEOWORLD_SCENE_API_SLOTS"


def scene_worker(config, work_dir, isaac_python=None):
    if os.environ.get(POOL_ENV):
        return PoolClient(os.environ[POOL_ENV], config, work_dir, isaac_python)
    from . import worker_client
    return worker_client.SceneWorker(config=config, work_dir=work_dir, isaac_python=isaac_python)


def request_priority(request, final_duration):
    return (0 if request["mode"] == "full" and request["duration_s"] >= final_duration else
            1 if request["mode"] == "render" else 2)


@contextmanager
def api_slot():
    """Cross-controller cap; the slot covers the complete streaming request."""
    root = os.environ.get(API_ENV)
    if not root:
        yield
        return
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    handles = [(root / f"slot_{i}.lock").open("a") for i in range(4)]
    acquired = None
    try:
        while acquired is None:
            for handle in handles:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = handle
                    break
                except BlockingIOError:
                    pass
            if acquired is None:
                time.sleep(.05)
        yield
    finally:
        if acquired is not None:
            fcntl.flock(acquired, fcntl.LOCK_UN)
        for handle in handles:
            handle.close()


class PoolClient:
    def __init__(self, root, config, work_dir, isaac_python=None):
        self.root, self.work_dir = Path(root).resolve(), Path(work_dir).resolve()
        self.config = physics_settings(config)
        self.session = uuid.uuid4().hex
        self.launcher = Path(isaac_python).resolve()
        self.sources = source_hashes()
        self.runtime = native_runtime_binding(self.launcher)
        self.owner = json.loads((self.root / "pool.json").read_text())
        if (self.owner["config"] != self.config or self.owner["source_hashes"] != self.sources
                or self.owner["runtime_binding"] != self.runtime):
            raise RuntimeError("Worker pool source/runtime/configuration mismatch")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def render(self, candidate, out_dir):
        return self.evaluate(candidate, 0., out_dir=out_dir, mode="render")

    def evaluate(self, candidate, duration_s, out_dir=None, output_dir=None, mode="full"):
        if mode not in {"physics", "render", "full", "terminal_settle"}:
            raise ValueError("Unknown worker mode")
        request_id = uuid.uuid4().hex
        output = Path(out_dir or output_dir).resolve()
        inputs = {key: sha256(candidate[key]) for key in ("usd_path", "scene_manifest_path", "layout_path")
                  if candidate.get(key)}
        request = {"pool_id": self.owner["pool_id"], "request_id": request_id,
                   "session_id": self.session, "run_id": str(self.work_dir),
                   "candidate_id": candidate["candidate_id"], "candidate": candidate,
                   "mode": mode, "duration_s": float(duration_s), "output": str(output),
                   "input_hashes": inputs, "submitted_unix": time.time(),
                   "source_hashes": self.sources, "runtime_binding": self.runtime,
                   "config": self.config}
        request["digest"] = content_hash(request)
        atomic_json(self.root / "requests" / (request_id + ".json"), request)
        result_path = self.root / "responses" / (request_id + ".json")
        while not result_path.exists():
            if (self.root / "closed.json").exists():
                raise RuntimeError("Worker pool closed before completing request")
            try:
                os.kill(self.owner["pid"], 0)
            except ProcessLookupError as exc:
                raise RuntimeError("Worker pool coordinator exited") from exc
            time.sleep(.05)
        result = json.loads(result_path.read_text())
        for key in ("pool_id", "session_id", "run_id", "request_id", "candidate_id", "digest"):
            if result.get(key) != request[key]:
                raise RuntimeError("Mismatched worker pool response: " + key)
        if self.sources != source_hashes():
            raise RuntimeError("Controller sources changed during pool evaluation")
        report = result["report"]
        report["pool_evidence"] = {key: result[key] for key in result if key != "report"}
        report.setdefault("timings", {})["pool_queue_seconds"] = result["queue_seconds"]
        if report.get("status") != "environment_error":
            if report.get("source_hashes") != self.sources:
                raise RuntimeError("Native source binding differs from pool client")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "physics_report.json", report)
        return report


class WorkerPool:
    def __init__(self, root, config, workers=1, isaac_python=None, *, output_roots,
                 cache_seed=None, minimum_free_mib=2048, allow_external_contention=False):
        if workers not in (1, 2, 3):
            raise ValueError("workers must be 1, 2, or 3")
        self.root, self.config = Path(root).resolve(), config
        self.count, self.launcher = workers, Path(isaac_python).resolve()
        self.output_roots = [Path(p).resolve() for p in output_roots]
        self.cache_seed, self.minimum_free_mib = cache_seed, minimum_free_mib
        self.allow_external_contention = allow_external_contention
        self.workers, self.threads = [], []
        self.tasks = queue.PriorityQueue()
        self.stop = threading.Event()
        self.sources = source_hashes()
        self.pool_id = uuid.uuid4().hex
        self.lock = None
        self.failure_lock = threading.Lock()
        self.failures = []
        self.active_indices = set()

    def __enter__(self):
        from urdf_core.isaac_cache import gpu_worker_slot
        self.root.mkdir(parents=True, exist_ok=False)
        admission_path = self.root/'batch_gpu.lock' if self.allow_external_contention else Path('/tmp/neoworld_isaac_gpu.lock')
        self.lock = gpu_worker_slot(_lock_path=admission_path)
        waiting = time.monotonic()
        self.lock_fd = self.lock.__enter__()
        queue_seconds = time.monotonic()-waiting
        started = time.monotonic()
        try:
            self.owner = {"schema_version": "scene_worker_pool.v1", "pool_id": self.pool_id,
                "pid": os.getpid(), "config": physics_settings(self.config), "source_hashes": self.sources,
                "runtime_binding": native_runtime_binding(self.launcher), "requested_workers": self.count,
                "external_gpu_queue_seconds": queue_seconds,
                "allow_external_contention": self.allow_external_contention,
                "admission_lock_path": str(admission_path),
                "output_roots": list(map(str, self.output_roots)), "minimum_free_mib": self.minimum_free_mib}
            atomic_json(self.root / "pool.json", self.owner)
            for index in range(self.count):
                cache = self.root / "private_caches" / str(index)
                prepared = time.monotonic()
                if self.cache_seed:
                    shutil.copytree(self.cache_seed, cache, symlinks=False)
                else:
                    cache.mkdir(parents=True)
                worker = SceneWorker(self.config, self.root / f"worker_{index}", self.launcher,
                                     gpu_lock_fd=self.lock_fd, cache_root=cache,
                                     gpu_lock_path=admission_path)
                # Workers launch on their first request, so initialization admission
                # cannot block dispatch of the first worker's physics request.
                worker.minimum_gpu_headroom = self.minimum_free_mib
                self.workers.append(worker)
            self.owner.update(active_workers=len(self.workers), startup_seconds=time.monotonic()-started)
            atomic_json(self.root / "pool.json", self.owner)
            for index, worker in enumerate(self.workers):
                self.active_indices.add(index)
                thread = threading.Thread(target=self._consume, args=(index, worker), daemon=True)
                thread.start(); self.threads.append(thread)
            dispatcher = threading.Thread(target=self._dispatch, daemon=True)
            dispatcher.start(); self.threads.append(dispatcher)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _failure(self, failure):
        with self.failure_lock:
            self.failures.append(dict(failure, time_unix=time.time()))
            atomic_json(self.root / "infrastructure_failures.json", self.failures)

    def _dispatch(self):
        seen = set()
        while not self.stop.is_set():
            for path in sorted((self.root / "requests").glob("*.json")):
                if path.name in seen:
                    continue
                seen.add(path.name)
                request = json.loads(path.read_text())
                if not self.active_indices:
                    report = {"status": "environment_error", "passed": False, "complete": False,
                              "acceptance_complete": False, "errors": ["All native workers retired"]}
                    self._respond(request, report, -1, None, time.time()-request["submitted_unix"])
                    continue
                self.tasks.put((request_priority(request, self.config["final_duration_s"]),
                                request["submitted_unix"], path.name, request))
            self.stop.wait(.025)

    def _validate(self, request):
        unbound = {k: v for k, v in request.items() if k != "digest"}
        if (request["digest"] != content_hash(unbound) or request["pool_id"] != self.pool_id
                or request["source_hashes"] != self.sources or request["config"] != self.owner["config"]
                or request["runtime_binding"] != self.owner["runtime_binding"]):
            raise ValueError("Pool request binding mismatch")
        output = Path(request["output"]).resolve()
        if not any(output.is_relative_to(root) for root in self.output_roots):
            raise ValueError("Pool output is outside this batch")
        if request["candidate_id"] != request["candidate"]["candidate_id"]:
            raise ValueError("Pool candidate ID mismatch")
        for key, digest in request["input_hashes"].items():
            if sha256(request["candidate"][key]) != digest:
                raise ValueError("Candidate changed while queued")

    def _consume(self, index, worker):
        while not self.stop.is_set():
            try:
                _, _, _, request = self.tasks.get(timeout=.1)
            except queue.Empty:
                continue
            queued = max(0., time.time() - request["submitted_unix"])
            try:
                self._validate(request)
                report = worker.evaluate(request["candidate"], request["duration_s"],
                                         out_dir=request["output"], mode=request["mode"])
            except Exception as exc:
                report = {"status": "environment_error", "passed": False, "complete": False,
                          "acceptance_complete": False, "errors": [f"{type(exc).__name__}: {exc}"]}
            if report.get("status") == "environment_error":
                self._failure({"stage": request["mode"], "worker": index,
                               "request_id": request["request_id"], "errors": report.get("errors")})
                error_text = ' '.join(map(str, report.get('errors', []))).lower()
                if any(token in error_text for token in ('out of memory', 'out_of_memory',
                        'out_of_device_memory', 'cudaerroroutofmemory', 'device_lost')):
                    worker.failure = 'Retired after GPU/memory infrastructure failure: '+error_text[:500]
            self._respond(request, report, index, worker.session, queued)
            self.tasks.task_done()
            # A crashed worker retires. Other independent workers continue; an
            # infrastructure failure never becomes a physical training sample.
            if worker.failure:
                worker.close()
                with self.failure_lock:
                    self.active_indices.discard(index)
                    if not self.active_indices:
                        while not self.tasks.empty():
                            try:
                                _, _, _, pending = self.tasks.get_nowait()
                            except queue.Empty:
                                break
                            self._respond(pending, {"status": "environment_error", "passed": False,
                                "complete": False, "acceptance_complete": False,
                                "errors": ["All native workers retired"]}, -1, None,
                                time.time()-pending["submitted_unix"])
                            self.tasks.task_done()
                return

    def _respond(self, request, report, index, session, queued):
        result = {key: request[key] for key in ("pool_id", "session_id", "run_id", "request_id",
                                                 "candidate_id", "digest")}
        result.update(worker_index=index, worker_session=session, queue_seconds=queued, report=report)
        result.update(evidence_producer_session=report.get('session_id'),
                      native_request_executed=report.get('cache_hit') is not True)
        atomic_json(self.root / "responses" / (request["request_id"] + ".json"), result)

    def __exit__(self, *args):
        self.stop.set()
        atomic_json(self.root / "closed.json", {"pool_id": self.pool_id, "time_unix": time.time()})
        for thread in self.threads:
            thread.join(timeout=10)
        for worker in self.workers:
            worker.close()
        if self.lock is not None:
            self.lock.__exit__(None, None, None)
            self.lock = None
