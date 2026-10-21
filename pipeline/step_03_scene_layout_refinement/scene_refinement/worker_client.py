"""Owned resident Isaac subprocess with hash-bound filesystem requests and receipts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from .metrics import physics_settings
from .physics import observation_ready
from .hashing import file_hash


HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def native_runtime_binding(launcher):
    """Bind cached evidence to the installed engine and driver, not just its path."""
    launcher = Path(launcher).resolve(); root = launcher.parent
    files = [launcher, root / "VERSION", root / "kit/kit", Path("/proc/driver/nvidia/version")]
    files.extend(root.glob("extscache/omni.physx-*/bin/libomni.physx.plugin.so"))
    files.extend(root.glob("extscache/omni.physics.physx-*/bin/libomni.physics.physx.plugin.so"))
    return {"launcher": str(launcher), "files": {str(p): file_hash(p) for p in files if p.is_file()}}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes():
    stage2 = HERE.parents[1] / "step_02_object_generation"
    paths = [*HERE.glob("*.py"),
        stage2 / "physics_feedback/physics_runtime.py",
        stage2 / "kinematics_validation/verify_urdf_isaac.py",
        stage2 / "kinematics_validation/isaac_motion.py",
        stage2 / "urdf_core/isaac_cache.py",
        stage2 / "urdf_core/isaac_admission.py",
        stage2 / "urdf_core/gpu_queue.py",
        stage2 / "urdf_core/environment.py",
        stage2 / "urdf_core/content_errors.py",
    ]
    return {str(path): sha256(path) for path in sorted(paths) if path.is_file()}


# This snapshot describes the modules this interpreter imported, not whatever
# source may be edited onto disk during a resident worker's lifetime.
IMPORTED_SOURCE_HASHES = source_hashes()


def assert_source_binding(frozen, requested=None):
    if (frozen != IMPORTED_SOURCE_HASHES or frozen != source_hashes()
            or requested is not None and requested != frozen):
        raise RuntimeError("Source changed since interpreter startup; restart with unchanged sources")


def valid_asset_rejection(report):
    error = report.get('asset_error')
    return (report.get('status') == 'asset_rejected' and report.get('passed') is False
            and report.get('complete') is False and report.get('acceptance_complete') is False
            and report.get('violation') is None and report.get('physical_residual') is None
            and isinstance(error, dict) and isinstance(error.get('reason'), str) and bool(error['reason'])
            and error.get('stage') in {'collision_cook', 'scene_asset_topology'}
            and report.get('asset_scope') == ('asset_topology' if error['stage'] == 'scene_asset_topology' else 'candidate_geometry'))


def valid_terminal_evidence(report):
    from .terminal_settle import POLICY as TERMINAL_POLICY
    try:
        if (report.get('policy') != TERMINAL_POLICY or report.get('passed') is not False
                or report.get('acceptance_complete') is not False or report.get('native_observation_complete') is not True
                or report.get('native_dynamics') is not True or report.get('complete') is not True
                or report.get('status') not in ('settled_window_passed', 'terminal_not_settled')
                or bool(report.get('window_passed')) != (report['status'] == 'settled_window_passed')
                or report.get('errors') or report.get('window_duration_s', 0) < 1.-1e-8
                or not 1. <= report['duration_s'] <= 10. + 1e-8):
            return False
        references = [report[k] for k in ('trace', 'terminal_state', 'initial_state_receipt', 'cooked_geometry')]
        candidate = report.get('terminal_candidate')
        if report.get('window_passed'):
            if not candidate or not candidate.get('settled_artifacts'): return False
            references += [{'path': p, 'sha256': h} for p, h in candidate['settled_artifacts'].items()]
        references += report.get('images', [])
        return all(Path(r['path']).is_file() and sha256(r['path']) == r['sha256'] for r in references)
    except (OSError, ValueError, TypeError, KeyError):
        return False


def valid_cached_evidence(report, duration_s, config, require_images=True):
    """Require the complete initial-state and final-video evidence on reuse."""
    try:
        images = report.get("images", [])
        if require_images and len(images) != 2:
            return False
        artifacts = [*images, report["trace"], report["initial_state_receipt"],
                     report["terminal_state"], report["cooked_geometry"]]
        settled = report.get("settled_candidate")
        if settled:
            hashes = settled.get("settled_artifacts", {})
            if not hashes:
                return False
            artifacts.extend({"path": path, "sha256": digest} for path, digest in hashes.items())
        if duration_s >= config["final_duration_s"] - config["physics_dt_s"]:
            video = report["video"]
            artifacts.extend([{"path": video["video_path"], "sha256": video["sha256"]},
                              {"path": video["frame_list_path"], "sha256": video["frame_list_sha256"]}])
            listing = json.loads(Path(video["frame_list_path"]).read_text())
            frames, hashes = listing["frames"], listing["frame_sha256"]
            if (len(frames) != video["frame_count"] or len(hashes) != len(frames) or not frames
                    or listing["fps"] != video["fps"]):
                return False
            artifacts.extend({"path": path, "sha256": digest} for path, digest in zip(frames, hashes))
        return all(Path(item["path"]).is_file() and sha256(item["path"]) == item["sha256"] for item in artifacts)
    except (OSError, ValueError, TypeError, KeyError):
        return False


class SceneWorker:
    """One worker per caller, serialized with the existing stage-2 GPU lock.

    Only this client's process group may be terminated. Reuse accepts complete
    receipts with exact inputs, sources, duration and image/trace hashes.
    """
    def __init__(self, config, work_dir, isaac_python=None, *, gpu_lock_fd=None, cache_root=None,
                 gpu_lock_path=None):
        self.config = physics_settings(config)
        self.work_dir = Path(work_dir).resolve()
        if not isaac_python:
            raise ValueError("Runtime must explicitly provide isaac_python")
        self.launcher = Path(isaac_python).resolve()
        self.session = uuid.uuid4().hex
        self.queue = self.work_dir / "workers" / self.session
        self.process = self.log = None
        self.failure = None
        self.failure_retryable = False
        self.frozen_sources = dict(IMPORTED_SOURCE_HASHES)
        self.runtime_binding = native_runtime_binding(self.launcher)
        self.gpu_lock_fd, self.cache_root = gpu_lock_fd, cache_root
        self.gpu_lock_path = gpu_lock_path
        self.coordination_lock = None

    def __enter__(self):
        # Lazy launch allows fully validated receipts to be replayed without GPU.
        return self

    def _start(self):
        if self.process is not None and hasattr(self.process, 'set_idle') and not self.process.set_idle(False):
            self.close()
            self.session = uuid.uuid4().hex
            self.queue = self.work_dir / 'workers' / self.session
            self.process = self.log = None
        if self.process is not None or self.failure:
            return
        try:
            assert_source_binding(self.frozen_sources)
        except RuntimeError as exc:
            self.failure = str(exc)
            return
        self.queue.mkdir(parents=True, exist_ok=True)
        from .process_identity import process_identity
        startup = {"session_id": self.session, "config": self.config,
                   "coordinator_identity": process_identity(),
                   "runtime_binding": self.runtime_binding,
                   "source_hashes": self.frozen_sources, "launcher": str(self.launcher)}
        atomic_json(self.queue / "startup.json", startup)
        if not self.launcher.is_file():
            self.failure = "Isaac launcher does not exist: " + str(self.launcher)
            return
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(HERE.parent) + os.pathsep + str(HERE.parents[1] / "step_02_object_generation")
        if self.cache_root is not None:
            env["NEOWORLD_ISAAC_CACHE_ROOT"] = str(self.cache_root)
        self.log = (self.queue / "worker.log").open("w", encoding="utf-8")
        command = [str(self.launcher), str(HERE / "isaac_worker.py"), "--queue", str(self.queue)]
        try:
            from urdf_core.isaac_cache import GPU_LOCK, gpu_worker_slot
            lock_fd, lock_path = self.gpu_lock_fd, self.gpu_lock_path or GPU_LOCK
            if lock_fd is None:
                if env.get('NEOWORLD_SCENE_ALLOW_EXTERNAL_CONTENTION') == '1':
                    if not env.get('NEOWORLD_ISAAC_CACHE_ROOT'):
                        raise ValueError('Standalone overlap requires an isolated writable cache root')
                    lock_path = self.queue / 'private_gpu.lock'
                elif lock_path != GPU_LOCK:
                    raise ValueError('A private admission lease must be inherited from its coordinator')
                atomic_json(self.queue / 'progress.json', {'stage': 'waiting_gpu_lock'})
                context = gpu_worker_slot(_lock_path=lock_path)
                lock_fd = context.__enter__()
                self.coordination_lock = context
            command += ['--gpu-lock-fd', str(lock_fd), '--gpu-lock-path', str(lock_path)]
            from urdf_core.isaac_admission import IsaacProcess, resource_key
            self.process = IsaacProcess(command, resource_key=resource_key(getattr(self, "resource_asset", self.queue / "startup.json"), self.launcher), resource_output=self.queue, gpu_headroom=getattr(self, "minimum_gpu_headroom", 2048), stdout=self.log, stderr=subprocess.STDOUT,
                                            env=env, start_new_session=True, pass_fds=(lock_fd,))
        except OSError as exc:
            self.failure = f"Isaac launch failed: {exc}"
            self.close()
            return
        except BaseException:
            self.close()
            raise
        started = time.monotonic()
        while not (self.queue / "ready.json").exists():
            if self.process.poll() is not None:
                self.failure_retryable = True
                self.failure = f"Isaac exited during startup with code {self.process.returncode}"
                self.close()
                return
            if time.monotonic() - started > self.config["worker_timeout_s"]:
                self.failure_retryable = True
                self.failure = "Isaac startup timed out after resource admission"
                self.close()
                return
            time.sleep(.1)
        ready = json.loads((self.queue / "ready.json").read_text())
        if ready.get("session_id") != self.session or ready.get("status") != "ready":
            self.failure = ready.get("error", "Invalid worker startup receipt")
            self.close()

    def evaluate(self, candidate, duration_s, out_dir=None, output_dir=None, mode="full"):
        """Retry the same native operation at most five times, including across resume."""
        output = Path(out_dir or output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        binding = content_hash({'candidate': candidate, 'duration_s': duration_s, 'mode': mode,
            'config': self.config, 'sources': self.frozen_sources, 'runtime': self.runtime_binding,
            'inputs': {key: sha256(candidate[key]) for key in ('usd_path', 'scene_manifest_path', 'layout_path') if candidate.get(key)}})
        ledger_path = output / 'technical_attempts.json'
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'binding': binding, 'attempts': []}
        if ledger['binding'] != binding:
            raise RuntimeError('Native technical retry inputs changed')
        if ledger.get('status') in {'completed', 'failed'}:
            report_path = output / 'native_operation_result.json'
            if sha256(report_path) != ledger['report_sha256']:
                raise RuntimeError('Native retry receipt changed')
            report = json.loads(report_path.read_text())
            if ledger['status'] == 'completed':
                if report.get('status') == 'asset_rejected':
                    valid = valid_asset_rejection(report)
                elif mode == 'render':
                    images = report.get('images', [])
                    valid = len(images) == 2 and all(sha256(item['path']) == item['sha256'] for item in images)
                elif mode == 'terminal_settle':
                    valid = valid_terminal_evidence(report)
                elif report.get('status') in {'certified_pruned', 'invalid_candidate'}:
                    valid = True  # The signed failure receipt never becomes acceptance.
                else:
                    valid = valid_cached_evidence(report, duration_s, self.config, require_images=mode != 'physics')
                if not valid:
                    raise RuntimeError('Completed native evidence changed')
            return dict(report, cache_hit=True)
        if len(ledger['attempts']) >= 5:
            raise RuntimeError('Five native technical attempts exhausted; resume cannot reset the limit')
        while len(ledger['attempts']) < 5:
            if ledger['attempts']:
                self.close()
                self.session = uuid.uuid4().hex
                self.queue = self.work_dir / 'workers' / self.session
                self.process = self.log = None
                self.failure = None
                self.failure_retryable = False
            attempt = {'attempt': len(ledger['attempts']) + 1, 'status': 'pending', 'session_id': self.session}
            ledger['attempts'].append(attempt)
            atomic_json(ledger_path, ledger)
            report = self._evaluate_once(candidate, duration_s, out_dir=output, mode=mode)
            failed = report.get('status') in {'environment_error', 'worker_error', 'timeout', 'infrastructure_error'}
            attempt.update(status='failed' if failed else 'completed', errors=report.get('errors', []))
            atomic_json(output / 'technical_attempts' / f"attempt_{attempt['attempt']:03d}.json", report)
            retryable = bool(report.get('retryable_technical', self.failure_retryable))
            if not failed or not retryable or len(ledger['attempts']) == 5:
                report['technical_attempts'] = len(ledger['attempts'])
                report['technical_attempt_limit'] = 5
                atomic_json(output / 'physics_report.json', report)
                atomic_json(output / 'native_operation_result.json', report)
                ledger.update(status='failed' if failed else 'completed', report_sha256=sha256(output / 'native_operation_result.json'))
                atomic_json(ledger_path, ledger)
                return report
            atomic_json(ledger_path, ledger)
        raise RuntimeError('Native retry loop exhausted')

    def _evaluate_once(self, candidate, duration_s, out_dir=None, output_dir=None, mode="full"):
        if mode not in {"full", "physics", "render", "terminal_settle"}:
            raise ValueError("Unknown scene worker evaluation mode")
        output = Path(out_dir or output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        inputs = {key: {"path": str(Path(candidate[key]).resolve()), "sha256": sha256(candidate[key])}
                  for key in ("usd_path", "scene_manifest_path", "layout_path") if candidate.get(key)}
        binding = {"candidate_id": candidate["candidate_id"], "inputs": inputs,
                   "runtime_binding": self.runtime_binding,
                   "mode": mode, "pruning_witness": candidate.get("pruning_witness"),
                   "duration_s": float(duration_s), "config": self.config, "source_hashes": self.frozen_sources}
        try:
            assert_source_binding(self.frozen_sources)
            if native_runtime_binding(self.launcher) != self.runtime_binding:
                raise RuntimeError("Native runtime changed since client startup")
        except RuntimeError as exc:
            self.failure = str(exc)
        digest = content_hash(binding)
        report_path = output / "physics_report.json"
        if not self.failure and report_path.is_file():
            prior = json.loads(report_path.read_text())
            if prior.get('request_digest') == digest and valid_asset_rejection(prior):
                return dict(prior, cache_hit=True)
            if mode == 'terminal_settle' and prior.get('request_digest') == digest and valid_terminal_evidence(prior):
                return dict(prior, cache_hit=True)
            if mode == "render" and prior.get("request_digest") == digest and prior.get("status") == "rendered":
                images = prior.get("images", [])
                if len(images) == 2 and all(sha256(i["path"]) == i["sha256"] for i in images):
                    return dict(prior, cache_hit=True)
            if (prior.get("request_digest") == digest and prior.get("acceptance_complete") is True
                    and observation_ready(prior) and prior.get("native_dynamics") is True
                    and prior.get("status") in {"passed", "physics_failed"}
                    and prior.get("source_hashes") == self.frozen_sources
                    and valid_cached_evidence(prior, duration_s, self.config, require_images=mode != "physics")):
                return dict(prior, cache_hit=True)
        startup_started = time.monotonic()
        was_started = self.process is not None
        self.resource_asset = candidate.get("usd_path", self.queue / "startup.json")
        self._start()
        startup_seconds = 0. if was_started else time.monotonic() - startup_started
        request_id = uuid.uuid4().hex
        if not self.failure:
            request = dict(binding, session_id=self.session, request_id=request_id,
                           request_digest=digest, out_dir=str(output), candidate=candidate)
            atomic_json(self.queue / f"request_{request_id}.json", request)
            response_path = self.queue / f"response_{request_id}.json"
            started = time.monotonic()
            while not response_path.exists():
                if self.process.poll() is not None:
                    self.failure_retryable = True
                    self.failure = f"Isaac exited with code {self.process.returncode}"
                    break
                if time.monotonic() - started > self.config["worker_timeout_s"]:
                    self.failure_retryable = True
                    self.failure = "Isaac candidate evaluation timed out"
                    self.close()
                    break
                time.sleep(.1)
            if not self.failure:
                report = json.loads(response_path.read_text())
                try:
                    assert_source_binding(self.frozen_sources)
                except RuntimeError as exc:
                    self.failure = str(exc)
                if (report.get("request_digest") != digest or report.get("request_id") != request_id
                        or report.get("session_id") != self.session
                        or report.get("source_hashes") != self.frozen_sources):
                    self.failure = "Stale/mismatched Isaac response"
                if not self.failure:
                    report.setdefault("timings", {})["worker_startup_seconds"] = startup_seconds
                    atomic_json(report_path, report)
                    if hasattr(self.process, "set_idle"):
                        self.process.set_idle(True)
                    return report
        report = {"schema_version": "scene_physics_report.v1", "status": "environment_error", "passed": False,
                  "acceptance_complete": False, "complete": False, "request_digest": digest, "binding": binding,
                  "request_id": request_id, "session_id": self.session, "violation": 1.0,
                  "physical_residual": 1.0, "images": [], "errors": [self.failure],
                  "worker_log": str(self.queue / "worker.log"), "queue_path": str(self.queue),
                  "retryable_technical": self.failure_retryable}
        atomic_json(report_path, report)
        return report

    def render(self, candidate, out_dir):
        return self.evaluate(candidate, 0., out_dir=out_dir, mode="render")

    def close(self):
        if self.process is not None and self.process.poll() is None:
            atomic_json(self.queue / "stop.json", {"session_id": self.session})
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # start_new_session=True gives this owned process its own group.
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=10)
        if self.log:
            self.log.close()
            self.log = None
        if self.coordination_lock is not None:
            self.coordination_lock.__exit__(None, None, None)
            self.coordination_lock = None

    def __exit__(self, *_):
        self.close()
