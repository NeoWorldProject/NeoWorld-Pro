"""Supervise an isolated Isaac 5.1 dynamic object validation worker.

The report distinguishes a completed physical rejection from an unavailable or
incomplete observation. A timeout, renderer failure, or stale report cannot pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from urdf_core.isaac_cache import prepare_runtime_cache

DEFAULT_ISAAC_PYTHON = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [Path(__file__), HERE / "physics_runtime.py", HERE / "geometry_checks.py",
             HERE / "metrics.py", HERE / "control.py", HERE / "protocol.py", HERE / "collision_planning.py",
             HERE / "physicalize.py", HERE / "lightweight_collision.py", HERE / "collision_reuse.py", HERE / "support_rest_pose.py",
             HERE / "trajectory_io.py", HERE / "motion_scope.py",
             HERE / "capture_retention.py",
             HERE / "stream_video.py", HERE / "trial_order.py", HERE / "representative_motion.py", HERE / "joint_groups.py", HERE / "joint_group_geometry.py",
             HERE / "deferred_capture.py", HERE / "resident_worker.py", HERE / "incidental_motion.py",
             HERE / "control_diagnostics.py",
             HERE / "measurement_plan.py", HERE / "motion_acceptance.py", HERE / 'test_modes.py', HERE / 'trial_checkpoint.py', HERE / 'parallel_trials.py',
             HERE / 'complete_evaluation.py', HERE / 'verification.py', HERE / 'verification_receipt.py', HERE / 'measurement_reuse.py',
             HERE.parent / "urdf_core/kinematics.py", HERE.parent / "urdf_core/urdf_validation.py",
             HERE.parent / "urdf_core/isaac_cache.py",
             HERE.parent / "urdf_core/isaac_admission.py",
             HERE.parent / "urdf_core/content_errors.py",
             HERE.parent / "urdf_core/degraded_asset.py",
             HERE.parent / "urdf_core/gpu_queue.py", HERE.parent / "urdf_core/execution.py",
             HERE.parent / "urdf_core/media_execution.py",
             HERE.parent / "urdf_core/native_progress.py",
             HERE.parent / "kinematics_validation/isaac_motion.py",
             HERE.parent / "kinematics_validation/verify_urdf_isaac.py"]
    return {str(path): sha256(path) for path in paths if path.is_file()}


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    if report.get('run_id') and report.get('stage') not in (None, 'starting', 'input_validation', 'simulation_app_launch'):
        from urdf_core.native_progress import write_execution_progress, FIELDS
        write_execution_progress(path.parent, report['run_id'], {k: report.get(k) for k in FIELDS})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("asset-dir", "spec", "profile", "contract", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--ffmpeg", default=os.environ.get("NEOWORLD_FFMPEG", "ffmpeg"))
    parser.add_argument("--trial-order", type=Path)
    parser.add_argument("--representative-motion", type=Path)
    parser.add_argument('--incidental-motion', type=Path)
    parser.add_argument('--capture-mode', choices=('deferred', 'live'), default='deferred')
    parser.add_argument('--measurement-mode', choices=('optimized', 'reference'), default='optimized')
    parser.add_argument('--trial-cache-root', type=Path, help='Exact-candidate completed-trial checkpoint directory')
    parser.add_argument('--no-trial-resume', action='store_true')
    parser.add_argument('--evaluation-policy', choices=('first-failure','complete','grouping'), default='first-failure')
    parser.add_argument('--trial-workers', type=int, choices=range(1,5), default=1)
    parser.add_argument('--trial-selection', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--resident-server', type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--expected-sources", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.isaac_python is None:
        from urdf_core.environment import resolve_tool
        try:
            args.isaac_python = Path(resolve_tool("isaac_python"))
        except FileNotFoundError as error:
            parser.error(str(error))
    for name in ("asset_dir", "spec", "profile", "contract", "out_dir", "isaac_python"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.trial_cache_root=None if args.no_trial_resume else (args.trial_cache_root or args.out_dir).resolve()
    return args


def initial_report(args, run_id):
    inputs = {"urdf": args.asset_dir / "object.urdf", "spec": args.spec,
              "profile": args.profile, "contract": args.contract,
              "asset_manifest": args.asset_dir / "asset_manifest.json"}
    report = {"schema_version": 1, "run_id": run_id, "status": "running", "passed": False,
              "numerical_passed": False, "evaluation_complete": False, "acceptance_complete": False, "stage": "starting",
              "trials": [], "errors": [], "issues": [], "source_hashes": source_hashes(),
              "reload_check": {"passed": False}, "inputs": {}}
    for name, path in inputs.items():
        report["inputs"][name] = {"path": str(path), "sha256": sha256(path) if path.is_file() else None}
        report[name + "_sha256"] = report["inputs"][name]["sha256"]
    order = getattr(args, 'trial_order', None)
    if order is not None:
        path = Path(order).resolve()
        report['inputs']['trial_order'] = {'path': str(path), 'sha256': sha256(path)}
    representative = getattr(args, "representative_motion", None)
    if representative is not None:
        path = Path(representative).resolve()
        report["inputs"]["representative_motion"] = {"path": str(path), "sha256": sha256(path)}
    incidental = getattr(args, 'incidental_motion', None)
    if incidental is not None:
        path = Path(incidental).resolve()
        report['inputs']['incidental_motion'] = {'path': str(path), 'sha256': sha256(path)}
        from physics_feedback.incidental_motion import POLICY
        report['incidental_motion_policy'] = POLICY
    return report


def worker_workload(contract):
    """Describe simulated work; elapsed runtime never limits its coverage."""
    from physics_feedback.control import scoped_motion_joint, target_schedule, post_operation_observation_s
    from physics_feedback.protocol import free_base_required
    config = contract['config']
    settle = float(config['settle_duration_s'])
    total = settle if contract['stable_required'] else 0.
    per_joint = {}
    for name in contract['joints']:
        _, motion = target_schedule(scoped_motion_joint(contract, name), config, contract=contract)
        duration = settle + motion + post_operation_observation_s(config, contract=contract)
        per_joint[name] = duration
        total += duration * (2 if free_base_required(contract) else 1)
    from urdf_core.execution import watchdog_settings
    return {'policy': 'physics_workload.v2', 'simulated_duration_upper_bound_s': total + settle,
            'full_joint_duration_upper_bounds_s': per_joint,
            **watchdog_settings(), 'total_runtime_limit': None}


def supervised_run(args):
    if getattr(args,'trial_workers',1)>1:
        from physics_feedback.parallel_trials import ParallelPhysicsSession
        with ParallelPhysicsSession(args.out_dir/'workers',workers=args.trial_workers) as session:
            path=session.run(args)
        report=json.loads(path.read_text())
        print(f"Isaac physics: {report['status']}; report: {path}",flush=True)
        return 0 if report['passed'] or report['status'] == 'grouping_ready' else 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    report_path = args.out_dir / "report.json"
    report = initial_report(args, run_id)
    if any(not value["sha256"] for value in report["inputs"].values()) or not args.isaac_python.is_file():
        report.update(status="environment_error", stage="input_validation")
        report["errors"].append("A required asset, protocol, or Isaac launcher is missing")
        write_report(report_path, report)
        return 3
    contract = json.loads(args.contract.read_text())
    worker_budget = worker_workload(contract)
    report['worker_workload'] = worker_budget
    runtime = args.out_dir / "runtime"
    env = os.environ.copy()
    for key, folder in {"XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data",
                        "CUDA_CACHE_PATH": "cuda", "__GL_SHADER_DISK_CACHE_PATH": "shader", "TMPDIR": "tmp"}.items():
        path = runtime / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    from urdf_core.execution import native_owner_environment
    env.update(native_owner_environment())
    command = [str(args.isaac_python), str(Path(__file__).resolve()), "--worker", "--run-id", run_id,
               "--expected-sources", json.dumps(report["source_hashes"], sort_keys=True),
               '--capture-mode', args.capture_mode, '--measurement-mode', args.measurement_mode,
               '--evaluation-policy', getattr(args,'evaluation_policy','first-failure')]
    for name in ("asset_dir", "spec", "profile", "contract", "out_dir", "isaac_python", "ffmpeg"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.trial_order is not None:
        command.extend(['--trial-order', str(args.trial_order.resolve())])
    if getattr(args, "representative_motion", None) is not None:
        command.extend(["--representative-motion", str(args.representative_motion.resolve())])
    if args.trial_cache_root is not None:
        command.extend(['--trial-cache-root',str(args.trial_cache_root)])
    else:
        command.append('--no-trial-resume')
    if args.incidental_motion is not None:
        command.extend(['--incidental-motion',str(args.incidental_motion.resolve())])
    if args.trial_selection is not None:
        raise ValueError('Trial fragments must be dispatched by the isolated pool')
    write_report(report_path, report)
    timed_out, returncode = False, None
    cache_policy = None
    started = time.monotonic()
    try:
        from contextlib import nullcontext
        queued = getattr(args, '_gpu_queue_owned', False)
        from urdf_core.isaac_cache import gpu_worker_slot
        with (nullcontext(None) if queued else gpu_worker_slot()) as lock_fd:
            if queued:
                from urdf_core.gpu_queue import private_runtime_cache
                cache_policy = private_runtime_cache(args.out_dir, args.isaac_python, environ=env)
            else:
                cache_policy = prepare_runtime_cache(args.out_dir, args.isaac_python,
                                                     gpu_lock_fd=lock_fd, environ=env)
            report["runtime_cache_policy"] = cache_policy
            # A CLI can wait behind a long 3.1 check while implementation files
            # change. Bind the worker sources at dispatch, not at queue entry.
            report["source_hashes"] = source_hashes()
            command[command.index("--expected-sources") + 1] = json.dumps(report["source_hashes"], sort_keys=True)
            write_report(report_path, report)
            with (args.out_dir / "isaac_stdout.log").open("w", encoding="utf-8") as log:
                from urdf_core.isaac_admission import IsaacProcess, resource_key
                process = IsaacProcess(command, resource_key=resource_key(args.asset_dir / 'object.urdf', args.isaac_python), resource_output=args.out_dir, cwd=args.out_dir, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                if getattr(args, '_gpu_task', None) is not None:
                    args._gpu_task.bind_process(process)
                from urdf_core.execution import wait_process, TechnicalExecutionError
                from urdf_core.native_progress import read_execution_progress
                try:
                    returncode = wait_process(process,
                        progress=lambda: read_execution_progress(args.out_dir, run_id))
                except TechnicalExecutionError as exc:
                    timed_out = True
                    report['errors'].append(str(exc))
    except (OSError, ValueError) as exc:
        report["errors"].append(f"Worker setup or launch failed: {exc}")
        write_report(report_path, report)
    try:
        saved = json.loads(report_path.read_text())
        if saved.get("run_id") != run_id or saved.get("source_hashes") != report["source_hashes"]:
            raise ValueError("Worker report does not match this invocation/source")
        report = saved
    except (OSError, ValueError) as exc:
        report["errors"].append(str(exc))
    report['worker_workload'] = worker_budget
    if cache_policy is not None:
        # The native worker creates its own report; retain the supervisor's
        # resource policy alongside its measurements without changing them.
        report["runtime_cache_policy"] = cache_policy
    if timed_out or returncode != 0 or report.get("status") not in {"passed", "physics_failed", "grouping_ready", "asset_rejected"}:
        report.update(status="environment_error", passed=False, evaluation_complete=False, acceptance_complete=False)
        report["errors"].append("Worker timeout" if timed_out else f"Worker incomplete or failed (exit={returncode})")
    elif report["status"] == "passed" and not (report.get("numerical_passed") and report.get("reload_check", {}).get("passed")
                                               and isinstance(report.get("trials"), list) and report.get("delivered_usd_sha256")):
        report.update(status="environment_error", passed=False, evaluation_complete=False, acceptance_complete=False)
        report["errors"].append("Incomplete worker acceptance evidence")
    report.update(worker_returncode=returncode, elapsed_seconds=round(time.monotonic() - started, 3))
    write_report(report_path, report)
    print(f"Isaac physics: {report['status']}; report: {report_path}", flush=True)
    return {"passed": 0, "physics_failed": 2, "grouping_ready": 0, "asset_rejected": 2, "environment_error": 3}[report["status"]]


def queued_run(args):
    from urdf_core.gpu_queue import (GpuTask, GpuOutOfMemory, GpuResourceError,
                                    oom_evidence, archive_attempt, record_resource_error)
    if getattr(args, 'trial_workers', 1) > 1:
        try:
            return supervised_run(args)
        except GpuResourceError as error:
            record_resource_error(args.out_dir, error)
            return 3
    identity = initial_report(args, 'gpu_admission')
    if any(not v['sha256'] for v in identity['inputs'].values()) or not args.isaac_python.is_file():
        return supervised_run(args)
    task = GpuTask(args.out_dir / 'gpu_admission.json',
        identity={'inputs': identity['inputs'], 'sources': identity['source_hashes']})
    def execute(_fd):
        args._gpu_queue_owned = True
        args._gpu_task = task
        code = supervised_run(args)
        if code == 3 and oom_evidence(args.out_dir / 'isaac_stdout.log', args.out_dir / 'report.json'):
            archive_attempt(args.out_dir, task.state['oom_count'])
            raise GpuOutOfMemory('Isaac GPU OOM; resume completed trials in a new worker')
        if code == 3:
            from urdf_core.execution import TechnicalExecutionError
            archive_attempt(args.out_dir, task.state.get('technical_failures', 0), prefix='technical_attempts')
            raise TechnicalExecutionError('Isaac worker did not complete; resume the same candidate')
        return code
    try:
        return task.run(execute)
    except GpuResourceError as error:
        record_resource_error(args.out_dir, error)
        return 3


def main(argv=None):
    args = parse_args(argv)
    if args.resident_server is not None:
        from physics_feedback.resident_worker import serve
        return serve(args)
    if args.worker:
        from urdf_core.execution import start_owner_monitor
        start_owner_monitor()
        from physics_feedback.physics_runtime import run_worker
        return run_worker(args)
    return queued_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
