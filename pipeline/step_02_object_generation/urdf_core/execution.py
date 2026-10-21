"""Detect startup failures and stalled workers without a total runtime limit."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import os
import signal
import subprocess
import time

STARTUP_TIMEOUT_S = 300.
NO_PROGRESS_TIMEOUT_S = 600.


def watchdog_settings():
    return {'startup_timeout_s': float(os.environ.get('NEOWORLD_STARTUP_TIMEOUT', STARTUP_TIMEOUT_S)),
            'no_progress_timeout_s': float(os.environ.get('NEOWORLD_NO_PROGRESS_TIMEOUT', NO_PROGRESS_TIMEOUT_S))}


class TechnicalExecutionError(RuntimeError):
    pass


class ExecutionCancelled(RuntimeError):
    pass


def terminal_failure(error):
    """Environment/transport failures must not become normal best-effort completion."""
    from .api_requests import APIAttemptsExhausted, ServiceIssue
    from .gpu_queue import GpuResourceError
    return isinstance(error, (OSError, APIAttemptsExhausted, ServiceIssue,
                              GpuResourceError, TechnicalExecutionError, ExecutionCancelled))


class ProgressWatchdog:
    def __init__(self, *, startup_timeout=None,
                 no_progress_timeout=None, now=time.monotonic):
        import math
        startup_timeout = float(os.environ.get('NEOWORLD_STARTUP_TIMEOUT', STARTUP_TIMEOUT_S)) if startup_timeout is None else startup_timeout
        no_progress_timeout = float(os.environ.get('NEOWORLD_NO_PROGRESS_TIMEOUT', NO_PROGRESS_TIMEOUT_S)) if no_progress_timeout is None else no_progress_timeout
        if any(not math.isfinite(v) or v <= 0 for v in (startup_timeout, no_progress_timeout)):
            raise ValueError('Watchdog intervals must be positive and finite')
        self.now = now
        self.started = self.last_progress = now()
        self.startup_timeout, self.no_progress_timeout = startup_timeout, no_progress_timeout
        self.token = None
        self.ready = False

    def check(self, token=None):
        current = self.now()
        if token is not None and (not self.ready or token != self.token):
            self.ready = True
            self.token = copy.deepcopy(token)
            self.last_progress = current
        if not self.ready and current - self.started >= self.startup_timeout:
            raise TechnicalExecutionError('Worker did not finish startup within 300 seconds' if self.startup_timeout == 300 else 'Worker startup stalled')
        if self.ready and current - self.last_progress >= self.no_progress_timeout:
            raise TechnicalExecutionError('Worker made no progress within 600 seconds' if self.no_progress_timeout == 600 else 'Worker progress stalled')

    @contextmanager
    def suspended(self):
        started = self.now()
        try:
            yield
        finally:
            elapsed = self.now() - started
            self.started += elapsed
            self.last_progress += elapsed


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        grouped = os.getpgid(process.pid) == process.pid
        if grouped:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if grouped:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)
    except ProcessLookupError:
        pass


def wait_process(process, *, progress, cancelled=None, startup_timeout=None,
                 no_progress_timeout=None, poll_interval=.2):
    """Return the exit code; stop only this process group on timeout/cancellation."""
    watchdog = ProgressWatchdog(startup_timeout=startup_timeout,
                                no_progress_timeout=no_progress_timeout)
    try:
        while True:
            code = process.poll()
            if code is not None:
                return code
            if cancelled is not None and cancelled():
                raise ExecutionCancelled('Worker cancelled')
            watchdog.check(progress())
            time.sleep(poll_interval)
    except BaseException:
        stop_process(process)
        raise


_OWNER_MONITOR = None


def native_owner_environment():
    from .gpu_queue import process_identity
    return {'NEOWORLD_NATIVE_OWNER_PID': str(os.getpid()),
            'NEOWORLD_NATIVE_OWNER_START': process_identity(os.getpid())}


def start_owner_monitor(owner_pid=None, owner_start=None):
    """Stop this isolated worker group when its exact owner or launcher exits."""
    import threading
    from .gpu_queue import process_identity
    global _OWNER_MONITOR
    owner_pid = owner_pid or os.environ.get('NEOWORLD_NATIVE_OWNER_PID')
    owner_start = owner_start or os.environ.get('NEOWORLD_NATIVE_OWNER_START')
    if owner_pid is None or owner_start is None:
        return
    owner_pid = int(owner_pid)
    binding = (owner_pid, owner_start)
    if _OWNER_MONITOR is not None:
        if _OWNER_MONITOR != binding:
            raise RuntimeError('Native worker owner changed')
        return
    group = os.getpgrp()
    try:
        if os.getpgid(owner_pid) == group:
            raise RuntimeError('Native owner monitor requires an isolated worker process group')
    except ProcessLookupError:
        pass
    leader_start = process_identity(group)
    def watch():
        while True:
            if process_identity(owner_pid) != owner_start or process_identity(group) != leader_start:
                os.killpg(group, signal.SIGKILL)
                return
            time.sleep(.2)
    _OWNER_MONITOR = binding
    threading.Thread(target=watch, name='native-owner-monitor', daemon=True).start()
