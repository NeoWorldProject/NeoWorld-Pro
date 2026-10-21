"""Host-wide admission for native Isaac processes; queue time is not execution time."""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

from .gpu_queue import process_identity
from .run_state import write_json

POLICY = 'isaac_host_resources.v1'
MIB = 1024 * 1024


def root_path():
    return Path(os.environ.get('NEOWORLD_ISAAC_ADMISSION_DIR', '/tmp/neoworld_isaac_admission'))


def psi(kind):
    line = Path('/proc/pressure/' + kind).read_text().splitlines()[0]
    return float(dict(s.split('=') for s in line.split()[1:])['avg10'])


def host_sample():
    cpu = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    mem = {k: int(v.split()[0]) / 1024 for k, v in
           (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return dict(epoch=time.time(), cpu_total=sum(cpu), cpu_idle=cpu[3] + cpu[4],
                cpu_psi=psi('cpu'), memory_psi=psi('memory'),
                available_mib=mem['MemAvailable'], total_mib=mem['MemTotal'],
                swap_bytes=(int(vm['pswpin']) + int(vm['pswpout'])) * os.sysconf('SC_PAGE_SIZE'))


def process_groups():
    groups = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            pgid = int(fields[2]); pid = int(path.parent.name)
            row = groups.setdefault(pgid, {'rss_mib': 0., 'pids': [], 'gpu_mib': 0.})
            row['rss_mib'] += int(fields[21]) * os.sysconf('SC_PAGE_SIZE') / MIB
            row['pids'].append(pid)
        except (OSError, ValueError, IndexError):
            continue
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError('Cannot sample Isaac GPU processes: ' + result.stderr.strip())
    by_pid = {pid: row for row in groups.values() for pid in row['pids']}
    for line in result.stdout.splitlines():
        try:
            pid, memory = [int(x.strip()) for x in line.split(',')]
            if pid in by_pid:
                by_pid[pid]['gpu_mib'] += memory
        except ValueError:
            continue
    return groups


def evaluate(samples, leases, gpu_free, estimates, disk_free, gpu_headroom=2048):
    """Pure admission decision; already resident usage is not subtracted twice."""
    if not samples:
        return ['collecting_cpu_history']
    s = samples[-1]; now = s['epoch']; reasons = []
    old = next((x for x in reversed(samples) if now - x['epoch'] >= 30), None)
    if old is None:
        reasons.append('collecting_cpu_history')
    else:
        total = s['cpu_total'] - old['cpu_total']
        utilization = 100 * (1 - (s['cpu_idle'] - old['cpu_idle']) / max(1, total))
        if utilization >= 80:
            reasons.append('cpu_busy')
    window = [x for x in samples if now - x['epoch'] <= 10]
    before = next((x for x in reversed(samples) if now - x['epoch'] >= 10), None)
    if before is None:
        reasons.append('collecting_pressure_history')
    else:
        window = [before] + window
        if any(x['cpu_psi'] >= 10 for x in window):
            reasons.append('cpu_pressure')
        if any(x['memory_psi'] >= 1 for x in window):
            reasons.append('memory_pressure')
        if (s['swap_bytes'] - before['swap_bytes']) / (now - before['epoch']) >= MIB:
            reasons.append('swap_activity')
    if any(not v.get('ready_at') or now - v['ready_at'] < 10 for v in leases):
        reasons.append('initialization_in_progress')
    ram_reserved = sum(max(0, v['expected_ram_mib'] - v.get('rss_mib', 0)) for v in leases)
    gpu_reserved = sum(max(0, v['expected_gpu_mib'] - v.get('gpu_mib', 0)) for v in leases)
    if s['available_mib'] - ram_reserved - estimates[0] < max(8192, s['total_mib'] * .15):
        reasons.append('host_memory')
    if gpu_free - gpu_reserved - estimates[1] < max(2048, gpu_headroom):
        reasons.append('gpu_memory')
    if disk_free < 1200:
        reasons.append('disk_space')
    return reasons


class Registry:
    def __init__(self, root=None):
        self.root = Path(root or root_path()); self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with (self.root / 'registry.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'registry.json'
            state = json.loads(path.read_text()) if path.exists() else dict(
                policy=POLICY, samples=[], leases={}, waiting={}, history={}, peak_workers=0)
            try:
                yield state
            finally:
                write_json(path, state)

    def refresh(self, state):
        now = time.time()
        if state['samples'] and now - state['samples'][-1]['epoch'] < 2:
            return
        groups = process_groups(); sample = host_sample()
        state['samples'] = [x for x in state['samples'] if now - x['epoch'] < 40] + [sample]
        for key, lease in list(state['leases'].items()):
            pid = lease.get('pid')
            live = process_identity(pid) == lease.get('start_tick') if pid else (
                process_identity(lease['owner']) == lease['owner_tick'])
            if not live:
                if lease.get('ready_at'):
                    history = state['history'].setdefault(lease['key'], {})
                    for metric in ('ram', 'gpu'):
                        history[metric] = max(history.get(metric, 0), lease.get('peak_' + metric, 0))
                del state['leases'][key]
                continue
            observed = groups.get(pid, {})
            lease['rss_mib'] = observed.get('rss_mib', 0)
            lease['gpu_mib'] = observed.get('gpu_mib', 0)
            lease['peak_ram'] = max(lease.get('peak_ram', 0), lease['rss_mib'])
            lease['peak_gpu'] = max(lease.get('peak_gpu', 0), lease['gpu_mib'])
            ready = self.root / (key + '.initialized.json')
            if ready.exists() and not lease.get('ready_at'):
                lease['ready_at'] = now
        state['waiting'] = {k: v for k, v in state['waiting'].items()
                            if process_identity(v['owner']) == v['owner_tick']}
        count = sum(bool(v.get('pid')) for v in state['leases'].values())
        state['peak_workers'] = max(state['peak_workers'], count)
        with (self.root / 'samples.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(sample, active_workers=count, peak_workers=state['peak_workers'],
                workers=[{'pid': v.get('pid'), 'rss_mib': v.get('rss_mib'), 'gpu_mib': v.get('gpu_mib'),
                          'label': v.get('label')} for v in state['leases'].values()])) + '\n')


def initialized():
    """Called by the native worker after its first physics context is initialized."""
    target = os.environ.get('NEOWORLD_ISAAC_INITIALIZED_FILE')
    if target and not Path(target).exists():
        write_json(Path(target), {'pid': os.getpid(), 'epoch': time.time()})


def thread_args():
    return ['--/plugins/carb.tasking.plugin/threadCount=8',
            '--/plugins/omni.tbb.globalcontrol/maxThreadCount=8']


class IsaacProcess(subprocess.Popen):
    def __init__(self, command, *, resource_key, resource_output, cancelled=None, gpu_headroom=2048, **kwargs):
        import shutil
        from .gpu_queue import gpu_memory
        self.registry = Registry(); self.lease_id = uuid.uuid4().hex
        self.reclaimed_idle = False; self.idle = False; self.usage_lock = threading.RLock()
        self.cancelled = cancelled or (lambda: False)
        output = Path(resource_output); output.mkdir(parents=True, exist_ok=True)
        ticket = dict(owner=os.getpid(), owner_tick=process_identity(os.getpid()), epoch=time.time(),
                      label=str(output), key=resource_key, experiment=os.environ.get('NEOWORLD_EXPERIMENT_ROOT', str(output)))
        with self.registry.locked() as state:
            state['waiting'][self.lease_id] = ticket
        try:
            while True:
                if self.cancelled():
                    raise RuntimeError('Isaac resource wait cancelled')
                with self.registry.locked() as state:
                    self.registry.refresh(state)
                    history = state['history'].get(resource_key)
                    estimates = (max(1024, history['ram'] * 1.25), max(512, history['gpu'] * 1.25)) if history else (16384, 4096)
                    reasons = evaluate(state['samples'], list(state['leases'].values()),
                                       gpu_memory()['free_mib'], estimates, shutil.disk_usage(output).free / MIB, gpu_headroom)
                    if next(iter(state['waiting'])) != self.lease_id:
                        reasons.append('fifo')
                    write_json(output / 'resource_wait.json', dict(ticket, reasons=reasons,
                               expected_ram_mib=estimates[0], expected_gpu_mib=estimates[1], updated_at=time.time()))
                    if not reasons:
                        state['leases'][self.lease_id] = dict(ticket, expected_ram_mib=estimates[0],
                                                           expected_gpu_mib=estimates[1])
                        del state['waiting'][self.lease_id]
                        break
                time.sleep(2)
            env = dict(kwargs.pop('env', os.environ))
            env.update(NEOWORLD_ISAAC_INITIALIZED_FILE=str(self.registry.root / (self.lease_id + '.initialized.json')),
                       OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8', NUMEXPR_NUM_THREADS='8')
            super().__init__(command, env=env, **kwargs)
            with self.registry.locked() as state:
                state['leases'][self.lease_id].update(pid=self.pid, start_tick=process_identity(self.pid))
                live = [v for v in state['leases'].values() if v.get('pid') and
                        process_identity(v['pid']) == v['start_tick']]
                state['peak_workers'] = max(state['peak_workers'], len(live))
                peaks = state.setdefault('experiment_peaks', {})
                group = ticket['experiment']
                peaks[group] = max(peaks.get(group, 0), sum(v.get('experiment') == group for v in live))
                with (self.registry.root / 'launches.jsonl').open('a') as stream:
                    stream.write(json.dumps({'epoch': time.time(), 'pid': self.pid, 'experiment': group,
                                             'active_workers': len(live), 'experiment_peak': peaks[group]}) + '\n')
            self.monitor = threading.Thread(target=self._monitor, daemon=True)
            self.monitor.start()
        except BaseException:
            if getattr(self, 'pid', None) is not None and self.poll() is None:
                import signal
                os.killpg(self.pid, signal.SIGKILL)
                self.wait()
            with self.registry.locked() as state:
                state['waiting'].pop(self.lease_id, None)
                state['leases'].pop(self.lease_id, None)
            raise

    def set_idle(self, idle):
        with self.usage_lock:
            self.idle = idle
            return not self.reclaimed_idle

    def _monitor(self):
        import signal
        while True:
            try:
                with self.registry.locked() as state:
                    self.registry.refresh(state)
                    waiting = bool(state['waiting'])
                with self.usage_lock:
                    if waiting and self.idle and not self.reclaimed_idle and self.poll() is None:
                        self.reclaimed_idle = True
                        self.reclaim_started = time.monotonic()
                        os.killpg(self.pid, signal.SIGTERM)
                if self.reclaimed_idle and self.poll() is None and time.monotonic() - self.reclaim_started > 10:
                    os.killpg(self.pid, signal.SIGKILL)
                if self.poll() is not None:
                    # Refresh once more after exit so retained idle allocations are released.
                    time.sleep(2)
                    with self.registry.locked() as state:
                        self.registry.refresh(state)
                    return
            except Exception as error:
                with (self.registry.root / 'monitor_errors.jsonl').open('a') as stream:
                    stream.write(json.dumps({'pid': self.pid, 'error': repr(error), 'epoch': time.time()}) + '\n')
            time.sleep(2)


def resource_key(path, runtime):
    path = Path(path)
    content = path.read_bytes() if path.is_file() else str(path).encode()
    return hashlib.sha256(content + str(runtime).encode()).hexdigest()
