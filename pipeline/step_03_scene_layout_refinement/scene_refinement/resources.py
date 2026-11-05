"""Read-only resource observations; failures are explicit, never zero usage."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def resource_sample():
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    result.check_returncode()
    used, free, load = map(float, result.stdout.splitlines()[0].split(","))
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines()}
    processes = {}
    for entry in Path('/proc').glob('[0-9]*/stat'):
        try:
            values = entry.read_text().rsplit(')', 1)[1].split()
            processes[int(entry.parent.name)] = (int(values[1]), int(values[21]) * os.sysconf('SC_PAGE_SIZE'),
                                                 int(values[11]) + int(values[12]))
        except (OSError, ValueError, IndexError):
            pass
    descendants = {os.getpid()}
    while True:
        added = {pid for pid, (ppid, _, _) in processes.items() if ppid in descendants}
        if added <= descendants:
            break
        descendants |= added
    applications = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=10)
    applications.check_returncode()
    gpu_processes = []
    for line in applications.stdout.splitlines():
        if not line.strip():
            continue
        pid, mib = line.split(',')
        pid = int(pid)
        gpu_processes.append({'pid': pid, 'used_mib': float(mib), 'owned_by_batch': pid in descendants,
                              'cpu_ticks': processes.get(pid, (0, 0, None))[2]})
    cpu = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
    disk = {}
    for line in Path('/proc/diskstats').read_text().splitlines():
        values = line.split()
        if Path('/sys/block', values[2]).exists() and not values[2].startswith(('loop', 'ram')):
            disk[values[2]] = {'read_bytes': int(values[5])*512, 'write_bytes': int(values[9])*512,
                               'read_time_ms': int(values[6]), 'write_time_ms': int(values[10])}
    return {"time_unix": time.time(), "gpu": {"used_mib": used, "free_mib": free, "utilization_pct": load,
            'compute_processes': gpu_processes,
            'external_compute_present': any(not p['owned_by_batch'] for p in gpu_processes),
            'external_compute_memory_mib': sum(p['used_mib'] for p in gpu_processes if not p['owned_by_batch'])},
            "host": {"available_bytes": memory['MemAvailable'],
                     "used_bytes": memory['MemTotal']-memory['MemAvailable'],
                     "cpu_ticks": cpu, "clock_ticks_per_second": os.sysconf('SC_CLK_TCK'),
                     "batch_cpu_ticks": sum(processes[p][2] for p in descendants if p in processes),
                     "batch_rss_bytes": sum(processes[p][1] for p in descendants if p in processes)},
            "disk_devices": disk}


class ResourceSampler:
    def __init__(self, path, interval=1.):
        self.path, self.interval = Path(path), interval
        self.stop = threading.Event()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        with self.path.open("a") as output:
            while not self.stop.is_set():
                try:
                    sample = resource_sample()
                except Exception as error:
                    sample = {"time_unix": time.time(), "sampling_error": str(error)}
                output.write(json.dumps(sample) + "\n"); output.flush()
                self.stop.wait(self.interval)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=12)


def summarize_resources(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    valid = [r for r in rows if 'gpu' in r]
    identified = [r for r in valid if 'external_compute_present' in r['gpu']]
    return {'sample_count': len(rows), 'sampling_errors': len(rows)-len(valid),
        'gpu_peak_mib': max((r['gpu']['used_mib'] for r in valid), default=None),
        'gpu_minimum_free_mib': min((r['gpu']['free_mib'] for r in valid), default=None),
        'batch_peak_rss_bytes': max((r['host']['batch_rss_bytes'] for r in valid), default=None),
        'process_identification_samples': len(identified),
        'external_compute_overlap_fraction': (sum(r['gpu']['external_compute_present'] for r in identified)/len(identified)
                                               if identified else None),
        'external_compute_peak_mib': max((r['gpu']['external_compute_memory_mib'] for r in identified), default=None),
        'timing_interpretation': 'Observed wall times include contention. Process presence is evidence of overlap, not a causal estimate of slowdown.'}


def available_cpus():
    """Conservative CPU count from affinity and hierarchical Linux quotas."""
    import math
    count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    cgroups = Path('/proc/self/cgroup')
    if not cgroups.is_file():
        return count
    for line in cgroups.read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if not controllers:
            base = Path('/sys/fs/cgroup')
            leaf = base / relative.lstrip('/')
            for folder in [leaf, *leaf.parents]:
                if not folder.is_relative_to(base):
                    break
                quota = folder / 'cpu.max'
                if quota.is_file():
                    value, period = quota.read_text().split()[:2]
                    if value != 'max':
                        count = min(count, max(1, math.floor(int(value) / int(period))))
        elif 'cpu' in controllers.split(','):
            for base in (Path('/sys/fs/cgroup/cpu'), Path('/sys/fs/cgroup/cpu,cpuacct')):
                leaf = base / relative.lstrip('/')
                for folder in [leaf, *leaf.parents]:
                    if not folder.is_relative_to(base):
                        break
                    quota, period = folder/'cpu.cfs_quota_us', folder/'cpu.cfs_period_us'
                    if quota.is_file() and period.is_file() and int(quota.read_text()) > 0:
                        count = min(count, max(1, math.floor(int(quota.read_text())/int(period.read_text()))))
    return max(1, count)
