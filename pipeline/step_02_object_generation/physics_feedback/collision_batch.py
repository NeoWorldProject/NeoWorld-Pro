"""Bounded native collision jobs with verified translation-only geometry reuse."""
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import time

import numpy as np

from urdf_core.run_state import write_json
from urdf_core.error_recovery import VerifiedExternalError
from . import coacd_worker, collision_reuse, lightweight_collision

GIB = 1024 ** 3
RESERVE_BYTES = 6 * GIB
JOB_BYTES = 3 * GIB
REUSE_TOLERANCE_M = 1e-10


def available_resources():
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    memory = None
    if Path('/proc/meminfo').is_file():
        fields = {r.split(':')[0]: int(r.split()[1]) * 1024
                  for r in Path('/proc/meminfo').read_text().splitlines()}
        memory = fields['MemAvailable']
    cgroup = Path('/proc/self/cgroup')
    if cgroup.is_file():
        for row in cgroup.read_text().splitlines():
            _, controllers, relative = row.split(':', 2)
            if controllers:
                continue
            base = Path('/sys/fs/cgroup')
            leaf = base / relative.lstrip('/')
            for folder in [leaf, *leaf.parents]:
                if not folder.is_relative_to(base):
                    break
                quota = folder / 'cpu.max'
                if quota.is_file():
                    value, period = quota.read_text().split()[:2]
                    if value != 'max':
                        cpus = min(cpus, max(1, int(value) // int(period)))
                limit, used = folder / 'memory.max', folder / 'memory.current'
                if limit.is_file() and used.is_file() and limit.read_text().strip() != 'max':
                    free = max(0, int(limit.read_text()) - int(used.read_text()))
                    memory = free if memory is None else min(memory, free)
    return cpus, memory


def translation_match(reference, candidate):
    if len(reference) != len(candidate) or not reference:
        return None
    if any(not len(m.vertices) or not np.isfinite(m.vertices).all()
           for m in [*reference, *candidate]):
        return None
    shift = candidate[0].vertices[0] - reference[0].vertices[0]
    error = 0.
    for a, b in zip(reference, candidate):
        if a.vertices.shape != b.vertices.shape or not np.array_equal(a.faces, b.faces):
            return None
        error = max(error, float(np.max(np.abs(a.vertices + shift - b.vertices))))
        if error > REUSE_TOLERANCE_M:
            return None
    return shift, error


def translated_result(result, parts, shift, source_identity, error):
    _, original, original_audit = result
    proxies = []
    for proxy in original:
        item = {k: copy.deepcopy(v) for k, v in proxy.items() if k != 'mesh'}
        item['mesh'] = proxy['mesh'].copy()
        item['mesh'].apply_translation(shift)
        if 'box' in item:
            item['box']['center_m'] = (np.asarray(item['box']['center_m']) + shift).tolist()
        proxies.append(item)
    audit = copy.deepcopy(original_audit)
    for row in audit.get('components', []):
        if 'source_bounds_m' in row:
            row['source_bounds_m'] = (np.asarray(row['source_bounds_m']) + shift).tolist()
    bounds = np.array([p.bounds for p in parts])
    source_bounds = np.array([bounds[:, 0].min(0), bounds[:, 1].max(0)])
    bounds = np.array([p['mesh'].bounds for p in proxies])
    proxy_bounds = np.array([bounds[:, 0].min(0), bounds[:, 1].max(0)])
    audit['bounds_diagnostic'] = dict(frame='authored_link_local_m',
        source_bounds_m=source_bounds.tolist(), proxy_bounds_m=proxy_bounds.tolist(),
        outward_excess_lower_m=np.maximum(0, source_bounds[0]-proxy_bounds[0]).tolist(),
        outward_excess_upper_m=np.maximum(0, proxy_bounds[1]-source_bounds[1]).tolist())
    audit['geometry_reuse'] = dict(policy='verified_translation_only.v1', source=source_identity,
        translation_m=shift.tolist(), max_vertex_error_m=error, tolerance_m=REUSE_TOLERANCE_M,
        source_meshes=[collision_reuse._mesh_identity(p) for p in parts])
    return [p['mesh'] for p in proxies], proxies, audit


def execute(parts, settings, cache_root, identity, threads):
    started = time.monotonic()
    with coacd_worker.isolated_worker(threads) as lane:
        if cache_root is None:
            result = lightweight_collision.plan_lightweight_collisions(parts, config=settings)
        else:
            result = collision_reuse.plan_cached_collisions(parts, config=settings,
                cache_root=cache_root, identity=identity)
        process = lane.worker.process if lane.worker is not None else None
        resource = {'cpu_ids': lane.cpu_ids, 'pid': process.pid if process else None}
        if process and Path(f'/proc/{process.pid}/status').is_file():
            fields = dict(line.split(':', 1) for line in Path(f'/proc/{process.pid}/status').read_text().splitlines() if ':' in line)
            resource.update(peak_rss_kib=int(fields['VmHWM'].split()[0]),
                            native_threads=int(fields['Threads']),
                            allowed_cpus=fields['Cpus_allowed_list'].strip())
        result[2]['native_resources'] = resource
    result[2]['execution_seconds'] = time.monotonic() - started
    return result


class CollisionBatch:
    def __init__(self, settings, cache_root, progress_path):
        self.settings, self.cache_root = settings, cache_root
        self.progress_path = Path(progress_path)
        cpus, memory = available_resources()
        self.threads = min(settings['collision_threads'], max(1, cpus-2))
        self.workers = min(settings['collision_workers'], max(1, (cpus-2)//self.threads))
        if memory is not None:
            self.workers = min(self.workers, max(0, (memory-RESERVE_BYTES)//JOB_BYTES))
        if self.workers < 1:
            raise VerifiedExternalError('Insufficient memory for collision generation plus host reserve',
                kind='resource_unavailable', evidence={'available_bytes': memory,
                    'reserve_bytes': RESERVE_BYTES, 'job_bytes': JOB_BYTES})
        self.resources = dict(available_cpus=cpus, available_bytes=memory,
            reserve_bytes=RESERVE_BYTES, per_job_estimate_bytes=JOB_BYTES)
        self.groups, self.pending, self.bindings = [], [], []
        self.pool = None
        self.started = time.monotonic()
        self.peak_pending = 0

    def __enter__(self):
        self.save('running')
        return self

    def save(self, status, error=None):
        write_json(self.progress_path, dict(status=status, workers=self.workers, threads=self.threads,
            resources=self.resources, submitted_objects=len(self.bindings),
            unique_jobs=len(self.groups), reused_objects=len(self.bindings)-len(self.groups),
            completed_jobs=sum('result' in g for g in self.groups), pending_jobs=len(self.pending),
            peak_pending=self.peak_pending, seconds=time.monotonic()-self.started, error=error,
            jobs=[dict(identity=g['identity'], completed='result' in g,
                seconds=g.get('result', (None,None,{}))[2].get('execution_seconds')) for g in self.groups]))

    def drain(self):
        group = self.pending.pop(0)
        try:
            group['result'] = group.pop('future').result()
        except (ValueError, RuntimeError) as error:
            self.failure(group, error)
        self.save('running')

    @staticmethod
    def failure(group, error):
        from .physicalize import PhysicalizationGeometryError
        raise PhysicalizationGeometryError('lightweight collision planning failed: '+str(error),
            {**group['identity'], 'code':'collision_planning_failed',
             'collision_error':getattr(error, 'diagnostics', {'reason':str(error)})}) from error

    def submit(self, info, parts):
        parts = list(parts)
        identity = {'link':info['link'], 'object':info['object']}
        if self.settings['collision_geometry_reuse']:
            for group in self.groups:
                match = translation_match(group['parts'], parts)
                if match is not None:
                    self.bindings.append((info, group, parts, match))
                    self.save('running')
                    return
        while len(self.pending) >= self.workers:
            self.drain()
        # Existing native jobs already occupy memory; reserve only the next job.
        _, memory = available_resources()
        while memory is not None and memory < RESERVE_BYTES + JOB_BYTES and self.pending:
            self.drain()
            _, memory = available_resources()
        if memory is not None and memory < RESERVE_BYTES + JOB_BYTES:
            raise VerifiedExternalError('Host memory pressure paused collision dispatch',
                kind='resource_unavailable', evidence={'available_bytes': memory,
                    'reserve_bytes': RESERVE_BYTES, 'job_bytes': JOB_BYTES})
        group = {'identity':identity, 'parts':parts}
        self.groups.append(group)
        self.bindings.append((info, group, parts, None))
        if self.workers == 1:
            try:
                group['result'] = execute(parts, self.settings, self.cache_root, identity, self.threads)
            except (ValueError, RuntimeError) as error:
                self.failure(group, error)
        else:
            if self.pool is None:
                self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='coacd')
            group['future'] = self.pool.submit(execute, parts, self.settings, self.cache_root, identity, self.threads)
            self.pending.append(group)
            self.peak_pending = max(self.peak_pending, len(self.pending))
        self.save('running')

    def finish(self):
        while self.pending:
            self.drain()
        for info, group, parts, match in self.bindings:
            result = group['result'] if match is None else translated_result(
                group['result'], parts, match[0], group['identity'], match[1])
            refs, proxies, audit = result
            info.update(collision_parts=refs, proxies=proxies, collision_audit=audit)
        self.save('completed')

    def __exit__(self, kind, error, traceback):
        try:
            if kind is None:
                self.finish()
            else:
                self.save('failed', str(error))
        except BaseException as failure:
            self.save('failed', str(failure))
            raise
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)
