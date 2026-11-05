"""Bounded candidate evaluation with controller-owned checkpoint updates."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import shutil
import threading
import time

from .protocol import read_json, write_json, digest, file_hash
from .physics import acceptance_ready, observation_ready
from .worker_client import valid_cached_evidence, valid_asset_rejection
from .candidate_storage import dependencies as candidate_dependencies


_DISPATCH_POLL_INTERVAL_S = 2.0


def _trajectory_evidence(physics, config, *, bound=None):
    from physics_feedback.compact_trajectory import is_compact_path
    if not is_compact_path(physics.get('trace', {}).get('path', '')):
        return {}
    from .trajectory_retention import validate
    references = {item['path']: item['sha256'] for item in validate(physics, config)}
    if bound is not None and any(bound.get(path) != checksum for path, checksum in references.items()):
        raise ValueError('Candidate evidence omits a processed trajectory dependency')
    return references


class EvaluationPipeline:
    def __init__(self, scene, worker, config, root, fingerprint):
        self.scene, self.worker, self.config = scene, worker, config
        self.root, self.fingerprint = Path(root), fingerprint
        self.concurrency = getattr(worker, 'max_concurrency', 1)
        self.worker_lock = threading.Lock() if self.concurrency == 1 else nullcontext()
        self.composition_lock = threading.Lock()
        self.cancel_pending = threading.Event()

    def folder(self, parameters):
        return self.root/'candidates'/digest({'scene':self.fingerprint, 'parameters':parameters})

    def cached(self, parameters):
        path = self.folder(parameters)/'evaluation.json'
        if not path.exists():
            return None
        receipt = read_json(path)
        if receipt['fingerprint'] != self.fingerprint or any(file_hash(p) != h for p,h in receipt['evidence_hashes'].items()):
            raise RuntimeError('Candidate native evidence changed')
        _trajectory_evidence(receipt['candidate'].get('physics', {}), self.config,
                             bound=receipt['evidence_hashes'])
        return receipt['candidate']

    def evaluate(self, parameters, witness=None):
        recovery = getattr(self, 'recovery', None)
        if recovery is None:
            return self._evaluate(parameters, witness)
        def invalid(error):
            from urdf_core.error_recovery import error_evidence
            folder = self.folder(parameters)
            result = {'candidate_id': folder.name, 'parameters': deepcopy(parameters),
                'physics': {'status': 'invalid_candidate', 'passed': False, 'complete': False,
                    'acceptance_complete': False, 'violation': None, 'physical_residual': None,
                    'reason': 'unclassified_evaluation_error', 'error_context': error_evidence(error)}}
            write_json(folder/'evaluation.json', {'fingerprint': self.fingerprint,
                'candidate': result, 'evidence_hashes': {}})
            return result
        return recovery.run('candidate_evaluation', lambda: self._evaluate(parameters, witness),
            evidence={'input': self.fingerprint, 'parameters': parameters},
            fallback=invalid, fallback_action='skip_candidate', attempts=5)

    def _evaluate(self, parameters, witness=None):
        from .composition import build_candidate
        from .workflow import InfrastructureError
        from .support import SupportUnresolved, AssetGeometryIncompatible
        cached = self.cached(parameters)
        if cached is not None:
            return cached
        folder = self.folder(parameters)
        folder.mkdir(parents=True, exist_ok=True)
        if self.scene.get('unresolved_bindings'):
            # Stage boundaries freeze identities/relations/assets. CEM still
            # records its proposals, but cannot claim this missing contract was
            # verified by native physics. Keep the original assembly for delivery.
            initial = all(not any(values) for values in parameters.values())
            candidate = build_candidate(self.scene, parameters, folder/'constructed') if initial else {'candidate_id': folder.name}
            return {**candidate, 'parameters': deepcopy(parameters),
                'physics': {'status': 'invalid_candidate', 'passed': False, 'complete': True,
                    'violation': 1e6, 'physical_residual': 1e6,
                    'errors': list(self.scene['unresolved_bindings'].values())}}
        if shutil.disk_usage(folder).free < 700 * 1024**2:
            from urdf_core.error_recovery import VerifiedExternalError
            raise VerifiedExternalError('Less than 700 MiB free', kind='storage',
                evidence={'free_bytes': shutil.disk_usage(folder).free})
        job_path = folder/'native_job.json'
        if job_path.exists():
            job = read_json(job_path)
            if job['fingerprint'] != self.fingerprint or any(file_hash(p) != h for p,h in job['evidence_hashes'].items()):
                raise InfrastructureError('Native job binding/evidence changed')
            candidate, physics = job['candidate'], read_json(job['native_path'])
            _trajectory_evidence(physics, self.config, bound=job['evidence_hashes'])
            composition_seconds, evidence = job['composition_seconds'], dict(job['evidence_hashes'])
            composition_queue_seconds = job.get('composition_queue_seconds', 0.)
        else:
            start = time.monotonic()
            try:
                with self.composition_lock:
                    if self.cancel_pending.is_set():
                        from .candidate_workers import CandidateCancelled
                        raise CandidateCancelled('Assembly cancelled before native dispatch')
                    composition_queue_seconds = time.monotonic()-start
                    start = time.monotonic()
                    candidate = build_candidate(self.scene, parameters, folder/'constructed')
            except SupportUnresolved as exc:
                if isinstance(exc, AssetGeometryIncompatible):
                    raise
                return {'candidate_id':folder.name, 'parameters':parameters,
                        'physics':{'status':'invalid_candidate', 'passed':False, 'complete':True,
                                   'violation':1e6, 'physical_residual':1e6, 'errors':[str(exc)]}}
            candidate['parameters'] = deepcopy(parameters)
            composition_seconds = time.monotonic()-start
            manifest = read_json(candidate['scene_manifest_path'])
            clearance = manifest.get('horizontal_clearance')
            if self.scene.get('search_specs') and not clearance:
                raise InfrastructureError('Missing whole-scene static clearance receipt')
            if clearance:
                if file_hash(clearance['path']) != clearance['sha256']:
                    raise InfrastructureError('Static clearance evidence changed')
                checked = read_json(clearance['path'])
                candidate['horizontal_clearance'] = clearance
                if checked['passed']:
                    placed = deepcopy(parameters)
                    for oid, uv in checked['local_uv_corrections_m'].items():
                        for axis in range(2):
                            placed[oid][axis] += uv[axis]
                    candidate['placement_parameters'] = placed
                if not checked['passed']:
                    last = checked['iterations'][-1] if checked.get('iterations') else {}
                    issues = [{'kind':'static_initial_penetration', **hit,
                               'penalty':max(0., hit['depth_m']/self.config['penetration_tolerance_m']-1.)}
                              for hit in last.get('collisions', [])]
                    issues += [{'kind':'static_support_missing', 'object_id':oid, 'penalty':1.}
                               for oid in last.get('unsupported', [])]
                    candidate.update(physics={'status':'invalid_candidate', 'passed':False, 'complete':True,
                        'acceptance_complete':False, 'native_evaluation_performed':False,
                        'violation':max(1., sum(i['penalty'] for i in issues)),
                        'issues':issues, 'errors':[], 'reason':checked['status']}, images=[],
                        timings={'composition_seconds':composition_seconds,
                                 'static_clearance_seconds':checked['seconds']})
                    evidence = {candidate[k]:file_hash(candidate[k]) for k in ('usd_path','scene_manifest_path','layout_path')}
                    evidence.update(candidate_dependencies(candidate['scene_manifest_path']))
                    evidence[clearance['path']] = clearance['sha256']
                    write_json(folder/'evaluation.json', {'fingerprint':self.fingerprint,
                        'candidate':candidate, 'evidence_hashes':evidence})
                    return candidate
            # Resume the original native request, including its pruning proof;
            # newly completed neighbours must not change a pending retry ledger.
            request_path = folder/'native_request.json'
            if request_path.exists():
                saved = read_json(request_path)
                if saved['fingerprint'] != self.fingerprint or saved['candidate'] != candidate:
                    raise InfrastructureError('Pending native request binding changed')
                witness = saved['witness']
            else:
                write_json(request_path, {'fingerprint':self.fingerprint,
                    'candidate':candidate, 'witness':deepcopy(witness)})
            with self.worker_lock:
                evaluate = getattr(self.worker, 'evaluate_cancellable', self.worker.evaluate)
                options = {'cancelled':self.cancel_pending.is_set} if hasattr(self.worker, 'evaluate_cancellable') else {}
                physics = evaluate(dict(candidate, pruning_witness=deepcopy(witness)),
                    self.config['rollout_max_duration_s'], out_dir=folder/'native', mode='physics', **options)
            if physics['status'] not in ('passed', 'physics_failed', 'certified_pruned', 'invalid_candidate', 'asset_rejected'):
                from urdf_core.error_recovery import report_error
                report_error(physics)
            if physics['status'] == 'asset_rejected' and not valid_asset_rejection(physics):
                raise InfrastructureError('Malformed native asset rejection')
            if physics['status'] in ('passed', 'physics_failed') and (
                    not observation_ready(physics) or
                    not valid_cached_evidence(physics, self.config['rollout_max_duration_s'], self.config, require_images=False)):
                raise InfrastructureError('Incomplete native measurements or saved evidence')
            native_path = folder/'native/physics_report.json'
            evidence = {str(native_path):file_hash(native_path)}
            for field in ('usd_path', 'scene_manifest_path', 'layout_path'):
                evidence[candidate[field]] = file_hash(candidate[field])
            evidence.update(candidate_dependencies(candidate['scene_manifest_path']))
            if candidate.get('horizontal_clearance'):
                ref = candidate['horizontal_clearance']
                evidence[ref['path']] = ref['sha256']
            for field in ('trace', 'cooked_geometry', 'initial_state_receipt', 'terminal_state', 'rollout_evidence'):
                if field in physics:
                    evidence[physics[field]['path']] = physics[field]['sha256']
            if physics.get('cooked_geometry'):
                from .geometry_cache import geometry_references
                evidence.update({r['path']:r['sha256'] for r in geometry_references(physics['cooked_geometry']['path'])})
            evidence.update(_trajectory_evidence(physics, self.config))
            write_json(job_path, {'fingerprint':self.fingerprint, 'candidate':candidate,
                'native_path':str(native_path), 'composition_seconds':composition_seconds,
                'composition_queue_seconds':composition_queue_seconds,
                'evidence_hashes':evidence})
        constructed = {key:candidate[key] for key in ('candidate_id', 'usd_path', 'layout_path', 'scene_manifest_path')}
        if acceptance_ready(physics):
            settled = physics.get('settled_candidate')
            if not settled or any(file_hash(p) != h for p,h in settled['settled_artifacts'].items()):
                raise InfrastructureError('Native settled state evidence is missing or changed')
            candidate.update(settled)
            evidence.update(settled['settled_artifacts'])
        candidate.update(physics=physics, images=[], constructed_candidate=constructed,
                         timings={'composition_seconds':composition_seconds,
                                  'composition_queue_seconds':composition_queue_seconds, **physics.get('timings', {})})
        write_json(folder/'evaluation.json', {'fingerprint':self.fingerprint,
            'candidate':candidate, 'evidence_hashes':evidence})
        return candidate

    def population(self, missing, witness, stage):
        if self.concurrency > 1:
            yield from self._parallel_population(missing, witness, stage)
            return
        for index, parameters in missing:
            proof = witness()
            stage(index, {'stage':'native_running', 'witness':proof})
            candidate = self.evaluate(parameters, proof)
            stage(index, {'stage':'native_saved', 'witness':proof,
                          'native_job':str(self.folder(parameters)/'native_job.json')})
            yield index, candidate

    def _parallel_population(self, missing, witness, stage):
        from .candidate_workers import CandidateCancelled
        self.cancel_pending.clear()
        unique = {}
        for index, parameters in missing:
            unique.setdefault(digest(parameters), (index, parameters))
        remaining = iter(unique.values())
        executor = ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix='scene_candidate')
        pending = {}
        def submit():
            while (not self.cancel_pending.is_set()
                   and len(pending) < getattr(self.worker, 'dispatch_capacity', self.concurrency)):
                try:
                    index, parameters = next(remaining)
                except StopIteration:
                    break
                proof = deepcopy(witness())
                stage(index, {'stage':'native_pending', 'witness':proof})
                future = executor.submit(self.evaluate, parameters, proof)
                pending[future] = (index, parameters, proof)
        def saved(future):
            index, parameters, proof = pending.pop(future)
            try:
                result = future.result()
            except Exception as exc:
                stage(index, {'stage':'cancelled_before_native_dispatch'} if isinstance(exc, CandidateCancelled)
                      else {'stage':'technical_failure', 'error':f'{type(exc).__name__}: {exc}'})
                raise
            stage(index, {'stage':'native_saved', 'witness':proof,
                'native_job':str(self.folder(parameters)/'native_job.json')})
            return index, result
        try:
            submit()
            while pending:
                done, _ = wait(pending, timeout=_DISPATCH_POLL_INTERVAL_S, return_when=FIRST_COMPLETED)
                # Prefer potential acceptance when several results arrive together.
                ordered = sorted(done, key=lambda f:(not (not f.exception() and
                    f.result().get('physics', {}).get('passed')), pending[f][0]))
                for future in ordered:
                    result = saved(future)
                    if hasattr(self.worker, 'pause_search'):
                        self.worker.pause_search()
                    try:
                        yield result
                    except GeneratorExit:
                        self.cancel_pending.set()
                        raise
                    finally:
                        if hasattr(self.worker, 'resume_search'):
                            self.worker.resume_search()
                # Include results that finished during serial acceptance before replenishing.
                if not any(future.done() for future in pending):
                    submit()
        finally:
            self.cancel_pending.set()
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            for future, (index, _, _) in list(pending.items()):
                try:
                    if future.cancelled():
                        raise CandidateCancelled()
                    saved(future)
                except CandidateCancelled:
                    stage(index, {'stage':'cancelled_before_native_dispatch'})
                except Exception as exc:
                    stage(index, {'stage':'technical_failure', 'error':f'{type(exc).__name__}: {exc}'})
            self.cancel_pending.clear()

    def close(self):
        pass
