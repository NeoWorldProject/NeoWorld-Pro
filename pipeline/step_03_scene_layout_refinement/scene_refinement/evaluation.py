"""Evaluate a native candidate before dispatching the next search request."""
from copy import deepcopy
from pathlib import Path
import shutil
import threading
import time

from .protocol import read_json, write_json, digest, file_hash
from .physics import acceptance_ready, observation_ready
from .worker_client import valid_cached_evidence, valid_asset_rejection


class EvaluationPipeline:
    def __init__(self, scene, worker, config, root, fingerprint):
        self.scene, self.worker, self.config = scene, worker, config
        self.root, self.fingerprint = Path(root), fingerprint
        self.worker_lock = threading.Lock()

    def folder(self, parameters):
        return self.root/'candidates'/digest({'scene':self.fingerprint, 'parameters':parameters})

    def cached(self, parameters):
        path = self.folder(parameters)/'evaluation.json'
        if not path.exists():
            return None
        receipt = read_json(path)
        if receipt['fingerprint'] != self.fingerprint or any(file_hash(p) != h for p,h in receipt['evidence_hashes'].items()):
            raise RuntimeError('Candidate native evidence changed')
        return receipt['candidate']

    def evaluate(self, parameters, witness=None):
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
            raise InfrastructureError('Less than 700 MiB free; run evidence retained. No history deleted.')
        job_path = folder/'native_job.json'
        if job_path.exists():
            job = read_json(job_path)
            if job['fingerprint'] != self.fingerprint or any(file_hash(p) != h for p,h in job['evidence_hashes'].items()):
                raise InfrastructureError('Native job binding/evidence changed')
            candidate, physics = job['candidate'], read_json(job['native_path'])
            composition_seconds, evidence = job['composition_seconds'], dict(job['evidence_hashes'])
        else:
            start = time.monotonic()
            try:
                candidate = build_candidate(self.scene, parameters, folder/'constructed')
            except SupportUnresolved as exc:
                if isinstance(exc, AssetGeometryIncompatible):
                    raise
                return {'candidate_id':folder.name, 'parameters':parameters,
                        'physics':{'status':'invalid_candidate', 'passed':False, 'complete':True,
                                   'violation':1e6, 'physical_residual':1e6, 'errors':[str(exc)]}}
            candidate['parameters'] = deepcopy(parameters)
            composition_seconds = time.monotonic()-start
            with self.worker_lock:
                physics = self.worker.evaluate(dict(candidate, pruning_witness=deepcopy(witness)),
                    self.config['search_duration_s'], out_dir=folder/'native', mode='physics')
            if physics['status'] not in ('passed', 'physics_failed', 'certified_pruned', 'invalid_candidate', 'asset_rejected'):
                raise InfrastructureError(str(physics.get('errors', physics['status'])))
            if physics['status'] == 'asset_rejected' and not valid_asset_rejection(physics):
                raise InfrastructureError('Malformed native asset rejection')
            if physics['status'] in ('passed', 'physics_failed') and (
                    not observation_ready(physics) or
                    not valid_cached_evidence(physics, self.config['search_duration_s'], self.config, require_images=False)):
                raise InfrastructureError('Incomplete native measurements or saved evidence')
            native_path = folder/'native/physics_report.json'
            evidence = {str(native_path):file_hash(native_path)}
            for field in ('usd_path', 'scene_manifest_path', 'layout_path'):
                evidence[candidate[field]] = file_hash(candidate[field])
            for field in ('trace', 'cooked_geometry', 'initial_state_receipt', 'terminal_state'):
                if field in physics:
                    evidence[physics[field]['path']] = physics[field]['sha256']
            write_json(job_path, {'fingerprint':self.fingerprint, 'candidate':candidate,
                'native_path':str(native_path), 'composition_seconds':composition_seconds,
                'evidence_hashes':evidence})
        constructed = {key:candidate[key] for key in ('candidate_id', 'usd_path', 'layout_path', 'scene_manifest_path')}
        if acceptance_ready(physics):
            settled = physics.get('settled_candidate')
            if not settled or any(file_hash(p) != h for p,h in settled['settled_artifacts'].items()):
                raise InfrastructureError('Native settled state evidence is missing or changed')
            candidate.update(settled)
            evidence.update(settled['settled_artifacts'])
        candidate.update(physics=physics, images=[], constructed_candidate=constructed,
                         timings={'composition_seconds':composition_seconds, **physics.get('timings', {})})
        write_json(folder/'evaluation.json', {'fingerprint':self.fingerprint,
            'candidate':candidate, 'evidence_hashes':evidence})
        return candidate

    def population(self, missing, witness, stage):
        for index, parameters in missing:
            proof = witness()
            stage(index, {'stage':'native_running', 'witness':proof})
            candidate = self.evaluate(parameters, proof)
            stage(index, {'stage':'native_saved', 'witness':proof,
                          'native_job':str(self.folder(parameters)/'native_job.json')})
            yield index, candidate

    def close(self):
        pass
