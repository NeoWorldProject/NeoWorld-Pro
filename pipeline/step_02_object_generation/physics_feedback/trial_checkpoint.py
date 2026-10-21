"""Atomic completed-trial checkpoints for one exact native candidate.

An absent checkpoint means remeasure. A corrupted committed checkpoint is an
error, never a pass. Partial trajectories, other candidates and other runtime
versions cannot satisfy a trial. This module owns no retry/time budget.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
from collections.abc import Sequence

from urdf_core.run_state import file_hash, write_json
from .trajectory_io import iter_trajectory_samples

POLICY = 'exact_candidate_completed_trials.v1'
REPORT_MAPS = ('trial_work_budgets', 'controller_selections', 'post_operation_observation')


class RecordedTrace(Sequence):
    """Read already verified samples only when a downstream pose needs them."""
    def __init__(self, path):
        self.path, self.samples = path, None

    def _samples(self):
        if self.samples is None:
            self.samples = list(iter_trajectory_samples(self.path))
        return self.samples

    def __getitem__(self, key):
        return self._samples()[key]

    def __len__(self):
        return len(self._samples())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def file_bindings(value):
    """Collect the existing file/hash pairs in a numeric trial and its receipts."""
    result = {}
    if isinstance(value, list):
        for item in value:result.update(file_bindings(item))
    elif isinstance(value, dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            result[value['path']] = value['sha256']
        for key, path in value.items():
            if key.endswith('_path') and isinstance(path,str) and isinstance(value.get(key[:-5]+'_sha256'),str):
                result[path] = value[key[:-5]+'_sha256']
            if isinstance(path,(list,dict)):result.update(file_bindings(path))
    return result


def trial_evidence_root(trial, report, *, report_dir, root):
    """Authorize reused artifacts only through the exact completed-candidate cache."""
    report_dir, root = Path(report_dir).resolve(), Path(root).resolve()
    files = file_bindings(trial)
    if all(Path(path).resolve().is_relative_to(report_dir) for path in files):
        return report_dir
    resume = report.get('trial_resume', {})
    key = resume.get('candidate_key')
    if (resume.get('policy') != POLICY or trial.get('id') not in resume.get('reused', [])
            or not isinstance(key, str) or len(key) != 64 or any(c not in '0123456789abcdef' for c in key)):
        raise ValueError('External trial evidence requires an exact candidate checkpoint')
    trajectory = Path(trial.get('trajectory_path', '')).resolve()
    if not trajectory.is_relative_to(root):
        raise ValueError('Checkpoint trajectory is outside the current object run')
    name = digest([trial['kind'], trial.get('joint_name')]) + '.json'
    candidates = [parent / 'trial_checkpoints' / key / name for parent in trajectory.parents
                  if parent.is_relative_to(root)]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError('Reused trial must have one exact completed checkpoint')
    checkpoint = matches[0]
    if not checkpoint.resolve().is_relative_to(root):
        raise ValueError('Completed checkpoint is outside the current object run')
    saved = json.loads(checkpoint.read_text())
    binding = saved.get('binding', {})
    if (binding.get('policy') != POLICY or binding.get('inputs') != report['inputs']
            or binding.get('source_hashes') != report['source_hashes'] or digest(binding) != key
            or binding.get('measurement_mode') not in {'optimized', 'reference'}
            or binding.get('capture_mode') not in {'deferred', 'live'}):
        raise ValueError('Reused trial checkpoint belongs to another input, source or measurement mode')
    payload = saved.get('payload', {})
    if saved.get('payload_sha256') != digest(payload):
        raise ValueError('Reused trial checkpoint content changed')
    original = payload.get('trial', {})
    from .deferred_capture import MEDIA_FIELDS
    mutable_media = {*MEDIA_FIELDS, 'media_policy'}
    stable = lambda value: {k: v for k, v in value.items() if k not in mutable_media}
    if (stable(original) != stable(trial) or type(original.get('passed')) is not bool
            or original.get('terminated_early')
            or type(original.get('expected_physics_steps')) is not int or original['expected_physics_steps'] <= 0
            or original.get('completed_physics_steps') != original['expected_physics_steps']):
        raise ValueError('Reused trial differs from its completed numerical checkpoint')
    originals = payload.get('files', {})
    if originals.get(str(trajectory)) != trial.get('trajectory_sha256'):
        raise ValueError('Checkpoint does not bind the reused numerical trajectory')
    for path, expected in originals.items():
        path = Path(path).resolve()
        if not path.is_relative_to(root) or file_hash(path) != expected:
            raise ValueError('Completed trial artifact changed or left the object run: ' + str(path))
    for path, expected in files.items():
        path = Path(path).resolve()
        local_media = path.is_relative_to(trajectory.parent) or path.is_relative_to(report_dir)
        if (not path.is_relative_to(root) or (not local_media and originals.get(str(path)) != expected)
                or file_hash(path) != expected):
            raise ValueError('Reused trial references an unauthorized artifact: ' + str(path))
    return root


class TrialCheckpoint:
    def __init__(self, root, report, *, measurement_mode='optimized', capture_mode='deferred'):
        self.root=Path(root).resolve()
        self.binding={'policy':POLICY,'inputs':report['inputs'],'source_hashes':report['source_hashes'],
                      'measurement_mode':measurement_mode,'capture_mode':capture_mode}
        for item in self.binding['inputs'].values():
            if not item['sha256'] or file_hash(item['path'])!=item['sha256']:raise ValueError('Checkpoint input changed')
        manifest=self.binding['inputs'].get('asset_manifest')
        self.assets=file_bindings(json.loads(Path(manifest['path']).read_text())) if manifest else {}
        for path,expected in self.assets.items():
            if file_hash(path)!=expected:raise ValueError('Checkpoint mesh asset changed')
        self.key=digest(self.binding);self.folder=self.root/'trial_checkpoints'/self.key
        self.folder.mkdir(parents=True,exist_ok=True)

    def path(self,kind,name):
        # Hash the identity; never let a supplied joint name escape the directory.
        return self.folder/(digest([kind,name])+'.json')

    def peek(self,kind,name):
        path=self.path(kind,name)
        if not path.exists():return None
        saved=json.loads(path.read_text())
        if saved.get('binding')!=self.binding or saved.get('payload_sha256')!=digest(saved.get('payload')):
            raise ValueError('Completed trial checkpoint binding/content changed')
        payload=saved['payload'];trial=payload['trial']
        if (trial['kind'],trial.get('joint_name'))!=(kind,name):raise ValueError('Checkpoint trial identity mismatch')
        if type(trial.get('passed')) is not bool or trial.get('terminated_early'):
            raise ValueError('Incomplete trial checkpoint')
        if (type(trial.get('expected_physics_steps')) is not int or trial['expected_physics_steps'] <= 0
                or trial.get('completed_physics_steps') != trial['expected_physics_steps']):
            raise ValueError('Completed checkpoint requires every scheduled sample')
        for name, expected in payload['files'].items():
            if file_hash(name)!=expected:raise ValueError('Completed trial artifact changed: '+name)
        return copy.deepcopy(payload)

    def load(self,kind,name,runtime):
        payload=self.peek(kind,name)
        if payload is None:return None
        trial=payload['trial']
        trace=RecordedTrace(trial['trajectory_path'])
        from .verification import validate_trial_metrics
        from .verification_receipt import find_verified_metrics
        scoped = bool(runtime.contract.get('motion_scope_policy') and kind in {'fixed_joint','free_joint'})
        reviewed = find_verified_metrics(trial['trajectory_path'], root=self.root,
            kind=kind, joint_name=name, contract=runtime.contract, inputs=runtime.report['inputs'],
            include_initial_sample=scoped)
        numeric, receipt = reviewed if reviewed else (None, {'status': 'pending_owner_review'})
        if scoped and reviewed:
            numeric, _ = numeric
        runtime.report.setdefault('checkpoint_metric_reviews', []).append(dict(receipt, trial_id=trial['id']))
        if reviewed and trial.get('incidental_motion_review'):
            from .incidental_motion import verify_receipt,apply_decision,pending_names
            if 'incidental_motion' not in runtime.report['inputs'] or not pending_names(numeric,kind):
                raise ValueError('Unauthorized checkpoint incidental-motion decision')
            decision,_=verify_receipt(trial['incidental_motion_review'],raw=numeric,
                trial=trial,report=runtime.report,root=self.root)
            numeric=apply_decision(numeric,kind,decision)
        if reviewed:
            validate_trial_metrics(trial,numeric,runtime.contract)
        delivery=payload.get('delivery')
        if delivery:
            target=runtime.delivered
            if target.exists() and file_hash(target)!=delivery['sha256']:
                raise ValueError('Checkpoint delivered USD conflicts with current candidate')
            if not target.exists():shutil.copy2(delivery['path'],target)
            runtime.delivered_written=True
            runtime.report.update(delivered_usd_path=str(target),delivered_usd_sha256=delivery['sha256'])
        for key, values in payload['report_maps'].items():runtime.report.setdefault(key,{}).update(values)
        runtime.report.setdefault('trial_resume',{'policy':POLICY,'candidate_key':self.key,'reused':[]})['reused'].append(trial['id'])
        return trial,trace

    def restore_delivery(self,runtime):
        path=self.folder/'delivery.json'
        if not path.exists():return False
        saved=json.loads(path.read_text())
        if saved['binding']!=self.binding or file_hash(saved['path'])!=saved['sha256']:
            raise ValueError('Checkpoint delivery binding changed')
        if runtime.delivered.exists() and file_hash(runtime.delivered)!=saved['sha256']:
            raise ValueError('Different exported asset in the same candidate')
        if not runtime.delivered.exists():shutil.copy2(saved['path'],runtime.delivered)
        runtime.delivered_written=True
        runtime.report.update(delivered_usd_path=str(runtime.delivered),delivered_usd_sha256=saved['sha256'])
        return True

    def save_delivery(self,runtime):
        path=self.folder/'delivery.json'
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['binding']!=self.binding or file_hash(runtime.delivered)!=saved['sha256']:
                raise ValueError('A candidate exported different native assets')
            return
        write_json(path,{'binding':self.binding,'path':str(runtime.delivered),'sha256':file_hash(runtime.delivered)})

    def commit(self,runtime,trial):
        if type(trial.get('passed')) is not bool or trial.get('terminated_early'):return
        if not trial.get('trajectory_path'):return
        if (not trial.get('trajectory_sha256') or not trial.get('expected_physics_steps')
                or trial.get('completed_physics_steps')!=trial['expected_physics_steps']):
            raise ValueError('Only complete full-sample trials can be checkpointed')
        files=file_bindings(trial)
        delivery=None
        if runtime.delivered.is_file():
            self.save_delivery(runtime)
            delivery={'path':str(runtime.delivered),'sha256':file_hash(runtime.delivered)}
            files[delivery['path']]=delivery['sha256']
        for path,h in files.items():
            if file_hash(path)!=h:raise ValueError('Cannot checkpoint changed trial artifact')
        payload={'trial':copy.deepcopy(trial),'files':files,'delivery':delivery,
                 'report_maps':{key:{trial['id']:runtime.report[key][trial['id']]}
                    for key in REPORT_MAPS if trial['id'] in runtime.report.get(key,{})}}
        target=self.path(trial['kind'],trial.get('joint_name'))
        if target.exists():
            previous=self.peek(trial['kind'],trial.get('joint_name'))
            if previous!=payload:raise ValueError('Completed trial cannot be overwritten by another measurement')
            return
        write_json(target,{'binding':self.binding,'payload':payload,'payload_sha256':digest(payload)})


def for_runtime(runtime):
    args=getattr(runtime,'args',None)
    root=getattr(args,'trial_cache_root',None)
    if root is None:return None
    return TrialCheckpoint(root,runtime.report,
        measurement_mode=getattr(args,'measurement_mode','optimized'),
        capture_mode=getattr(args,'capture_mode','deferred'))


def observe(runtime,cache,kind,name=None,**kwargs):
    saved=cache.load(kind,name,runtime) if cache else None
    if saved is not None:return saved
    trial,trace=runtime.trial(kind,name,**kwargs)
    if trial.get('status')=='not_applicable_ground_blocked':return trial,trace
    from .incidental_motion import await_review
    await_review(runtime,trial,trace)
    if cache:cache.commit(runtime,trial)
    return trial,trace
