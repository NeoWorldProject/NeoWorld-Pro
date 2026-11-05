"""Reuse measured trials only when their simulation inputs are unchanged."""
import copy
import json
from pathlib import Path

from urdf_core.run_state import file_hash
from .trial_checkpoint import RecordedTrace, digest, file_bindings, REPORT_MAPS

POLICY = 'inactive_schedule_trial_reuse.v1'


def contract_dependency(contract, kind, name):
    value = copy.deepcopy(contract)
    for key in ('revision', 'previous_contract_sha256'):
        value.pop(key, None)
    # The executor schedules only its operated joint. Other gains and all
    # physical properties remain dependencies, including coupled branches.
    for joint, definition in value['joints'].items():
        if kind == 'ground_settle' or joint != name:
            definition.pop('speed_fraction', None)
    return value


def read_bound(binding):
    path = Path(binding['path'])
    if file_hash(path) != binding['sha256']:
        raise ValueError('Trial reuse input changed: ' + str(path))
    return json.loads(path.read_text())


def dependency(report, kind, name):
    inputs = report['inputs']
    required = {'urdf', 'spec', 'profile', 'contract', 'asset_manifest'}
    optional = {'representative_motion', 'incidental_motion', 'trial_order'}
    if not required <= inputs.keys() or inputs.keys() - required - optional:
        return None
    for binding in inputs.values():
        if file_hash(binding['path']) != binding['sha256']:
            raise ValueError('Trial reuse input changed')
    manifest = read_bound(inputs['asset_manifest'])
    for path, expected in file_bindings(manifest).items():
        if file_hash(path) != expected:
            raise ValueError('Trial reuse mesh changed')
    for key in ('profile_path', 'spec_path', 'urdf_path'):
        manifest.pop(key, None)
    policies = {}
    for key in ('representative_motion', 'incidental_motion'):
        if key not in inputs:
            continue
        value = read_bound(inputs[key])
        for field in ('candidate_id', 'contract_sha256'):
            value.pop(field, None)
        for path, expected in file_bindings(value).items():
            if file_hash(path) != expected:
                raise ValueError('Trial reuse policy dependency changed')
        if key == 'incidental_motion':
            if value.get('contract') != inputs['contract']:
                raise ValueError('Incidental policy belongs to another contract')
            value.pop('contract', None)
        def content(v):
            if isinstance(v, dict):
                return {k: content(x) for k, x in v.items() if not (k == 'path' and 'sha256' in v)}
            if isinstance(v, list):
                return [content(x) for x in v]
            return v
        policies[key] = content(value)
    return {'policy': POLICY, 'kind': kind, 'name': name,
            'source_hashes': report['source_hashes'],
            'execution_environment': report.get('execution_environment'),
            'assets': {k: inputs[k]['sha256'] for k in ('urdf', 'spec', 'profile')},
            'manifest': manifest, 'policies': policies,
            'contract': contract_dependency(read_bound(inputs['contract']), kind, name)}


def eligible(trial):
    return (trial.get('kind') in {'ground_settle', 'fixed_joint'}
            and trial.get('passed') is True and not trial.get('terminated_early')
            and not trial.get('incidental_motion_review')
            and type(trial.get('expected_physics_steps')) is int
            and trial['expected_physics_steps'] > 0
            and trial.get('completed_physics_steps') == trial['expected_physics_steps'])


def source_report(runtime, binding):
    cached = getattr(runtime, '_trial_reuse_source', None)
    if cached is not None and cached[0] == binding:
        if file_hash(binding['path']) != binding['sha256']:
            raise ValueError('Trial reuse source report changed')
        return cached[1]
    value = read_bound(binding)
    runtime._trial_reuse_source = (dict(binding), value)
    return value


def load(runtime, kind, name):
    if kind not in {'ground_settle', 'fixed_joint'}:
        return None
    plan_path = Path(runtime.args.contract).parent / 'trial_reuse.json'
    if not plan_path.is_file():
        return None
    plan = json.loads(plan_path.read_text())
    source = source_report(runtime, plan['source_report'])
    if not source.get('evaluation_complete') or source.get('errors'):
        return None
    trial = next((t for t in source['trials'] if (t['kind'], t.get('joint_name')) == (kind, name)), None)
    if trial is None or not eligible(trial):
        return None
    if not source.get('execution_environment') or not runtime.report.get('execution_environment'):
        return None
    before, after = dependency(source, kind, name), dependency(runtime.report, kind, name)
    if before is None or before != after:
        return None
    for path, expected in file_bindings(trial).items():
        if file_hash(path) != expected:
            raise ValueError('Reusable trajectory/evidence changed')
    if kind == 'ground_settle':
        # The next isolated worker reloads this native asset before actuation.
        # Reusing the measurement alone leaves that worker without its USD.
        asset, expected = source.get('delivered_usd_path'), source.get('delivered_usd_sha256')
        if not asset or not expected or file_hash(asset) != expected:
            raise ValueError('Reusable ground asset is missing or changed')
        target = runtime.delivered
        if target.exists() and file_hash(target) != expected:
            raise ValueError('Reusable ground asset conflicts with current candidate')
        if not target.exists():
            from urdf_core.storage import copy_asset
            copy_asset(asset, target)
        runtime.delivered_written = True
        runtime.report.update(delivered_usd_path=str(target), delivered_usd_sha256=expected)
    runtime.report.setdefault('trial_reuse', {})[trial['id']] = {
        'policy': POLICY, 'source_report': plan['source_report'],
        'dependency_sha256': digest(after), 'original_trial_sha256': digest(trial)}
    for key in REPORT_MAPS:
        if trial['id'] in source.get(key, {}):
            runtime.report.setdefault(key, {})[trial['id']] = copy.deepcopy(source[key][trial['id']])
    return copy.deepcopy(trial), RecordedTrace(trial['trajectory_path'])


def verify(trial, report, root):
    receipt = report['trial_reuse'][trial['id']]
    root = Path(root).resolve()
    if receipt.get('policy') != POLICY or not Path(receipt['source_report']['path']).resolve().is_relative_to(root):
        raise ValueError('Invalid trial reuse source')
    source = read_bound(receipt['source_report'])
    original = next(t for t in source['trials'] if t['id'] == trial['id'])
    if not eligible(original) or digest(original) != receipt['original_trial_sha256']:
        raise ValueError('Trial reuse source is not a completed passed measurement')
    from .deferred_capture import MEDIA_FIELDS
    mutable = {*MEDIA_FIELDS, 'media_policy'}
    stable = lambda t: {k: v for k, v in t.items() if k not in mutable}
    if stable(trial) != stable(original):
        raise ValueError('Reused native measurements were modified')
    if not source.get('execution_environment') or not report.get('execution_environment'):
        raise ValueError('Trial reuse requires native runtime identity')
    before = dependency(source, trial['kind'], trial.get('joint_name'))
    after = dependency(report, trial['kind'], trial.get('joint_name'))
    if before is None or before != after or digest(after) != receipt['dependency_sha256']:
        raise ValueError('Reused measurement has different simulation dependencies')
    for path, expected in file_bindings(trial).items():
        if not Path(path).resolve().is_relative_to(root) or file_hash(path) != expected:
            raise ValueError('Reused evidence left run or changed')
    return root


def motion_scope_asset_dir(trial, report, root, current_asset_dir):
    """Resolve the original URDF location after validating each reuse link."""
    root = Path(root).resolve()
    seen = set()
    asset_dir = Path(current_asset_dir)
    while trial['id'] in report.get('trial_reuse', {}):
        verify(trial, report, root)
        binding = report['trial_reuse'][trial['id']]['source_report']
        identity = (str(Path(binding['path']).resolve()), binding['sha256'])
        if identity in seen:
            raise ValueError('Cyclic trial reuse provenance')
        seen.add(identity)
        report = read_bound(binding)
        trial = next(t for t in report['trials'] if t['id'] == trial['id'])
        urdf = report['inputs']['urdf']
        path = Path(urdf['path']).resolve()
        if not path.is_relative_to(root) or file_hash(path) != urdf['sha256']:
            raise ValueError('Motion scope source URDF changed or left the object run')
        asset_dir = path.parent
    return asset_dir
