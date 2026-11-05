"""Passive native assessment of a deterministic derivative, without model repairs."""
import copy
import json
from pathlib import Path

import numpy as np

from urdf_core.run_state import file_hash, write_json

POLICY = 'limited_dynamic_assessment.v1'


def inputs(folder, root):
    from .protocol import load_config, GROUND_HOLD_ACCEPTANCE_POLICY, CURRENT_GROUND_POLICY, MOTION_SCOPE_POLICY, json_hash
    from .motion_acceptance import freeze
    from .test_modes import FIXED_BASE_POLICY
    folder, root = Path(folder), Path(root)
    read = lambda p: json.loads(p.read_text())
    spec, manifest = read(folder/'kinematics.json'), read(folder/'asset_manifest.json')
    state = read(folder/'physics_profile.json')
    proposal = read(root/'physics_initial_proposal.json') if (root/'physics_initial_proposal.json').is_file() else {}
    previous = read(root/'physics_test_contract.json') if (root/'physics_test_contract.json').is_file() else {}
    retained = read(root/'retained_profile.json') if (root/'retained_profile.json').is_file() else proposal.get('profile', {})
    stable = previous.get('stable_required', retained.get('stable_required', True))
    support = root/'scene_support_applicability.json'
    if support.is_file():
        stable = read(support).get('decision', {}).get('stable_required', stable)
    if type(stable) is not bool:
        raise ValueError('Limited assessment needs explicit stability applicability')
    from urdf_core.dynamic_asset import inspect_dynamic
    dynamic = inspect_dynamic(folder/'object.usd', folder/'asset_manifest.json', folder/'kinematics.json')
    points = []
    for part in dynamic['geometry']['collision_meshes']:
        transform = np.asarray(dynamic['rest'][part['link']])
        points.extend(np.asarray(part['vertices']) @ transform[:3, :3].T + transform[:3, 3])
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError('Invalid dynamic derivative bounds')
    config = load_config()
    if previous.get('spec_sha256') == json_hash(spec):
        from .protocol import validate_current_contract
        try:
            validate_current_contract(previous)
        except ValueError:
            pass
        else:
            config = copy.deepcopy(previous['config'])
    config.update(position_tolerance_m=max(.001,.001*diagonal), release_height_m=0.,
                  free_slip_tolerance_m=max(.005,.02*diagonal))
    profile = {'schema_version': 3, 'stable_required': stable, 'stability_reason': 'Retained applicability or conservative default',
               'collision': {'approximation': 'adaptive'}, 'links': {}, 'joints': {}}
    for link, data in manifest['link_properties'].items():
        profile['links'][link] = {'material': data.get('material', {'static_friction': .5, 'dynamic_friction': .5, 'restitution': 0.})}
    joints = {}
    for joint in spec['joints']:
        if joint['type'] == 'fixed':continue
        name = joint['name']
        q = state['initial_joint_positions'][name]
        profile['joints'][name] = {key: retained.get('joints', {}).get(name, {}).get(key, 0.)
                                  for key in ('static_friction_effort', 'dynamic_friction_effort', 'damping')}
        joints[name] = {'type': joint['type'], 'initial_position': q,
                        'lower': joint.get('limit', {}).get('lower', q-2*np.pi),
                        'upper': joint.get('limit', {}).get('upper', q+2*np.pi)}
    contract = {'schema_version': 3, 'revision': 1, 'assessment_scope': POLICY,
        'spec_sha256': json_hash(spec), 'diagonal_m': diagonal, 'config': config, 'joints': joints,
        'stable_required': stable, 'required_links': [x['name'] for x in spec['links']],
        'ground_policy': CURRENT_GROUND_POLICY, 'acceptance_policy': GROUND_HOLD_ACCEPTANCE_POLICY,
        'penetration_policy': 'diagnostic_only', 'joint_peak_velocity_policy': 'diagnostic_only',
        'external_probe_policy': 'not_required', 'motion_reference_policy': 'pre_motion_stable_root',
        'collision_policy': 'external_and_ground_only', 'motion_scope_policy': MOTION_SCOPE_POLICY,
        'passive_tail_velocity_policy': 'diagnostic_only', 'ground_initial_rotary_tolerance_deg': 10.,
        'orientation_policy': proposal.get('orientation_policy', previous.get('orientation_policy',
            {'upright_required': True, 'reason': 'Conservative default; no classification available'})),
        'free_base_policy': {'required': False, 'reason': 'Active joint functionality is not evaluated in limited assessment'},
        'fixed_base_policy': FIXED_BASE_POLICY}
    orientation = previous.get('scene_orientation')
    context = root/'scene_context.json'
    if orientation is None and context.is_file():
        from .scene_context import load_context
        from .scene_orientation import from_context
        orientation = from_context(load_context(context))
    if orientation is not None:contract['scene_orientation'] = orientation
    freeze(contract, spec)
    return profile, contract


class PassiveExecutor:
    """Zero actuation; no effort budget or target motion is inferred."""
    needs_native_inertia = False
    controller_selection = None
    def __init__(self, contract):
        self.names = list(contract['joints'])
        self.last_targets = {n: j['initial_position'] for n,j in contract['joints'].items()}
    def step(self, *args, **kwargs):
        return {name: 0. for name in self.names}


def evaluate(runtime, report, contract, report_path):
    from .trial_checkpoint import for_runtime, observe
    from .verify_physics_isaac import write_report
    checkpoint = for_runtime(runtime)
    stable = contract['stable_required']
    report.update(assessment_scope=POLICY, acceptance_policy=contract['acceptance_policy'], evaluation_complete=False, acceptance_complete=False,
                  unassessed_checks=['active_joint_functionality'] if contract['joints'] else [], passed=False)
    pose = None
    if stable:
        trial, trace = observe(runtime, checkpoint, 'ground_settle', None)
        report['trials'].append(trial)
        write_report(report_path, report)
        if not trial['passed']:
            report.update(limited_passed=False, limited_evaluation_complete=True, status='physics_failed', stage='complete')
            report['reload_check'] = {'passed': False, 'status': 'skipped_by_ground_failure'}
            return
        pose = runtime.make_rest_candidate(trial, trace)
    elif checkpoint is None or not checkpoint.restore_delivery(runtime):
        runtime.create_trial(fixed=False)
        if checkpoint:checkpoint.save_delivery(runtime)
    trial, trace = observe(runtime, checkpoint, 'reload' if stable else 'reload_supported', None,
                           reload=True, **({'support_pose': pose} if pose else {}))
    report['reload_check'] = {'passed': trial['passed'], 'trial': trial, 'skipped_ground': not stable}
    if stable:
        report['support_rest_pose'].update(verified=trial['passed'], status='verified' if trial['passed'] else 'rejected')
    report.update(limited_passed=trial['passed'], limited_evaluation_complete=True,
                  status='limited_assessed', stage='complete', passed=False,
                  evaluation_complete=False, acceptance_complete=False)


def run(folder, root, *, isaac_python=None, ffmpeg=None):
    from .verify_physics_isaac import parse_args, supervised_run, source_hashes
    folder, root = Path(folder), Path(root)
    target = root/'limited_evaluation'
    target.mkdir(parents=True, exist_ok=True)
    binding = {'asset': {str(p):file_hash(p) for p in folder.rglob('*') if p.is_file()}, 'sources':source_hashes()}
    receipt_path = target/'completion.json'
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt['binding'] != binding or any(file_hash(p)!=h for p,h in receipt['artifacts'].items()):
            raise ValueError('Limited assessment evidence changed')
        return target/'native/report.json'
    profile, contract = inputs(folder,root)
    write_json(target/'profile.json',profile);write_json(target/'contract.json',contract)
    argv = ['--asset-dir',str(folder),'--spec',str(folder/'kinematics.json'),
            '--profile',str(target/'profile.json'),'--contract',str(target/'contract.json'),
            '--out-dir',str(target/'native'),'--evaluation-policy','limited']
    if isaac_python:argv += ['--isaac-python',str(isaac_python)]
    if ffmpeg:argv += ['--ffmpeg',str(ffmpeg)]
    args=parse_args(argv)
    result=supervised_run(args)
    report_path=target/'native/report.json'
    report=json.loads(report_path.read_text())
    if not report.get('limited_evaluation_complete'):
        from urdf_core.error_recovery import VerifiedExternalError
        raise VerifiedExternalError('Limited native assessment unavailable: '+str(report.get('errors')),
            kind='worker_exit', evidence={'report':str(report_path), 'status':report.get('status'),
                                        'returncode':report.get('worker_returncode')})
    artifacts={str(p):file_hash(p) for p in target.rglob('*') if p.is_file()
               and 'runtime' not in p.relative_to(target).parts and p != receipt_path}
    write_json(receipt_path,{'binding':binding,'artifacts':artifacts})
    return report_path


def verify(report_path, *, root, asset, contract):
    report_path, root = Path(report_path), Path(root)
    receipt = json.loads((report_path.parent.parent/'completion.json').read_text())
    report = json.loads(report_path.read_text())
    if (report_path != root/'limited_evaluation/native/report.json'
            or not report.get('limited_evaluation_complete') or report.get('passed')
            or report.get('evaluation_complete') or report.get('acceptance_complete')
            or report.get('assessment_scope') != POLICY):
        raise ValueError('Invalid limited native assessment')
    for path, expected in receipt['artifacts'].items():
        if file_hash(path) != expected:
            raise ValueError('Limited native evidence changed: '+path)
    for path, expected in receipt['binding']['asset'].items():
        if file_hash(path) != expected:
            raise ValueError('Limited asset source changed: '+path)
    if (Path(report['inputs']['contract']['path']).resolve() != Path(contract).resolve()
            or Path(report['inputs']['urdf']['path']).parent.resolve() != Path(asset).resolve()):
        raise ValueError('Limited assessment source differs from publication')
    return report
