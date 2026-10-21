"""Fixed-threshold rewards and certified nonnegative bounds for first acceptance."""
from copy import deepcopy
import math
import numpy as np
from scipy.spatial.transform import Rotation

POLICY = 'physx_native_scene.v2_pose_stability'


def excess(value, limit):
    return max(0., float(value) / limit - 1.)


def observation_ready(report):
    return (not report.get('errors') and report.get('policy') == POLICY
            and report.get('complete') is True and report.get('acceptance_complete') is True
            and report.get('native_observation_complete') is True)


def acceptance_ready(report):
    return report.get('passed') is True and observation_ready(report)


class PrefixBound:
    """Only monotone prefix measurements; never settling, support or drift."""
    def __init__(self, manifest, config):
        self.manifest, self.config = manifest, config
        self.depths, self.angles = {}, {}
        self.owners = {path: oid for oid, o in manifest['objects'].items() for path in o['body_paths'].values()}

    def add(self, sample):
        from .metrics import _contact_owner
        contacts = sample['contacts']
        if contacts.get('observed') is not True:
            raise ValueError('Missing native contact measurement')
        for pair in contacts.get('pairs', []) + contacts.get('events', []):
            ids = [_contact_owner(pair.get(f'body{i}', ''), self.owners) or
                   _contact_owner(pair.get(f'collider{i}', ''), self.owners) for i in (0, 1)]
            if not all(ids):
                raise ValueError('Unbound native contact body/collider')
            if ids[0] == ids[1]:
                continue
            key = tuple(sorted(ids))
            for point in pair.get('points', []):
                depth = -float(point['separation_m'])
                if not math.isfinite(depth):
                    raise ValueError('Nonfinite native depth')
                self.depths[key] = max(self.depths.get(key, 0.), depth)
        for oid, obj in self.manifest['objects'].items():
            if obj.get('is_environment_support'):
                continue
            pose = np.asarray(sample['body_poses'][f"{oid}::{obj['root_link']}"])
            reference = np.asarray(obj.get('orientation_reference_world', pose[:3, :3]))
            angle = float(np.degrees(Rotation.from_matrix(reference.T @ pose[:3, :3]).magnitude()))
            self.angles[oid] = max(self.angles.get(oid, 0.), angle)
        return self.value

    @property
    def value(self):
        return (sum(excess(d, self.config['penetration_tolerance_m']) for d in self.depths.values()) +
                sum(excess(a, self.config['tilt_limit_deg']) for a in self.angles.values()))

    def issues(self):
        return ([{'kind': 'native_contact_penetration', 'object_ids': list(pair), 'depth_m': depth,
                  'penalty': excess(depth, self.config['penetration_tolerance_m'])}
                 for pair, depth in sorted(self.depths.items()) if depth > self.config['penetration_tolerance_m'] + 1e-12] +
                [{'kind': 'major_orientation_failure', 'object_id': oid, 'major_orientation_deg': angle,
                  'penalty': excess(angle, self.config['tilt_limit_deg'])}
                 for oid, angle in sorted(self.angles.items()) if angle > self.config['tilt_limit_deg'] + 1e-6])


def prunable(bound, witness, known_failure=True):
    if not witness or not known_failure or not math.isfinite(bound):
        return False
    rows = witness.get('witnesses', [])
    if len(rows) != witness.get('elite_count', 4) or len({r['key'] for r in rows}) != len(rows):
        return False
    if any(r.get('known_physical_failure') is not True or not math.isfinite(r['violation']) for r in rows):
        return False
    threshold = max(r['violation'] for r in rows)
    return bound > threshold + max(1e-8, abs(threshold) * 1e-10)


def pruned_report(bound, witness, duration, required, issues, stage):
    if not prunable(bound, witness, bool(issues)):
        raise ValueError('Uncertified pruning')
    return {'passed': False, 'complete': False, 'acceptance_complete': False, 'status': 'certified_pruned',
            'policy': POLICY, 'known_infeasible': True, 'trajectory_complete': duration >= required,
            'duration_s': duration, 'required_duration_s': required, 'violation': None,
            'physical_residual': None, 'violation_lower_bound': bound, 'issues': issues, 'errors': [],
            'pruning_certificate': {'stage': stage, 'bound': bound, **deepcopy(witness)}}


def needs_extension(report, config):
    return bool(report.get('native_observation_complete', report.get('complete')) and not report.get('passed')
        and report.get('issues') and all(i['kind'] == 'unstable_or_unsupported' for i in report['issues'])
        and any(any(v['translation_m'] > d['position_limit_m'] or
                    v['rotation_deg'] > config['orientation_tolerance_deg']
                    for v in d['link_settling'].values()) for d in report.get('objects', {}).values()))


def evaluate(trace, manifest, *, duration_s, config, native_articulations=None):
    from .metrics import reduce_native_trace
    base = reduce_native_trace(trace, manifest, duration_s=duration_s, config=config)
    if base.get('status') == 'environment_error':
        return base
    try:
        keys = set(trace[0]['body_poses'])
        joint_names = ({oid: list(art['dof_names']) for oid, art in native_articulations.items()
                        if art['dof_count']} if native_articulations is not None else
                       {oid: list(state['joint_names']) for oid, state in trace[0].get('native_joint_states', {}).items()})
        if any(len(names) != len(set(names)) for names in joint_names.values()):
            raise ValueError('Duplicate native joint names')
        for sample in trace:
            if set(sample.get('body_velocities', {})) != keys:
                raise ValueError('Missing native link velocity observation')
            for value in sample['body_velocities'].values():
                for field in ('linear_m_s', 'angular_rad_s'):
                    vector = np.asarray(value[field], float)
                    if vector.shape != (3,) or not np.isfinite(vector).all():
                        raise ValueError('Invalid native link velocity observation')
            states = sample.get('native_joint_states', {})
            if set(states) != set(joint_names):
                raise ValueError('Missing native articulation measurements')
            for oid, state in states.items():
                names = state['joint_names']
                if names != joint_names[oid]:
                    raise ValueError('Native joint names changed during observation')
                for field in ('positions', 'velocities'):
                    value = np.asarray(state[field], float).reshape(-1)
                    if len(value) != len(names) or not np.isfinite(value).all():
                        raise ValueError('Missing named joint measurements')
        bound = PrefixBound(manifest, config)
        for sample in trace:
            bound.add(sample)
        p = sum(excess(v, config['penetration_tolerance_m']) for v in bound.depths.values())
        issues = [i for i in bound.issues() if i['kind'] == 'native_contact_penetration']
        stability = 0.
        for oid, detail in base['objects'].items():
            links = list(detail['link_settling'].values())
            terms = {
                'major_orientation': excess(max(detail['max_tilt_deg'], detail['max_major_orientation_deg']), config['tilt_limit_deg']),
                'tail_translation': excess(max(v['translation_m'] for v in links), detail['position_limit_m']),
                'tail_rotation': excess(max(v['rotation_deg'] for v in links), config['orientation_tolerance_deg']),
                'target_support': 1. - detail['tail_support_fraction']}
            if manifest.get('state_kind') == 'settled':
                terms['reload_drift'] = excess(detail['max_root_displacement_m'], detail['drift_limit_m'])
            if detail['relation'] == 'attached':
                terms['attachment_position'] = excess(max(float(s['attachments'][oid]['position_error_m']) for s in trace), detail['position_limit_m'])
                terms['attachment_rotation'] = excess(max(float(s['attachments'][oid]['rotation_error_deg']) for s in trace), config['orientation_tolerance_deg'])
            detail['penalties'] = terms
            stability += sum(terms.values())
            if not detail['passed']:
                issues.append({'kind': 'unstable_or_unsupported', 'object_id': oid,
                               'penalty': sum(terms.values()), 'categories': terms})
        base.pop('unresolved_geometry', None)
        base.update(policy=POLICY, complete=True,
                    extra_geometry={'required':False, 'executed':False},
                    native_observation_complete=True, acceptance_complete=True,
                    passed=not issues, status='physics_failed' if issues else 'passed', issues=issues,
                    known_infeasible=bool(issues), violation=p + stability, physical_residual=p + stability,
                    reward_terms={'penetration': -p, 'stable': -stability, 'mllm': None},
                    native_pair_maxima=[{'object_ids': list(k), 'depth_m': v} for k, v in sorted(bound.depths.items())])
        return base
    except (ValueError, TypeError, KeyError) as exc:
        return {'status': 'environment_error', 'passed': False, 'complete': False,
                'acceptance_complete': False, 'errors': [str(exc)]}
