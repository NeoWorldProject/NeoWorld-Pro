"""Measured stationary windows terminate rollouts; acceptance remains separate."""
from collections import deque
import math
import numpy as np
from scipy.spatial.transform import Rotation

from .metrics import _anchored_objects, _contact_owner, stability_root_key

POLICY = 'scene_adaptive_rollout.v2_root_stability'
CHECK_INTERVAL_S = .1


class StationaryWindow:
    def __init__(self, manifest, config):
        self.objects, self.config = manifest['objects'], config
        self.dt, self.window = config['physics_dt_s'], config['settle_window_s']
        self.keys = tuple(f'{oid}::{link}' for oid, obj in self.objects.items() for link in obj['body_paths'])
        self.roots = [self.keys.index(stability_root_key(oid, obj)) for oid, obj in self.objects.items()
                      if not obj.get('is_environment_support')]
        self.owners = {path: oid for oid, obj in self.objects.items() for path in obj['body_paths'].values()}
        self.owner_cache = {}
        self.samples = deque(maxlen=round(self.window/self.dt)+1)
        self.previous = None
        self.next_check = round(self.window/self.dt)+1
        self.count = 0
        self.last = None

    def owner(self, path):
        if path not in self.owner_cache:
            self.owner_cache[path] = _contact_owner(path, self.owners)
        return self.owner_cache[path]

    def add(self, sample):
        t = float(sample['time_s'])
        if (not math.isfinite(t) or (self.previous is None and abs(t) > 1e-8)
                or (self.previous is not None and abs(t-self.previous-self.dt) > max(1e-8, self.dt*.001))):
            raise ValueError('Incomplete adaptive rollout sampling cadence')
        if set(sample.get('body_poses', {})) != set(self.keys):
            raise ValueError('Incomplete adaptive body observation')
        poses = np.asarray([sample['body_poses'][k] for k in self.keys], dtype=float)
        velocities = sample.get('body_velocities', {})
        if set(velocities) != set(self.keys):
            raise ValueError('Incomplete native velocity observation')
        v = np.asarray([[*velocities[k]['linear_m_s'], *velocities[k]['angular_rad_s']] for k in self.keys])
        if poses.shape != (len(self.keys),4,4):
            raise ValueError('Invalid adaptive body transform shape')
        r = poses[:, :3, :3]
        if (poses.shape != (len(self.keys),4,4) or not np.isfinite(poses).all()
                or not np.allclose(poses[:,3,:], [0,0,0,1], atol=1e-6)
                or not np.allclose(r.transpose(0,2,1) @ r, np.eye(3), atol=1e-4)
                or np.any(np.linalg.det(r) < .999) or v.shape != (len(self.keys),6) or not np.isfinite(v).all()):
            raise ValueError('Invalid adaptive native observation')
        contact = sample.get('contacts', {})
        if contact.get('observed') is not True:
            raise ValueError('Missing native contact observation')
        pairs = set()
        for pair in contact.get('pairs', []):
            ids = [self.owner(pair.get(f'body{i}', '')) or self.owner(pair.get(f'collider{i}', '')) for i in (0,1)]
            if not all(ids): raise ValueError('Unbound adaptive contact')
            if ids[0] != ids[1]: pairs.add(frozenset(ids))
        attachments = sample.get('attachments', {})
        anchored = _anchored_objects(self.objects, pairs, attachments)
        supported = all(oid in anchored for oid, obj in self.objects.items() if not obj.get('is_environment_support'))
        for oid, obj in self.objects.items():
            if obj.get('relation') == 'attached':
                a = attachments.get(oid, {})
                limit = max(self.config['position_tolerance_min_m'], self.config['position_tolerance_fraction']*obj['initial_diagonal_m'])
                supported &= (a.get('observed') is True and 0 <= a.get('position_error_m', math.inf) <= limit
                              and 0 <= a.get('rotation_error_deg', math.inf) <= self.config['orientation_tolerance_deg'])
        self.previous = t
        self.count += 1
        if t > 0:
            self.samples.append((t, poses[self.roots].copy(), supported))
        index = round(t/self.dt)
        if index < self.next_check: return None
        self.next_check = index + max(1, round(CHECK_INTERVAL_S/self.dt))
        if len(self.samples) < 2 or self.samples[-1][0]-self.samples[0][0] < self.window-1e-8: return None
        data = np.asarray([s[1] for s in self.samples])
        if self.roots:
            translation = float(np.linalg.norm(data[:,:,:3,3]-data[0,:,:3,3], axis=-1).max())
            relative = data[0,:,:3,:3].transpose(0,2,1)[None] @ data[:,:,:3,:3]
            angle = float(np.degrees(Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude()).max())
        else: translation = angle = 0.
        self.last = {'start_s':self.samples[0][0], 'end_s':t, 'samples':len(self.samples),
                     'dynamic_root_count':len(self.roots), 'supported':all(s[2] for s in self.samples),
                     'max_translation_m':translation, 'max_rotation_deg':angle}
        return self.last if (self.last['supported'] and translation <= self.config['early_stop_translation_m']
                             and angle <= self.config['early_stop_rotation_deg']) else None

    def receipt(self, reason, duration):
        return {'policy':POLICY, 'reason':reason, 'actual_duration_s':duration,
                'max_duration_s':self.config['rollout_max_duration_s'], 'observation_samples':self.count,
                'window':self.last, 'window_s':self.window,
                'translation_limit_m':self.config['early_stop_translation_m'],
                'rotation_limit_deg':self.config['early_stop_rotation_deg']}


def valid_stop(stop, config):
    try:
        duration, cap, dt = stop['actual_duration_s'], config['rollout_max_duration_s'], config['physics_dt_s']
        if (stop['policy'] != POLICY or stop['max_duration_s'] != cap or not math.isfinite(duration)
                or duration < config['settle_window_s'] or duration > cap+1e-8
                or stop['observation_samples'] != round(duration/dt)+1
                or abs(duration/dt-round(duration/dt)) > 1e-6
                or stop['window_s'] != config['settle_window_s']
                or stop['translation_limit_m'] != config['early_stop_translation_m']
                or stop['rotation_limit_deg'] != config['early_stop_rotation_deg']): return False
        if stop['reason'] == 'max_duration': return abs(duration-cap) < 1e-8
        if stop['reason'] != 'stationary': return False
        w = stop['window']
        return (w['supported'] is True and w['start_s'] > 0 and abs(w['end_s']-duration) < 1e-8
                and abs(w['end_s']-w['start_s']-config['settle_window_s']) < 1e-8
                and w['samples'] == round(config['settle_window_s']/dt)+1
                and w['dynamic_root_count'] >= 0
                and 0 <= w['max_translation_m'] <= config['early_stop_translation_m']
                and 0 <= w['max_rotation_deg'] <= config['early_stop_rotation_deg'])
    except (KeyError, ValueError, TypeError): return False
