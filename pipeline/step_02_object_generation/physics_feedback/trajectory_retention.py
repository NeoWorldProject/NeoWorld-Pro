"""Commit full-rate measurements as verified summaries and exact replay samples."""
from __future__ import annotations

from pathlib import Path
import time

from urdf_core.run_state import file_hash


def compact_path(path):
    return str(path).endswith('.compact.json')


def contract_identity(contract, kind, joint_name):
    from .trial_dependencies import contract_dependency
    from .protocol import json_hash
    return json_hash(contract_dependency(contract, kind, joint_name))


def load_compact(path, *, kind=None, joint_name=None, contract=None):
    from .compact_trajectory import CompactTrajectory
    from .verification_receipt import verifier_identity
    saved = CompactTrajectory(path)
    if contract is not None:
        expected = {'kind': kind, 'joint_name': joint_name,
                    'contract_sha256': contract_identity(contract, kind, joint_name),
                    'verifier_files': verifier_identity()['files']}
        if any(saved.binding.get(k) != v for k, v in expected.items()):
            raise ValueError('Processed trajectory belongs to different inputs or evaluator')
    return saved


class RetainedTrace:
    """Original indices for stored poses; iterating a complete trace is forbidden."""
    def __init__(self, saved):
        self.saved = saved
        self.retention_metadata = saved.extras

    def __len__(self):
        return self.saved.original_sample_count

    def __iter__(self):
        raise ValueError('Full-rate trajectory is not retained; use verified summaries or replay samples')

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index == 0:
            return self.saved.initial_sample
        if index == len(self) - 1:
            return self.saved.terminal_sample
        return self.saved.sample_at(index, kind='replay')


def load_trace(path):
    if compact_path(path):
        return RetainedTrace(load_compact(path))
    from .trajectory_io import iter_trajectory_samples
    return list(iter_trajectory_samples(path))


def compact_trial(runtime, trial, trace):
    """Run the existing reducer on the closed raw file before committing summaries."""
    if (runtime.config.get('trajectory_retention', 'compact') == 'full'
            or getattr(runtime.args, 'capture_mode', 'deferred') != 'deferred'
            or compact_path(trial['trajectory_path'])):
        return None
    from .verification import recompute_trial_metrics, validate_trial_metrics
    from .verification_receipt import verifier_identity
    from .evidence import trajectory_windows
    from .deferred_capture import frame_schedule
    from .capture_retention import keyframe_indices
    from .incidental_motion import pending_names, summarize_events
    from .compact_trajectory import build_compact_trajectory, CompactTrajectory

    started = time.monotonic()
    raw = Path(trial['trajectory_path'])
    expected = trial['trajectory_sha256']
    recomputed, verified_trace = recompute_trial_metrics(raw, kind=trial['kind'],
        joint_name=trial.get('joint_name'), contract=runtime.contract, _return_trace=True)
    validate_trial_metrics(trial, recomputed, runtime.contract)
    windows = (trajectory_windows(trial, runtime.contract, _verified_trace=verified_trace)
               if recomputed['passed'] is not True else
               {'available': False, 'reason': 'passed trial has no failure diagnostic windows',
                'sample_count': len(verified_trace), 'radius_steps': 3, 'samples': {}})
    frames = frame_schedule(verified_trace, runtime.config['video_fps'])
    indices, anchors = keyframe_indices(frames, verified_trace, runtime.contract,
                                       joint_name=trial.get('joint_name'))
    extras = {'diagnostic_windows': windows, 'frame_schedule': frames,
              'keyframe_indices': indices, 'numerical_anchors': anchors,
              'video_fps': runtime.config['video_fps']}
    if pending_names(recomputed, trial['kind']):
        extras['incidental_events'] = summarize_events(verified_trace, recomputed,
            runtime.contract, runtime.spec, trial.get('joint_name'))
    destination = raw.parent / 'trajectory.compact.json'
    build_compact_trajectory(raw, destination, verified_metrics=recomputed, validated_samples=verified_trace,
        diagnostic_indices=[int(i) for i in windows.get('samples', {})],
        replay_indices=[f['sample_index'] for f in frames],
        expected_raw_sha256=expected, fps=runtime.config['video_fps'],
        binding={'kind': trial['kind'], 'joint_name': trial.get('joint_name'),
                 'contract_sha256': contract_identity(runtime.contract, trial['kind'], trial.get('joint_name')),
                 'inputs': runtime.report['inputs'],
                 'verifier_files': verifier_identity()['files'],
                 'verifier_environment': {k: v for k, v in verifier_identity().items() if k != 'files'},
                 'review_scope': 'complete_raw_native_samples_before_retention'}, extras=extras)
    saved = CompactTrajectory(destination)
    if saved.initial_sample != trace[0] or saved.terminal_sample != trace[-1]:
        raise ValueError('Processed trajectory changed initial or terminal state')
    trial.update(trajectory_path=str(destination), trajectory_sha256=file_hash(destination),
                 trajectory_storage={'policy': 'verified_summaries_and_native_replay.v1',
                    'full_rate_samples_evaluated': len(trace), 'raw_retained': False,
                    'dependencies': [{'path': str(p), 'sha256': h} for p, h in saved.files.items()],
                    'raw_content_sha256': expected})
    trial.setdefault('timing_wall_seconds', {})['trajectory_retention'] = time.monotonic() - started
    return raw, expected


def remove_temporary_raw(pending):
    if pending is not None:
        path, expected = pending
        if path.exists():
            if path.is_symlink() or file_hash(path) != expected:
                raise ValueError('Temporary native trajectory changed before cleanup')
            path.unlink()
