"""Finalize scene evidence after full native reduction, preserving exact endpoints."""
from pathlib import Path

from .hashing import file_hash
from .protocol import digest

METRIC_KEYS = ('passed', 'complete', 'acceptance_complete', 'status', 'violation',
               'reward_terms', 'issues', 'errors', 'window_passed', 'duration_s')


def retain(report, manifest, config, native_articulations, *, terminal=False):
    if config.get('trajectory_retention', 'compact') == 'full':
        return
    from .trajectory import read_trace
    from .metrics import evaluate_scene_trace
    from .terminal_settle import evaluate_window
    from physics_feedback.verification import _same_metrics
    from physics_feedback.compact_trajectory import build_compact_trajectory, CompactTrajectory
    raw = Path(report['trace']['path'])
    expected = report['trace']['sha256']
    if file_hash(raw) != expected:
        raise ValueError('Native scene trajectory changed before review')
    samples = read_trace(raw)
    reviewed = (evaluate_window(samples, manifest, config, native_articulations) if terminal else
                evaluate_scene_trace(samples, manifest, duration_s=samples[-1]['time_s'],
                                     config=config, native_articulations=native_articulations))
    keys = sorted(reviewed)
    if not _same_metrics({k: report[k] for k in keys}, {k: reviewed[k] for k in keys}):
        raise ValueError('Scene report differs from complete native measurements')
    # Event samples and the complete acceptance tail remain at native frequency.
    cutoff = samples[-1]['time_s'] - config['settle_window_s']
    selected = {i for i, sample in enumerate(samples) if sample['time_s'] >= cutoff}
    for issue in reviewed.get('issues', []):
        for key in ('time_s', 'first_time_s', 'max_time_s'):
            value = issue.get(key)
            if isinstance(value, (int, float)):
                index = min(range(len(samples)), key=lambda i: abs(samples[i]['time_s'] - value))
                selected.update(range(max(0, index-3), min(len(samples), index+4)))
    target = raw.parent / 'trajectory.compact.json'
    build_compact_trajectory(raw, target, verified_metrics=reviewed,
        validated_samples=samples, expected_raw_sha256=expected,
        diagnostic_indices=sorted(selected), fps=config['video_fps'],
        replay_fields=('time_s', 'body_poses'),
        binding={'manifest_sha256': digest(manifest), 'config': config,
                 'native_articulations': native_articulations, 'terminal': terminal},
        extras={'metric_keys': keys})
    saved = CompactTrajectory(target)
    report['trace'] = {'path': str(target), 'sha256': file_hash(target)}
    report['trajectory_storage'] = {'policy': 'verified_summaries_and_native_replay.v1',
        'full_rate_samples_evaluated': len(samples), 'raw_retained': False,
        'manifest_sha256': digest(manifest),
        'dependencies': [{'path': str(p), 'sha256': h} for p, h in saved.files.items()]}
    from .protocol import write_json
    write_json(raw.parent/'completed_measurements.json', report)
    if file_hash(raw) != expected:
        raise ValueError('Native scene trajectory changed during evidence commit')
    raw.unlink()


def validate(report, config=None):
    from physics_feedback.compact_trajectory import is_compact_path, CompactTrajectory
    reference = report['trace']
    if not is_compact_path(reference['path']):
        return []
    saved = CompactTrajectory(reference['path'], expected_sha256=reference['sha256'])
    if saved.binding['manifest_sha256'] != report['trajectory_storage']['manifest_sha256']:
        raise ValueError('Processed scene input manifest changed')
    if 'native_articulations' in report and saved.binding['native_articulations'] != report['native_articulations']:
        raise ValueError('Processed native articulation binding changed')
    if config is not None:
        from .metrics import physics_settings
        if saved.binding['config'] != physics_settings(config):
            raise ValueError('Processed scene physics configuration changed')
    for key in saved.extras['metric_keys']:
        if report.get(key) != saved.verified_metrics[key]:
            raise ValueError('Processed scene metric changed: ' + key)
    if abs(saved.terminal_sample['time_s'] - report['effective_duration_s']) > 1e-8:
        raise ValueError('Processed scene terminal time changed')
    return [{'path': str(p), 'sha256': h} for p, h in saved.files.items()]
