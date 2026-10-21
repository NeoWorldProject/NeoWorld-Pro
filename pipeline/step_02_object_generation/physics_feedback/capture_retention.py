"""Finalize only new, unpublished trial captures after complete video encoding.

Numerical samples, video cadence, pixels and acceptance are unchanged. Historical
reports have no retention binding and continue to use their original selector.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from itertools import chain
import copy
import json
import math
from pathlib import Path

from urdf_core.run_state import file_hash, write_json

POLICY = "physics_keyframes_full_video.v1"


def _base_indices(frames, joint_name):
    if len(frames) < 2:
        raise ValueError("Retention requires at least two captured frames")
    times = [float(f["time_seconds"]) for f in frames]
    if not all(math.isfinite(t) for t in times) or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("Retention capture times must strictly advance")
    if [f["frame_index"] for f in frames] != list(range(len(frames))):
        raise ValueError("Retention only accepts this new trial's complete capture sequence")
    # Keep the same initial/end/extreme/uniform views as the historical critic.
    indices = {0, len(frames) - 1}
    if joint_name:
        q = [float(f["positions"][joint_name]) for f in frames]
        indices.update((min(range(len(q)), key=q.__getitem__), max(range(len(q)), key=q.__getitem__)))
    indices.update(round(i * (len(frames) - 1) / 6) for i in range(7))
    failed = [i for i, f in enumerate(frames) if f.get("geometry_passed") is False]
    if failed:
        indices.update((max(0, failed[0] - 1), failed[0], min(len(frames) - 1, failed[0] + 1), failed[-1]))
    return indices, times


def _with_anchors(indices, times, anchors):
    for value in anchors.values():
        for key in ("first", "max"):
            if value[key] is None:
                continue
            t = value[key + "_time_s"]
            # The native initial state is sampled at t=0 before gravity; the
            # first image is taken after the first physics step. An initial
            # maximum legitimately precedes that image. Keep the first real
            # image rather than fabricating an image at the numerical time.
            if (not isinstance(t, (int, float)) or isinstance(t, bool)
                    or not math.isfinite(t) or not 0. <= t <= times[-1]):
                raise ValueError("Invalid numerical capture anchor time")
            indices.add(max(0, bisect_right(times, t) - 1))
            indices.add(min(len(times) - 1, bisect_left(times, t)))
    return sorted(indices)


def keyframe_indices(frames, trace, contract, *, joint_name=None):
    """Preserve original critic views plus bracketing native anomaly captures."""
    samples = iter(trace)
    try:
        first = next(samples)
    except StopIteration:
        raise ValueError("Retention requires complete captured and numerical evidence")
    indices, times = _base_indices(frames, joint_name)
    anchors = {}
    cfg = contract["config"]
    initial = first["root_position"]
    for sample_index, step in enumerate(chain((first,), samples)):
        signals = {}
        for name, joint in contract.get("joints", {}).items():
            q = float(step["joint_positions"][name])
            target = float(step["joint_targets"][name])
            tolerance = max(1e-8, (joint["upper"] - joint["lower"]) * cfg["endpoint_tolerance_fraction"])
            signals[name + "/tracking"] = abs(q - target) / tolerance
            signals[name + "/effort"] = abs(float(step["joint_efforts"][name])) / joint["max_effort"]
            # Speed remains diagnostic; preserving a picture does not gate it.
            signals[name + "/speed_diagnostic"] = abs(float(step["joint_velocities"][name])) / joint["max_velocity"]
        root = step["root_position"]
        signals["root/displacement"] = math.hypot(root[0] - initial[0], root[1] - initial[1]) / cfg.get("free_slip_tolerance_m", .005)
        if step.get("geometry", {}).get("passed") is False:
            signals["geometry/invalid"] = 2.
        if step.get("contacts", {}).get("observed") is False:
            signals["contact/missing_observation"] = 2.
        for name, value in signals.items():
            if not math.isfinite(value):
                raise ValueError("Retention does not conceal nonfinite numerical signals")
            anchor = anchors.setdefault(name, {"first": None, "max": None, "max_ratio": -1.})
            if value > 1. and anchor["first"] is None:
                anchor["first"] = sample_index
                anchor["first_time_s"] = float(step["time_s"])
            if value > anchor["max_ratio"]:
                anchor.update(max=sample_index, max_ratio=value)
                anchor["max_time_s"] = float(step["time_s"])
    # A 240 Hz event may lie between images: preserve both real neighbors.
    return _with_anchors(indices, times, anchors), anchors


def finalize_trial_captures(trial, trace, contract, *, report_root):
    """Delete only exact new trial-local inputs after verified full encoding."""
    root = Path(report_root).resolve()
    video = Path(trial["video_path"]).resolve()
    folder = video.parent
    if folder != root / "trials" / trial["id"]:
        raise ValueError("Retention cannot touch another trial or historical directory")
    report_path = root / "report.json"
    if report_path.exists():
        published = json.loads(report_path.read_text())
        if any(t.get("id") == trial["id"] for t in published.get("trials", [])):
            raise ValueError("Retention cannot rewrite a previously published trial")
    receipt_path = folder / "capture_retention.json"
    if receipt_path.exists():
        raise ValueError("A finalized retention receipt cannot be overwritten")
    frames = trial["frames"]
    if (not video.is_file() or file_hash(video) != trial["video_sha256"]
            or int(trial["video_probe"].get("nb_read_frames", -1)) != len(frames)):
        raise ValueError("Full video must be successfully encoded and count-verified before pruning")
    paths = []
    for index, frame in enumerate(frames):
        path = Path(frame["path"])
        if (path.is_symlink() or path.resolve() != folder / f"frame_{index:06d}.png"
                or not path.is_file() or file_hash(path) != frame["sha256"]):
            raise ValueError("Retention only removes unchanged, whitelisted new capture files")
        paths.append(path)
    listing = video.with_suffix(".frames.json")
    original_listing = json.loads(listing.read_text())
    if original_listing.get("frames") != [str(p) for p in paths]:
        raise ValueError("Video encoder input list differs from captured frames")
    indices, anchors = keyframe_indices(frames, trace, contract, joint_name=trial.get("joint_name"))
    selected = set(indices)
    receipt = {"policy": POLICY, "captured_frame_count": len(frames), "retained_indices": indices,
               "all_capture_metadata": copy.deepcopy(frames), "numerical_anchors": anchors,
               "trajectory_sha256": trial["trajectory_sha256"], "video_sha256": trial["video_sha256"],
               "video_probe": trial["video_probe"], "encoder_input_list_sha256": file_hash(listing),
               "removed_png_bytes": sum(p.stat().st_size for i, p in enumerate(paths) if i not in selected),
               "full_video_retained": True, "full_numerical_trajectory_retained": True}
    write_json(receipt_path, receipt)
    # The physics-only sidecar now explicitly distinguishes retained PNGs from
    # encoded inputs. The shared generation encoder is unchanged.
    write_json(listing, {"fps": original_listing["fps"], "policy": POLICY,
                        "encoded_capture_count": len(frames), "frames": [str(paths[i]) for i in indices],
                        "retention": {"path": str(receipt_path), "sha256": file_hash(receipt_path)}})
    for i, path in enumerate(paths):
        if i not in selected:
            path.unlink()
    trial["frames"] = [frames[i] for i in indices]
    trial["capture_retention"] = {"policy": POLICY, "path": str(receipt_path), "sha256": file_hash(receipt_path)}
    return receipt


def verify_retained_captures(trial, report_root):
    """Check sparse captures against their immutable full-video input record."""
    binding = trial.get("capture_retention")
    if binding is None:
        return None
    from .stream_video import POLICY as STREAM_POLICY, DIRECT_POLICY
    if binding.get('policy') in (STREAM_POLICY, DIRECT_POLICY):
        return _verify_stream_captures(trial, report_root)
    from .critic import bound_file
    if binding.get("policy") != POLICY:
        raise ValueError("Unknown physics capture retention policy")
    path = bound_file(binding["path"], binding["sha256"], report_root)
    if path != Path(trial["video_path"]).resolve().parent / "capture_retention.json":
        raise ValueError("Retention receipt belongs to another video")
    receipt = json.loads(path.read_text())
    all_frames, indices = receipt["all_capture_metadata"], receipt["retained_indices"]
    base_indices, times = _base_indices(all_frames, trial.get("joint_name"))
    expected_indices = _with_anchors(base_indices, times, receipt["numerical_anchors"])
    if (receipt.get("policy") != POLICY or receipt["captured_frame_count"] != len(all_frames)
            or indices != sorted(set(indices)) or len(indices) < 2
            or indices != expected_indices
            or indices[0] != 0 or indices[-1] != len(all_frames) - 1
            or any(type(i) is not int or not 0 <= i < len(all_frames) for i in indices)
            or trial["frames"] != [all_frames[i] for i in indices]
            or receipt["trajectory_sha256"] != trial["trajectory_sha256"]
            or receipt["video_sha256"] != trial["video_sha256"]
            or receipt["video_probe"] != trial["video_probe"]
            or int(trial["video_probe"]["nb_read_frames"]) != len(all_frames)
            or receipt.get("full_video_retained") is not True
            or receipt.get("full_numerical_trajectory_retained") is not True):
        raise ValueError("Retained capture evidence differs from full video/trajectory")
    return path


def _verify_stream_captures(trial, report_root):
    import hashlib
    import numpy as np
    from PIL import Image
    from .critic import bound_file
    from .stream_video import POLICY as STREAM_POLICY, DIRECT_POLICY
    binding = trial['capture_retention']
    path = bound_file(binding['path'], binding['sha256'], report_root)
    if path != Path(trial['video_path']).resolve().parent / 'capture_retention.json':
        raise ValueError('Stream capture receipt belongs to another trial')
    receipt = json.loads(path.read_text())
    frames = receipt['all_capture_metadata']
    base, times = _base_indices(frames, trial.get('joint_name'))
    indices = _with_anchors(base, times, receipt['numerical_anchors'])
    if (binding['policy'] not in (STREAM_POLICY, DIRECT_POLICY)
            or receipt['policy'] != binding['policy'] or receipt['retained_indices'] != indices
            or receipt['captured_frame_count'] != len(frames)
            or int(trial['video_probe']['nb_read_frames']) != len(frames)
            or receipt['video_probe'] != trial['video_probe']
            or receipt['video_sha256'] != trial['video_sha256']
            or receipt['trajectory_sha256'] != trial['trajectory_sha256']
            or receipt['retained_frames'] != trial['frames']
            or len(trial['frames']) != len(indices)
            or any(receipt.get(key) is not True for key in ('full_video_retained',
                   'full_numerical_trajectory_retained', 'diagnostic_pixels_verified_lossless'))):
        raise ValueError('Streamed captures differ from their numerical/video evidence')
    if binding['policy'] == DIRECT_POLICY and (
            receipt.get('keyframe_source') != 'renderer_rgb_before_video_encoding'
            or receipt.get('video_encode_passes') != 1
            or receipt.get('delivery_codec') != 'h264_yuv420p'):
        raise ValueError('Direct capture must retain original RGB with one delivery encode')
    for index, retained in zip(indices, trial['frames']):
        metadata = frames[index]
        if {k: v for k, v in retained.items() if k not in ('path', 'sha256')} != metadata:
            raise ValueError('Streamed keyframe metadata changed')
        png = bound_file(retained['path'], retained['sha256'], report_root)
        if png != path.parent / f'frame_{index:06d}.png':
            raise ValueError('Streamed keyframe path changed')
        with Image.open(png) as img:
            raw = np.asarray(img.convert('RGB')).tobytes()
        if hashlib.sha256(raw).hexdigest() != metadata['rgb_sha256']:
            raise ValueError('Streamed keyframe pixels changed')
    return path
