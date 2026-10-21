"""Stream videos; retain exact diagnostic pixels separately from lossy delivery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from urdf_core.run_state import file_hash, write_json
from urdf_core.execution import wait_process
from urdf_core.media_execution import run_ffmpeg, progress_monitor

POLICY = 'physics_lossless_stream.v1'
DIRECT_POLICY = 'physics_direct_video_exact_keyframes.v1'


class StreamVideo:
    def __init__(self, output, fps, *, width=640, height=480, ffmpeg='ffmpeg',
                 direct_delivery=False):
        self.output = Path(output).resolve()
        self.width, self.height, self.fps = width, height, fps
        self.policy = DIRECT_POLICY if direct_delivery else POLICY
        self.ffmpeg = shutil.which(str(ffmpeg))
        self.probe = str(Path(self.ffmpeg).with_name('ffprobe')) if self.ffmpeg else None
        if not self.ffmpeg or not Path(self.probe).is_file():
            raise RuntimeError('ffmpeg and ffprobe are required for verified streaming capture')
        if self.output.exists():
            raise ValueError('Cannot overwrite an existing trial video')
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.output.with_suffix('.encode.log').open('wb')
        self.count, self.encode_write_s, self.finalized = 0, 0., False
        codec = (['-c:v', 'libx264', '-crf', '23', '-preset', 'medium', '-pix_fmt', 'yuv420p']
                 if direct_delivery else ['-c:v', 'libx264rgb', '-crf', '0', '-preset', 'veryfast'])
        self.progress_path = self.output.with_suffix('.encode.progress')
        self.progress_path.unlink(missing_ok=True)
        self.process = subprocess.Popen([
            self.ffmpeg, '-progress', str(self.progress_path), '-nostats', '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', f'{width}x{height}',
            '-r', str(fps), '-i', 'pipe:0', '-an', *codec, '-threads', '2',
            '-movflags', '+faststart', str(self.output)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self.finalized:
            self.abort()

    def append(self, pixels):
        pixels = np.asarray(pixels)
        if pixels.dtype != np.uint8 or pixels.shape != (self.height, self.width, 3):
            raise ValueError('Streaming RGB shape or dtype changed')
        raw = pixels.tobytes()
        started = time.monotonic()
        self.process.stdin.write(raw)
        self.encode_write_s += time.monotonic() - started
        self.count += 1
        return hashlib.sha256(raw).hexdigest()

    def abort(self):
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            self.process.stdin.close()
            self.log.close()

    def finish(self):
        if self.finalized or self.count < 2:
            raise ValueError('Streaming video must contain at least two captures and finish once')
        self.process.stdin.close()
        try:
            code = wait_process(self.process, progress=progress_monitor(self.progress_path))
        except BaseException:
            self.abort()
            raise
        finally:
            self.log.close()
        if code or not self.output.is_file() or not self.output.stat().st_size:
            raise RuntimeError('Streaming video encoding failed; see encode log')
        result = subprocess.run([self.probe, '-v', 'error', '-select_streams', 'v:0',
            '-count_frames', '-show_entries', 'stream=nb_read_frames,width,height,codec_name,r_frame_rate',
            '-of', 'json', str(self.output)], capture_output=True, text=True, timeout=120, check=True)
        streams = json.loads(result.stdout)['streams']
        if len(streams) != 1:
            raise ValueError('Video must contain exactly one stream')
        record = streams[0]
        num, den = map(int, record['r_frame_rate'].split('/'))
        if (int(record['nb_read_frames']) != self.count or record['width'] != self.width
                or record['height'] != self.height or not den or abs(num / den - self.fps) > 1e-6):
            raise ValueError('Encoded video does not match the actual capture sequence')
        self.finalized = True
        return {'video_path': str(self.output), 'sha256': file_hash(self.output),
                'probe': record, 'stream_policy': self.policy, 'encoded_frames': self.count,
                'encode_pipe_wall_seconds': self.encode_write_s, 'fps': self.fps}


def save_captured_keyframe(folder, frame, pixels):
    """Save selected original RGB, before any video codec/color conversion."""
    pixels = np.asarray(pixels)
    if (pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3
            or hashlib.sha256(pixels.tobytes()).hexdigest() != frame['rgb_sha256']):
        raise ValueError('Keyframe pixels differ from the captured RGB')
    path = Path(folder) / f"frame_{frame['frame_index']:06d}.png"
    if path.exists():
        raise ValueError('Cannot overwrite a retained frame')
    Image.fromarray(pixels).save(path)
    return dict(frame, path=str(path), sha256=file_hash(path))


def finalize_direct_keyframes(trial, trace, contract):
    """Bind already captured exact PNGs and a once-encoded delivery video."""
    from .capture_retention import keyframe_indices, verify_retained_captures
    frames = trial['frames']
    indices, anchors = keyframe_indices(frames, trace, contract, joint_name=trial.get('joint_name'))
    if [i for i, f in enumerate(frames) if 'path' in f or 'sha256' in f] != indices:
        raise ValueError('Direct capture does not contain exactly the selected keyframes')
    video = Path(trial['video_path']).resolve()
    if (file_hash(video) != trial['video_sha256']
            or int(trial['video_probe']['nb_read_frames']) != len(frames)):
        raise ValueError('Delivery video differs from the complete capture sequence')
    selected = [frames[i] for i in indices]
    metadata = [{k: v for k, v in f.items() if k not in ('path', 'sha256')} for f in frames]
    receipt = {'policy': DIRECT_POLICY, 'captured_frame_count': len(frames),
               'retained_indices': indices, 'all_capture_metadata': metadata,
               'retained_frames': selected, 'numerical_anchors': anchors,
               'trajectory_sha256': trial['trajectory_sha256'],
               'video_sha256': trial['video_sha256'], 'video_probe': trial['video_probe'],
               'full_video_retained': True, 'full_numerical_trajectory_retained': True,
               'diagnostic_pixels_verified_lossless': True,
               'keyframe_source': 'renderer_rgb_before_video_encoding',
               'video_encode_passes': 1, 'delivery_codec': 'h264_yuv420p'}
    path = video.parent / 'capture_retention.json'
    if path.exists():
        raise ValueError('Cannot overwrite a capture receipt')
    write_json(path, receipt)
    trial['frames'] = selected
    trial['capture_retention'] = {'policy': DIRECT_POLICY, 'path': str(path), 'sha256': file_hash(path)}
    # Independently decode only the sparse PNGs, never the video. This verifies
    # exact diagnostic pixels and hashes before publishing the evidence.
    verify_retained_captures(trial, video.parent)
    write_json(video.with_suffix('.frames.json'), {'policy': DIRECT_POLICY,
               'encoded_capture_count': len(frames), 'frames': [f['path'] for f in selected],
               'retention': trial['capture_retention']})
    return receipt


def materialize_keyframes(trial, trace, contract, *, ffmpeg='ffmpeg'):
    """Decode only selected PNGs and prove they equal original renderer pixels."""
    from .capture_retention import keyframe_indices
    frames = trial['frames']
    indices, anchors = keyframe_indices(frames, trace, contract, joint_name=trial.get('joint_name'))
    video = Path(trial['video_path']).resolve()
    expression = '+'.join(f'eq(n\\,{i})' for i in indices)
    command = [str(ffmpeg), '-nostdin', '-v', 'error', '-threads', '2', '-i', str(video),
               '-vf', 'select=' + expression, '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    raw = run_ffmpeg(command, progress_path=video.with_suffix('.decode.progress'), check=True).stdout
    height, width = trial['video_probe']['height'], trial['video_probe']['width']
    stride = height * width * 3
    if len(raw) != len(indices) * stride:
        raise ValueError('Keyframe decode count differs from numerical anchor selection')
    selected = []
    for offset, index in enumerate(indices):
        pixels = raw[offset * stride:(offset + 1) * stride]
        metadata = frames[index]
        if hashlib.sha256(pixels).hexdigest() != metadata['rgb_sha256']:
            raise ValueError('Lossless video pixels differ from measured capture')
        path = video.parent / f'frame_{index:06d}.png'
        if path.exists():
            raise ValueError('Cannot overwrite a retained frame')
        Image.fromarray(np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)).save(path)
        selected.append(dict(metadata, path=str(path), sha256=file_hash(path)))
    # Keep the usual broadly playable, compact H.264 delivery video. The
    # temporary RGB stream exists only until exact diagnostic PNGs are proven.
    lossless_hash = trial['video_sha256']
    compact = video.with_name('motion.delivery.mp4')
    if compact.exists():
        raise ValueError('Cannot overwrite a pending delivery video')
    run_ffmpeg([str(ffmpeg), '-nostdin', '-v', 'error', '-i', str(video),
        '-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p', '-vsync', '0',
        '-movflags', '+faststart', str(compact)], progress_path=compact.with_suffix('.encode.progress'), check=True)
    executable = shutil.which(str(ffmpeg))
    probe = str(Path(executable).with_name('ffprobe'))
    result = subprocess.run([probe, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=nb_read_frames,width,height,codec_name,r_frame_rate', '-of', 'json',
        str(compact)], capture_output=True, text=True, timeout=120, check=True)
    if json.loads(result.stdout).get('streams') != [trial['video_probe']]:
        raise ValueError('Compact delivery video changed capture count, dimensions or cadence')
    compact.replace(video)
    trial['video_sha256'] = file_hash(video)
    receipt = {'policy': POLICY, 'captured_frame_count': len(frames), 'retained_indices': indices,
               'all_capture_metadata': frames, 'retained_frames': selected, 'numerical_anchors': anchors,
               'trajectory_sha256': trial['trajectory_sha256'], 'video_sha256': trial['video_sha256'],
               'video_probe': trial['video_probe'], 'full_video_retained': True,
               'full_numerical_trajectory_retained': True, 'diagnostic_pixels_verified_lossless': True,
               'temporary_lossless_capture_sha256': lossless_hash,
               'delivery_codec': 'h264_yuv420p', 'temporary_lossless_capture_removed': True}
    path = video.parent / 'capture_retention.json'
    if path.exists():
        raise ValueError('Cannot overwrite a capture receipt')
    write_json(path, receipt)
    trial['frames'] = selected
    trial['capture_retention'] = {'policy': POLICY, 'path': str(path), 'sha256': file_hash(path)}
    write_json(video.with_suffix('.frames.json'), {'policy': POLICY, 'encoded_capture_count': len(frames),
               'frames': [f['path'] for f in selected], 'retention': trial['capture_retention']})
    return receipt
