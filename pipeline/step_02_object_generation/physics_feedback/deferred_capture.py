"""Render recorded native link poses, after numerical evaluation, without dynamics.

Replay is labelled and leaves measured states unchanged. Rejected candidates
retain failure keyframes; accepted numeric delivery uses the complete native
trajectories and independent reload evidence without new rendered media.
"""
from __future__ import annotations

import hashlib
import copy
import json
from pathlib import Path
import time
import uuid

import numpy as np

from urdf_core.run_state import file_hash, write_json
from .trajectory_io import iter_trajectory_samples
from .capture_retention import keyframe_indices

POLICY = 'native_numeric_then_recorded_pose_render.v1'
NUMERIC_DELIVERY_POLICY = 'native_numeric_delivery_failure_keyframes.v2'
MEDIA_FIELDS=('frames','video_path','video_sha256','video_probe','capture_policy','recording_mode',
              'recorded_pose_render','timing_wall_seconds','media_timing_wall_seconds','wall_seconds',
              'capture_retention')


def media_checkpoint(runtime,trial,full_video):
    folder=Path(trial['trajectory_path']).parent
    prefix='delivery_media' if full_video else 'diagnostic_media'
    binding={'policy':POLICY,'inputs':runtime.report['inputs'],'sources':runtime.report['source_hashes'],
             'trajectory_sha256':trial['trajectory_sha256'],'scene':trial['replay_scene'],
             'camera':trial['camera'],'full_video':full_video,'fps':runtime.config['video_fps']}
    return folder/(prefix+'.completed.json'),binding,folder/prefix


def restore_media(runtime,trial,path,binding):
    if not path.exists():return False
    from .trial_checkpoint import digest,file_bindings
    saved=json.loads(path.read_text())
    if (saved['binding']!=binding or saved['payload_sha256']!=digest(saved['fields'])
            or set(saved['fields'])-set(MEDIA_FIELDS)):
        raise ValueError('Completed media checkpoint binding changed')
    for name,expected in file_bindings(saved['fields']).items():
        if file_hash(name)!=expected:raise ValueError('Completed media artifact changed: '+name)
    trial.update(copy.deepcopy(saved['fields']))
    runtime.report.setdefault('media_resume',[]).append(trial['id'])
    return True


def save_media(trial,path,binding):
    from .trial_checkpoint import digest,file_bindings
    fields={key:copy.deepcopy(trial[key]) for key in MEDIA_FIELDS if key in trial}
    for name,expected in file_bindings(fields).items():
        if file_hash(name)!=expected:raise ValueError('Cannot commit changed media artifact')
    write_json(path,{'binding':binding,'fields':fields,'payload_sha256':digest(fields)})


def state_hash(sample):
    return hashlib.sha256(json.dumps(sample['link_transforms'], sort_keys=True,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def frame_schedule(trace, fps):
    indices, last = [], -1
    for i, sample in enumerate(trace):
        frame = int(np.floor(float(sample['time_s']) * fps + 1e-8))
        if frame > last or i == len(trace) - 1:
            indices.append(i)
            last = frame
    return [{'frame_index': k, 'sample_index': i, 'time_seconds': trace[i]['time_s'],
             'positions': trace[i]['joint_positions'], 'native_link_state_sha256': state_hash(trace[i]),
             'geometry_passed': trace[i]['geometry']['passed']}
            for k, i in enumerate(indices)]


def save_scene(runtime, trial):
    path = Path(trial['trajectory_path']).parent / 'replay_scene.usdc'
    runtime.stage.Flatten().Export(str(path))
    trial.update(recording_mode='numeric_pending_media', media_policy=POLICY,
                 replay_scene={'path': str(path), 'sha256': file_hash(path),
                               'links': {n: str(p.GetPath()) for n, p in runtime.bodies.items()}})


class RecordedPoseWriter:
    """Trial-local USD handles; change only matrix values on subsequent frames."""
    def __init__(self, stage, bodies):
        from pxr import UsdGeom
        self.prims, self.ops = {}, {}
        for name, path in bodies.items():
            prim = stage.GetPrimAtPath(path)
            if not prim or not UsdGeom.Xformable(prim):
                raise ValueError('Replay link is missing or not transformable')
            xform = UsdGeom.Xformable(prim)
            attr = prim.GetAttribute('xformOp:transform:nativeRecordedPose')
            op = (UsdGeom.XformOp(attr) if attr
                  else xform.AddTransformOp(opSuffix='nativeRecordedPose'))
            xform.SetXformOpOrder([op], resetXformStack=True)
            self.prims[name], self.ops[name] = prim, op
        if not self.prims:
            raise ValueError('Replay must contain measured links')
        self.cache = UsdGeom.XformCache()

    def apply(self, sample):
        from pxr import Gf
        if set(self.ops) != set(sample['link_transforms']):
            raise ValueError('Replay must contain EVERY measured link')
        for name, op in self.ops.items():
            matrix = np.asarray(sample['link_transforms'][name])
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError('Invalid recorded link transform')
            op.Set(Gf.Matrix4d(matrix.T.tolist()))
        # USD XformCache does not invalidate automatically after attribute edits.
        self.cache.Clear()
        error = max(float(np.abs(np.asarray(self.cache.GetLocalToWorldTransform(prim)).T
                                - np.asarray(sample['link_transforms'][name])).max())
                    for name, prim in self.prims.items())
        if not np.isfinite(error) or error > 1e-7:
            raise ValueError('Rendered world transforms differ from native evidence')
        return error


def render_trial(runtime, trial, *, full_video):
    checkpoint,checkpoint_binding,folder=media_checkpoint(runtime,trial,full_video)
    if restore_media(runtime,trial,checkpoint,checkpoint_binding):return
    # A killed encoder must never overwrite either its partial video or PNGs.
    if folder.exists() and any(folder.iterdir()):folder=folder.with_name(folder.name+'.attempt_'+uuid.uuid4().hex)
    import carb.settings
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep
    from pxr import UsdPhysics, PhysxSchema
    from .physics_runtime import capture_rgb, configure_fractional_opacity
    from .stream_video import StreamVideo, save_captured_keyframe, finalize_direct_keyframes
    from contextlib import nullcontext

    started = time.monotonic()
    timings = {name: 0. for name in ('setup', 'pose_apply_and_verify', 'capture_rgb',
                                    'png_write', 'video_pipe', 'video_finalize', 'keyframe_receipt')}
    trace = list(iter_trajectory_samples(trial['trajectory_path']))
    if file_hash(trial['trajectory_path']) != trial['trajectory_sha256']:
        raise ValueError('Numerical trajectory changed before replay')
    frames = frame_schedule(trace, runtime.config['video_fps'])
    selected, anchors = keyframe_indices(frames, trace, runtime.contract, joint_name=trial.get('joint_name'))
    chosen = list(range(len(frames))) if full_video else selected
    selected_set = set(selected)
    runtime.close_trial()
    omni.timeline.get_timeline_interface().stop()
    binding = trial['replay_scene']
    if file_hash(binding['path']) != binding['sha256']:
        raise ValueError('Replay scene changed')
    if not omni.usd.get_context().open_stage(binding['path']):
        raise RuntimeError('Cannot load recorded trial scene')
    stage = omni.usd.get_context().get_stage()
    # A detached stage only. Never modify the delivery asset or run physics here.
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.Scene):
            stage.RemovePrim(prim.GetPath())
            continue
        for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.ArticulationRootAPI,
                    PhysxSchema.PhysxArticulationAPI, PhysxSchema.PhysxRigidBodyAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
    config = trial['camera']
    camera = rep.create.camera(position=config['position'], look_at=config['look_at'],
        clipping_range=config['clipping_range'], focal_length=config['focal_length'],
        horizontal_aperture=config['horizontal_aperture'])
    product = rep.create.render_product(camera, tuple(config['resolution']))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach([product])
    runtime.capture.keep(rgb, product)
    settings = carb.settings.get_settings(); configure_fractional_opacity(settings)
    folder.mkdir(parents=True,exist_ok=True)
    pose_writer = RecordedPoseWriter(stage, binding['links'])
    recording = StreamVideo(folder / 'motion.mp4', runtime.config['video_fps'],
                            ffmpeg=runtime.args.ffmpeg, direct_delivery=True) if full_video else nullcontext(None)
    captured, errors = [], []
    timings['setup'] = time.monotonic() - started
    with recording as encoder:
        for index in chosen:
            frame = dict(frames[index]); sample = trace[frame['sample_index']]
            timer = time.monotonic()
            errors.append(pose_writer.apply(sample))
            timings['pose_apply_and_verify'] += time.monotonic() - timer
            timer = time.monotonic()
            pixels = np.ascontiguousarray(capture_rgb(rep.orchestrator, rgb, settings=settings,
                                                     rt_subframes=4 if not captured else 1)[:, :, :3])
            frame['rgb_sha256'] = hashlib.sha256(pixels.tobytes()).hexdigest()
            timings['capture_rgb'] += time.monotonic() - timer
            if index in selected_set:
                timer = time.monotonic()
                frame = save_captured_keyframe(folder, frame, pixels)
                timings['png_write'] += time.monotonic() - timer
            if encoder is not None:
                timer = time.monotonic()
                if encoder.append(pixels) != frame['rgb_sha256']:
                    raise ValueError('Replay encoding changed pixels')
                timings['video_pipe'] += time.monotonic() - timer
            captured.append(frame)
        if encoder is not None:
            timer = time.monotonic()
            video = encoder.finish()
            trial.update(video_path=video['video_path'], video_sha256=video['sha256'], video_probe=video['probe'])
            timings['video_finalize'] += time.monotonic() - timer
    runtime.capture.suspend(product)
    trial['frames'] = captured
    if full_video:
        timer = time.monotonic()
        finalize_direct_keyframes(trial, trace, runtime.contract)
        timings['keyframe_receipt'] += time.monotonic() - timer
    trial['capture_policy']['capture_codec'] = 'original_rgb_selected_pngs'
    trial['capture_policy']['video_encode_passes'] = 1 if full_video else 0
    receipt = {'policy': POLICY, 'mode': 'full_video' if full_video else 'failure_keyframes',
               'trajectory_sha256': trial['trajectory_sha256'], 'scene': binding,
               'selected_indices': chosen, 'frame_schedule': frames, 'numerical_anchors': anchors,
               'max_world_transform_error': max(errors), 'physics_steps_during_render': 0,
               'render_source': 'all_measured_link_world_transforms', 'camera': config,
               'retained_frames': trial['frames']}
    path = folder / 'recorded_pose_render.json'; write_json(path, receipt)
    trial.update(recording_mode=receipt['mode'], recorded_pose_render={'path': str(path), 'sha256': file_hash(path)})
    trial.setdefault('timing_wall_seconds', {})['deferred_render'] = time.monotonic() - started
    trial['media_timing_wall_seconds'] = timings
    trial['wall_seconds'] += time.monotonic() - started
    save_media(trial,checkpoint,checkpoint_binding)


def finalize_evidence(runtime, report):
    report['media_policy'] = POLICY
    complete = report.get('passed') is True and report.get('acceptance_complete') is True
    trials = [*report['trials']]
    if report.get('reload_check', {}).get('trial'):
        trials.append(report['reload_check']['trial'])
    if complete:
        # Numerical acceptance and the independently reloaded USD are sufficient.
        # Incidental-motion review owns its separately bound diagnostic images.
        if (report.get('evaluation_complete') is not True
                or report.get('reload_check', {}).get('passed') is not True
                or any(t.get('passed') is not True for t in trials)):
            raise ValueError('Numeric delivery requires complete passing measurements and reload')
        report['media_policy'] = NUMERIC_DELIVERY_POLICY
        for trial in trials:
            for key in ('video_path', 'video_sha256', 'video_probe', 'recorded_pose_render',
                        'capture_retention'):
                trial.pop(key, None)
            trial.update(media_policy=NUMERIC_DELIVERY_POLICY,
                         recording_mode='numeric_delivery', frames=[])
        report['stage'] = 'complete'
        report.pop('active_trial', None)
        return
    for trial in trials:
        if complete or trial.get('passed') is not True:
            report.update(stage='deferred_render', active_trial=trial['id'])
            write_json(runtime.args.out_dir / 'report.json', report)
            if complete or trial.get('recording_mode') != 'failure_keyframes':
                render_trial(runtime, trial, full_video=complete)
        else:
            if not trial.get('incidental_motion_review'):
                trial['recording_mode'] = 'numeric_only'
    report['stage'] = 'complete'


def verify_media(trial, report, contract, report_root):
    """No missing video exemption without explicit policy and complete raw data."""
    from .critic import bound_file
    if report.get('media_policy') == NUMERIC_DELIVERY_POLICY:
        if (trial.get('media_policy') != NUMERIC_DELIVERY_POLICY
                or report.get('passed') is not True
                or report.get('evaluation_complete') is not True
                or report.get('acceptance_complete') is not True
                or report.get('reload_check', {}).get('passed') is not True
                or trial.get('passed') is not True
                or trial.get('recording_mode') != 'numeric_delivery'
                or trial.get('frames') or trial.get('video_path')):
            raise ValueError('Invalid complete numeric-only delivery evidence')
        # Full raw trajectories, exact trial coverage and reload are recomputed
        # by verify_simulation_report; this branch exempts only rendered media.
        return set()
    if report.get('media_policy') != POLICY or trial.get('media_policy') != POLICY:
        raise ValueError('Unbound numeric/deferred media policy')
    complete = report.get('passed') is True
    mode = trial.get('recording_mode')
    if mode == 'numeric_only':
        if complete or trial.get('passed') is not True or trial.get('frames') or trial.get('video_path'):
            raise ValueError('Numeric-only media can only omit passed trials of a rejected candidate')
        return set()
    if mode != ('full_video' if complete else 'failure_keyframes'):
        raise ValueError('Wrong deferred evidence scope')
    record = trial['recorded_pose_render']
    path = bound_file(record['path'], record['sha256'], report_root)
    receipt = json.loads(path.read_text())
    scene = receipt['scene']; scene_path = bound_file(scene['path'], scene['sha256'], report_root)
    trace = list(iter_trajectory_samples(trial['trajectory_path']))
    frames = frame_schedule(trace, contract['config']['video_fps'])
    indices, anchors = keyframe_indices(frames, trace, contract, joint_name=trial.get('joint_name'))
    chosen = list(range(len(frames))) if complete else indices
    if (receipt['policy'] != POLICY or receipt['mode'] != mode or scene != trial['replay_scene']
            or receipt['trajectory_sha256'] != trial['trajectory_sha256']
            or receipt['frame_schedule'] != frames or receipt['selected_indices'] != chosen
            or receipt['numerical_anchors'] != anchors or receipt['physics_steps_during_render'] != 0
            or not 0 <= receipt['max_world_transform_error'] <= 1e-7
            or receipt['render_source'] != 'all_measured_link_world_transforms'
            or receipt['camera'] != trial['camera'] or receipt['retained_frames'] != trial['frames']
            or {f['frame_index'] for f in trial['frames']} != set(indices)):
        raise ValueError('Rendered evidence differs from complete native trajectory')
    for f in trial['frames']:
        expected = frames[f['frame_index']]
        if any(f.get(k) != v for k, v in expected.items()):
            raise ValueError('Replay frame misidentifies its measured state')
    return {path, scene_path}
