import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

from urdf_core.run_state import file_hash, write_json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event
import time
import io


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must contain an object: {path.name}")
    return value


def sanitize_name(name: str) -> str:
    safe = []
    for ch in name.lower().replace(" ", "_"):
        if ch.isalnum() or ch == "_":
            safe.append(ch)
    text = "".join(safe).strip("_")
    return text if text else "object"


def crop_from_bbox(image: Image.Image, bbox_norm: list[float]) -> tuple[Image.Image, list[int]]:
    """Use the final visible bbox exactly, matching stage-one pixel coverage."""
    if len(bbox_norm) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox_norm):
        raise ValueError("bbox requires four finite numbers")
    x_min, y_min, x_max, y_max = bbox_norm
    if not (0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1):
        raise ValueError("bbox must be ordered and normalized; no silent crop repair")
    width, height = image.size
    pixels = [math.floor(x_min * width), math.floor(y_min * height),
              math.ceil(x_max * width), math.ceil(y_max * height)]
    return image.crop(pixels), pixels


def add_batch_arguments(parser):
    from generate_object_urdf import add_runtime_arguments
    from urdf_core.paths import RESOURCE_ROOT
    add_runtime_arguments(parser)
    parser.add_argument('--prompts-dir', '--prompts_dir', type=Path, default=RESOURCE_ROOT / 'prompts/urdf')
    parser.add_argument('--object-workers', type=int, default=0,
                        help='0 advances all objects; API/GPU queues control admission. Positive values impose an explicit object cap.')
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--skip-environment', '--skip_environment', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--require-accepted', action='store_true', help='After completing the batch, return a nonzero status for unaccepted assets')
    parser.add_argument('--max-objects', '--max_objects', type=int, default=0)


def build_parser():
    parser = argparse.ArgumentParser(description='Concurrent objects using the single-object state machine')
    parser.add_argument('--layout-json', '--layout_json', required=True, type=Path)
    parser.add_argument('--scene-image', '--scene_image', required=True, type=Path)
    parser.add_argument('--batch-out-dir', '--batch_out_dir', '--output', required=True, type=Path,
                        help='Output directory for the asset manifest, object crops, and object runs')
    add_batch_arguments(parser)
    return parser


def prepare_batch(args, layout_path, scene_image, output, *, scene_name=None):
    from physics_feedback.scene_context import build_context
    from urdf_core.delivery_policy import POLICY
    layout_path, scene_image, output = map(lambda p: Path(p).resolve(), (layout_path, scene_image, output))
    scene_name = scene_name or layout_path.parent.name
    layout = read_json(layout_path)
    objects = layout.get('objects_data', [])
    if not isinstance(objects, list):
        raise ValueError('layout objects_data must be a list')
    identities = [str(obj.get('object_id', f'obj_{index:03d}'))
                  for index, obj in enumerate(objects, 1)]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate object identity in layout')
    layout_quality = verify_layout_handoff(layout_path, scene_image)
    if args.max_objects:
        objects = objects[:args.max_objects]
    identity = {'layout': {'path': str(layout_path), 'sha256': file_hash(layout_path)},
                'scene_image': {'path': str(scene_image), 'sha256': file_hash(scene_image)},
                'skip_environment': args.skip_environment, 'max_objects': args.max_objects,
                'layout_delivery': layout_quality}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'asset_manifest.json'
    if manifest_path.exists():
        prior = read_json(manifest_path)
        if prior.get('schema_version') != POLICY or prior.get('inputs') != identity:
            raise ValueError('Batch inputs or harness changed; start a new experiment')
    crops = output / 'object_crops'
    crops.mkdir(exist_ok=True)
    with Image.open(scene_image) as source:
        image = source.convert('RGB')
    records, tasks, names = [], [], set()
    for index, obj in enumerate(objects, 1):
        object_id = str(obj.get('object_id', f'obj_{index:03d}'))
        name = str(obj.get('name', object_id))
        import hashlib
        stem = sanitize_name(object_id)[:60] + '_' + hashlib.sha256((object_id + '\0' + name).encode()).hexdigest()[:16]
        if stem in names:
            raise ValueError('Duplicate object identity: ' + object_id)
        names.add(stem)
        record = {'object_id': object_id, 'name': name, 'description': obj.get('description', ''),
                  'is_environment_support': bool(obj.get('is_environment_support')),
                  'is_occluded': bool(obj.get('is_occluded')), 'bbox_3d_meters': obj.get('3d_bbox_meters')}
        records.append(record)
        if args.skip_environment and record['is_environment_support']:
            record['status'] = 'skipped_environment'
            continue
        try:
            try:
                crop, pixels = crop_from_bbox(image, obj.get('2d_bbox_normalized', []))
                crop_fallback = None
            except (ValueError, TypeError) as error:
                crop, pixels = image.copy(), [0, 0, image.width, image.height]
                crop_fallback = {'policy': 'full_image_fallback', 'reason': str(error)}
            crop_path = crops / (stem + '.png')
            contents = io.BytesIO()
            crop.save(contents, format='PNG')
            if crop_path.exists() and crop_path.read_bytes() != contents.getvalue():
                raise ValueError('Crop identity changed')
            if not crop_path.exists():
                crop_path.write_bytes(contents.getvalue())
            context = build_context(layout_path, scene_image, object_id, crop_path)
            if crop_fallback:
                context['crop_fallback'] = crop_fallback
                record['crop_fallback'] = crop_fallback
            context_path = crops / (stem + '.scene_context.json')
            if context_path.exists() and read_json(context_path) != context:
                raise ValueError('Scene context changed')
            write_json(context_path, context)
            run = output / 'objects' / stem
            if args.fresh and run.exists() and any(run.iterdir()):
                raise ValueError('--fresh requires an empty object directory')
            record.update(status='pending', object_run_dir=str(run), crop_path=str(crop_path),
                          crop_bbox_pixels=pixels, bbox_2d_normalized=obj.get('2d_bbox_normalized'),
                          scene_context_path=str(context_path))
            tasks.append(record)
        except (ValueError, OSError, TypeError) as exc:
            from urdf_core.error_recovery import external_failure, error_evidence
            record.update(status='failed', error=str(exc), error_context=error_evidence(exc),
                          failure_kind='external_failure' if external_failure(exc) else 'content_failure')
    manifest = {'schema_version': POLICY, 'scene_name': scene_name, 'layout_json': str(layout_path),
                'scene_image': str(scene_image), 'inputs': identity, 'objects': records}
    save_manifest(manifest_path, manifest)
    return manifest_path, manifest, tasks


def verify_layout_handoff(layout_path, image_path):
    from urdf_core.layout_handoff import verify_layout_delivery
    return verify_layout_delivery(layout_path, image_path)[1]


def save_manifest(path, manifest):
    rows = manifest['objects']
    manifest['summary'] = {key: sum(row.get('status') == key for row in rows)
                           for key in ('success', 'delivered_unaccepted', 'failed', 'cancelled', 'pending', 'skipped_environment')}
    manifest['summary']['total'] = len(rows)
    write_json(path, manifest)


def child_command(args, record):
    command = [sys.executable, '-m', 'urdf_core.cli',
               '--image', record['crop_path'], '--scene-context', record['scene_context_path'],
               '--out-dir', record['object_run_dir'], '--prompts-dir', str(args.prompts_dir.resolve())]
    for name in ('max_error_diagnoses', 'model', 'reasoning_effort', 'num_model_candidates', 'max_generation_repairs', 'max_articulation_repairs',
                 'max_physics_repairs', 'max_geometry_repairs', 'joint_test_scope', 'physics_trial_workers', 'blender',
                 'isaac_python', 'ffmpeg', 'startup_timeout', 'no_progress_timeout', 'api_concurrency'):
        value = getattr(args, name, None)
        if value is not None:
            command.extend(['--' + name.replace('_', '-'), str(value)])
    if args.physics_free_base is not None:
        command.append('--physics-free-base' if args.physics_free_base else '--no-physics-free-base')
    if args.fresh:
        command.append('--fresh')
    return command


def execute_task(args, record, cancel):
    from urdf_core.error_recovery import object_recovery, ReportedError, external_failure, cancelled
    from urdf_core.asset_delivery import verify_run
    from physics_feedback.loop import _audited_ask
    import generate_object_urdf as api
    run = Path(record['object_run_dir'])
    result = _execute_task_once(args, record, cancel)
    if result['status'] != 'failed' or result.get('failure_kind') in ('external_failure', 'worker_exit'):
        return result
    recovery = object_recovery(args, api, run, _audited_ask(args, api, run))
    first = [result]
    def retry():
        current = first.pop() if first else _execute_task_once(args, record, cancel)
        if current['status'] == 'failed':
            if current.get('failure_kind') in ('external_failure', 'worker_exit'):
                from urdf_core.error_recovery import VerifiedExternalError
                raise VerifiedExternalError('Object child technical interruption',
                    kind=current['failure_kind'], evidence=current.get('error'))
            raise ReportedError({'errors': [current.get('error')], 'error_context': {'object_id': record['object_id']}})
        return current
    def existing(error):
        checked = verify_run(run, {'source_images': {record['crop_path']: file_hash(Path(record['crop_path']))}})
        return dict(record, **checked, status='success' if checked.get('passed') else 'delivered_unaccepted')
    try:
        return recovery.run('object_batch_handoff', retry, fallback=existing, attempts=2,
            evidence={'object_id': record['object_id'], 'image_sha256': file_hash(Path(record['crop_path']))},
            validate_result=lambda value: value == existing(None))
    except Exception as error:
        if cancelled(error):
            raise
        return dict(result, failure_kind='external_failure' if external_failure(error) else 'unclassified_error',
                    error=f'{type(error).__name__}: {error}')


def _execute_task_once(args, record, cancel):
    from urdf_core.execution import stop_process
    from urdf_core.asset_delivery import verify_run
    run = Path(record['object_run_dir'])
    log_path = run.parent.parent / 'logs' / (run.name + '.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if cancel.is_set():
        return dict(record, status='cancelled')
    try:
        with log_path.open('a') as log:
            env = dict(os.environ)
            env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(Path(__file__).resolve().parent), env.get('PYTHONPATH'))))
            process = subprocess.Popen(child_command(args, record), stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=env)
            try:
                while process.poll() is None:
                    if cancel.wait(.25):
                        stop_process(process)
                        return dict(record, status='cancelled', run_log_path=str(log_path))
            finally:
                if process.poll() is None:
                    stop_process(process)
        failure = json.loads((run/'stage_failure.json').read_text()) if (run/'stage_failure.json').exists() else {}
        if (failure.get('kind') in ('external_failure', 'cancelled')
                and not failure.get('status', '').startswith('resolved_')) or process.returncode < 0:
            return dict(record, status='failed', failure_kind=failure.get('kind', 'worker_exit'),
                        return_code=process.returncode, error=failure, run_log_path=str(log_path))
        checked = verify_run(run, {'source_images': {record['crop_path']: file_hash(Path(record['crop_path']))}})
        status = 'success' if checked.get('passed') else 'delivered_unaccepted'
        return dict(record, **checked, status=status, return_code=process.returncode,
                    elapsed_seconds=time.monotonic() - started, run_log_path=str(log_path))
    except Exception as exc:
        from urdf_core.error_recovery import external_failure, cancelled
        if cancelled(exc):
            raise
        return dict(record, status='failed', failure_kind='external_failure' if external_failure(exc) else 'unclassified_error', error=f'{type(exc).__name__}: {exc}',
                    elapsed_seconds=time.monotonic() - started, run_log_path=str(log_path))


def execute_batches(args, batches):
    cancel = Event()
    tasks_count = sum(len(tasks) for _, _, tasks in batches)
    workers = args.object_workers or max(1, tasks_count)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(execute_task, args, record, cancel): (path, manifest, record)
                   for path, manifest, tasks in batches for record in tasks}
        try:
            for future in as_completed(pending):
                path, manifest, record = pending[future]
                record.update(future.result())
                if record['status'] == 'failed' and record.get('failure_kind') in ('external_failure', 'worker_exit'):
                    cancel.set()
                save_manifest(path, manifest)
                print(f'{manifest["scene_name"]}/{record["object_id"]}: {record["status"]}', flush=True)
        except BaseException:
            cancel.set()
            for future, (path, manifest, record) in pending.items():
                record.update(future.result())
                save_manifest(path, manifest)
            raise
    if any(row.get('status') in ('failed', 'cancelled') for _, manifest, _ in batches for row in manifest['objects']):
        raise RuntimeError('Object batch stopped after an execution/input error; inspect retained manifests and logs')


def main(argv=None):
    from generate_object_urdf import parse_configured_args
    from urdf_core.environment import load_project_env, configure_runtime
    load_project_env()
    args = parse_configured_args(build_parser(), argv)
    if args.object_workers < 0 or min(args.num_model_candidates, args.physics_trial_workers) < 1 or args.max_objects < 0:
        raise ValueError('Object workers and max-objects must be nonnegative; candidate/trial-worker counts must be positive')
    configure_runtime(args)
    batch = prepare_batch(args, args.layout_json, args.scene_image, args.batch_out_dir)
    execute_batches(args, [batch])
    print(json.dumps(batch[1]['summary'], ensure_ascii=False))
    if args.require_accepted and any(row['status'] not in ('success', 'skipped_environment') for row in batch[1]['objects']):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
