"""Explicit inputs, algorithm settings, runtime settings and serial batch state."""
import argparse
from contextlib import nullcontext
import fcntl
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from .protocol import SCHEMA_VERSION, read_json, write_json, load_config, file_hash


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    for name in ('run', 'batch'):
        sub = commands.add_parser(name)
        sub.add_argument('--input', type=Path, required=True, help='Input JSON; batch uses {scenes: [...]}')
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--config', type=Path, required=True)
        sub.add_argument('--runtime', type=Path, required=True)
        if name == 'run':
            sub.add_argument('--seed', type=int, default=0)
        else:
            sub.add_argument('--seeds', type=int, nargs='+', default=[0])
    commands.add_parser('resume').add_argument('--output', type=Path, required=True)
    return p


def configure_runtime(value, base):
    from urdf_core.environment import resolve_tool, load_project_env
    from .resources import available_cpus
    allowed = {'isaac_python', 'blender', 'ffmpeg', 'cache_dir', 'project_env_root',
               'allow_external_contention', 'sample_resources'}
    if not isinstance(value, dict) or set(value)-allowed:
        raise ValueError('Unsupported runtime configuration fields')
    runtime = dict(value)
    def path(value):
        item = Path(value).expanduser()
        return str((base/item).resolve() if not item.is_absolute() else item.resolve())
    if 'cache_dir' not in runtime:
        raise ValueError('Runtime requires an explicit writable cache_dir')
    runtime['cache_dir'] = path(runtime['cache_dir'])
    if runtime.get('project_env_root'):
        runtime['project_env_root'] = path(runtime['project_env_root'])
        load_project_env(Path(runtime['project_env_root']))
    for name in ('isaac_python', 'blender', 'ffmpeg'):
        value = runtime.get(name)
        if value and ('/' in value or value.startswith('.')):
            value = path(value)
        runtime[name] = resolve_tool(name, value)
        os.environ['NEOWORLD_'+name.upper()] = runtime[name]
    os.environ['NEOWORLD_ISAAC_ROOT'] = str(Path(runtime['isaac_python']).parent)
    os.environ['NEOWORLD_SCENE_STATIC_CACHE'] = runtime['cache_dir']
    os.environ['NEOWORLD_SCENE_HASH_CACHE'] = str(Path(runtime['cache_dir'])/'file_hashes.sqlite3')
    os.environ['NEOWORLD_ISAAC_CACHE_ROOT'] = str(Path(runtime['cache_dir'])/'isaac')
    runtime.setdefault('allow_external_contention', False)
    runtime.setdefault('sample_resources', True)
    if any(type(runtime[k]) is not bool for k in ('allow_external_contention', 'sample_resources')):
        raise ValueError('Runtime contention/resource settings must be booleans')
    os.environ['NEOWORLD_SCENE_ALLOW_EXTERNAL_CONTENTION'] = '1' if runtime['allow_external_contention'] else '0'
    runtime['available_cpus'] = available_cpus()
    return runtime


def _inputs(value, base):
    allowed = {'scene_id', 'layout_json', 'asset_manifest', 'scene_image'}
    if not isinstance(value, dict) or set(value)-allowed or not allowed-{'scene_id'} <= set(value):
        raise ValueError('Input requires layout_json, asset_manifest and scene_image')
    result = {}
    for key in allowed-{'scene_id'}:
        file = Path(value[key]).expanduser()
        result[key] = str((base/file).resolve() if not file.is_absolute() else file.resolve())
        if not Path(result[key]).is_file():
            raise ValueError('Missing input: ' + result[key])
    result['scene_id'] = value.get('scene_id', 'scene')
    if not result['scene_id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in result['scene_id']):
        raise ValueError('scene_id must be a simple directory name')
    return result


def execute(invocation, root, resume):
    from .resources import ResourceSampler
    from .workflow import run_refinement
    from .worker_pool import WorkerPool, POOL_ENV
    config, runtime = invocation['config'], invocation['runtime']
    scenes, seeds = invocation['scenes'], invocation['seeds']
    previous = read_json(root/'batch_report.json') if resume and (root/'batch_report.json').exists() else {}
    rows = previous.get('results', [])
    task_starts = previous.get('task_started_unix', {})
    completed = {(r['scene_id'], r['seed']) for r in rows if r['status'] != 'system_paused'}
    is_batch = invocation['kind'] == 'batch'
    def save(status, **details):
        write_json(root/'batch_report.json', {'schema_version': SCHEMA_VERSION, 'results': rows,
            'status': status, 'elapsed_seconds': time.time()-invocation['started_unix'],
            'task_started_unix': task_starts, 'counts': counts(rows), **details})
    if is_batch:
        for row in rows:
            if (row['scene_id'], row['seed']) in completed:
                verify_completed(row)
        pending = [(s, seed) for s in scenes for seed in seeds if (s['scene_id'], seed) not in completed]
        if not pending:
            return 0 if all(r['accepted'] for r in rows) else 2
        first_scene, first_seed = pending[0]
        task_starts.setdefault(f"{first_scene['scene_id']}_seed{first_seed}",
                               time.time() if completed else invocation['started_unix'])
        save('starting')
    context = WorkerPool(root/'workers'/str(time.time_ns()), config, workers=1,
        isaac_python=runtime['isaac_python'], output_roots=[root],
        allow_external_contention=runtime['allow_external_contention']) if is_batch else nullcontext()
    sampler = ResourceSampler(root/'resources.jsonl') if runtime['sample_resources'] else nullcontext()
    from contextlib import ExitStack
    with ExitStack() as stack:
        try:
            stack.enter_context(sampler)
            if is_batch and shutil.disk_usage(root).free < 1200 * 1024**2:
                raise RuntimeError('Less than 1200 MiB free for scene startup')
            pool = stack.enter_context(context)
        except (OSError, RuntimeError) as exc:
            if not is_batch:
                raise
            save('paused', infrastructure_error=str(exc))
            return 1
        if pool:
            os.environ[POOL_ENV] = str(pool.root)
        try:
            for scene in scenes:
                for seed in seeds:
                    if (scene['scene_id'], seed) in completed:
                        continue
                    task_key = f"{scene['scene_id']}_seed{seed}"
                    output = root/'runs'/task_key if is_batch else root
                    task_starts.setdefault(task_key, time.time() if is_batch else invocation['started_unix'])
                    if is_batch:
                        save('running')
                    args = SimpleNamespace(**{k: Path(scene[k]) for k in ('layout_json', 'asset_manifest', 'scene_image')},
                        config=config, runtime=runtime, seed=seed, out_dir=output, isaac_python=Path(runtime['isaac_python']),
                        resume=resume, started_unix=task_starts[task_key],
                        native_runtime_binding=invocation.get('native_runtime'),
                        batch_enqueued_unix=invocation['started_unix'] if is_batch else None,
                        model='gpt-5.6-sol', reasoning_effort='medium')
                    report = run_refinement(args)
                    print(f"{scene['scene_id']} seed={seed}: {report['status']}; delivered={report.get('delivered')}; accepted={report.get('accepted')}", flush=True)
                    for error in report.get('errors', []):
                        print(error, flush=True)
                    row = {'scene_id': scene['scene_id'], 'seed': seed, 'status': report['status'],
                           'accepted': report.get('accepted', False), 'delivered': report.get('delivered', False),
                           'report': str(output/'scene_refinement_report.json'),
                           'report_sha256': file_hash(output/'scene_refinement_report.json')}
                    rows = [r for r in rows if (r['scene_id'], r['seed']) != (scene['scene_id'], seed)]+[row]
                    if is_batch:
                        save('paused' if report['status'] == 'system_paused' else 'running')
                    if report['status'] == 'system_paused':
                        return 1
        finally:
            if pool:
                os.environ.pop(POOL_ENV, None)
    if is_batch:
        save('completed')
    return 0 if all(r['accepted'] for r in rows) else 2


def verify_completed(row):
    from .workflow import _verify_hashes
    report = Path(row['report'])
    if not row.get('report_sha256') or file_hash(report) != row['report_sha256']:
        raise ValueError('Completed scene report changed')
    manifest = report.parent/'run_manifest.json'
    if manifest.exists():
        value = read_json(manifest)
        if value.get('status') != 'completed' or not _verify_hashes(value.get('artifacts', {})):
            raise ValueError('Completed scene evidence changed')


def counts(rows):
    return {'accepted': sum(r['accepted'] for r in rows),
            'delivered_unaccepted': sum(r['delivered'] and not r['accepted'] for r in rows),
            'not_delivered': sum(not r['delivered'] for r in rows)}


def main(argv=None):
    from .worker_client import native_runtime_binding
    started = time.time()
    options = parser().parse_args(argv)
    root = options.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root/'invocation.json'
    resume = options.command == 'resume'
    if resume:
        invocation = read_json(path)
        if invocation.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Only current-version checkpoints can resume')
        for group in ('source_configs', 'input_hashes', 'implementation'):
            if any(file_hash(p) != h for p, h in invocation[group].items()):
                raise ValueError('Resume binding changed: ' + group)
        original_runtime = invocation['runtime']
        configure_runtime({k: v for k, v in original_runtime.items() if k != 'available_cpus'}, root)
        if invocation.get('native_runtime') != native_runtime_binding(original_runtime['isaac_python']):
            raise ValueError('Resume native runtime changed or is unbound; use the original frozen version')
    else:
        if path.exists() or any(root.iterdir()):
            raise ValueError('Use a new output directory or resume')
        config_value = read_json(options.config)
        runtime = configure_runtime(read_json(options.runtime), options.runtime.resolve().parent)
        config = load_config(config_value)
        value = read_json(options.input)
        scenes = value['scenes'] if options.command == 'batch' else [value]
        scenes = [_inputs(s, options.input.resolve().parent) for s in scenes]
        seeds = options.seeds if options.command == 'batch' else [options.seed]
        if not scenes or len({s['scene_id'] for s in scenes}) != len(scenes) or len(set(seeds)) != len(seeds):
            raise ValueError('Scenes and seeds must be nonempty and unique')
        invocation = {'schema_version': SCHEMA_VERSION, 'kind': options.command, 'scenes': scenes, 'seeds': seeds,
            'config': config, 'runtime': runtime, 'started_unix': started,
            'source_configs': {str(p.resolve()): file_hash(p) for p in (options.input, options.config, options.runtime)}}
        invocation['input_hashes'] = {s[k]: file_hash(s[k]) for s in scenes for k in ('layout_json','asset_manifest','scene_image')}
        invocation['native_runtime'] = native_runtime_binding(runtime['isaac_python'])
        from .worker_client import source_hashes
        stage = Path(__file__).resolve().parents[1]
        invocation['implementation'] = {**source_hashes(), **{str(p):file_hash(p) for p in [*stage.rglob('*.py'), *stage.rglob('*.md')]}}
        write_json(path, invocation)
    if invocation['kind'] == 'batch':
        with (root/'.batch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return execute(invocation, root, resume)
    return execute(invocation, root, resume)


if __name__ == '__main__':
    raise SystemExit(main())
