"""Run a scene image through parsing, object generation with Isaac refinement, and scene CEM."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARSER = ROOT/'pipeline/step_01_scene_parsing'
OBJECTS = ROOT/'pipeline/step_02_object_generation'
SCENE = ROOT/'pipeline/step_03_scene_layout_refinement'
for path in (PARSER, OBJECTS, SCENE): sys.path.insert(0, str(path))
from urdf_core.run_state import write_json, file_hash
from urdf_core.execution import stop_process

POLICY = 'full_pipeline_degraded_delivery.v1'
OPTIONS = {
    'parsing': {'model': '--model', 'reasoning_effort': '--reasoning_effort', 'max_refine_iters': '--max_refine_iters',
                'pass_score': '--pass_score', 'thinking_budget': '--thinking_budget'},
    'objects': {k: '--'+k.replace('_', '-') for k in ('model', 'reasoning_effort', 'object_workers', 'num_model_candidates',
                'max_generation_repairs', 'max_articulation_repairs', 'max_physics_repairs', 'joint_test_scope', 'physics_trial_workers')},
}


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    files = [Path(__file__), ROOT/'run_scene.py', ROOT/'run_object.py', *ROOT.joinpath('pipeline').rglob('*.py'), *ROOT.joinpath('pipeline').rglob('*.md'),
             *ROOT.joinpath('prompts').rglob('*.md'), *ROOT.joinpath('resources').rglob('*.md'),
             ROOT/'skills/isaac-physics-refine/SKILL.md']
    return {str(p): file_hash(p) for p in sorted(set(files)) if 'archive' not in p.parts and '__pycache__' not in p.parts}


def validate_config(value):
    from scene_refinement.protocol import load_config
    if not isinstance(value, dict) or set(value)-{'parsing', 'objects', 'scene'}:
        raise ValueError('Config sections must be parsing, objects and scene')
    result = {key: dict(value.get(key, {})) for key in ('parsing', 'objects', 'scene')}
    for key in OPTIONS:
        if set(result[key])-set(OPTIONS[key]): raise ValueError('Unsupported '+key+' options')
    result['scene'] = load_config(result['scene'])
    return result


def runtime_config(path):
    value = read(path); base = Path(path).resolve().parent
    if not isinstance(value, dict) or set(value) != {'objects', 'scene'}:
        raise ValueError('Runtime requires separate objects and scene sections')
    value = json.loads(json.dumps(value))
    for section in value.values():
        if not isinstance(section, dict): raise ValueError('Runtime section must be an object')
        for key in ('blender', 'isaac_python', 'ffmpeg', 'cache_dir', 'project_env_root'):
            name = section.get(key)
            if name and ('/' in name or key in ('cache_dir', 'project_env_root')):
                p = Path(name).expanduser(); section[key] = str((base/p).resolve() if not p.is_absolute() else p.resolve())
    return value


def execute_stage(command, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            return process.wait()
        finally:
            if process.poll() is None: stop_process(process)


def verify_stage(name, output, invocation):
    image, scene_id = invocation['image'], invocation['scene_id']
    layout = output/'01_parsing'/scene_id/'final_layout.json'
    if name == 'parsing':
        from urdf_core.layout_handoff import verify_layout_delivery
        data, quality = verify_layout_delivery(layout, image)
        return {'quality': quality, 'foreground_ids': [o['object_id'] for o in data['objects_data'] if not o['is_environment_support']],
                'receipt': str(layout), 'sha256': file_hash(layout)}
    if name == 'objects':
        from urdf_core.asset_delivery import verify_run
        path = output/'02_objects'/scene_id/'asset_manifest.json'; manifest = read(path)
        expected = {o['object_id'] for o in read(layout)['objects_data'] if not o['is_environment_support']}
        rows = [r for r in manifest['objects'] if not r.get('is_environment_support')]
        if len(rows) != len(expected) or {r['object_id'] for r in rows} != expected:
            raise ValueError('Object delivery inventory differs from the parsing inventory')
        for row in rows:
            if row.get('status') not in ('success', 'delivered_unaccepted'):
                raise RuntimeError('Object stage has a technical interruption: '+row['object_id'])
            checked = verify_run(Path(row['object_run_dir']))
            if Path(checked['usd_path']).resolve() != Path(row['usd_path']).resolve():
                raise ValueError('Object publication binding changed')
            physical = Path(checked.get('asset_manifest_path') or Path(checked['usd_path']).parent/'asset_manifest.json')
            if not physical.is_file(): raise ValueError('Object delivery lacks dynamic physicalization')
        return {'receipt': str(path), 'sha256': file_hash(path), 'summary': manifest.get('summary', {}), 'foreground_ids': sorted(expected)}
    path = output/'03_scene/delivery/receipt.json'; receipt = read(path)
    if not receipt.get('delivered'): raise RuntimeError('Scene delivery was not completed')
    for relative, expected in receipt['artifacts'].items():
        p = (path.parent/relative).resolve()
        if not p.is_relative_to(path.parent.resolve()) or file_hash(p) != expected:
            raise ValueError('Scene delivered dependency changed')
    return {'receipt': str(path), 'sha256': file_hash(path), 'accepted': receipt['accepted'],
            'terminal_window_passed': receipt.get('terminal_window_passed', False), 'delivered': True,
            'physical_status': receipt['physical_status']}


def run(invocation, output, *, resume=False):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    if invocation['sources'] != sources(): raise ValueError('Source changed: use a new pipeline output')
    if file_hash(invocation['image']) != invocation['image_sha256']: raise ValueError('Original image changed')
    report_path = output/'pipeline_report.json'
    report = read(report_path) if report_path.exists() else {'policy': POLICY, 'stages': {}, 'started_unix': invocation['started_unix']}
    from urdf_core.environment import load_project_env
    env_root = invocation['runtime']['scene'].get('project_env_root')
    load_project_env(Path(env_root) if env_root else None)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = os.pathsep.join([str(PARSER), str(OBJECTS), str(SCENE), *filter(None, [env.get('PYTHONPATH')])])
    cache = Path(invocation['runtime']['scene']['cache_dir']); cache.mkdir(parents=True, exist_ok=True)
    temporary = cache/'temporary'; temporary.mkdir(exist_ok=True)
    env.update(TMPDIR=str(temporary), NEOWORLD_ISAAC_CACHE_ROOT=str(cache/'isaac'))
    scene_id = invocation['scene_id']; configs = output/'configs'
    for kind, value in [('scene', invocation['config']['scene']), ('object_runtime', invocation['runtime']['objects']),
                        ('scene_runtime', invocation['runtime']['scene'])]:
        path = configs/(kind+'.json')
        if path.exists() and read(path) != value: raise ValueError('Pipeline configuration changed')
        write_json(path, value)
    layout = output/'01_parsing'/scene_id/'final_layout.json'
    assets = output/'02_objects'/scene_id/'asset_manifest.json'
    try:
        if resume and 'error' in report:
            report.setdefault('interruption_history', []).append({
                **{key: report.get(key) for key in ('error', 'active_stage', 'elapsed_seconds', 'resume_command')},
                'resumed_unix': time.time()})
            for key in ('error', 'resume_command', 'elapsed_seconds'):
                report.pop(key, None)
        for name in ('parsing', 'objects', 'scene'):
            report['active_stage'] = name
            if report['stages'].get(name, {}).get('status') == 'completed':
                checked = verify_stage(name, output, invocation)
                if checked != report['stages'][name]['delivery']: raise ValueError('Completed stage receipt changed')
                continue
            if shutil.disk_usage(output).free < 1200*1024**2:
                raise OSError('Less than 1200 MiB free for pipeline stage startup')
            report.update(status='running', active_stage=name, delivered=False)
            old = report['stages'].get(name, {})
            stage = {**old, 'status': 'running', 'started_unix': old.get('started_unix', time.time())}
            report['stages'][name] = stage; write_json(report_path, report)
            if name == 'parsing':
                command = [sys.executable, '-m', 'generate_scene_layout', '--image', invocation['image'],
                           '--out_dir', str(output/'01_parsing'), '--scene_id', scene_id]
            elif name == 'objects':
                command = [sys.executable, '-m', 'generate_urdf_batch', '--layout-json', str(layout),
                    '--scene-image', invocation['image'], '--batch-out-dir', str(output/'02_objects'),
                    '--runtime-config', str(configs/'object_runtime.json')]
                if not report['stages']['parsing']['delivery']['foreground_ids']:
                    write_json(assets, {'schema_version': 'object_pipeline.v1', 'layout_json': str(layout),
                        'scene_image': invocation['image'], 'objects': [], 'summary': {'total': 0},
                        'reason': 'Environment-only parsing delivery; object generation not needed'})
                    command = None
            else:
                scene_input = {'scene_id': scene_id, 'layout_json': str(layout), 'asset_manifest': str(assets), 'scene_image': invocation['image']}
                write_json(configs/'scene_input.json', scene_input)
                command = [sys.executable, '-m', 'scene_refinement.cli']
                if (output/'03_scene/invocation.json').exists():
                    command += ['resume', '--output', str(output/'03_scene')]
                else:
                    command += ['run', '--input', str(configs/'scene_input.json'), '--output', str(output/'03_scene'),
                                '--config', str(configs/'scene.json'), '--runtime', str(configs/'scene_runtime.json'), '--seed', '0']
            if command and name in OPTIONS:
                for key, value in invocation['config'][name].items(): command += [OPTIONS[name][key], str(value)]
            print(json.dumps({'stage': name, 'status': 'running', 'output': str(output)}, ensure_ascii=False), flush=True)
            code = execute_stage(command, output/'logs'/(name+'.log'), env) if command else 0
            stage.update(returncode=code, elapsed_seconds=time.time()-stage['started_unix'])
            if code not in (0, 2): raise RuntimeError(f'{name} technical interruption; see logs/{name}.log')
            stage.update(status='completed', delivery=verify_stage(name, output, invocation))
            write_json(report_path, report)
        report.update(status='completed', delivered=True, accepted=report['stages']['scene']['delivery']['accepted'],
                      terminal_window_passed=report['stages']['scene']['delivery']['terminal_window_passed'],
                      elapsed_seconds=time.time()-report['started_unix'], active_stage=None)
        write_json(report_path, report)
        return 0
    except Exception as error:
        from scene_refinement.provider import safe_error
        report.update(status='technical_paused', delivered=False, accepted=False, terminal_window_passed=False,
                      error=safe_error(error), resume_command=f'run_scene.py resume --output {output}',
                      elapsed_seconds=time.time()-report['started_unix'])
        write_json(report_path, report)
        print(report['error'], file=sys.stderr, flush=True)
        return 1


def build_parser():
    p = argparse.ArgumentParser(prog='run_scene.py', description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    create = sub.add_parser('run')
    for name in ('image', 'output', 'config', 'runtime'): create.add_argument('--'+name, type=Path, required=True)
    create.add_argument('--scene-id', required=True)
    sub.add_parser('resume').add_argument('--output', type=Path, required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    with (output/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output/'pipeline_invocation.json'
        if args.command == 'run':
            if any(p.name != '.pipeline.lock' for p in output.iterdir()): raise ValueError('Use an empty output or resume')
            if not re.fullmatch(r'[A-Za-z0-9_-]+', args.scene_id): raise ValueError('Invalid scene_id')
            invocation = {'policy': POLICY, 'image': str(args.image.resolve()), 'image_sha256': file_hash(args.image),
                'scene_id': args.scene_id, 'config': validate_config(read(args.config)), 'runtime': runtime_config(args.runtime),
                'sources': sources(), 'started_unix': time.time()}
            write_json(path, invocation)
        else:
            invocation = read(path)
            if invocation.get('policy') != POLICY: raise ValueError('Unsupported pipeline checkpoint version')
        return run(invocation, output, resume=args.command == 'resume')

