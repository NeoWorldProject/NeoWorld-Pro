"""Execute each object from its own immutable code and prompt snapshot."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from .delivery_policy import POLICY
from .paths import OBJECT_GENERATION, RESOURCE_ROOT
from .run_state import file_hash, write_json


def verified_runtime(binding):
    path = Path(binding['path']).resolve()
    if file_hash(path) != binding['sha256']:
        raise ValueError('Frozen runtime manifest changed')
    manifest = json.loads(path.read_text())
    if (manifest.get('release_family') != 'neoworld-source-20261019'
            or manifest.get('harness_policy') != POLICY or manifest.get('kind') not in {'physics', 'generation'}):
        raise ValueError('Unknown frozen object runtime')
    files = manifest.get('snapshot_files', {})
    if not files:
        raise ValueError('Frozen object runtime has no source inventory')
    for name, expected in files.items():
        source = Path(name).resolve()
        if not source.is_relative_to(path.parent) or file_hash(source) != expected:
            raise ValueError('Frozen runtime source changed: ' + name)
    package = path.parent / 'pipeline/step_02_object_generation'
    if str(package / 'urdf_core/runtime_snapshot.py') not in files:
        raise ValueError('Frozen runtime is missing its dispatcher')
    return package


def prepare_runtime(args):
    root = Path(args.out_dir).resolve()
    destination = root.with_name(root.name + '.runtime')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_name(destination.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = destination / 'source_snapshot.json'
        if manifest_path.is_file():
            binding = {'path': str(manifest_path), 'sha256': file_hash(manifest_path)}
            verified_runtime(binding)
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('object_run_dir') != str(root):
                raise ValueError('Frozen runtime belongs to another object run')
            if manifest.get('prompt_source_dir') != str(Path(args.prompts_dir).resolve()):
                raise ValueError('Prompt source selection changed; use a new object run')
            return binding
        if destination.exists():
            raise ValueError('Incomplete runtime snapshot; preserve it and use a new output directory')
        if (root / 'run_manifest.json').exists():
            raise ValueError('Existing object has no frozen runtime; keep its original verifier and use a new run')
        sources = {}
        for directory in ('urdf_core', 'physics_feedback', 'kinematics_validation'):
            for source in (OBJECT_GENERATION / directory).rglob('*.py'):
                if '__pycache__' not in source.parts and source.name not in {'synthetic_regression.py', 'performance_regression.py'}:
                    sources[Path('pipeline/step_02_object_generation') / source.relative_to(OBJECT_GENERATION)] = source
        for name in ('generate_object_urdf.py', 'generate_urdf_batch.py'):
            sources[Path('pipeline/step_02_object_generation') / name] = OBJECT_GENERATION / name
        for source in Path(args.prompts_dir).glob('*.md'):
            sources[Path('prompts/urdf') / source.name] = source
        for source in (RESOURCE_ROOT / 'prompts/isaac').glob('physics_*.md'):
            sources[Path('prompts/isaac') / source.name] = source
        skill = RESOURCE_ROOT / 'skills/isaac-physics-refine/SKILL.md'
        sources[Path('skills/isaac-physics-refine/SKILL.md')] = skill
        originals = {str(source.resolve()): file_hash(source) for source in sources.values()}
        with tempfile.TemporaryDirectory(prefix=destination.name + '.pending_', dir=destination.parent) as staging:
            staging = Path(staging)
            for relative, source in sources.items():
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                if file_hash(target) != originals[str(source.resolve())] or file_hash(source) != originals[str(source.resolve())]:
                    raise ValueError('Object implementation changed while freezing: ' + str(source))
            write_json(staging / 'source_snapshot.json', {'schema_version': 'object_runtime_snapshot.v1',
                'release_family': 'neoworld-source-20261019', 'harness_policy': POLICY, 'kind': 'physics' if args.enable_physics else 'generation',
                'object_run_dir': str(root), 'prompt_source_dir': str(Path(args.prompts_dir).resolve()),
                'original_hashes': originals,
                'snapshot_files': {str(destination / relative): originals[str(source.resolve())]
                                   for relative, source in sources.items()}})
            staging.rename(destination)
        binding = {'path': str(manifest_path), 'sha256': file_hash(manifest_path)}
        verified_runtime(binding)
        return binding


def _encode(value):
    if isinstance(value, Path):
        return {'__path__': str(value)}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    return value


def _decode(value):
    if isinstance(value, dict):
        return Path(value['__path__']) if set(value) == {'__path__'} else {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _environment(package):
    return dict(os.environ, PYTHONPATH=str(package), PYTHONDONTWRITEBYTECODE='1',
                NEOWORLD_RESOURCE_ROOT=str(package.parents[1]))


def dispatch(args):
    existing = getattr(args, '_runtime_snapshot', None)
    if existing:
        if verified_runtime(existing) != OBJECT_GENERATION.resolve():
            raise ValueError('Object execution is outside its frozen runtime')
        return False
    root = Path(args.out_dir).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock outside the output directory so --fresh remains meaningful.
    # The inherited descriptor keeps ownership alive if the launcher exits first.
    with root.with_name(root.name + '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This object run already has a live owner: ' + str(root)) from error
        return _dispatch_owned(args, lock.fileno())


def _dispatch_owned(args, owner_fd):
    binding = prepare_runtime(args)
    package = verified_runtime(binding)
    values = dict(vars(args), _runtime_snapshot=binding,
                  prompts_dir=package.parents[1] / 'prompts/urdf')
    process = subprocess.Popen([sys.executable, '-B', '-m', 'urdf_core.runtime_snapshot', 'run'],
        stdin=subprocess.PIPE, text=True, start_new_session=True, env=_environment(package),
        pass_fds=(owner_fd,))
    try:
        process.communicate(json.dumps(_encode(values)))
    except BaseException:
        from .execution import stop_process
        stop_process(process)
        raise
    if process.returncode:
        raise RuntimeError(f'Frozen object process exited with code {process.returncode}')
    return True


def verify_frozen_run(root, binding, expected_config=None, *, require_pass=False):
    package = verified_runtime(binding)
    request = {'root': str(Path(root).resolve()), 'expected_config': expected_config, 'require_pass': require_pass}
    environment = _environment(package)
    config = json.loads((Path(root)/'run_manifest.json').read_text()).get('config', {})
    launcher = config.get('runtime', {}).get('isaac_python') or config.get('isaac_python')
    if launcher:
        environment.update(NEOWORLD_ISAAC_PYTHON=launcher,
                           NEOWORLD_ISAAC_ROOT=str(Path(launcher).parent))
    process = subprocess.run([sys.executable, '-B', '-m', 'urdf_core.runtime_snapshot', 'verify'],
        input=json.dumps(request), text=True, capture_output=True, env=environment)
    if process.returncode:
        raise ValueError('Frozen object verification failed: ' + process.stderr[-4000:])
    verified_runtime(binding)
    return json.loads(process.stdout)


def main():
    request = json.loads(sys.stdin.read())
    if sys.argv[1:] == ['run']:
        import generate_object_urdf as api
        from .orchestration import run_pipeline
        run_pipeline(SimpleNamespace(**_decode(request)), api)
    elif sys.argv[1:] == ['verify']:
        from .asset_delivery import verify_run
        print(json.dumps(verify_run(Path(request['root']), request['expected_config'], require_pass=request['require_pass'])))
    else:
        raise ValueError('Select run or verify')


if __name__ == '__main__':
    main()
