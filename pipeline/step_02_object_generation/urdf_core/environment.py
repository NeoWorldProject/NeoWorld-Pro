"""Environment loading and portable executable discovery."""
import json
import os
from pathlib import Path
import shutil

from .paths import PROJECT


def load_project_env(project_root: Path | None = None) -> None:
    from dotenv import load_dotenv

    root = project_root or PROJECT
    # An explicitly supplied process environment takes precedence over local defaults.
    load_dotenv(root / ".env", override=False)


TOOLS = {'blender': 'NEOWORLD_BLENDER', 'isaac_python': 'NEOWORLD_ISAAC_PYTHON',
         'ffmpeg': 'NEOWORLD_FFMPEG'}


def resolve_tool(name, value=None, *, required=True):
    candidate = value or os.environ.get(TOOLS[name])
    if not candidate and name == 'isaac_python':
        root = os.environ.get('NEOWORLD_ISAAC_ROOT')
        candidate = str(Path(root).expanduser() / 'python.sh') if root else shutil.which('isaac-sim-python')
    if not candidate and name != 'isaac_python':
        candidate = shutil.which(name)
    found = shutil.which(str(candidate)) if candidate else None
    if found:
        return str(Path(found).absolute())
    if required:
        raise FileNotFoundError(f'{name} executable unavailable; configure --{name.replace("_", "-")} or {TOOLS[name]}')
    return None


def isaac_mdl_paths():
    root = os.environ.get('NEOWORLD_ISAAC_ROOT')
    launcher = resolve_tool('isaac_python', required=False)
    if root is None and launcher:
        root = str(Path(launcher).parent)
    candidates = [Path(root) / 'kit/mdl/core/Base'] if root else []
    return [p.resolve() for p in candidates if p.is_dir()]


def configure_runtime(args):
    config = json.loads(Path(args.runtime_config).read_text()) if getattr(args, 'runtime_config', None) else {}
    allowed = {*TOOLS, 'startup_timeout', 'no_progress_timeout', 'api_concurrency', 'reasoning_effort'}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError('Runtime configuration contains unsupported keys')
    for name in TOOLS:
        value = getattr(args, name, None) or config.get(name)
        resolved = resolve_tool(name, value)
        setattr(args, name, resolved)
        os.environ[TOOLS[name]] = resolved
    os.environ['NEOWORLD_ISAAC_ROOT'] = str(Path(args.isaac_python).parent)
    for name, env, default in (
            ('startup_timeout', 'NEOWORLD_STARTUP_TIMEOUT', 300.),
            ('no_progress_timeout', 'NEOWORLD_NO_PROGRESS_TIMEOUT', 600.),
            ('api_concurrency', 'NEOWORLD_API_CONCURRENCY', 64)):
        value = getattr(args, name, None)
        value = value if value is not None else config.get(name, os.environ.get(env, default))
        value = int(value) if name == 'api_concurrency' else float(value)
        import math
        if value <= 0 or not math.isfinite(value):
            raise ValueError(name + ' must be finite and positive')
        setattr(args, name, value)
        os.environ[env] = str(value)
    effort = getattr(args, 'reasoning_effort', None) or config.get('reasoning_effort') or os.environ.get('URDF_REASONING_EFFORT', 'high')
    args.reasoning_effort = effort
    os.environ['URDF_REASONING_EFFORT'] = effort
    return args


def inspect_environment():
    result = {name: resolve_tool(name, required=False) for name in TOOLS}
    result['isaac_mdl_paths'] = list(map(str, isaac_mdl_paths()))
    result['api_credentials_configured'] = bool(os.environ.get('OPENAI_API_KEY'))
    result['ready'] = all(result[name] for name in TOOLS)
    return result


def runtime_identity(args):
    return {name: getattr(args, name, default) for name, default in (
        ('blender', None), ('isaac_python', None), ('ffmpeg', None),
        ('startup_timeout', 300.), ('no_progress_timeout', 600.), ('api_concurrency', 64),
        ('physics_trial_workers', 2))}
