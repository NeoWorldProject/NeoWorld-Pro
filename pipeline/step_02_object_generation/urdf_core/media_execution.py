"""Supervise ffmpeg using encoded frame progress, without a total runtime cap."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

from .execution import wait_process
from .native_progress import write_execution_progress


def progress_monitor(path, *, root=None):
    path = Path(path)
    binding = None
    for folder in ([Path(root)] if root else path.parents):
        try:
            report = json.loads((folder / 'report.json').read_text())
            if report.get('run_id'):
                binding = (folder, report['run_id'])
                break
        except (OSError, ValueError, AttributeError):
            pass
    previous = None
    def read():
        nonlocal previous
        try:
            values = dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line)
            token = {k: int(values[k]) for k in ('frame', 'out_time_us') if values.get(k, '').lstrip('-').isdigit()}
            if not token:
                return None
            if token != previous:
                if binding is not None:
                    write_execution_progress(*binding, {'media': str(path), **token})
                blender_progress = os.environ.get('NEOWORLD_BLENDER_PROGRESS')
                if blender_progress:
                    target = Path(blender_progress)
                    pending = target.with_suffix('.media.tmp')
                    pending.write_text(json.dumps({'media': str(path), **token}) + '\n')
                    pending.replace(target)
            previous = token
            return token
        except (OSError, ValueError):
            return None
    return read


def run_ffmpeg(command, *, progress_path, root=None, text=False, check=False, start_new_session=False):
    path = Path(progress_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    command = [str(command[0]), '-progress', str(path), '-nostats', *map(str, command[1:])]
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=start_new_session)
        code = wait_process(process, progress=progress_monitor(path, root=root))
        stdout.seek(0); stderr.seek(0)
        out, err = stdout.read(), stderr.read()
    if text:
        out, err = out.decode(errors='replace'), err.decode(errors='replace')
    result = subprocess.CompletedProcess(command, code, out, err)
    if check:
        result.check_returncode()
    return result


def run_capture(command, *, ledger_path, binding, outputs):
    """Retry host-side extraction crashes without spending articulation repairs."""
    from .execution import TechnicalExecutionError
    from .run_state import fingerprint, write_json, file_hash
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = fingerprint({'command': list(map(str, command)), 'inputs': binding})
    data = json.loads(path.read_text()) if path.exists() else {'identity': identity, 'attempts': []}
    if data['identity'] != identity:
        raise RuntimeError('Media extraction inputs changed')
    if 'returncode' in data:
        if any(file_hash(p) != h for p, h in data['outputs'].items()):
            raise RuntimeError('Media extraction output changed')
        return data['returncode']
    while len(data['attempts']) < 5:
        row = {'attempt': len(data['attempts']) + 1, 'status': 'pending'}
        data['attempts'].append(row)
        write_json(path, data)
        try:
            result = run_ffmpeg(command, progress_path=path.with_suffix('.progress'),
                                text=True, start_new_session=True)
            path.with_suffix(f'.attempt_{row["attempt"]}.log').write_text(result.stdout + result.stderr)
            if result.returncode < 0:
                raise TechnicalExecutionError(f'Media worker exited with signal {-result.returncode}')
        except TechnicalExecutionError as exc:
            row.update(status='technical_failure', error=str(exc))
        else:
            code = result.returncode or int(any(not Path(p).is_file() for p in outputs))
            row.update(status='completed', returncode=code)
            data.update(returncode=code, outputs={str(p): file_hash(p) for p in outputs if Path(p).is_file()})
            write_json(path, data)
            return code  # Decode/content errors are candidate feedback, not crashes.
        write_json(path, data)
    raise TechnicalExecutionError('Media extraction exhausted five technical attempts')
