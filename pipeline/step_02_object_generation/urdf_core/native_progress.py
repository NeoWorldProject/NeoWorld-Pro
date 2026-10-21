"""Small, non-authoritative worker heartbeats; full reports remain evidence."""
from __future__ import annotations

import json
from pathlib import Path
import time

FIELDS = ('status', 'stage', 'active_trial', 'active_trial_time_s', 'active_trial_physics_step')
SCHEMA = 'physics_worker_progress.v1'


def _checkpoint(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size]


def write_progress(report_path, report):
    """Do not serialize completed trials and frames for each simulated second."""
    report_path = Path(report_path)
    payload = {'schema_version': SCHEMA, 'run_id': report['run_id'],
               'report_checkpoint': _checkpoint(report_path),
               'written_at_unix_s': time.time(),
               **{key: report.get(key) for key in FIELDS}}
    path = report_path.with_name('progress.json')
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)
    write_execution_progress(report_path.parent, report['run_id'], {k: report.get(k) for k in FIELDS})


def observed_progress(report_path, report):
    """A heartbeat may enrich a running display, never an acceptance result."""
    result = {key: report.get(key) for key in FIELDS}
    if report.get('status') != 'running' or not report.get('run_id'):
        return result
    report_path = Path(report_path)
    try:
        value = json.loads(report_path.with_name('progress.json').read_text())
        if (value.get('schema_version') != SCHEMA
                or value.get('run_id') != report['run_id']
                or value.get('status') != 'running'
                or value.get('report_checkpoint') != _checkpoint(report_path)):
            return result
        result.update({key: value.get(key) for key in FIELDS})
        result['heartbeat_unix_s'] = value['written_at_unix_s']
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return result


def write_execution_progress(root, run_id, token):
    """Only a changed algorithm step refreshes the worker watchdog."""
    from .run_state import write_json
    write_json(Path(root) / 'execution_progress.json', {'run_id': run_id, 'token': token})


def read_execution_progress(root, run_id):
    try:
        data = json.loads((Path(root) / 'execution_progress.json').read_text())
        return data.get('token') if data.get('run_id') == run_id else None
    except (OSError, ValueError, AttributeError):
        return None
