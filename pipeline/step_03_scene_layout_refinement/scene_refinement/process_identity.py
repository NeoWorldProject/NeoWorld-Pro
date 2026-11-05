"""Linux process lifetime identity, preventing orphaned resident GPU leases."""
import os
from pathlib import Path


def process_identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    values = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(values[19])}


def owner_alive(identity):
    try:
        return process_identity(identity['pid']) == identity
    except (OSError, KeyError, ValueError, IndexError):
        return False
