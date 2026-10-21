"""Cross-process resource coordination and content-free API timing evidence."""
import fcntl
import json
import os
from pathlib import Path
import time


def record_api_event(event, **fields):
    value = os.environ.get("URDF_API_METRICS_PATH")
    if not value:
        return
    path = Path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "time": time.time(), "pid": os.getpid(),
              "task_id": os.environ.get("URDF_TASK_ID"), **fields}
    # Do not record prompts, images, response bodies, credentials or error text.
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
