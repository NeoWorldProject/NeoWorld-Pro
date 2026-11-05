"""Content-bound checkpoints. A file's presence is never a success signal."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock


# Process-local only: a new worker always reads the content again. Identity,
# size, nanosecond mtime AND ctime invalidate replaced or edited evidence.
_HASH_CACHE = OrderedDict()
_HASH_CACHE_LOCK = Lock()
_HASH_CACHE_LIMIT = 16384


def _file_version(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _cacheable(version):
    # Even nanosecond fields can have coarse effective clock resolution. Never
    # reuse hashes of recently written files whose edits may share one tick.
    return time.time_ns() - version[-1] >= 2_000_000_000


def file_hash(path: Path) -> str:
    path = Path(path).resolve()
    before = _file_version(path.stat())
    key = (str(path), before)
    with _HASH_CACHE_LOCK:
        cached = _HASH_CACHE.get(key)
        if cached is not None:
            _HASH_CACHE.move_to_end(key)
    if cached is not None and _cacheable(before) and _file_version(path.stat()) == before:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = _file_version(os.fstat(stream.fileno()))
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if (opened != _file_version(os.fstat(stream.fileno()))
                or opened != _file_version(path.stat())):
            raise ValueError("File changed while hashing: " + str(path))
    value = digest.hexdigest()
    if _cacheable(opened):
        with _HASH_CACHE_LOCK:
            _HASH_CACHE[(str(path), opened)] = value
            while len(_HASH_CACHE) > _HASH_CACHE_LIMIT:
                _HASH_CACHE.popitem(last=False)
    return value


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


class RunState:
    def __init__(self, root: Path, fresh: bool, config: dict):
        self.root = root.resolve()
        if fresh and self.root.exists() and any(self.root.iterdir()):
            raise ValueError("--fresh requires a nonexistent or empty output directory; nothing was deleted")
        self.path = self.root / "run_manifest.json"
        if self.root.exists() and any(self.root.iterdir()) and not self.path.is_file():
            raise ValueError("Output has no versioned run manifest. Use --fresh with a new directory.")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data.get("schema_version") != 1:
                raise ValueError("Unsupported run manifest version; use a new output directory")
        else:
            self.data = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                         "stages": {}}
        self.data.update(config=config, status="running")
        self.save()

    def save(self):
        write_json(self.path, self.data)

    def current(self, name: str, key: str) -> bool:
        stage = self.data["stages"].get(name, {})
        if stage.get("key") != key or stage.get("status") != "passed" or not stage.get("artifacts"):
            return False
        for relative, digest in stage["artifacts"].items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root) or not path.is_file() or file_hash(path) != digest:
                return False
        return True

    def begin(self, name: str, key: str):
        # Invalidate before executing: a failed attempt cannot retain an old pass.
        self.data["stages"][name] = {"key": key, "status": "running"}
        self.data["status"] = "running"
        self.save()

    def passed(self, name: str, key: str, paths: list[Path], **details):
        artifacts = {}
        for path in paths:
            path = path.resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"Checkpoint artifact must be a current-run file: {path}")
            artifacts[str(path.relative_to(self.root))] = file_hash(path)
        if not artifacts:
            raise ValueError("A passed checkpoint must have artifacts")
        self.data["stages"][name] = {"key": key, "status": "passed", "artifacts": artifacts, **details}
        self.save()

    def fail(self, error: str):
        # Preserve verified upstream checkpoints, but do not leave an active
        # stage looking as if its worker were still running after an exception.
        for stage in self.data.get("stages", {}).values():
            if isinstance(stage, dict) and stage.get("status") == "running":
                stage.update(status="failed", error=error)
        self.data.update(status="failed", error=error)
        self.save()
