"""Processed native evidence: exact retained samples, never a full-rate trace."""
from __future__ import annotations

from bisect import bisect_left
from collections import OrderedDict
from copy import deepcopy
import gzip
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
import tempfile
import time
import zlib

import numpy as np

SCHEMA = "native_processed_trajectory.v1"
DEFAULT_REPLAY_FIELDS = ("time_s", "link_transforms", "joint_positions", "geometry.passed")


def is_compact_path(path):
    return str(path).endswith(".compact.json")


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _reject_constant(value):
    raise ValueError("Nonfinite constant in native trajectory: " + value)


def _read_json(value):
    return json.loads(value, parse_constant=_reject_constant)


def _raw_samples(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else lzma.open if path.suffix == ".xz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            for ordinal, line in enumerate(stream):
                sample = _read_json(line)
                if not isinstance(sample, dict):
                    raise ValueError(f"Native sample {ordinal} is not an object")
                yield sample
    except (OSError, EOFError, lzma.LZMAError, UnicodeError, zlib.error) as error:
        raise ValueError("Invalid or truncated native trajectory") from error


def _indices(values):
    values = list(values)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Retained sample indices must be nonnegative integers")
    return sorted(set(values))


def _project(sample, fields):
    result = {}
    for path in fields:
        source, target = sample, result
        names = path.split(".")
        for name in names[:-1]:
            if not isinstance(source, dict) or name not in source:
                raise ValueError("Replay field is absent: " + path)
            source = source[name]
            target = target.setdefault(name, {})
        if not isinstance(source, dict) or names[-1] not in source:
            raise ValueError("Replay field is absent: " + path)
        target[names[-1]] = source[names[-1]]
    return result


def _number_kinds(value):
    if isinstance(value, list):
        return [_number_kinds(item) for item in value]
    if type(value) not in (int, float):
        raise TypeError
    return isinstance(value, int)


def _pack(values, arrays):
    """Columnar fixed structures; variable event payloads remain exact JSON."""
    first = values[0]
    if isinstance(first, dict) and all(isinstance(v, dict) and set(v) == set(first) for v in values):
        return {"kind": "dict", "items": {key: _pack([v[key] for v in values], arrays) for key in sorted(first)}}
    if all(type(v) is bool for v in values):
        array = np.asarray(values, dtype=np.bool_)
        kind = "bool"
    else:
        try:
            kinds = np.asarray([_number_kinds(v) for v in values], dtype=np.bool_)
            array = np.asarray(values)
            if array.dtype.kind not in "ifu" or array.dtype == object:
                raise TypeError
            if array.dtype.kind in "iu":
                array = np.asarray(values, dtype="<i8")
            else:
                array = np.asarray(values, dtype="<f8")
                if not np.isfinite(array).all():
                    raise ValueError("Nonfinite retained native value")
                # Mixed huge integers cannot be represented exactly by float64.
                if kinds.any() and np.any(np.abs(array[kinds]) > 2**53):
                    raise TypeError
            kind = "numeric"
        except (TypeError, OverflowError, ValueError):
            encoded = [_json_bytes(value) for value in values]
            offsets = np.concatenate(([0], np.cumsum([len(v) for v in encoded], dtype=np.int64)))
            key = "a" + str(len(arrays))
            arrays[key] = np.frombuffer(b"".join(encoded), dtype=np.uint8)
            arrays[key + "_offsets"] = offsets
            return {"kind": "json", "array": key}
    key = "a" + str(len(arrays))
    arrays[key] = array
    spec = {"kind": kind, "array": key}
    if kind == "numeric" and array.dtype.kind == "f" and kinds.any():
        arrays[key + "_ints"] = kinds
        spec["int_mask"] = key + "_ints"
    return spec


def _restore_ints(value, mask):
    if isinstance(value, list):
        return [_restore_ints(v, m) for v, m in zip(value, mask)]
    return int(value) if mask else value


def _unpack(spec, arrays, row):
    kind = spec["kind"]
    if kind == "dict":
        return {name: _unpack(child, arrays, row) for name, child in spec["items"].items()}
    key = spec["array"]
    if kind == "json":
        offsets = arrays[key + "_offsets"]
        return _read_json(arrays[key][int(offsets[row]):int(offsets[row + 1])].tobytes())
    value = arrays[key][row].tolist()
    if "int_mask" in spec:
        value = _restore_ints(value, arrays[spec["int_mask"]][row].tolist())
    return value


def _publish_arrays(folder, arrays):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".native_samples_", suffix=".npz", dir=folder, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        digest = _hash(temporary)
        path = folder / ("native_samples_" + digest + ".npz")
        if path.exists():
            if _hash(path) != digest:
                raise ValueError("Existing native sample block changed")
            temporary.unlink()
        else:
            os.replace(temporary, path)
        return {"path": path.name, "sha256": digest, "size_bytes": path.stat().st_size}
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class _BlockWriter:
    def __init__(self, folder, block_size):
        self.folder, self.block_size = folder, block_size
        self.indices, self.samples, self.blocks = [], [], []

    def add(self, index, sample):
        self.indices.append(index)
        self.samples.append(sample)
        if len(self.samples) >= self.block_size:
            self.flush()

    def flush(self):
        if not self.samples:
            return
        arrays = {"sample_indices": np.asarray(self.indices, dtype="<i8")}
        structure = _pack(self.samples, arrays)
        item = _publish_arrays(self.folder, arrays)
        item.update(count=len(self.indices), first_index=self.indices[0], last_index=self.indices[-1], structure=structure)
        self.blocks.append(item)
        self.indices, self.samples = [], []


def build_compact_trajectory(raw_path, manifest_path, *, verified_metrics,
                             diagnostic_indices=(), replay_indices=None, fps=12.,
                             expected_raw_sha256=None, binding=None, extras=None,
                             replay_fields=DEFAULT_REPLAY_FIELDS, block_size=128,
                             validated_samples=None):
    """Commit processed evidence after the caller's full-rate independent check.

    Raw data is fully decoded and hashed before commit, but is never deleted here.
    A caller may reuse samples from its just-completed full decode and independent
    evaluation; the original file's verified hash is then required and rechecked.
    The caller commits its checkpoint before removing its own temporary raw file.
    """
    started = time.monotonic()
    raw_path, manifest_path = Path(raw_path), Path(manifest_path)
    if not isinstance(verified_metrics, dict):
        raise ValueError("Processed evidence requires verified full-rate metrics")
    if validated_samples is not None and expected_raw_sha256 is None:
        raise ValueError("Reusing validated samples requires the verified original trajectory hash")
    if type(block_size) is not int or not 1 <= block_size <= 4096:
        raise ValueError("Invalid native sample block size")
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid replay rate")
    replay_fields = tuple(replay_fields)
    if not replay_fields or any(not isinstance(v, str) or not v for v in replay_fields):
        raise ValueError("Replay fields must be explicit field paths")
    _json_bytes({"metrics": verified_metrics, "binding": binding, "extras": extras})
    diagnostic = _indices(diagnostic_indices)
    replay = None if replay_indices is None else _indices(replay_indices)
    diagnostic_set, replay_set = set(diagnostic), None if replay is None else set(replay)
    before = raw_path.stat()
    raw_digest = _hash(raw_path)
    if expected_raw_sha256 is not None and raw_digest != expected_raw_sha256:
        raise ValueError("Raw native trajectory hash changed before compaction")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_writer = _BlockWriter(manifest_path.parent, block_size)
    replay_writer = _BlockWriter(manifest_path.parent, block_size)
    count, previous_time, last_frame = 0, None, -1
    initial = terminal = None
    replay_count, diagnostic_count, last_replay_index = 0, 0, None
    source_samples = _raw_samples(raw_path) if validated_samples is None else iter(validated_samples)
    for index, sample in enumerate(source_samples):
        if not isinstance(sample, dict):
            raise ValueError("Validated native sample is not an object")
        value = sample.get("time_s")
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or previous_time is not None and value <= previous_time):
            raise ValueError("Native sample times must strictly advance")
        if initial is None:
            initial = deepcopy(sample)
        terminal, previous_time, count = sample, value, index + 1
        if index in diagnostic_set:
            diagnostic_writer.add(index, sample)
            diagnostic_count += 1
        frame = math.floor(float(value) * fps + 1e-8)
        selected = index in replay_set if replay_set is not None else frame > last_frame
        if selected:
            replay_writer.add(index, _project(sample, replay_fields))
            last_frame, last_replay_index = frame, index
            replay_count += 1
    if not count:
        raise ValueError("Cannot compact an empty native trajectory")
    if diagnostic and diagnostic[-1] >= count or replay and replay[-1] >= count:
        raise ValueError("Retained sample schedule exceeds the complete native trace")
    if replay_set is None and last_replay_index != count - 1:
        replay_writer.add(count - 1, _project(terminal, replay_fields))
        replay_count += 1
    diagnostic_writer.flush()
    replay_writer.flush()
    endpoint_writer = _BlockWriter(manifest_path.parent, 2)
    endpoint_writer.add(0, initial)
    if count > 1:
        endpoint_writer.add(count - 1, terminal)
    endpoint_writer.flush()
    after = raw_path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or _hash(raw_path) != raw_digest:
        raise ValueError("Raw native trajectory changed during compaction")
    payload = {"schema": SCHEMA, "full_rate_samples_retained": False,
               "source": {"sha256": raw_digest, "size_bytes": before.st_size,
                          "sample_count": count, "first_time_s": initial["time_s"], "last_time_s": terminal["time_s"]},
               "binding": deepcopy(binding or {}), "verified_metrics": deepcopy(verified_metrics),
               "extras": deepcopy(extras or {}),
               "replay": {"fps": fps, "fields": list(replay_fields), "count": replay_count, "blocks": replay_writer.blocks},
               "diagnostic": {"count": diagnostic_count, "blocks": diagnostic_writer.blocks},
               "endpoints": {"count": min(count, 2), "blocks": endpoint_writer.blocks}}
    encoded = _json_bytes(payload)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="." + manifest_path.name + ".", dir=manifest_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if manifest_path.exists():
            if manifest_path.read_bytes() != encoded:
                raise ValueError("Processed native evidence cannot overwrite another receipt")
            temporary.unlink()
        else:
            os.replace(temporary, manifest_path)
        directory_fd = os.open(manifest_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        result = CompactTrajectory(manifest_path)
        return {"schema": SCHEMA, "path": str(manifest_path.resolve()), "sha256": _hash(manifest_path),
                "raw_sha256": raw_digest, "sample_count": count,
                "replay_count": replay_count, "diagnostic_count": diagnostic_count,
                "size_bytes": result.size_bytes, "source_size_bytes": before.st_size,
                "build_wall_seconds": time.monotonic() - started}
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class CompactTrajectory:
    """Checked sparse evidence; no full-rate iteration interface is provided."""
    def __init__(self, path, expected_sha256=None, *, expected_binding=None):
        self.path = Path(path)
        if expected_sha256 is not None and _hash(self.path) != expected_sha256:
            raise ValueError("Processed native evidence hash changed")
        self.manifest = _read_json(self.path.read_bytes())
        if (self.manifest.get("schema") != SCHEMA
                or self.manifest.get("full_rate_samples_retained") is not False):
            raise ValueError("Not supported processed native evidence")
        self.original_sample_count = self.manifest["source"]["sample_count"]
        self.original_sha256 = self.manifest["source"]["sha256"]
        self.original_stats = deepcopy(self.manifest["source"])
        if type(self.original_sample_count) is not int or self.original_sample_count < 1:
            raise ValueError("Invalid original sample count")
        if expected_binding is not None and self.manifest["binding"] != expected_binding:
            raise ValueError("Processed native evidence belongs to different evaluation inputs")
        self.binding = deepcopy(self.manifest["binding"])
        self.verified_metrics = deepcopy(self.manifest["verified_metrics"])
        self.extras = deepcopy(self.manifest["extras"])
        self._cache = OrderedDict()
        self._block_indices = {}
        self.size_bytes = self.path.stat().st_size
        self.files = {str(self.path.resolve()): _hash(self.path)}
        files = set()
        for section in ("replay", "diagnostic", "endpoints"):
            total, previous = 0, -1
            for block in self.manifest[section]["blocks"]:
                resolved = self._block_path(block)
                if resolved not in files:
                    self.size_bytes += resolved.stat().st_size
                    files.add(resolved)
                    self.files[str(resolved.resolve())] = block["sha256"]
                arrays = self._load_block(block)
                indices = arrays["sample_indices"]
                if (indices.dtype.kind not in "iu" or indices.ndim != 1 or len(indices) != block["count"]
                        or not len(indices) or int(indices[0]) <= previous
                        or np.any(indices[1:] <= indices[:-1]) or int(indices[-1]) >= self.original_sample_count
                        or int(indices[0]) != block["first_index"] or int(indices[-1]) != block["last_index"]):
                    raise ValueError("Invalid retained native sample schedule")
                for row in range(len(indices)):
                    sample = _unpack(block["structure"], arrays, row)
                    if not isinstance(sample, dict):
                        raise ValueError("Retained native sample is not an object")
                self._block_indices[(section, block["path"])] = indices.tolist()
                total += len(indices)
                previous = int(indices[-1])
            if total != self.manifest[section]["count"]:
                raise ValueError("Processed native sample count differs from blocks")
        endpoint_indices = [i for i, _ in self._samples("endpoints")]
        if endpoint_indices != sorted({0, self.original_sample_count - 1}):
            raise ValueError("Processed native endpoints are incomplete")

    def __iter__(self):
        raise TypeError("Processed native evidence is sparse; it cannot be replayed as full-rate physics measurements")

    def _block_path(self, block):
        name = block["path"]
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz"):
            raise ValueError("Native sample block must be local to its receipt")
        path = self.path.parent / name
        if path.is_symlink() or path.stat().st_size != block["size_bytes"] or _hash(path) != block["sha256"]:
            raise ValueError("Native sample block is missing or changed")
        return path

    def _load_block(self, block):
        key = block["path"]
        if key in self._cache:
            return self._cache[key]
        path = self._block_path(block)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if any(value.dtype.hasobject for value in arrays.values()):
            raise ValueError("Object arrays are not native numeric evidence")
        self._cache[key] = arrays
        while len(self._cache) > 2:
            self._cache.popitem(last=False)
        return arrays

    def _samples(self, section):
        for block in self.manifest[section]["blocks"]:
            arrays = self._load_block(block)
            for row, index in enumerate(arrays["sample_indices"]):
                yield int(index), _unpack(block["structure"], arrays, row)

    def replay_samples(self):
        return self._samples("replay")

    def diagnostic_samples(self):
        return self._samples("diagnostic")

    def sample_at(self, index, *, kind="replay"):
        if kind not in {"replay", "diagnostic", "endpoints"}:
            raise ValueError("Unknown retained sample kind")
        for block in self.manifest[kind]["blocks"]:
            if block["first_index"] <= index <= block["last_index"]:
                arrays = self._load_block(block)
                indices = arrays["sample_indices"]
                row = bisect_left(indices, index)
                if row < len(indices) and indices[row] == index:
                    return _unpack(block["structure"], arrays, row)
        raise KeyError(f"Native sample {index} was not retained as {kind}")

    @property
    def initial_sample(self):
        return self.sample_at(0, kind="endpoints")

    @property
    def terminal_sample(self):
        return self.sample_at(self.original_sample_count - 1, kind="endpoints")
