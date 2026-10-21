"""Opt-in rebuildable Isaac shader caches, separate from immutable run evidence.

No simulation, asset, material or acceptance parameter is changed here. Existing
bound/local caches are never removed. Call preparation only while holding the
same exclusive GPU lock used for the whole native worker lifetime.
"""
from __future__ import annotations

import fcntl
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

CACHE_ENV = "NEOWORLD_ISAAC_CACHE_ROOT"
GPU_LOCK = Path("/tmp/neoworld_isaac_gpu.lock")
POLICY_FILE = "runtime_cache_policy.json"
POLICY_VERSION = "isaac_runtime_cache.v1"
SHARED_DIRS = ("nv_shadercache", "shadercache")
# These directories contain disposable engine/compiler state, never trial data.
REBUILDABLE = ("runtime/cache", "runtime/cuda", "runtime/shader", "runtime/kit/cache")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _require_gpu_lock(fd, lock_path=GPU_LOCK):
    if type(fd) is not int or fd < 0:
        raise ValueError("Shared Isaac cache requires the live exclusive GPU lock descriptor")
    expected, actual = Path(lock_path).stat(), os.fstat(fd)
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("Cache lock descriptor does not identify the shared GPU lock")
    # A separate open description must be blocked; the supplied description
    # must already own (or inherit) that same lock. Do not acquire a fresh lock
    # implicitly and leave it held after this helper returns.
    with Path(lock_path).open("a") as probe:
        try:
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
            raise ValueError("GPU lock must already be held before cache preparation")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise ValueError("The supplied descriptor does not own the GPU lock") from error


@contextmanager
def gpu_worker_slot(inherited_fd=None, *, _lock_path=GPU_LOCK):
    """Hold one GPU slot, or borrow the descriptor passed by orchestration.

    The parent must pass inherited_fd with subprocess pass_fds. A child never
    unlocks that borrowed description; its owner releases it after supervision.
    """
    if inherited_fd is not None:
        _require_gpu_lock(inherited_fd, _lock_path)
        yield inherited_fd
        return
    with Path(_lock_path).open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield lock.fileno()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def cache_identity(isaac_python, *, gpu_devices=None):
    """Fingerprint installed build bytes and visible device/driver identities.

    nvidia-smi is an informational query, not a simulation. Tests can inject its
    already parsed device identity without accessing a GPU or subprocess.
    """
    launcher = Path(isaac_python).resolve()
    if not launcher.is_file():
        raise ValueError("Cannot identify missing Isaac launcher")
    package = launcher.parent
    build = {"python.sh": _sha(launcher)}
    for relative in ("VERSION", "version.txt", "kit/VERSION", "kit/kit", "kit/kernel/BUILD_INFO"):
        path = package / relative
        if path.is_file():
            build[relative] = _sha(path)
    if gpu_devices is None:
        result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version",
                                 "--format=csv,noheader,nounits"], check=True, capture_output=True,
                                text=True, timeout=10)
        gpu_devices = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    if (not isinstance(gpu_devices, list) or not gpu_devices
            or any(not isinstance(v, str) or not v.strip() for v in gpu_devices)):
        raise ValueError("Cannot identify GPU and driver for shared cache")
    return {"build_sha256": build, "device_driver": sorted(gpu_devices),
            "visibility": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")}}


def prepare_runtime_cache(out_dir, isaac_python, *, gpu_lock_fd, environ=None, identity=None, _lock_path=GPU_LOCK):
    """Create only the two shader symlinks; return a per-run immutable receipt.

    No configured root means isolated caches. Missing driver identification
    also falls back to isolated caches, with an explicit reason. Existing local
    directories stay local; foreign symlinks are rejected rather than replaced.
    """
    out_dir = Path(out_dir).resolve()
    environment = os.environ if environ is None else environ
    configured = environment.get(CACHE_ENV)
    requested_root = str(Path(configured).expanduser().resolve()) if configured else None
    receipt = out_dir / POLICY_FILE
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old.get("implementation_sha256") != _sha(__file__) or old.get("requested_cache_root") != requested_root:
            raise ValueError("Existing runtime cache policy is immutable; use a new run for a new policy")
    policy = {"schema_version": POLICY_VERSION, "cache_environment": CACHE_ENV,
              "requested_cache_root": requested_root,
              "enabled": False, "reason": "not_configured", "shared": {}, "retained_local": [],
              "rebuildable_excluded_from_new_checkpoints": list(REBUILDABLE),
              "private_run_state": ["runtime/config", "runtime/data", "runtime/tmp", "runtime/kit/cache/DerivedDataCache",
                                    "runtime/cache/cookedmeshcache", "kit.log", "isaac_stdout.log", "trials"],
              "implementation_sha256": _sha(__file__), "physical_parameters_changed": False}
    if configured:
        _require_gpu_lock(gpu_lock_fd, _lock_path)
        configured = Path(configured).expanduser()
        if not configured.is_absolute():
            raise ValueError(CACHE_ENV + " must be an absolute path outside the individual run")
        cache_root = configured.resolve()
        if cache_root == out_dir or cache_root.is_relative_to(out_dir) or out_dir.is_relative_to(cache_root):
            raise ValueError("Shared cache root must be independent of the individual run output")
        if identity is None:
            try:
                identity = cache_identity(isaac_python)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                policy.update(reason="identity_unavailable", identity_error_type=type(error).__name__)
        if identity is not None:
            if not isinstance(identity, dict) or not identity:
                raise ValueError("Shared cache needs a nonempty build/device-driver identity")
            bucket = cache_root / ("isaac_" + _json_hash(identity))
            policy.update(reason="configured", cache_root=str(cache_root), bucket=str(bucket), identity=identity)
            for name in SHARED_DIRS:
                local = out_dir / "runtime/kit/cache" / name
                target = bucket / name
                if local.is_symlink():
                    if local.resolve() != target.resolve():
                        raise ValueError("Existing shader cache symlink belongs to a different identity/root")
                elif local.exists():
                    # Even an empty directory might belong to an earlier run.
                    # Never delete/rewrite it or alter its immutable receipt.
                    policy["retained_local"].append(str(local.relative_to(out_dir)))
                    continue
                else:
                    target.mkdir(parents=True, exist_ok=True)
                    local.parent.mkdir(parents=True, exist_ok=True)
                    local.symlink_to(target, target_is_directory=True)
                policy["shared"][str(local.relative_to(out_dir))] = str(target)
            policy["enabled"] = bool(policy["shared"])
    if receipt.exists() and json.loads(receipt.read_text()) != policy:
        raise ValueError("Existing runtime cache policy is immutable; use a new run for a new policy")
    if not receipt.exists():
        _write(receipt, policy)
    return policy


def _declared_cache_roots(directory):
    policy_path = Path(directory) / POLICY_FILE
    if not policy_path.is_file():
        return ()
    policy = json.loads(policy_path.read_text())
    if (policy.get("schema_version") != POLICY_VERSION
            or policy.get("rebuildable_excluded_from_new_checkpoints") != list(REBUILDABLE)):
        raise ValueError("Unsupported or modified runtime cache collection policy")
    return tuple(Path(directory) / name for name in REBUILDABLE)


def evidence_files(root):
    """Exclude only caches declared by this policy; preserve all legacy files.

    os.walk does not follow directory symlinks. Logs/config/data/tmp and native
    evidence stay in the immutable checkpoint, including the cache policy file.
    """
    root = Path(root)
    excluded = []
    result = []
    for directory, folders, names in os.walk(root, followlinks=False):
        directory = Path(directory)
        excluded.extend(_declared_cache_roots(directory))
        folders[:] = sorted(name for name in folders
                            if name != "__pycache__" and directory / name not in excluded)
        for name in sorted(names):
            path = directory / name
            if path.suffix not in {".pyc", ".tmp"} and path.is_file():
                result.append(path)
    return sorted(result)


def _manifest_bound_paths(source):
    path = source / "run_manifest.json"
    if not path.is_file():
        return set()
    manifest = json.loads(path.read_text())
    result = set()
    for stage in manifest.get("stages", {}).values():
        for relative in stage.get("artifacts", {}):
            relative = Path(relative)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Source manifest artifact escapes its run")
            result.add(source / relative)
    return result


def copy_verified_run(source, destination):
    """Copy a previously verified source without expanding new shared caches.

    Caller must verify the source before and the copied run after this operation.
    Caches required by a historical manifest are preserved. No source file or
    manifest is edited and no derived acceptance claim is created.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Verified source and copied run must be independent")
    bound, excluded, retained = _manifest_bound_paths(source), [], []
    # Historical runs have no cache receipt but may bind every cache file. Make
    # their retained storage explicit without changing their old manifests.
    for path in bound:
        parts = path.relative_to(source).parts
        for cache in REBUILDABLE:
            pattern = Path(cache).parts
            for offset in range(len(parts) - len(pattern) + 1):
                if parts[offset:offset + len(pattern)] == pattern:
                    retained.append(str(Path(*parts[:offset + len(pattern)])))
    def ignore(directory, names):
        directory = Path(directory)
        # Policies can belong to nested Isaac evaluations in the source run.
        for candidate in _declared_cache_roots(directory):
            if any(p == candidate or p.is_relative_to(candidate) for p in bound):
                retained.append(str(candidate.relative_to(source)))
            else:
                excluded.append(candidate)
        return [name for name in names if directory / name in excluded or name == "__pycache__"]
    shutil.copytree(source, destination, ignore=ignore)
    receipt = {"schema_version": "isaac_cache_copy.v1", "source": str(source),
               "source_manifest_sha256": _sha(source / "run_manifest.json"),
               "excluded_rebuildable_cache_roots": sorted(str(p.relative_to(source)) for p in excluded),
               "retained_legacy_bound_cache_roots": sorted(set(retained)),
               "source_modified": False, "physical_parameters_changed": False}
    _write(destination / "cache_copy_receipt.json", receipt)
    return receipt
