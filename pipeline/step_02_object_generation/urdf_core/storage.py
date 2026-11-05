"""Bounded copies and immutable delivery-file reuse."""
from contextlib import contextmanager
import errno
import fcntl
import json
import os
import re
from pathlib import Path
import shutil
import uuid
from .run_state import file_hash, write_json

RESERVE_BYTES = 700 * 2**20


def require_space(destination, additional_bytes, reserve=RESERVE_BYTES):
    destination = Path(destination).resolve()
    while not destination.exists():
        destination = destination.parent
    free = shutil.disk_usage(destination).free
    needed = int(additional_bytes) + reserve
    if free < needed:
        raise OSError(errno.ENOSPC, f'Storage preparation requires {needed} bytes '
                      f'(copy {additional_bytes}, reserve {reserve}); available {free}', str(destination))


def regular_files(root):
    root = Path(root).resolve()
    result = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Asset package must not contain symlinks: ' + str(path))
        if path.is_file():
            result[str(path.relative_to(root))] = path
    return result


@contextmanager
def publication_lock(store):
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    with (store / '.publication.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


@contextmanager
def blob_lock(pool):
    pool = Path(pool)
    pool.mkdir(parents=True, exist_ok=True)
    with (pool.parent/('.' + pool.name.lstrip('.') + '.publication.lock')).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def cleanup_pending(store):
    """Only remove our uncommitted staging directories while holding the store lock."""
    store = Path(store).resolve()
    protected = []
    for manifest in store.glob('*/delivery_manifest.json'):
        protected.append(manifest.read_text())
    for pending in store.glob('.pending_*'):
        marker = pending / '.storage_owner.json'
        if pending.is_symlink() or not marker.is_file() or (pending/'delivery_manifest.json').exists():
            continue
        try:
            owner = json.loads(marker.read_text())
        except (ValueError, OSError):
            continue
        if owner != {'policy': 'delivery_staging.v1', 'directory': pending.name}:
            continue
        if any(str(pending) in text for text in protected):
            continue
        shutil.rmtree(pending)


def asset_pool(destination):
    """Keep blobs within the owning run; source runs remain independent."""
    destination = Path(destination).absolute()
    for parent in destination.parents:
        if (parent/'run_manifest.json').is_file() or (parent/'pipeline_report.json').is_file() or (parent/'invocation.json').is_file():
            return parent/'.asset_blobs'
    parent = destination.parent
    while not parent.exists():
        parent = parent.parent
    return parent/'.asset_blobs'


def _stored_source(source, digest):
    source = Path(source).resolve()
    info = source.stat()
    if info.st_mode & 0o222:
        return False
    for parent in source.parents:
        pool = parent/'.asset_blobs'
        marker = pool/'.storage_owner.json'
        blob = pool/digest
        if not marker.is_file() or not blob.is_file() or blob.is_symlink():
            continue
        try:
            if json.loads(marker.read_text()) != {'policy':'immutable_delivery_blobs.v1'}:
                continue
        except (ValueError, OSError):
            continue
        if os.path.samefile(source, blob):
            return True
    return False


def _clone_file(source, target):
    """Use filesystem copy-on-write where available, otherwise copy bytes."""
    try:
        with Path(source).open('rb') as src, Path(target).open('wb') as dst:
            fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # Linux FICLONE.
        return True
    except OSError as error:
        if error.errno not in (errno.EOPNOTSUPP, errno.ENOTTY, errno.EXDEV, errno.EINVAL):
            raise
        Path(target).unlink(missing_ok=True)
        return False


def storage_bytes(source, pool, digest):
    pool = Path(pool)
    if (pool/digest).exists():
        return 0
    if _stored_source(source, digest):
        ancestor = pool
        while not ancestor.exists():
            ancestor = ancestor.parent
        if Path(source).stat().st_dev == ancestor.stat().st_dev:
            return 0
    return Path(source).stat().st_size


def _ensure_blob(source, pool, digest):
    source, pool = Path(source).resolve(), Path(pool)
    blob = pool/digest
    marker = pool/'.storage_owner.json'
    if not marker.exists():
        write_json(marker, {'policy':'immutable_delivery_blobs.v1'})
    elif marker.is_symlink() or json.loads(marker.read_text()) != {'policy':'immutable_delivery_blobs.v1'}:
        raise ValueError('Unrecognized immutable asset store: ' + str(pool))
    if blob.exists():
        if blob.is_symlink() or blob.stat().st_mode & 0o222 or file_hash(blob) != digest:
            raise ValueError('Immutable asset store changed: ' + str(blob))
    else:
        temporary = pool/('.pending_' + uuid.uuid4().hex)
        try:
            linked = False
            if _stored_source(source, digest):
                try:
                    os.link(source, temporary)
                    linked = True
                except OSError as error:
                    if error.errno != errno.EXDEV:
                        raise
            if not linked:
                require_space(pool, source.stat().st_size)
                if not _clone_file(source, temporary):
                    shutil.copyfile(source, temporary)
                temporary.chmod(0o444)
            if file_hash(temporary) != digest or file_hash(source) != digest:
                raise ValueError('Asset changed during publication: ' + str(source))
            os.replace(temporary, blob)
        finally:
            temporary.unlink(missing_ok=True)
    if file_hash(source) != digest:
        raise ValueError('Asset changed during publication: ' + str(source))
    return blob


def copy_asset(source, target, *, pool=None, digest=None, writable=False):
    """Atomic copy; immutable targets share bytes, writable targets never share inodes."""
    source, target = Path(source).resolve(), Path(target).absolute()
    if not source.is_file() or target.is_symlink():
        raise ValueError('Asset copy requires regular file endpoints')
    digest = digest or file_hash(source)
    if not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise ValueError('Invalid asset content digest')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name('.' + target.name + '.pending_' + uuid.uuid4().hex)
    try:
        if writable:
            require_space(target, source.stat().st_size)
            if not _clone_file(source, temporary):
                shutil.copyfile(source, temporary)
            temporary.chmod(0o644)
            if file_hash(temporary) != digest or file_hash(source) != digest:
                raise ValueError('Asset changed during copy: ' + str(source))
        else:
            pool = Path(pool) if pool is not None else asset_pool(target)
            with blob_lock(pool):
                blob = _ensure_blob(source, pool, digest)
                try:
                    os.link(blob, temporary)
                except OSError as error:
                    if error.errno != errno.EXDEV:
                        raise
                    require_space(target, blob.stat().st_size)
                    shutil.copyfile(blob, temporary)
                    temporary.chmod(0o444)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def make_writable(path):
    """Detach a shared asset before a tool that writes its input in place."""
    path = Path(path)
    if path.exists() and (path.stat().st_nlink > 1 or not path.stat().st_mode & 0o200):
        copy_asset(path, path, writable=True)
    return path


def freeze_file(path, *, pool=None):
    """Publish a newly finished local output without changing its bytes or pathname."""
    return copy_asset(path, path, pool=pool)


def copy_asset_tree(source, target, *, writable=(), pool=None):
    source, target = Path(source).resolve(), Path(target).absolute()
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Asset copies need independent directories')
    files = regular_files(source)
    if target.exists():
        raise ValueError('Asset tree destination already exists: ' + str(target))
    pool = Path(pool) if pool is not None else asset_pool(target)
    pending = target.with_name('.' + target.name + '.copy_' + uuid.uuid4().hex)
    pending.mkdir(parents=True)
    try:
        for relative, path in files.items():
            copy_asset(path, pending/relative, pool=pool, writable=relative in writable)
        pending.rename(target)
    except BaseException:
        shutil.rmtree(pending)
        collect_unreferenced_blobs(pool)
        raise
    return target


def link_immutable(source, target, pool, digest):
    if Path(target).exists():
        raise ValueError('Publication target already exists: ' + str(target))
    return copy_asset(source, target, pool=pool, digest=digest)


def _copy_bound_files(source, target, files, *, prepare=None):
    """Bounded import with immutable shared files; no recursive experiment copy."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists():
        raise ValueError('Import destination already exists: ' + str(target))
    hashes, unique = {}, {}
    for relative in sorted(files):
        path = source/relative
        if not path.resolve().is_relative_to(source) or not path.is_file():
            raise ValueError('Import dependency missing or outside source: ' + relative)
        digest = file_hash(path)
        hashes[relative] = digest
        unique.setdefault(digest, path)
    pool = asset_pool(target)
    require_space(target, sum(storage_bytes(path, pool, digest) for digest, path in unique.items()))
    staged = target.with_name('.' + target.name + '.import_' + uuid.uuid4().hex)
    try:
        staged.mkdir(parents=True)
        write_json(staged/'.storage_owner.json', {'policy':'selected_import_staging.v1', 'target':str(target)})
        for relative, digest in hashes.items():
            copy_asset(source/relative, staged/relative, pool=pool, digest=digest)
        if prepare is not None:
            prepare(staged, hashes)
        (staged/'.storage_owner.json').unlink()
        staged.rename(target)
    except BaseException:
        if staged.exists():
            shutil.rmtree(staged)
        collect_unreferenced_blobs(pool)
        raise
    return hashes


def copy_bound_files(source, target, files, *, prepare=None):
    target = Path(target).resolve()
    with publication_lock(target.parent):
        for staged in target.parent.glob('.' + target.name + '.import_*'):
            marker = staged/'.storage_owner.json'
            if staged.is_symlink() or not marker.is_file():
                continue
            try:
                owner = json.loads(marker.read_text())
            except (ValueError, OSError):
                continue
            if owner == {'policy':'selected_import_staging.v1', 'target':str(target)}:
                shutil.rmtree(staged)
        return _copy_bound_files(source, target, files, prepare=prepare)


def _collect_unreferenced_blobs(pool):
    """Called under the publication lock, after pending deliveries are removed."""
    pool = Path(pool)
    marker = pool/'.storage_owner.json'
    if not marker.is_file() or marker.is_symlink():
        return
    try:
        if json.loads(marker.read_text()) != {'policy':'immutable_delivery_blobs.v1'}:
            return
    except (ValueError, OSError):
        return
    for path in pool.iterdir():
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            continue
        if re.fullmatch(r'(?:[0-9a-f]{64}|\.pending_[0-9a-f]{32})', path.name):
            path.unlink()


def collect_unreferenced_blobs(pool):
    pool = Path(pool)
    if pool.exists():
        with blob_lock(pool):
            _collect_unreferenced_blobs(pool)
