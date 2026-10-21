"""Reuse immutable physicalization only for exactly equal physical inputs.

Controller/test-order changes do not change geometry, CoACD, mass or friction.
Copies are independent (no writable hardlinks), and every producer byte is
verified before reuse. Physics measurements are never reused by this cache.
"""
import json
from pathlib import Path
import shutil

from urdf_core.run_state import file_hash, fingerprint, write_json


def check_physical_proposal(base, profile, root, config):
    """Check physicalization while a proposal can still spend its content budget.

    A successful preflight is the producer for the normal exact-input cache;
    simulation uses an independent copy. Rejected evidence remains on disk.
    """
    from .physicalize import physicalize_asset
    root = Path(root)
    folder = root / 'physical_preflight' / fingerprint({'profile': profile, 'config': config})
    receipt_path = folder / 'receipt.json'
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if any(file_hash(p) != h for p, h in receipt['files'].items()):
            raise RuntimeError('Frozen physical preflight evidence changed')
        if receipt['status'] == 'rejected':
            raise ValueError(receipt['error'])
        return
    folder.mkdir(parents=True, exist_ok=True)
    # A partial producer has never been published. Retain it before replay.
    asset = folder / 'asset'
    if asset.exists():
        import time
        asset.rename(folder / ('interrupted_' + str(time.time_ns())))
    error = None
    try:
        prepare_asset(base, profile, asset, config=config,
            cache_root=root/'physical_asset_cache', physicalize=physicalize_asset)
    except ValueError as exc:
        error = str(exc)
    files = {str(p): file_hash(p) for p in asset.rglob('*') if p.is_file()}
    write_json(receipt_path, {'status': 'rejected' if error else 'passed', 'error': error, 'files': files})
    if error:
        raise ValueError(error)


def prepare_asset(base, profile, out_dir, *, config, cache_root, physicalize):
    from importlib.metadata import version
    from .protocol import load_config
    from . import physicalize as physical_module, lightweight_collision, coacd_worker, collision_reuse, collision_planning, protocol, geometry_checks
    from urdf_core import urdf_validation, kinematics
    base, out_dir, cache_root = map(lambda p: Path(p).resolve(), (base, out_dir, cache_root))
    # Exact source inventory, including authored meshes/part geometry. Never trust
    # a manifest hash alone when its referenced files might have changed.
    sources = [base / 'kinematics.json', *sorted((base / 'urdf_export').rglob('*'))]
    if (base / 'geometry_inventory').is_dir():
        sources += sorted((base / 'geometry_inventory').rglob('*'))
    key = fingerprint({'source': {str(p): file_hash(p) for p in sources if p.is_file()},
                       'profile': profile, 'config': load_config(config),
                       'implementation': {m.__name__: file_hash(m.__file__)
                                          for m in (physical_module, lightweight_collision, coacd_worker, collision_reuse, collision_planning,
                                                    protocol, geometry_checks, urdf_validation, kinematics)},
                       'versions': {n: version(n) for n in ('numpy', 'scipy', 'trimesh', 'coacd', 'fast-simplification')}})
    index = cache_root / f'{key}.json'
    if index.is_file():
        record = json.loads(index.read_text()); source = Path(record['producer'])
        manifest = json.loads((source / 'asset_manifest.json').read_text())
        if record['manifest_sha256'] != file_hash(source / 'asset_manifest.json'):
            raise ValueError('Cached physicalization manifest changed')
        for relative, digest in manifest['artifacts'].items():
            path = source / relative
            if not path.resolve().is_relative_to(source) or file_hash(path) != digest:
                raise ValueError('Cached physicalization artifact changed')
        if out_dir.exists() and any(out_dir.iterdir()):
            raise ValueError('Cannot replace an existing candidate asset')
        out_dir.mkdir(parents=True, exist_ok=True)
        for relative in manifest['artifacts']:
            target = out_dir / relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
        for field, name in [('urdf_path', 'object.urdf'), ('profile_path', 'physics_profile.json'),
                            ('spec_path', 'kinematics.json')]:
            manifest[field] = str(out_dir / name)
        write_json(out_dir / 'asset_manifest.json', manifest)
        return {'policy': 'exact_physicalization_inputs.v1', 'reused': True,
                'producer': str(source), 'key': key}
    physicalize(base, profile, out_dir, config=config, collision_cache_root=cache_root / 'collision_parts')
    cache_root.mkdir(parents=True, exist_ok=True)
    write_json(index, {'producer': str(out_dir), 'manifest_sha256': file_hash(out_dir / 'asset_manifest.json')})
    return {'policy': 'exact_physicalization_inputs.v1', 'reused': False, 'key': key}
