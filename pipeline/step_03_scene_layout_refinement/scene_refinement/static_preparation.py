"""Content-bound, cross-process preparation cache. Acceptance is always reverified."""
import fcntl
import inspect
import json
import os
from pathlib import Path

from .protocol import digest, file_hash, read_json, write_json


def shared_asset(record, rest_state_reader=None):
    from pxr import Usd
    import numpy
    import trimesh
    from .frozen_assets import verify_record
    from .assets import _load_published_asset
    checked = verify_record(record)
    here = Path(__file__).parent
    binding = {"record": record, "native_verification": checked,
        "implementation": {name: file_hash(here/name) for name in
                           ('assets.py', 'rest_state.py', 'initial_assembly.py', 'static_preparation.py')},
        "rest_reader": file_hash(inspect.getfile(rest_state_reader)) if rest_state_reader else None,
        "runtime": {"usd": list(Usd.GetVersion()), "numpy": numpy.__version__, "trimesh": trimesh.__version__}}
    key = digest(binding)
    root = Path(os.environ['NEOWORLD_SCENE_STATIC_CACHE']).resolve()/'assets'
    root.mkdir(parents=True, exist_ok=True)
    cache = root / (key+'.json')
    with (root/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.exists():
            receipt = read_json(root/(key+'.receipt.json'))
            if receipt['binding'] != binding or file_hash(cache) != receipt['sha256']:
                raise ValueError('Shared static asset cache changed')
            asset = read_json(cache)
            for path, expected in asset['source_hashes'].items():
                if file_hash(path) != expected:
                    raise ValueError('Shared static source changed: '+path)
            return asset
        asset = _load_published_asset(record, rest_state_reader=rest_state_reader, checked=checked)
        write_json(cache, asset)
        write_json(root/(key+'.receipt.json'), {'binding': binding, 'sha256': file_hash(cache)})
        return asset
