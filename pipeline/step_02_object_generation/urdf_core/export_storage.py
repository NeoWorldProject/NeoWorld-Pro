"""Lossless array storage for mesh-identity maps; legacy JSON remains readable."""
import json
from pathlib import Path
import numpy as np
from .run_state import file_hash, write_json
from .storage import require_space

ARRAY_FIELDS = ('source_vertex_indices', 'source_triangle_vertices', 'backface_source_triangle_indices')


def write_export_manifest(path, payload):
    path = Path(path)
    data = {**payload, 'meshes': [dict(mesh) for mesh in payload['meshes']]}
    arrays = {}
    for i, mesh in enumerate(data['meshes']):
        for field in ARRAY_FIELDS:
            if field not in mesh:
                continue
            key = f'mesh_{i}_{field}'
            value = np.asarray(mesh[field], dtype=np.int64)
            arrays[key] = value
            mesh[field] = {'array': key}
    sidecar = path.with_name('export_arrays.npz')
    require_space(path.parent, sum(a.nbytes for a in arrays.values()))
    temporary = sidecar.with_suffix('.npz.pending')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    data['array_storage'] = {'schema_version': 1, 'path': sidecar.name, 'sha256': file_hash(sidecar)}
    write_json(path, data)


def array_dependency(path, data=None):
    path = Path(path).resolve()
    data = data if data is not None else json.loads(path.read_text())
    ref = data.get('array_storage')
    if ref is None:
        return None
    if ref.get('schema_version') != 1:
        raise ValueError('Unknown export array format')
    sidecar = (path.parent/ref['path']).resolve()
    if not sidecar.is_relative_to(path.parent) or file_hash(sidecar) != ref['sha256']:
        raise ValueError('Export array identity changed')
    return sidecar


def read_export_manifest(path):
    data = json.loads(Path(path).read_text())
    sidecar = array_dependency(path, data)
    if sidecar is None:
        return data
    with np.load(sidecar, allow_pickle=False) as arrays:
        for mesh in data['meshes']:
            for field in ARRAY_FIELDS:
                if field not in mesh:
                    continue
                ref = mesh[field]
                if not isinstance(ref, dict) or set(ref) != {'array'}:
                    raise ValueError('Invalid export array reference')
                a = arrays[ref['array']]
                if a.dtype.kind not in 'iu' or (a < 0).any():
                    raise ValueError('Invalid export indices')
                if field == 'source_triangle_vertices':
                    if a.size and (a.ndim != 2 or a.shape[1] != 3):
                        raise ValueError('Invalid export triangle shape')
                elif a.ndim != 1:
                    raise ValueError('Invalid export index shape')
                mesh[field] = a.tolist()
    return data
