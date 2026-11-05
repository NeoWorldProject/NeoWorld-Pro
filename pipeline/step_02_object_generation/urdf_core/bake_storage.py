"""Baked corner UVs reference the exported OBJ; geometry is not copied."""
from pathlib import Path
import hashlib
import os
import uuid

import numpy as np

from .run_state import file_hash
from .storage import require_space


def read_obj(path):
    points, faces = [], []
    with Path(path).open() as stream:
        for line in stream:
            tokens = line.split('#', 1)[0].split()
            if not tokens:
                continue
            if tokens[0] == 'v':
                if len(tokens) != 4:
                    raise ValueError('Expected three coordinates per exported OBJ vertex')
                points.append([float(value) for value in tokens[1:]])
            elif tokens[0] == 'f':
                if len(tokens) != 4:
                    raise ValueError('Expected triangular exported OBJ faces')
                indices = [int(value.split('/')[0]) for value in tokens[1:]]
                if any(index == 0 for index in indices):
                    raise ValueError('Invalid exported OBJ vertex index')
                faces.append([index-1 if index > 0 else len(points)+index for index in indices])
    points, faces = np.asarray(points, dtype=float), np.asarray(faces, dtype=np.int64)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces)
            or faces.min() < 0 or faces.max() >= len(points)):
        raise ValueError('Invalid exported OBJ geometry')
    return points, faces


def _backface_hash(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def store_part(directory, entry, references, uvs):
    path = Path(entry['path']).resolve()
    points, faces = read_obj(path)
    refs, uvs = np.asarray(references, dtype='<f8'), np.asarray(uvs, dtype='<f8')
    count = len(entry['source_triangle_vertices'])
    if (refs.shape != (count, 3, 3) or uvs.shape != (count, 3, 2)
            or not np.isfinite(refs).all() or not np.isfinite(uvs).all()):
        raise ValueError('Invalid baked triangle/UV evidence')
    # Blender stores float32 vertices; the OBJ's 12 significant digits preserve them.
    recovered = points.astype(np.float32).astype('<f8')[faces[:count]]
    changed = np.flatnonzero(recovered.reshape(-1).view('<u8') != refs.reshape(-1).view('<u8'))
    arrays = {'uvs': uvs}
    if len(changed):
        arrays.update(reference_override_indices=changed.astype('<i8'),
                      reference_override_values=refs.reshape(-1)[changed])
    target = Path(directory)/(path.stem+'.uv.npz')
    target.parent.mkdir(parents=True, exist_ok=True)
    require_space(target.parent, sum(a.nbytes for a in arrays.values()) + 65536)
    temporary = target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.pending')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {'stem':path.stem, 'material_index':entry['material_index'],
            'geometry_reference':{'policy':'bound_obj_float32.v1', 'path':str(path),
                'sha256':file_hash(path), 'triangle_count':count,
                'backface_sha256':_backface_hash(entry.get('backface_source_triangle_indices', []))},
            'uv_storage':{'path':str(target.resolve()), 'sha256':file_hash(target)}}


def load_part(part, entry, points=None, faces=None):
    ref = part['geometry_reference']
    path = Path(entry['path']).resolve()
    backfaces = entry.get('backface_source_triangle_indices', [])
    count = len(entry['source_triangle_vertices'])
    if (ref.get('policy') != 'bound_obj_float32.v1' or Path(ref['path']).resolve() != path
            or file_hash(path) != ref['sha256'] or ref['triangle_count'] != count
            or ref['backface_sha256'] != _backface_hash(backfaces)):
        raise ValueError('Baked OBJ identity changed')
    if points is None:
        points, faces = read_obj(path)
    references = points.astype(np.float32).astype('<f8')[faces[:count]]
    storage = part['uv_storage']
    if file_hash(storage['path']) != storage['sha256']:
        raise ValueError('Baked UV dependency changed')
    with np.load(storage['path'], allow_pickle=False) as archive:
        allowed = {'uvs', 'reference_override_indices', 'reference_override_values'}
        if set(archive.files)-allowed or ('reference_override_indices' in archive) != ('reference_override_values' in archive):
            raise ValueError('Invalid baked UV storage')
        uv = archive['uvs']
        if uv.dtype != np.dtype('<f8') or uv.shape != (count, 3, 2) or not np.isfinite(uv).all():
            raise ValueError('Invalid baked UV array')
        if 'reference_override_indices' in archive:
            indices, values = archive['reference_override_indices'], archive['reference_override_values']
            if (indices.dtype.kind not in 'iu' or indices.ndim != 1 or values.shape != indices.shape
                    or indices.min(initial=0) < 0 or indices.max(initial=-1) >= references.size
                    or not np.isfinite(values).all() or len(np.unique(indices)) != len(indices)):
                raise ValueError('Invalid baked coordinate corrections')
            references.reshape(-1)[indices] = values
    return references, uv, backfaces


def dependencies(receipt):
    return {part['uv_storage']['path']:part['uv_storage']['sha256']
            for obj in receipt['objects'] for part in obj['parts'] if 'uv_storage' in part}
