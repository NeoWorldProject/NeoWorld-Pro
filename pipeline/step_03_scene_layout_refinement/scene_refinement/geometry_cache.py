"""Immutable local geometry and native cooking evidence; never simulation state."""
from collections import OrderedDict
from pathlib import Path
import time
import sys
import uuid
import zlib
import numpy as np

from .protocol import digest, read_json, write_json
from .hashing import file_hash

POLICY = 'scene_geometry_references.v1'


def freeze(meshes):
    result = []
    for mesh in meshes:
        v, f = np.asarray(mesh['vertices'], dtype=float), np.asarray(mesh['faces'], dtype=np.int64)
        if (v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all()
                or f.ndim != 2 or f.shape[1] != 3 or not len(f) or f.min() < 0 or f.max() >= len(v)):
            raise ValueError('Invalid cached mesh')
        v.flags.writeable = f.flags.writeable = False
        item = {'vertices':v, 'faces':f}
        if 'collider_index' in mesh:
            index = mesh['collider_index']
            if not isinstance(index, int) or index < 0:raise ValueError('Invalid collider index')
            item['collider_index'] = index
        result.append(item)
    if not result: raise ValueError('Empty cached geometry')
    return result


def storage_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size+(value.nbytes if not value.flags.owndata else 0)
    if isinstance(value, dict):return size+sum(storage_bytes(k,seen)+storage_bytes(v,seen) for k,v in value.items())
    if isinstance(value, (list,tuple)):return size+sum(storage_bytes(v,seen) for v in value)
    return size


class GeometryCache:
    def __init__(self, root, memory_mib=256):
        self.root = Path(root)
        self.limit = int(memory_mib*1024**2)
        self.memory, self.bytes = OrderedDict(), 0
        self.stats = {'memory_hits':0, 'disk_hits':0, 'misses':0, 'invalid_entries':0}

    def remember(self, key, meshes, receipt):
        size = storage_bytes((key,meshes,receipt))+128  # Include metadata and LRU bookkeeping.
        if key in self.memory: self.bytes -= self.memory.pop(key)[2]
        while self.memory and self.bytes+size > self.limit:
            _, (_,_,n) = self.memory.popitem(last=False); self.bytes -= n
        if size <= self.limit:
            self.memory[key] = (meshes, receipt, size); self.bytes += size

    def get(self, binding, build):
        key = digest(binding)
        directory = self.root/key
        pointer = directory/'current.json'
        try:
            if key in self.memory:
                meshes, receipt, _ = self.memory[key]
                if file_hash(receipt['path']) != receipt['sha256']: raise ValueError('Cached geometry changed')
                self.memory.move_to_end(key); self.stats['memory_hits'] += 1
                return meshes, receipt
            if pointer.is_file():
                receipt = read_json(pointer)
                path = Path(receipt['path'])
                if path.parent.resolve() != directory.resolve() or file_hash(path) != receipt['sha256']:
                    raise ValueError('Cached geometry content changed')
                value = read_json(path)
                if value['binding'] != binding: raise ValueError('Cached geometry binding changed')
                meshes = freeze(value['geometry']); self.remember(key, meshes, receipt)
                self.stats['disk_hits'] += 1
                return meshes, receipt
        except (OSError, EOFError, zlib.error, ValueError, KeyError, TypeError) as error:
            self.stats['invalid_entries'] += 1
            if key in self.memory:self.bytes -= self.memory.pop(key)[2]
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory/('invalid_'+uuid.uuid4().hex+'.json'), {'detected_unix':time.time(), 'binding':binding, 'error':type(error).__name__+': '+str(error)})
        meshes = freeze(build())
        path = directory/('geometry_'+uuid.uuid4().hex+'.json.gz')
        write_json(path, {'binding':binding, 'geometry':[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in m.items()} for m in meshes]})
        receipt = {'path':str(path.resolve()), 'sha256':file_hash(path)}
        write_json(pointer, receipt)
        self.remember(key, meshes, receipt); self.stats['misses'] += 1
        return meshes, receipt


def geometry_references(path):
    value = read_json(path)
    if value.get('policy') != POLICY:
        if 'geometry' in value: return []
        raise ValueError('Unknown cooked geometry evidence format')
    if not value.get('parts'):raise ValueError('Empty cooked geometry evidence')
    references, checked = [], set()
    for part in value['parts']:
        ref = part['artifact']
        identity = (ref['path'],ref['sha256'])
        if identity not in checked:
            if file_hash(ref['path']) != ref['sha256']: raise ValueError('Cooked geometry reference changed')
            references.append(ref); checked.add(identity)
    return references


def load_geometry(path):
    value = read_json(path)
    if 'geometry' in value: return value['geometry']
    geometry_references(path)
    output, decoded = {}, {}
    for part in value['parts']:
        path = part['artifact']['path']
        if path not in decoded:decoded[path] = read_json(path)['geometry']
        meshes = decoded[path]
        indices = part.get('mesh_indices',range(len(meshes)))
        if not indices or any(not isinstance(i,int) or i < 0 or i >= len(meshes) for i in indices):
            raise ValueError('Invalid collision mesh reference')
        for i, index in enumerate(indices):
            mesh = {k:v for k,v in meshes[index].items() if k != 'collider_index'}
            output.setdefault(part['body'], []).append({**mesh, 'name':part['prim_path']+f'#hull{i}'})
    return output
