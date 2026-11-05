"""Lossless candidate manifests with content-bound shared mesh arrays."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import uuid

from .hashing import file_hash

SCHEMA = 'scene_candidate_references.v1'


def metadata(path):
    """Read object and state descriptors without expanding shared mesh arrays."""
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        value = json.load(stream)
    return value['payload'] if value.get('storage_schema') == SCHEMA else value


def dependencies(path):
    """Validate and return external mesh bindings without decoding mesh arrays."""
    path = Path(path)
    data = path.read_bytes()
    value = json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)
    if value.get('storage_schema') != SCHEMA:
        return {}
    payload = value['payload']
    objects = list(payload['objects'].values())
    if isinstance(payload.get('instances'), list):
        objects += payload['instances']
    result = {}
    for obj in objects:
        for key in ('geometry_meshes', 'collision_meshes'):
            for mesh in obj.get(key, []):
                ref = mesh['mesh_arrays_reference']
                target = str((path.parent / ref['path']).resolve())
                if target not in result or result[target] != ref['sha256']:
                    if file_hash(target) != ref['sha256']:
                        raise ValueError('Referenced candidate geometry changed: ' + target)
                    result[target] = ref['sha256']
    return result


def _store(path):
    # Candidate siblings share geometry; standalone delivery bundles own theirs.
    for parent in path.parents:
        if parent.name == 'candidates':
            return parent.parent / 'candidate_geometry'
    return path.parent / 'candidate_geometry'


def pack(value, path):
    from urdf_core.storage import require_space
    path = Path(path).resolve()
    folder = _store(path)
    folder.mkdir(parents=True, exist_ok=True)
    cache = {}

    def object_record(obj):
        result = dict(obj)
        for key in ('geometry_meshes', 'collision_meshes'):
            if key not in obj:
                continue
            result[key] = []
            for mesh in obj[key]:
                arrays = {k: mesh[k] for k in ('vertices', 'faces')}
                raw = json.dumps(arrays, separators=(',', ':'), allow_nan=False).encode()
                digest = hashlib.sha256(raw).hexdigest()
                target = folder / (digest + '.json.gz')
                if digest not in cache:
                    if not target.exists():
                        data = gzip.compress(raw, compresslevel=1, mtime=0)
                        require_space(folder, len(data))
                        temporary = folder / ('.' + uuid.uuid4().hex + '.pending')
                        try:
                            temporary.write_bytes(data)
                            temporary.chmod(0o444)
                            os.replace(temporary, target)
                        finally:
                            temporary.unlink(missing_ok=True)
                    if gzip.decompress(target.read_bytes()) != raw:
                        raise ValueError('Candidate mesh store changed: ' + str(target))
                    cache[digest] = {'path': os.path.relpath(target, path.parent),
                                     'sha256': file_hash(target)}
                result[key].append({**{k: v for k, v in mesh.items() if k not in arrays},
                                    'mesh_arrays_reference': cache[digest]})
        return result

    objects = {oid: object_record(obj) for oid, obj in value['objects'].items()}
    payload = {**value, 'objects': objects}
    instances = value.get('instances')
    if instances is not None:
        ids = [obj.get('object_id') for obj in instances]
        if all(oid in value['objects'] and obj == value['objects'][oid]
               for oid, obj in zip(ids, instances)):
            payload['instances'] = {'from_objects': ids}
        else:
            payload['instances'] = [object_record(obj) for obj in instances]
    return {'storage_schema': SCHEMA, 'payload': payload}


def unpack(value, path):
    cache = {}

    def object_record(obj):
        result = copy.deepcopy(obj)
        for key in ('geometry_meshes', 'collision_meshes'):
            if key not in obj:
                continue
            result[key] = []
            for mesh in obj[key]:
                ref = mesh['mesh_arrays_reference']
                target = (Path(path).parent / ref['path']).resolve()
                identity = (str(target), ref['sha256'])
                if identity not in cache:
                    if file_hash(target) != ref['sha256']:
                        raise ValueError('Referenced candidate geometry changed: ' + str(target))
                    cache[identity] = gzip.decompress(target.read_bytes())
                result[key].append({**{k: copy.deepcopy(v) for k, v in mesh.items() if k != 'mesh_arrays_reference'},
                                    **json.loads(cache[identity])})
        return result

    payload = value['payload']
    objects = {oid: object_record(obj) for oid, obj in payload['objects'].items()}
    result = {**payload, 'objects': objects}
    if 'instances' in payload:
        instances = payload['instances']
        result['instances'] = ([object_record(payload['objects'][oid]) for oid in instances['from_objects']]
                               if isinstance(instances, dict) else [object_record(obj) for obj in instances])
    return result
