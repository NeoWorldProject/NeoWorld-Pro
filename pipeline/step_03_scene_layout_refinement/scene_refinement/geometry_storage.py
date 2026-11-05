"""Persist geometry references; reconstruct derived arrays on load."""
from pathlib import Path
import json
import hashlib
import os
import uuid

import numpy as np

from .hashing import file_hash

ASSET = 'scene_asset_references.v1'
SCENE = 'scene_prepared_references.v1'


def _implementations():
    root = Path(__file__).parent
    return {name:file_hash(root/name) for name in ('assets.py', 'support.py', 'geometry_storage.py')}


def _mesh(path, expected, cache):
    path = str(Path(path).resolve())
    key = path, expected
    if key not in cache:
        import trimesh
        if file_hash(path) != expected:
            raise ValueError('Referenced geometry changed: '+path)
        mesh = trimesh.load(path, force='mesh', process=False)
        if not len(mesh.faces):
            raise ValueError('Referenced geometry is empty: '+path)
        cache[key] = mesh
    return cache[key]


def _manifest(asset):
    # Existing in-memory assets already bind their published physical manifest.
    for path, expected in asset.get('source_hashes', {}).items():
        if Path(path).name != 'asset_manifest.json':
            continue
        if file_hash(path) != expected:
            raise ValueError('Referenced asset manifest changed')
        value = json.loads(Path(path).read_text())
        if {'geometry_meshes', 'collision_meshes', 'link_properties'} <= set(value):
            return Path(path).parent, value
    return None, None


def _pack_asset(asset, cache):
    parent, source = _manifest(asset)
    if source is None:
        return asset
    result = dict(asset)
    for key in ('geometry_meshes', 'collision_meshes'):
        if len(asset[key]) != len(source[key]):
            raise ValueError('Geometry inventory differs from the published manifest')
        result[key] = []
        for mesh, original in zip(asset[key], source[key]):
            path = Path(original['path'])
            path = path if path.is_absolute() else parent/path
            raw = _mesh(path, original['sha256'], cache)
            if (mesh['link'] != original['link'] or not np.array_equal(raw.vertices, mesh['vertices'])
                    or not np.array_equal(raw.faces, mesh['faces'])):
                # Explicitly edited geometry must keep its actual values.
                result[key].append(mesh)
                continue
            result[key].append({**{k:v for k,v in mesh.items() if k not in ('vertices', 'faces')},
                'mesh_reference':{'path':str(path.resolve()), 'sha256':original['sha256']}})
    return result


def _unpack_asset(asset, cache):
    result = dict(asset)
    for key in ('geometry_meshes', 'collision_meshes'):
        result[key] = []
        for mesh in asset[key]:
            if 'mesh_reference' not in mesh:
                result[key].append(mesh)
                continue
            ref = mesh['mesh_reference']
            raw = _mesh(ref['path'], ref['sha256'], cache)
            result[key].append({**{k:v for k,v in mesh.items() if k != 'mesh_reference'},
                                'vertices':raw.vertices.tolist(), 'faces':raw.faces.tolist()})
    return result


def pack_asset(asset):
    return {'storage_schema':ASSET, 'payload':_pack_asset(asset, {})}


def _catalog_hash(catalog):
    digest = hashlib.sha256()
    for region in catalog:
        metadata = {k:v for k,v in region.items() if k != 'triangles_local'}
        digest.update(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
        triangles = np.asarray(region['triangles_local'], dtype='<f8')
        digest.update(str(triangles.shape).encode())
        digest.update(triangles.tobytes())
    return digest.hexdigest()


def _catalog_references(catalog, meshes):
    indices, result, lookups = [], [], {}
    for region in catalog:
        mesh_index = int(region['region_id'].rsplit(':mesh_', 1)[1].split(':patch_', 1)[0])
        if mesh_index not in lookups:
            mesh = meshes[mesh_index]
            triangles = np.asarray(mesh['vertices'], dtype='<f8')[np.asarray(mesh['faces'], dtype=int)]
            keys = np.ascontiguousarray(triangles).reshape(-1, 9).view('V72').reshape(-1)
            order = np.argsort(keys)
            lookups[mesh_index] = keys[order], order
        keys, order = lookups[mesh_index]
        triangles = np.asarray(region['triangles_local'], dtype='<f8')
        query = np.ascontiguousarray(triangles).reshape(-1, 9).view('V72').reshape(-1)
        positions = np.searchsorted(keys, query)
        if np.any(positions >= len(keys)) or not np.array_equal(keys[positions], query):
            raise ValueError('Support triangles do not belong to their source mesh')
        selected = order[positions].tolist()
        result.append({**{k:v for k,v in region.items() if k != 'triangles_local'},
            'triangles_reference':{'mesh_index':mesh_index, 'start':len(indices), 'count':len(selected)}})
        indices.extend(selected)
    return result, np.asarray(indices, dtype='<i8')


def pack_scene(scene, path=None):
    from .assets import asset_points
    cache, objects, arrays, catalogs = {}, {}, {}, {}
    for oid, obj in scene['objects'].items():
        value = {**obj, 'asset':_pack_asset(obj['asset'], cache)}
        if np.array_equal(obj['rest_points'], asset_points(obj['asset'])):
            value['rest_points'] = {'derive':'asset_points'}
        regions, indices = _catalog_references(obj['surface_catalog'], obj['asset']['geometry_meshes'])
        arrays[oid] = indices
        catalogs[oid] = regions
        value['surface_catalog'] = {'regions':regions if path is None else oid, 'indices':indices.tolist() if path is None else oid,
                                    'sha256':_catalog_hash(obj['surface_catalog'])}
        objects[oid] = value
    payload = {**scene, 'objects':objects}
    stored = {'storage_schema':SCENE, 'implementations':_implementations(),
              'numpy_version':np.__version__, 'payload':payload}
    if path is not None:
        from urdf_core.storage import require_space
        arrays['__catalogs__'] = np.frombuffer(json.dumps(catalogs, separators=(',', ':'), allow_nan=False).encode(), dtype=np.uint8)
        digest = hashlib.sha256()
        for key in sorted(arrays):
            digest.update(key.encode()); digest.update(str(arrays[key].shape).encode()); digest.update(arrays[key].tobytes())
        folder = Path(path).parent/'geometry_refs'; folder.mkdir(exist_ok=True)
        target = folder/(digest.hexdigest()+'.npz')
        if not target.exists():
            require_space(folder, sum(a.nbytes for a in arrays.values())+65536)
            temporary = folder/('.'+uuid.uuid4().hex+'.pending')
            try:
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, **arrays)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        with np.load(target, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays) or any(not np.array_equal(archive[k], v) for k,v in arrays.items()):
                raise ValueError('Stored support indices differ from the source')
        stored['catalog_storage'] = {'path':str(target.resolve()), 'sha256':file_hash(target)}
    return stored


def unpack(value):
    kind = value.get('storage_schema') if isinstance(value, dict) else None
    if kind not in (ASSET, SCENE):
        return value
    cache = {}
    if kind == ASSET:
        return _unpack_asset(value['payload'], cache)
    if value['implementations'] != _implementations() or value['numpy_version'] != np.__version__:
        raise ValueError('Prepared geometry reconstruction runtime changed')
    from .assets import asset_points
    scene = value['payload']
    arrays = {}
    if 'catalog_storage' in value:
        storage = value['catalog_storage']
        if file_hash(storage['path']) != storage['sha256']:
            raise ValueError('Support index dependency changed')
        with np.load(storage['path'], allow_pickle=False) as archive:
            arrays = {key:archive[key] for key in archive.files}
    catalogs = json.loads(arrays.pop('__catalogs__').tobytes()) if arrays else {}
    for obj in scene['objects'].values():
        obj['asset'] = _unpack_asset(obj['asset'], cache)
        if isinstance(obj['rest_points'], dict):
            if obj['rest_points'] != {'derive':'asset_points'}:
                raise ValueError('Unknown rest-point reconstruction')
            obj['rest_points'] = asset_points(obj['asset']).tolist()
        recipe = obj['surface_catalog']
        indices = arrays[recipe['indices']] if isinstance(recipe['indices'], str) else np.asarray(recipe['indices'], dtype='<i8')
        if indices.dtype.kind not in 'iu' or indices.ndim != 1 or indices.min(initial=0) < 0:
            raise ValueError('Invalid support indices')
        regions, mesh_arrays = [], {}
        metadata = catalogs[recipe['regions']] if isinstance(recipe['regions'], str) else recipe['regions']
        for stored in metadata:
            ref = stored['triangles_reference']
            index = ref['mesh_index']
            if index < 0 or index >= len(obj['asset']['geometry_meshes']):
                raise ValueError('Invalid support mesh index')
            if index not in mesh_arrays:
                mesh = obj['asset']['geometry_meshes'][index]
                mesh_arrays[index] = np.asarray(mesh['vertices'], dtype=float), np.asarray(mesh['faces'], dtype=int)
            vertices, faces = mesh_arrays[index]
            start, count = ref['start'], ref['count']
            if start < 0 or count < 0 or start+count > len(indices):
                raise ValueError('Invalid support index range')
            selected = indices[start:start+count]
            if selected.max(initial=-1) >= len(faces):
                raise ValueError('Support index is outside its source mesh')
            triangles = vertices[faces[selected]]
            regions.append({**{k:v for k,v in stored.items() if k != 'triangles_reference'},
                            'triangles_local':triangles.tolist()})
        obj['surface_catalog'] = regions
        if _catalog_hash(regions) != recipe['sha256']:
            raise ValueError('Reconstructed support catalog differs from the recorded catalog')
    return scene
