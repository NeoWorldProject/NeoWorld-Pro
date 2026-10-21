"""Restore baked source colors onto verified USD visual meshes, with no physics edits."""
from copy import deepcopy
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .protocol import file_hash, digest, read_json, write_json
from urdf_core.content_errors import AppearanceContentError


def physical_signature(stage):
    """Every non-shading property, including all visual geometry and body/joint data."""
    from pxr import UsdShade
    def physical_property(name):
        namespace = name.split(':', 1)[0]
        return namespace == 'physics' or namespace.startswith('physx')

    rows = {}
    for prim in stage.Traverse():
        shading = (prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader)
                   or '/RestoredVisualMaterials' in str(prim.GetPath()))
        attrs = {}
        for attr in prim.GetAttributes():
            if attr.GetName().startswith('primvars:st') or shading and not physical_property(attr.GetName()):
                continue
            attrs[attr.GetName()] = [str(attr.Get()), [(t, str(attr.Get(t))) for t in attr.GetTimeSamples()],
                                     [str(p) for p in attr.GetConnections()]]
        rels = {}
        for relation in prim.GetRelationships():
            name = relation.GetName()
            physics_binding = name.startswith('material:binding') and 'physics' in name.split(':')
            if name.startswith('material:binding') and not physics_binding:
                continue
            if shading and not (physics_binding or physical_property(name)):
                continue
            targets = [str(p) for p in relation.GetTargets()]
            rels[name] = ({'targets': targets, 'bindMaterialAs': str(relation.GetMetadata('bindMaterialAs'))}
                          if physics_binding else targets)
        schemas = [str(x) for x in prim.GetAppliedSchemas() if str(x) != 'MaterialBindingAPI'
                   and (not shading or str(x).startswith(('Physics', 'Physx')))]
        if shading and not (attrs or rels or schemas):
            continue
        rows[str(prim.GetPath())] = {'type':prim.GetTypeName(), 'attributes':attrs, 'relationships':rels,
                                    'schemas':schemas}
    return digest(rows)


def match_uvs(points, face_counts, face_indices, reference_triangles, reference_uvs, tolerance=2e-6):
    """Map UVs by verified triangle geometry, including reversed export backfaces."""
    counts = np.asarray(face_counts)
    if (counts != 3).any():
        raise ValueError('Visual material binding needs the verified triangular export')
    triangles = np.asarray(points)[np.asarray(face_indices).reshape(-1, 3)]
    refs, uvs = np.asarray(reference_triangles), np.asarray(reference_uvs)
    if refs.shape[1:] != (3, 3) or uvs.shape != (len(refs), 3, 2):
        raise ValueError('Invalid baked triangle/UV evidence')
    tree = cKDTree(refs.mean(1)); output = []; errors = []
    permutations3 = list(permutations(range(3)))
    for triangle in triangles:
        choices = []
        for index in tree.query_ball_point(triangle.mean(0), tolerance):
            for order in permutations3:
                error = float(np.linalg.norm(triangle-refs[index, list(order)], axis=1).max())
                if error <= tolerance:
                    choices.append((error, index, order))
        if not choices:
            raise AppearanceContentError('USD visual geometry differs from the bound Blender/OBJ export')
        error, index, order = min(choices)
        values = uvs[index, list(order)]
        if any(not np.allclose(uvs[i, list(p)], values, rtol=0, atol=1e-6) for _, i, p in choices):
            raise AppearanceContentError('Coincident source triangles have ambiguous material UVs')
        output.extend(values.tolist()); errors.append(error)
    return output, max(errors, default=0.)


def _indices(values, shape, label):
    """Reject invalid identity evidence instead of silently coercing indices."""
    array = np.asarray(values)
    if array.size == 0:
        return np.empty(tuple(0 if size == -1 else size for size in shape), dtype=np.int64)
    if array.dtype.kind not in 'iu' or array.ndim != len(shape) or any(
            size != -1 and array.shape[i] != size for i, size in enumerate(shape)):
        raise ValueError('Invalid '+label+' topology indices')
    return array.astype(np.int64)


def _read_export_obj(path):
    points, faces = [], []
    for line in Path(path).read_text().splitlines():
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


def match_export_uvs(points, face_counts, face_indices, part, entry, tolerance=2e-6):
    """Use verified export identities before attempting an unambiguous spatial map.

    Close or coincident corners may have different UVs. Their identities survive
    in the OBJ and manifest even when coordinates alone cannot distinguish them.
    Invalid source provenance is never downgraded to a geometric fallback.
    """
    obj_path = Path(entry['path']).resolve()
    obj_hash = file_hash(obj_path)
    obj_points, obj_faces = _read_export_obj(obj_path)
    source_ids = _indices(entry['source_vertex_indices'], (-1,), 'source vertex')
    source_faces = _indices(entry['source_triangle_vertices'], (-1, 3), 'source triangle')
    backfaces = _indices(entry.get('backface_source_triangle_indices', []), (-1,), 'source backface')
    baked_backfaces = _indices(part.get('backface_source_triangle_indices', []), (-1,), 'baked backface')
    if (not len(source_faces) or len(source_ids) != len(obj_points)
            or len(set(source_ids.tolist())) != len(source_ids) or source_ids.min(initial=0) < 0
            or (len(backfaces) and (backfaces.min() < 0 or backfaces.max() >= len(source_faces)))
            or not np.array_equal(backfaces, baked_backfaces)):
        raise ValueError('Export manifest and baked topology identity differ')
    index_map = {int(source): index for index, source in enumerate(source_ids)}
    if set(source_faces.ravel().tolist()) != set(index_map):
        raise ValueError('Export manifest source vertex identities differ from faces')
    expected_faces = np.asarray([[index_map[int(v)] for v in face] for face in source_faces])
    expected_faces = np.concatenate((expected_faces, expected_faces[backfaces, ::-1]), axis=0)
    if (entry.get('triangles', len(expected_faces)) != len(expected_faces)
            or not np.array_equal(obj_faces, expected_faces)):
        raise ValueError('Exported OBJ topology differs from bound source identities')
    refs, uvs = np.asarray(part['triangle_vertices']), np.asarray(part['triangle_uvs'])
    if (refs.shape != (len(source_faces), 3, 3) or uvs.shape != (len(source_faces), 3, 2)
            or not np.isfinite(refs).all() or not np.isfinite(uvs).all()):
        raise ValueError('Invalid baked triangle/UV evidence')
    export_triangles = np.concatenate((refs, refs[backfaces, ::-1]), axis=0)
    export_uvs = np.concatenate((uvs, uvs[backfaces, ::-1]), axis=0)
    bake_error = float(np.linalg.norm(obj_points[obj_faces]-export_triangles, axis=2).max())
    if bake_error > tolerance:
        raise ValueError('Exported OBJ geometry differs from bound baked source')
    points = np.asarray(points)
    counts = _indices(face_counts, (-1,), 'USD face count')
    indices = _indices(face_indices, (-1,), 'USD face vertex')
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or (counts != 3).any() or len(indices) != len(counts)*3 or not len(indices)
            or indices.min() < 0 or indices.max() >= len(points)):
        raise AppearanceContentError('Invalid USD visual geometry')
    evidence = {'obj_path':str(obj_path), 'obj_sha256':obj_hash,
                'source_topology_verified':True, 'max_obj_bake_error_m':bake_error}
    same_indices = np.array_equal(indices.reshape(-1, 3), obj_faces)
    if same_indices and points.shape == obj_points.shape:
        position_error = float(np.linalg.norm(points-obj_points, axis=1).max())
        if position_error <= tolerance:
            evidence['mapping_mode'] = 'verified_export_topology'
            evidence['max_usd_obj_error_m'] = position_error
            # Check directly as well, so chained tolerances do not conceal drift.
            error = float(np.linalg.norm(points[indices.reshape(-1, 3)]-export_triangles, axis=2).max())
            if error > tolerance:
                raise AppearanceContentError('USD visual geometry differs from bound baked source')
            return export_uvs.reshape(-1, 2).tolist(), error, evidence
    uv, error = match_uvs(points, counts, indices, refs, uvs, tolerance)
    evidence['mapping_mode'] = 'unique_geometry_fallback'
    return uv, error, evidence


def restore(source_usd, bake_receipt, out_dir):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    source_usd, bake_receipt, out_dir = map(lambda p: Path(p).resolve(), (source_usd, bake_receipt, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    package = out_dir/'material_package.json'
    if package.exists():
        raise ValueError('Material package exists; use a content-bound cache or a new output directory')
    bake = read_json(bake_receipt)
    if file_hash(bake['source_blend']) != bake['source_blend_sha256'] or file_hash(bake['export_manifest']) != bake['export_manifest_sha256']:
        raise ValueError('Baked source bindings changed')
    export = read_json(bake['export_manifest'])
    asset_manifest = Path(source_usd).parent/'asset_manifest.json'
    derived_links = read_json(asset_manifest).get('derived_names', {}).get('links', {}) if asset_manifest.is_file() else {}
    derived_stems = read_json(asset_manifest).get('derived_names', {}).get('visual_stems', {}) if asset_manifest.is_file() else {}
    source_evidence = {bake['export_manifest']:bake['export_manifest_sha256']}
    original = Usd.Stage.Open(str(source_usd))
    signature = physical_signature(original)
    path = out_dir/'asset_with_materials.usdc'
    stage = Usd.Stage.CreateNew(str(path))
    default = original.GetDefaultPrim()
    root = stage.DefinePrim(default.GetPath(), default.GetTypeName())
    root.GetReferences().AddReference(str(source_usd), default.GetPath())
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(original))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
    # Other top-level authored physics prims must not disappear with a default-prim reference.
    for prim in original.GetPseudoRoot().GetChildren():
        if prim.GetPath() != default.GetPath():
            stage.DefinePrim(prim.GetPath(), prim.GetTypeName()).GetReferences().AddReference(str(source_usd), prim.GetPath())
    cache = UsdGeom.XformCache()
    mapping = []; images = {}; used = set()
    for obj in bake['objects']:
        image = Path(obj['image'])
        if file_hash(image) != obj['image_sha256']:
            raise ValueError('Baked image changed')
        images[str(image)] = obj['image_sha256']
        for part in obj['parts']:
            link_name = derived_links.get(obj['link'], obj['link'])
            token = '/'+link_name+'/visuals/'+derived_stems.get(part['stem'], part['stem'])+'/'
            matches = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and token in str(p.GetPath())]
            if len(matches) != 1:
                raise AppearanceContentError(f'Expected one native visual mesh for {token}, got {len(matches)}')
            prim = matches[0]; mesh = UsdGeom.Mesh(prim)
            link_prim = stage.GetPrimAtPath(default.GetPath().AppendChild(link_name))
            transform = np.asarray(cache.ComputeRelativeTransform(prim, link_prim)[0]).T
            points = np.asarray(mesh.GetPointsAttr().Get()) @ transform[:3, :3].T+transform[:3, 3]
            if 'meshes' in export:
                entries = [e for e in export['meshes'] if e['link'] == obj['link'] and e['object'] == obj['object']
                           and e['material_index'] == part['material_index'] and Path(e['path']).stem == part['stem']]
                if len(entries) != 1:
                    raise ValueError('Expected one bound export topology entry for '+token)
                uv, error, identity = match_export_uvs(points, mesh.GetFaceVertexCountsAttr().Get(),
                    mesh.GetFaceVertexIndicesAttr().Get(), part, entries[0])
                source_evidence[identity['obj_path']] = identity['obj_sha256']
            else:
                raise ValueError('Current material restoration requires export topology identities')
            UsdGeom.PrimvarsAPI(prim).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                                 UsdGeom.Tokens.faceVarying).Set(uv)
            material = UsdShade.Material.Define(stage, default.GetPath().AppendPath('RestoredVisualMaterials/'+part['stem']))
            reader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('UV'))
            reader.CreateIdAttr('UsdPrimvarReader_float2')
            reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
            reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
            texture = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('BaseColor'))
            texture.CreateIdAttr('UsdUVTexture')
            texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(image)))
            texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
            texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
            texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
            shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('Surface'))
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), 'rgb')
            constants = next(m for m in obj['materials'] if m['slot'] == part['material_index'])
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(constants['roughness'])
            shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(constants['metallic'])
            shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
            used.add(str(prim.GetPath()))
            mapping.append({'mesh':str(prim.GetPath()), 'source_object':obj['object'], 'max_geometry_error_m':error,
                            'image':str(image), 'material':str(material.GetPath()),
                            'color_scope':'baked_base_color', 'other_parameters':constants, **identity})
    expected = {str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and '/visuals/' in str(p.GetPath())}
    if used != expected:
        raise AppearanceContentError('Material restoration did not cover every visual mesh: '+str(expected-used))
    stage.GetRootLayer().Save()
    checked = Usd.Stage.Open(str(path))
    if physical_signature(checked) != signature:
        raise ValueError('Material restoration changed non-shading geometry/physics properties')
    receipt = {'schema_version':'visual_material_package.v1', 'source_usd':str(source_usd),
               'source_usd_sha256':file_hash(source_usd), 'usd_path':str(path), 'usd_sha256':file_hash(path),
               'bake_receipt':str(bake_receipt), 'bake_receipt_sha256':file_hash(bake_receipt),
               'images':images, 'source_evidence':source_evidence,
               'physical_signature_sha256':signature, 'meshes':mapping,
               'source_code_sha256':file_hash(__file__), 'scope':'Restored base colors; constant roughness/metallicity. Physical properties and all visual vertex/topology data unchanged.'}
    write_json(package, receipt)
    return receipt


def bind(asset, package_path, expected_hash):
    from pxr import Usd
    if file_hash(package_path) != expected_hash:
        raise ValueError('Material package receipt changed')
    package = read_json(package_path)
    if Path(package['source_usd']).resolve() != Path(asset['source_usd_path']).resolve():
        raise ValueError('Material package belongs to a different physical asset')
    bindings = {package['source_usd']:package['source_usd_sha256'], package['usd_path']:package['usd_sha256'],
                package['bake_receipt']:package['bake_receipt_sha256'], str(package_path):expected_hash,
                **package['images'], **package.get('source_evidence', {})}
    if any(file_hash(p) != h for p, h in bindings.items()):
        raise ValueError('Material package asset/texture evidence changed')
    if any(physical_signature(Usd.Stage.Open(str(p))) != package['physical_signature_sha256']
           for p in (package['source_usd'], package['usd_path'])):
        raise ValueError('Material package physics differs from the verified asset')
    result = deepcopy(asset)
    result['source_usd_path'] = package['usd_path']
    result['source_hashes'].update(bindings)
    result['visual_material_package'] = {'path':str(package_path), 'sha256':expected_hash}
    return result
