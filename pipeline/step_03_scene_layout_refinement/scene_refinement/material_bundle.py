"""Package scene state, shared geometry and relative texture dependencies.

The independently validated scene is never overwritten. Only shading asset paths
change; every other authored property must retain the same physical signature.
Isaac's built-in MDL modules remain runtime dependencies.
"""
from pathlib import Path

from .protocol import file_hash, write_json
from .visual_materials import physical_signature


def _dependencies(stage, source, allow_missing_textures):
    from pxr import UsdUtils, UsdShade, Sdf
    if stage.GetCompositionErrors():
        raise ValueError('Unresolved scene composition: ' + str(stage.GetCompositionErrors()))
    _, _, missing = UsdUtils.ComputeAllDependencies(str(source))
    textures = set()
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader): continue
        for attr in prim.GetAttributes():
            if attr.GetTypeName() not in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray): continue
            value = attr.Get()
            layer = next((spec.layer for spec in attr.GetPropertyStack() if spec.HasDefaultValue()), None)
            for item in ((value or []) if attr.GetTypeName() == Sdf.ValueTypeNames.AssetArray else [value]):
                if item and hasattr(item, 'path'):
                    textures.update((item.path, item.resolvedPath))
                    if layer:
                        textures.add(Sdf.ComputeAssetPathRelativeToLayer(layer, item.path))
    missing = [str(p) for p in missing if Path(str(p)).name not in ('OmniPBR.mdl', 'OmniPBR_Opacity.mdl')]
    required = [p for p in missing if not allow_missing_textures or p not in textures]
    if required: raise ValueError('Missing scene dependencies: ' + str(required))
    return missing


def _inventory(stage, objects):
    from pxr import Usd, UsdGeom, UsdPhysics, Tf
    rows = {}
    for oid, obj in objects.items():
        root = obj.get('root_prim_path') or obj.get('prim_path') or '/World/Objects/'+Tf.MakeValidIdentifier(oid)
        prim = stage.GetPrimAtPath(root)
        if not prim: raise ValueError('Missing declared object: '+oid)
        bodies = obj.get('body_paths', {})
        for link, path in bodies.items():
            body = stage.GetPrimAtPath(path)
            if not body or (not obj.get('is_environment_support') and not body.HasAPI(UsdPhysics.RigidBodyAPI)):
                raise ValueError('Missing declared body: '+oid+'::'+link)
        geometry = {}
        counts = {link: {'geometry_meshes': 0, 'collision_meshes': 0} for link in bodies}
        for child in Usd.PrimRange(prim):
            if not child.IsA(UsdGeom.Gprim): continue
            if child.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(child)
                if not mesh.GetPointsAttr().Get() or not mesh.GetFaceVertexIndicesAttr().Get():
                    raise ValueError('Empty published mesh: '+str(child.GetPath()))
            path = str(child.GetPath())
            collision = child.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get() is not False
            imageable = UsdGeom.Imageable(child)
            visible = imageable.ComputeVisibility() != UsdGeom.Tokens.invisible and imageable.ComputePurpose() != UsdGeom.Tokens.guide
            geometry[path] = {'type': child.GetTypeName(), 'collision': collision, 'visible': visible}
            owners = [(len(p), link) for link, p in bodies.items() if path == p or path.startswith(p+'/')]
            if owners:
                link = max(owners)[1]
                counts[link]['geometry_meshes'] += int(visible)
                counts[link]['collision_meshes'] += int(collision)
        if not geometry: raise ValueError('Missing object geometry: '+oid)
        for kind in ('geometry_meshes', 'collision_meshes'):
            for link in bodies:
                expected = sum(m['link'] == link and (kind != 'collision_meshes' or m.get('collision_enabled', True)) for m in obj.get(kind, []))
                if counts[link][kind] < expected:
                    raise ValueError('Incomplete '+kind+': '+oid+'::'+link)
        rows[oid] = {'root': root, 'bodies': bodies, 'geometry': geometry}
    return rows


def package(source, output, *, allow_missing_textures=False, objects=None):
    from pxr import Ar, Sdf, Usd, UsdShade
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new output directory; accepted sources stay immutable')
    from urdf_core.environment import isaac_mdl_paths
    context = Ar.DefaultResolverContext(list(map(str, isaac_mdl_paths())))
    with Ar.ResolverContextBinder(context):
        original_stage = Usd.Stage.Open(str(source))
        if original_stage is None: raise ValueError('Cannot load scene USD')
        _dependencies(original_stage, source, allow_missing_textures)
    expected_inventory = _inventory(original_stage, objects) if objects is not None else None
    stage = Usd.Stage.Open(original_stage.Flatten())
    import numpy as np
    from pxr import UsdGeom
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xformable):
            transform = np.asarray(cache.GetLocalToWorldTransform(prim))
            if not np.isfinite(transform).all() or abs(np.linalg.det(transform[:3, :3])) < 1e-12:
                raise ValueError('Invalid scene transform: ' + str(prim.GetPath()))
    signature = physical_signature(stage)
    output.mkdir(parents=True)
    target = output / 'scene.usdc'
    copied = stage
    textures, modules, degraded_materials = {}, set(), []
    for prim in copied.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() not in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                continue
            if attr.GetTimeSamples():
                raise ValueError('Animated asset dependencies require explicit packaging support')
            value = attr.Get()
            if value is None:
                continue
            values = [value] if attr.GetTypeName() == Sdf.ValueTypeNames.Asset else list(value)
            rebound = []
            for value in values:
                if not value or not value.path:
                    rebound.append(value)
                    continue
                if value.path in ('OmniPBR.mdl', 'OmniPBR_Opacity.mdl'):
                    modules.add(value.path)
                    rebound.append(value)
                    continue
                if not prim.IsA(UsdShade.Shader):
                    raise ValueError('Cannot rewrite a non-shading dependency: ' + str(attr.GetPath()))
                dependency = Path(value.resolvedPath or value.path)
                if not dependency.is_absolute():
                    dependency = source.parent / dependency
                if not dependency.is_file():
                    if allow_missing_textures:
                        degraded_materials.append({'shader': str(prim.GetPath()), 'missing_texture': str(dependency)})
                        rebound.append(Sdf.AssetPath(''))
                        continue
                    raise ValueError('Missing material dependency: ' + str(dependency))
                digest = file_hash(dependency)
                relative = Path('textures') / (digest + dependency.suffix.lower())
                destination = output / relative
                destination.parent.mkdir(exist_ok=True)
                if not destination.exists():
                    from urdf_core.storage import copy_asset, asset_pool
                    copy_asset(dependency, destination, digest=digest, pool=asset_pool(output))
                if file_hash(destination) != digest:
                    raise ValueError('Copied texture hash mismatch')
                textures[str(relative)] = digest
                rebound.append(Sdf.AssetPath(str(relative)))
            attr.Set(rebound[0] if attr.GetTypeName() == Sdf.ValueTypeNames.Asset else rebound)
    if degraded_materials:
        for prim in copied.Traverse():
            for rel in prim.GetRelationships():
                affected = any(row['shader'].startswith(str(target)+'/') for target in rel.GetTargets() for row in degraded_materials)
                if affected and rel.GetName().startswith('material:binding') and 'physics' not in rel.GetName().split(':'):
                    rel.ClearTargets(True)
    from physics_feedback.replay_storage import save_replay
    geometry_layers = save_replay(copied, target)
    with Ar.ResolverContextBinder(context):
        reloaded = Usd.Stage.Open(str(target))
        _dependencies(reloaded, target, False)
    if expected_inventory is not None and _inventory(reloaded, objects) != expected_inventory:
        raise ValueError('Packaging changed the object/link/geometry inventory')
    if physical_signature(reloaded) != signature:
        raise ValueError('Packaging changed a non-shading property')
    receipt = {'schema_version': 'scene_material_bundle.v1', 'source': str(source),
               'source_sha256': file_hash(source), 'scene': target.name,
               'scene_sha256': file_hash(target), 'non_shading_signature': signature,
               'textures': textures, 'runtime_material_modules': sorted(modules),
               'geometry_layers': {str(Path(row['path']).relative_to(output)): row['sha256']
                                   for row in geometry_layers},
               'material_degradation': degraded_materials,
               'static_checks': {'loadable': True, 'dependencies_complete': True,
                                 'objects_complete': expected_inventory is not None, 'physical_validation': False},
               'scope': 'Exact validated geometry and state; only texture references are relocated.'}
    write_json(output / 'receipt.json', receipt)
    return receipt
