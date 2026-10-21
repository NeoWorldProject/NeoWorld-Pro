"""Make a relocatable copy of a flattened scene and its texture dependencies.

The independently validated scene is never overwritten. Only shading asset paths
change; every other authored property must retain the same physical signature.
Isaac's built-in MDL modules remain runtime dependencies.
"""
from pathlib import Path
import shutil

from .protocol import file_hash, write_json
from .visual_materials import physical_signature


def package(source, output, *, allow_missing_textures=False):
    from pxr import Sdf, Usd, UsdShade
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new output directory; accepted sources stay immutable')
    original_stage = Usd.Stage.Open(str(source))
    if original_stage is None:
        raise ValueError('Cannot load scene USD')
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
    stage.Export(str(target))
    copied = Usd.Stage.Open(str(target))
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
                original = stage.GetPrimAtPath(prim.GetPath()).GetAttribute(attr.GetName()).Get()
                if attr.GetTypeName() == Sdf.ValueTypeNames.AssetArray:
                    original = original[len(rebound)]
                dependency = Path(original.resolvedPath or original.path)
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
                    shutil.copyfile(dependency, destination)
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
    copied.GetRootLayer().Save()
    reloaded = Usd.Stage.Open(str(target))
    if physical_signature(reloaded) != signature:
        raise ValueError('Packaging changed a non-shading property')
    receipt = {'schema_version': 'scene_material_bundle.v1', 'source': str(source),
               'source_sha256': file_hash(source), 'scene': target.name,
               'scene_sha256': file_hash(target), 'non_shading_signature': signature,
               'textures': textures, 'runtime_material_modules': sorted(modules),
               'material_degradation': degraded_materials,
               'scope': 'Exact validated geometry and state; only texture references are relocated.'}
    write_json(output / 'receipt.json', receipt)
    return receipt
