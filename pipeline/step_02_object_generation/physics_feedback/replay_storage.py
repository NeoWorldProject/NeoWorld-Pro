"""Share immutable mesh arrays while retaining every trial's USD state opinions."""
from pathlib import Path
import os
import uuid

from urdf_core.run_state import file_hash
from urdf_core.storage import copy_asset


def save_replay(stage, path):
    from pxr import Sdf, Usd, UsdGeom
    path = Path(path)
    layer = stage.Flatten()
    static = Sdf.Layer.CreateAnonymous('.usdc')
    composed = Usd.Stage.Open(layer)
    moved = 0
    for prim in composed.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        for attribute in prim.GetAttributes():
            name = attribute.GetName()
            if (name not in {'points', 'faceVertexCounts', 'faceVertexIndices', 'normals'}
                    and not name.startswith('primvars:')):
                continue
            prop = layer.GetPropertyAtPath(attribute.GetPath())
            if prop is None or not prop.HasDefaultValue() or attribute.GetNumTimeSamples():
                continue
            Sdf.CreatePrimInLayer(static, attribute.GetPrimPath())
            if not Sdf.CopySpec(layer, attribute.GetPath(), static, attribute.GetPath()):
                raise ValueError('Cannot separate static replay geometry')
            layer.GetPrimAtPath(attribute.GetPrimPath()).RemoveProperty(prop)
            moved += 1
    composed = prim = attribute = prop = None
    dependencies = []
    temporary = path.with_name('.replay_static_' + uuid.uuid4().hex + '.usdc')
    try:
        if moved:
            if not static.Export(str(temporary)):
                raise ValueError('Cannot save static replay geometry')
            digest = file_hash(temporary)
            destination = path.parent / 'replay_geometry' / (digest + '.usdc')
            copy_asset(temporary, destination)
            layer.subLayerPaths.append(os.path.relpath(destination, path.parent))
            dependencies.append({'path': str(destination), 'sha256': digest})
        if not layer.Export(str(path)):
            raise ValueError('Cannot save replay state')
    finally:
        temporary.unlink(missing_ok=True)
    return dependencies


def verify_replay_dependencies(binding, root):
    from .critic import bound_file
    return {bound_file(item['path'], item['sha256'], root)
            for item in binding.get('dependencies', [])}
