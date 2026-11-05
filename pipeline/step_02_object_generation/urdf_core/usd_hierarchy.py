"""Make mesh ancestry explicit without changing composed geometry transforms."""


def type_mesh_ancestors(stage):
    from pxr import UsdGeom
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    cache = UsdGeom.XformCache()
    before = {p.GetPath(): cache.GetLocalToWorldTransform(p) for p in meshes}
    changed = set()
    for mesh in meshes:
        parent = mesh.GetParent()
        while parent and not parent.IsPseudoRoot():
            if not parent.GetTypeName():
                parent.SetTypeName('Xform')
                changed.add(str(parent.GetPath()))
            parent = parent.GetParent()
    cache.Clear()
    for mesh in meshes:
        if cache.GetLocalToWorldTransform(mesh) != before[mesh.GetPath()]:
            raise ValueError('Typing mesh ancestry changed its world transform: ' + str(mesh.GetPath()))
    return sorted(changed)
