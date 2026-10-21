"""Bounded source winding normalization before candidate renders and modifiers."""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import time

POLICY = "source_winding.v1"


class GeometryPreparationError(RuntimeError):
    pass

def orientation_plan(faces):
    faces = [tuple(face) for face in faces]
    edges = defaultdict(list)
    vertex_faces = defaultdict(set)
    for index, face in enumerate(faces):
        for vertex in face:
            vertex_faces[vertex].add(index)
        for left, right in zip(face, face[1:] + face[:1]):
            edges[tuple(sorted((left, right)))].append((index, left < right))
    adjacency = defaultdict(list)
    connectivity = defaultdict(set)
    for edge, uses in edges.items():
        owners = [use[0] for use in uses]
        for face in owners:
            connectivity[face].update(owners)
        if len(uses) == 2:
            (left, direction_a), (right, direction_b) = uses
            different = direction_a == direction_b
            adjacency[left].append((right, different))
            adjacency[right].append((left, different))
    pending = set(range(len(faces)))
    components = []
    while pending:
        start = min(pending)
        component = {start}
        queue = [start]
        while queue:
            here = queue.pop()
            for other in connectivity[here] - component:
                component.add(other)
                queue.append(other)
        pending -= component
        labels = {}
        contradiction = False
        for seed in sorted(component):
            if seed in labels:
                continue
            labels[seed] = False
            queue = deque([seed])
            while queue:
                here = queue.popleft()
                for other, different in adjacency[here]:
                    required = labels[here] != different
                    if other in labels:
                        contradiction |= labels[other] != required
                    else:
                        labels[other] = required
                        queue.append(other)
        boundary = []
        nonmanifold = []
        vertices = {vertex for face in component for vertex in faces[face]}
        component_edges = {tuple(sorted((left, right))) for index in component
            for left, right in zip(faces[index], faces[index][1:] + faces[index][:1])}
        for edge in component_edges:
            uses = edges[edge]
            if len(uses) == 1:
                boundary.append(edge)
            elif len(uses) != 2:
                nonmanifold.append(edge)
        invalid_faces = [i for i in component if len(faces[i]) < 3 or len(set(faces[i])) != len(faces[i])]
        nonmanifold_vertices = []
        for vertex in vertices:
            uses = vertex_faces[vertex]
            local = {min(uses)}
            queue = list(local)
            while queue:
                here = queue.pop()
                for other, _ in adjacency[here]:
                    if other in uses and other not in local:
                        local.add(other)
                        queue.append(other)
            if local != uses:
                nonmanifold_vertices.append(vertex)
        reasons = []
        if invalid_faces: reasons.append('degenerate_polygon_topology')
        if boundary: reasons.append('open_component')
        if nonmanifold: reasons.append('nonmanifold_edges')
        if nonmanifold_vertices: reasons.append('nonmanifold_vertices')
        if contradiction: reasons.append('nonorientable_component')
        flips = sorted(i for i in component if labels[i])
        if len(flips) > len(component) - len(flips):
            flips = sorted(component - set(flips))
        components.append({'face_indices': sorted(component), 'closed': not boundary,
            'orientable': not contradiction, 'boundary_edge_count': len(boundary),
            'nonmanifold_edge_count': len(nonmanifold), 'nonmanifold_vertex_count': len(nonmanifold_vertices),
            'skipped_reasons': reasons, 'flip_faces': flips if not reasons else []})
    return {'components': components, 'flip_faces': sorted(i for c in components for i in c['flip_faces'])}


def _corner_identity(mesh):
    return [{'vertices': sorted(face.vertices), 'material': face.material_index,
        'uv': {mesh.loops[index].vertex_index: [tuple(layer.data[index].uv)
            for layer in mesh.uv_layers] for index in face.loop_indices}}
        for face in mesh.polygons]


def _attribute_identity(mesh):
    corners = {index: (face.index, mesh.loops[index].vertex_index)
        for face in mesh.polygons for index in face.loop_indices}
    fields = {'FLOAT': 'value', 'INT': 'value', 'INT8': 'value', 'BOOLEAN': 'value',
        'STRING': 'value', 'FLOAT_VECTOR': 'vector', 'FLOAT2': 'vector',
        'FLOAT_COLOR': 'color', 'BYTE_COLOR': 'color'}
    result = {}
    for attribute in mesh.attributes:
        if attribute.name.startswith('.') or attribute.name == 'position':
            continue
        field = fields.get(attribute.data_type)
        if field is None or attribute.domain not in {'POINT', 'EDGE', 'FACE', 'CORNER'}:
            raise GeometryPreparationError('Unsupported mesh attribute for winding normalization: ' + attribute.name)
        values = {}
        for index, item in enumerate(attribute.data):
            key = (corners[index] if attribute.domain == 'CORNER' else
                tuple(sorted(mesh.edges[index].vertices)) if attribute.domain == 'EDGE' else index)
            value = getattr(item, field)
            values[key] = tuple(value) if field in {'vector', 'color'} else value
        result[attribute.name] = (attribute.data_type, attribute.domain, values)
    return result


def normalize_mesh(mesh, plan=None):
    import bmesh
    plan = plan if plan is not None else orientation_plan([list(face.vertices) for face in mesh.polygons])
    if not plan['flip_faces']:
        return plan
    if mesh.shape_keys is not None or mesh.has_custom_normals:
        raise GeometryPreparationError('Winding normalization cannot safely preserve shape keys or custom split normals: ' + mesh.name)
    vertices = [tuple(vertex.co) for vertex in mesh.vertices]
    corners = _corner_identity(mesh)
    attributes = _attribute_identity(mesh)
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        for index in plan['flip_faces']:
            bm.faces[index].normal_flip()
        bm.normal_update()
        bm.to_mesh(mesh)
    finally:
        bm.free()
    mesh.update()
    if (vertices != [tuple(vertex.co) for vertex in mesh.vertices] or corners != _corner_identity(mesh)
            or attributes != _attribute_identity(mesh)):
        raise GeometryPreparationError('Winding normalization changed vertex, material, UV or attribute identity')
    return plan


def prepare_scene(*, validate_only=False):
    import bpy
    started = time.perf_counter()
    seen, records = set(), []
    for obj in sorted(bpy.context.scene.objects, key=lambda item: item.name):
        if obj.type != 'MESH' or obj.hide_render or obj.data.as_pointer() in seen:
            continue
        seen.add(obj.data.as_pointer())
        plan = orientation_plan([list(face.vertices) for face in obj.data.polygons])
        if any(not component['orientable'] for component in plan['components']):
            raise GeometryPreparationError('Nonorientable source mesh cannot be repaired by winding normalization: ' + obj.name)
        if validate_only and plan['flip_faces']:
            raise GeometryPreparationError('Baseline needs source winding preparation before candidate rendering; use a new generation run: ' + obj.name)
        if not validate_only:
            normalize_mesh(obj.data, plan)
        records.append({'object': obj.name, 'mesh': obj.data.name,
            'users': sorted(user.name for user in bpy.data.objects if user.type == 'MESH' and user.data == obj.data),
            'plan': plan})
    return {'objects': records, 'wall_seconds': time.perf_counter() - started, 'validate_only': validate_only}


def geometry_signature():
    import bpy
    started = time.perf_counter()
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    records = []
    for obj in sorted(bpy.context.scene.objects, key=lambda item: item.name):
        if obj.type not in {'MESH', 'CURVE', 'FONT'} or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=graph)
        try:
            if mesh is None or not mesh.vertices or not mesh.polygons:
                raise GeometryPreparationError('Visible object has no evaluated mesh: ' + obj.name)
            if abs(evaluated.matrix_world.to_3x3().determinant()) < 1e-12:
                raise GeometryPreparationError('Singular visible object transform: ' + obj.name)
            faces = [list(face.vertices) for face in mesh.polygons]
            plan = orientation_plan(faces)
            if any(not component['orientable'] for component in plan['components']):
                raise GeometryPreparationError('Evaluated mesh is nonorientable; an applied modifier cannot be repaired by source winding: ' + obj.name)
            records.append({'name': obj.name, 'world_matrix': [list(row) for row in evaluated.matrix_world],
                'vertices': [list(vertex.co) for vertex in mesh.vertices], 'polygons': faces,
                'materials': [face.material_index for face in mesh.polygons],
                'uv': [[list(corner.uv) for corner in layer.data] for layer in mesh.uv_layers]})
        finally:
            evaluated.to_mesh_clear()
    if not records:
        raise GeometryPreparationError('Model contains no visible geometry')
    encoded = json.dumps(records, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest(), time.perf_counter() - started


class RenderGeometryGuard:
    def __init__(self):
        import bpy
        self.errors, self.preparations, self.signature_seconds = [], [], []
        self.pending = self.rendered = None
        self.started = self.completed = 0

        @bpy.app.handlers.persistent
        def before(scene, *unused):
            self.started += 1
            self.pending = None
            try:
                self.preparations.append(prepare_scene())
                self.pending, seconds = geometry_signature()
                self.signature_seconds.append(seconds)
            except Exception as error:
                # Blender logs handler exceptions without failing the script.
                self.errors.append(type(error).__name__ + ': ' + str(error))

        @bpy.app.handlers.persistent
        def completed(scene, *unused):
            self.completed += 1
            if self.pending is not None:
                self.rendered = self.pending
            self.pending = None

        @bpy.app.handlers.persistent
        def cancelled(scene, *unused):
            self.errors.append('Render was cancelled')
            self.pending = None

        self.handlers = [('render_pre', before), ('render_complete', completed), ('render_cancel', cancelled)]
        for name, handler in self.handlers:
            getattr(bpy.app.handlers, name).append(handler)

    def finish(self, render_path):
        import bpy
        if self.errors:
            raise GeometryPreparationError('Render preparation failed: ' + '; '.join(self.errors))
        if self.rendered is None or self.pending is not None or self.started != self.completed:
            raise GeometryPreparationError('The last guarded render did not complete')
        self.preparations.append(prepare_scene())
        current, seconds = geometry_signature()
        self.signature_seconds.append(seconds)
        if current != self.rendered:
            raise GeometryPreparationError('Geometry changed after the last render; regenerate the candidate image before saving the baseline')
        result = bpy.data.images.get('Render Result')
        if result is None:
            raise GeometryPreparationError('The last render has no usable Render Result')
        render_path = Path(render_path)
        previous_format = bpy.context.scene.render.image_settings.file_format
        try:
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            result.save_render(filepath=str(render_path), scene=bpy.context.scene)
        except RuntimeError as error:
            raise GeometryPreparationError('The last Render Result cannot be saved: ' + str(error)) from error
        finally:
            bpy.context.scene.render.image_settings.file_format = previous_format
        if not render_path.is_file() or not render_path.stat().st_size:
            raise GeometryPreparationError('Canonical candidate image was not saved')
        encoded = bpy.data.images.load(str(render_path), check_existing=False)
        try:
            if min(encoded.size) < 1:
                raise GeometryPreparationError('Canonical candidate image has no pixels')
        finally:
            bpy.data.images.remove(encoded)
        return {'render_geometry_sha256': self.rendered, 'baseline_geometry_sha256': current,
            'preparations': self.preparations, 'identity_check_seconds': self.signature_seconds,
            'render_count': self.completed, 'additional_renders': 0,
            'render_path': str(render_path), 'render_sha256': _file_hash(render_path)}

    def close(self):
        import bpy
        for name, handler in self.handlers:
            collection = getattr(bpy.app.handlers, name)
            if handler in collection:
                collection.remove(handler)


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_geometry_script(script, render_path, blend_path, receipt_path):
    import bpy
    import runpy
    script, render_path, blend_path, receipt_path = map(Path, (script, render_path, blend_path, receipt_path))
    guard = RenderGeometryGuard()
    receipt = {'policy': POLICY, 'implementation_sha256': _file_hash(__file__),
        'source_script': str(script), 'source_script_sha256': _file_hash(script), 'status': 'failed'}
    try:
        runpy.run_path(str(script), run_name='__main__')
        receipt.update(guard.finish(render_path))
        save_handlers = bool(bpy.app.handlers.save_pre or bpy.app.handlers.save_post)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        prepare_scene(validate_only=True)
        saved_signature, seconds = geometry_signature()
        receipt['identity_check_seconds'].append(seconds)
        if saved_signature != guard.rendered:
            raise GeometryPreparationError('Geometry changed while saving the baseline')
        if save_handlers:
            # A save_post callback could restore memory after saving a changed mesh.
            handlers = {name: list(getattr(bpy.app.handlers, name)) for name in ('load_pre', 'load_post')}
            try:
                for name in handlers:
                    getattr(bpy.app.handlers, name).clear()
                bpy.ops.wm.open_mainfile(filepath=str(blend_path), use_scripts=False)
                prepare_scene(validate_only=True)
                saved_signature, seconds = geometry_signature()
                receipt['identity_check_seconds'].append(seconds)
                if saved_signature != guard.rendered:
                    raise GeometryPreparationError('Saved baseline differs from the rendered geometry')
            finally:
                for name, values in handlers.items():
                    getattr(bpy.app.handlers, name).clear()
                    getattr(bpy.app.handlers, name).extend(values)
        receipt['saved_geometry_sha256'] = saved_signature
        receipt.update(status='passed', baseline_path=str(blend_path), baseline_sha256=_file_hash(blend_path))
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error), handler_errors=guard.errors,
            preparations=guard.preparations)
        if isinstance(error, Exception) and not isinstance(error, OSError):
            guard.close()
            try:
                prepare_scene(validate_only=False)
                signature, _ = geometry_signature()
                salvage = blend_path.with_name('salvage.blend')
                bpy.ops.wm.save_as_mainfile(filepath=str(salvage))
                receipt['salvage'] = {'path': str(salvage), 'sha256': _file_hash(salvage),
                    'geometry_sha256': signature, 'preview_verified': False}
            except (GeometryPreparationError, ValueError, RuntimeError) as salvage_error:
                receipt['salvage_error'] = str(salvage_error)
        raise
    finally:
        guard.close()
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
