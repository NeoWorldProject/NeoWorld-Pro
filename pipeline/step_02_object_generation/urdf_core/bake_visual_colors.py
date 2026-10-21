"""Bake Blender's connected base-color graphs, preserving exported geometry.

Run in Blender. The accepted blend and OBJ exports remain read-only. UVs and
color images are new rendering evidence; no rigid body or collision data is edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urdf_core.content_errors import AppearanceContentError


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bake(blend, manifest, out, resolution=512, only_object=None):
    bpy.ops.wm.open_mainfile(filepath=str(blend), use_scripts=False)
    scene = bpy.context.scene
    scene.frame_set(1)
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 1
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 2
    scene.render.bake.margin = 4
    scene.render.bake.use_clear = True
    scene.render.bake.use_selected_to_active = False
    entries = json.loads(manifest.read_text())['meshes']
    groups = {}
    for entry in entries:
        if only_object is None or entry['object'] == only_object:
            groups.setdefault((entry['link'], entry['object']), []).append(entry)
    out.mkdir(parents=True, exist_ok=True)
    objects = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.hide_render = True
    for (link, name), parts in sorted(groups.items()):
        matches = [o for o in bpy.data.objects if o.type == 'MESH' and o.get('source_object') == name and o.get('urdf_link') == link]
        if len(matches) != 1:
            raise ValueError(f'Expected one bound source mesh: {link}/{name}, got {len(matches)}')
        obj = matches[0]
        if obj.modifiers:
            raise ValueError('Published rig geometry must already be evaluated: '+name)
        mesh = obj.data.copy()
        obj.data = mesh
        mesh.calc_loop_triangles()
        before = np.asarray([tuple(v.co) for v in mesh.vertices])
        original_uv = mesh.uv_layers.active.name if mesh.uv_layers.active else None
        materials = []
        for slot in range(len(mesh.materials)):
            original = mesh.materials[slot]
            if original is None or not original.use_nodes:
                material = bpy.data.materials.new('BakeConstant_'+name+'_'+str(slot))
                material.use_nodes = True
                material.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = original.diffuse_color if original else (.65, .65, .65, 1.)
            else:
                material = original.copy()
            mesh.materials[slot] = material
            # A bake UV atlas must not change the UV coordinates consumed by
            # the original source graph. Generated/object coordinates remain intact.
            if original_uv:
                nodes, links = material.node_tree.nodes, material.node_tree.links
                uv_node = nodes.new('ShaderNodeUVMap'); uv_node.uv_map = original_uv
                for node in list(nodes):
                    if node.type == 'TEX_IMAGE' and not node.inputs['Vector'].is_linked:
                        links.new(uv_node.outputs['UV'], node.inputs['Vector'])
                    elif node.type == 'TEX_COORD':
                        for connection in list(node.outputs['UV'].links):
                            links.new(uv_node.outputs['UV'], connection.to_socket)
            principals = [n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED']
            outputs = [n for n in material.node_tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output]
            if len(principals) != 1 or len(outputs) != 1:
                raise AppearanceContentError('Ambiguous base-color shader; explicit material conversion required: '+material.name)
            principal, output = principals[0], outputs[0]
            if not output.inputs['Surface'].is_linked or output.inputs['Surface'].links[0].from_node != principal:
                raise AppearanceContentError('Mixed/grouped surface shader requires explicit conversion: '+material.name)
            color = principal.inputs['Base Color']
            emission = material.node_tree.nodes.new('ShaderNodeEmission')
            if color.is_linked:
                material.node_tree.links.new(color.links[0].from_socket, emission.inputs['Color'])
            else:
                emission.inputs['Color'].default_value = color.default_value
            material.node_tree.links.new(emission.outputs[0], output.inputs['Surface'])
            materials.append({'slot':slot, 'source_material':original.name if original else None,
                              'base_color_linked':color.is_linked,
                              'roughness':float(principal.inputs['Roughness'].default_value),
                              'metallic':float(principal.inputs['Metallic'].default_value),
                              'roughness_linked':principal.inputs['Roughness'].is_linked,
                              'metallic_linked':principal.inputs['Metallic'].is_linked})
        bpy.ops.object.select_all(action='DESELECT')
        obj.hide_set(False); obj.hide_viewport = False; obj.hide_render = False
        obj.select_set(True); bpy.context.view_layer.objects.active = obj
        mesh.uv_layers.new(name='MaterialBakeUV')
        mesh.uv_layers.active_index = len(mesh.uv_layers)-1
        mesh.uv_layers.active.active_render = True
        bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=1.1519173, island_margin=.02, correct_aspect=True)
        bpy.ops.object.mode_set(mode='OBJECT')
        image = bpy.data.images.new('BakedColor_'+name, width=resolution, height=resolution, alpha=False)
        image.colorspace_settings.name = 'sRGB'
        for material in mesh.materials:
            nodes = material.node_tree.nodes
            node = nodes.new('ShaderNodeTexImage'); node.image = image
            nodes.active = node
        bpy.ops.object.bake(type='EMIT')
        image.filepath_raw = str(out/(link+'__'+name+'.png'))
        image.file_format = 'PNG'; image.save()
        if not np.array_equal(before, np.asarray([tuple(v.co) for v in mesh.vertices])):
            raise ValueError('Baking unexpectedly changed geometry')
        mesh.calc_loop_triangles()
        exported = []
        for part in parts:
            triangles = [t for t in mesh.loop_triangles if t.material_index == part['material_index']]
            if [list(t.vertices) for t in triangles] != part['source_triangle_vertices']:
                raise ValueError('Source blend and export topology mismatch: '+name)
            exported.append({'stem':Path(part['path']).stem, 'material_index':part['material_index'],
                             'triangle_vertices':[[list(mesh.vertices[v].co) for v in t.vertices] for t in triangles],
                             'triangle_uvs':[[list(mesh.uv_layers.active.data[i].uv) for i in t.loops] for t in triangles],
                             'backface_source_triangle_indices':part.get('backface_source_triangle_indices',[])})
        objects.append({'link':link, 'object':name, 'image':str(Path(image.filepath_raw).resolve()),
                        'image_sha256':sha(image.filepath_raw), 'materials':materials, 'parts':exported})
        obj.hide_render = True
        print(json.dumps({'object':name, 'status':'baked', 'image':image.filepath_raw}), flush=True)
    receipt = {'schema_version':'visual_color_bake.v1', 'source_blend':str(blend), 'source_blend_sha256':sha(blend),
               'export_manifest':str(manifest), 'export_manifest_sha256':sha(manifest),
               'script_sha256':sha(__file__), 'blender_version':bpy.app.version_string,
               'resolution':resolution, 'color_space':'sRGB', 'objects':objects,
               'scope':'Base color only, baked without illumination. Topology and vertex positions unchanged; no physical data authored.'}
    (out/'bake_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from urdf_core.execution import start_owner_monitor
    start_owner_monitor()
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend', type=Path, required=True)
    parser.add_argument('--export-manifest', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--only-object')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    try:
        bake(args.blend.resolve(), args.export_manifest.resolve(), args.out_dir.resolve(), args.resolution, args.only_object)
    except AppearanceContentError as error:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir/'appearance_rejection.json').write_text(json.dumps({'reason': str(error),
            'sources': {str(p.resolve()): sha(p) for p in (args.blend, args.export_manifest, Path(__file__))}}))
        raise
