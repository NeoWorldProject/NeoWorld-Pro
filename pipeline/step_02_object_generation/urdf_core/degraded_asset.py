"""Explicit dynamic derivatives for exhausted content repairs; never quality acceptance."""
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from .run_state import file_hash, write_json

POLICY = 'object_degraded_delivery.v1'
COLLISION_INPUT_POLICY = 'degraded_collision_input.v1'


def degradation_records(base):
    base = Path(base)
    paths = [base/'geometry_degradation.json']
    ledger = base/'generation_delivery_candidates.json'
    if ledger.is_file():
        data = json.loads(ledger.read_text())
        if data.get('best'):
            folder = (base/data['candidates'][data['best']]['folder']).resolve()
            if not folder.is_relative_to(base.resolve()): raise ValueError('Generation source escapes publication')
            paths.append(folder/'degradation.json')
    return [{'path': str(p.resolve()), 'sha256': file_hash(p), 'details': json.loads(p.read_text())}
            for p in paths if p.is_file()]


def proxy_description(args):
    size, color = [.2] * 3, [.65] * 3
    context = getattr(args, 'scene_context', None)
    if context and Path(context).is_file():
        from physics_feedback.scene_context import load_context
        data = load_context(context)
        candidates = [data.get(k, {}) for k in ('object', 'target', 'target_object')]
        for row in candidates:
            value = row.get('obb', {}).get('size_m') if isinstance(row, dict) else None
            if isinstance(value, list) and len(value) == 3 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in value):
                size = value; break
    images = getattr(args, 'image_paths', [])
    if images:
        from PIL import Image
        with Image.open(images[0]) as im:
            color = (np.asarray(im.convert('RGB').resize((1, 1)))[0, 0] / 255.).tolist()
    return {'size_m': size, 'color_rgb': color, 'policy': POLICY}


def proxy_script(args):
    info = proxy_description(args)
    return ("import bpy\nfrom mathutils import Vector\n"
        "bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)\n"
        "bpy.ops.mesh.primitive_cube_add(size=1)\nobj=bpy.context.object; obj.name='FallbackBody'\n"
        f"obj.dimensions={info['size_m']!r}\n"
        "bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)\n"
        "mat=bpy.data.materials.new('FallbackColor'); mat.diffuse_color=" + repr([*info['color_rgb'], 1.]) + "\n"
        "obj.data.materials.append(mat)\ns=bpy.context.scene; s.render.engine='BLENDER_WORKBENCH'\n"
        "s.display.shading.light='STUDIO'; s.display.shading.color_type='MATERIAL'\n"
        f"bpy.ops.object.camera_add(location=({max(info['size_m'])*2.5}, {-max(info['size_m'])*3}, {max(info['size_m'])*2}))\n"
        "cam=bpy.context.object; cam.rotation_euler=(-cam.location).to_track_quat('-Z','Y').to_euler(); s.camera=cam\n"
        "s.render.resolution_x=512; s.render.resolution_y=512; s.render.resolution_percentage=100\n"
        "s.render.image_settings.file_format='PNG'; s.render.filepath='Res1_3D.png'; bpy.ops.render.render(write_still=True)\n")


def _box(mesh):
    import trimesh
    points = np.asarray(mesh.vertices)
    if not len(points) or not np.isfinite(points).all():
        return trimesh.creation.box(extents=[.2]*3)
    extents = np.maximum(np.ptp(points, axis=0), 1e-4)
    result = trimesh.creation.box(extents=extents)
    result.apply_translation((points.max(0) + points.min(0)) / 2)
    return result


def collision_parts(mesh):
    from physics_feedback.lightweight_collision import plan_lightweight_collisions
    from .execution import terminal_failure
    try:
        defects = []
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            defects.append('empty_or_nonfinite_geometry')
        areas = np.asarray(mesh.area_faces)
        invalid_faces = int(np.count_nonzero(~np.isfinite(areas) | (areas <= 0)))
        if invalid_faces:
            defects.append(f'degenerate_faces={invalid_faces}')
        if not mesh.is_watertight:
            defects.append('open_or_nonmanifold_surface')
        if not mesh.is_winding_consistent:
            defects.append('inconsistent_winding')
        if defects:
            raise ValueError('invalid_fallback_surface: ' + ', '.join(defects))
        parts, _, audit = plan_lightweight_collisions([mesh])
        return parts, {'method': 'original_collision_algorithm', 'audit': audit}
    except (ValueError, RuntimeError) as error:
        if terminal_failure(error) or 'bad_alloc' in str(error) or 'out of memory' in str(error).lower():
            raise
        reason = str(error)
    try:
        hull = mesh.convex_hull
        if hull.volume <= 1e-15 or not np.isfinite(hull.vertices).all():
            raise ValueError('Degenerate hull')
        return [hull], {'method': 'part_convex_hull', 'reason': reason}
    except (ValueError, RuntimeError) as error:
        if terminal_failure(error):
            raise
        return [_box(mesh)], {'method': 'part_box', 'reason': reason}


def build_dynamic(base, output, *, reason, asset_error=None, physical_parameters=None):
    """Keep URDF link/joint identities and visual geometry; construct explicit physical defaults."""
    import trimesh
    from scipy.spatial.transform import Rotation
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    from .kinematics import fk, _pose, validate_spec
    base, output = Path(base), Path(output)
    source = base/'urdf_export'
    spec = validate_spec(json.loads((base/'kinematics.json').read_text()))
    physical_parameters = physical_parameters or {}
    binding = {'policy': POLICY, 'collision_input_policy': COLLISION_INPUT_POLICY,
        'topology_policy': 'explicit_root_body.v2',
        'reason': reason, 'asset_error': asset_error, 'physical_parameters': physical_parameters, 'sources': {
        str(p.resolve()): file_hash(p) for p in [base/'kinematics.json', *source.rglob('*')] if p.is_file()}}
    receipt_path = output/'degradation.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['binding'] != binding or any(file_hash(output/p) != h for p, h in receipt['artifacts'].items()):
            raise ValueError('Dynamic fallback binding or content changed')
        return output
    if output.exists(): shutil.rmtree(output)
    shutil.copytree(source, output)
    mesh_dir = output/'fallback_meshes'; mesh_dir.mkdir(exist_ok=True)
    tree = ET.parse(output/'object.urdf'); robot = tree.getroot()
    source_spec = deepcopy(spec)
    from pxr import Tf
    def mapped_names(values):
        used, result = set(values), {}
        for value in values:
            if Tf.IsValidIdentifier(value):
                result[value] = value
            else:
                candidate = Tf.MakeValidIdentifier(value)
                while candidate in used: candidate += '_'
                result[value] = candidate; used.add(candidate)
        return result
    link_names = mapped_names([r['name'] for r in spec['links']])
    joint_names = mapped_names([r['name'] for r in spec['joints']])
    spec['root'] = link_names[spec['root']]
    for row in spec['links']: row['name'] = link_names[row['name']]
    for row in spec['joints']:
        row.update(name=joint_names[row['name']], parent=link_names[row['parent']], child=link_names[row['child']])
    for row in robot.findall('link'): row.set('name', link_names[row.get('name')])
    for row in robot.findall('joint'):
        row.set('name', joint_names[row.get('name')])
        for key in ('parent', 'child'):
            child = row.find(key); child.set('link', link_names[child.get('link')])
    q = {j['name']: min(max(0., j['limit']['lower']), j['limit']['upper'])
         if j['type'] in ('revolute', 'prismatic') else 0. for j in spec['joints'] if j['type'] != 'fixed'}
    q_sources = {name: 'explicit_kinematic_zero_reference' for name in q}
    qd = {name: 0. for name in q}
    source_state_error = None
    if (source/'object.usd').is_file():
        try:
            authored_stage = Usd.Stage.Open(str(source/'object.usd'))
        except Tf.ErrorException as error:
            authored_stage = None
            source_state_error = str(error)
        for prim in authored_stage.Traverse() if authored_stage else []:
            name = joint_names.get(prim.GetName())
            if name not in q: continue
            info = next(j for j in spec['joints'] if j['name'] == name)
            scope = 'linear' if info['type'] == 'prismatic' else 'angular'
            unit = 1. if scope == 'linear' else math.pi/180.
            value = prim.GetAttribute(f'state:{scope}:physics:position').Get()
            if isinstance(value, (float, int)) and math.isfinite(value):
                q[name] = value*unit; q_sources[name] = 'bound_source_USD_state'
                velocity = prim.GetAttribute(f'state:{scope}:physics:velocity').Get()
                if isinstance(velocity, (float, int)) and math.isfinite(velocity): qd[name] = velocity*unit
    frames = fk(spec, q)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, 'Z'); UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, '/Asset').GetPrim(); stage.SetDefaultPrim(root)
    manifest = {'schema_version': 1, 'geometry_meshes': [], 'collision_meshes': [], 'link_properties': {},
                'source': {'degraded_source': str(base.resolve())}, 'degradation_policy': POLICY,
                'upstream_degradations': degradation_records(base),
                'derived_names': {'links': link_names, 'joints': joint_names}}
    publication = json.loads((base/'run_manifest.json').read_text()) if (base/'run_manifest.json').is_file() else {}
    legacy_export = publication.get('stages', {}).get('articulation_export', {}).get('artifacts', {})
    if (source/'export_manifest.json').is_file() and ((base/'generation_delivery_candidates.json').is_file()
            or {'pipeline_scene.blend', 'urdf_export/export_manifest.json'} <= legacy_export.keys()):
        manifest['source'].update(base_run=str(base.resolve()), export_manifest_sha256=file_hash(source/'export_manifest.json'))
    export = json.loads((source/'export_manifest.json').read_text()) if (source/'export_manifest.json').is_file() else {}
    names = {Path(r['path']).name: r.get('object') for r in export.get('meshes', [])}
    stems = [Path(e.get('filename')).stem for e in robot.iter('mesh') if e.get('filename')]
    visual_names = mapped_names(sorted(set(stems)))
    manifest['derived_names']['visual_stems'] = visual_names
    changes = []
    authored = []
    if export:
        from physics_feedback.physicalize import reconstruct_objects
        authored = reconstruct_objects(source, source_spec, clean=False)
        for row in authored: row['link'] = link_names[row['link']]
    def fmt(values): return ' '.join(str(float(v)) for v in values)
    def mesh_prim(path, mesh):
        obj = UsdGeom.Mesh.Define(stage, path)
        obj.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in mesh.vertices]))
        obj.CreateFaceVertexCountsAttr([3]*len(mesh.faces)); obj.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist())
        obj.CreateSubdivisionSchemeAttr('none')
        return obj
    material = UsdShade.Material.Define(stage, '/Asset/PhysicsMaterial')
    mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat.CreateStaticFrictionAttr(.5); mat.CreateDynamicFrictionAttr(.5); mat.CreateRestitutionAttr(0.)
    for link in robot.findall('link'):
        name = link.get('name'); body = UsdGeom.Xform.Define(stage, '/Asset/'+name)
        body.MakeMatrixXform().Set(Gf.Matrix4d(frames[name].T.tolist()))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        if name == spec['root'] and (spec['joints'] or len(spec['links']) > 1):
            UsdPhysics.ArticulationRootAPI.Apply(body.GetPrim())
        all_parts, mass_terms = [], []
        old_name = next(old for old, mapped in link_names.items() if mapped == name)
        parameters = physical_parameters.get('links', {}).get(old_name, {})
        contact = {'static_friction': .5, 'dynamic_friction': .5, 'restitution': 0.}
        for key, value in parameters.get('material', {}).items():
            if key in contact and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= (1 if key == 'restitution' else 2):
                contact[key] = value
        if contact['dynamic_friction'] > contact['static_friction']:
            contact['dynamic_friction'] = contact['static_friction']
        body_material = UsdShade.Material.Define(stage, '/Asset/ContactMaterials/'+name)
        contact_api = UsdPhysics.MaterialAPI.Apply(body_material.GetPrim())
        contact_api.CreateStaticFrictionAttr(contact['static_friction'])
        contact_api.CreateDynamicFrictionAttr(contact['dynamic_friction'])
        contact_api.CreateRestitutionAttr(contact['restitution'])
        for old in list(link.findall('collision')) + list(link.findall('inertial')): link.remove(old)
        collision_sources = []
        for i, visual in enumerate(link.findall('visual')):
            node = visual.find('geometry/mesh')
            if node is None: raise ValueError('Fallback expects the explicit mesh export contract')
            source_mesh = (source/node.get('filename')).resolve()
            if not source_mesh.is_relative_to(source.resolve()): raise ValueError('Mesh source escapes export')
            mesh = trimesh.load(str(source_mesh), force='mesh', process=False)
            mesh.apply_scale([float(v) for v in node.get('scale', '1 1 1').split()])
            origin = visual.find('origin')
            if origin is not None:
                transform = _pose({k: [float(v) for v in origin.get(k, '0 0 0').split()] for k in ('xyz', 'rpy')})
                mesh.apply_transform(transform)
            if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
                mesh = _box(mesh); changes.append({'link': name, 'part': i, 'visual_replaced': True})
            visual_path = mesh_dir/f'{name}_{i}_visual.obj'; mesh.export(visual_path)
            row = {'link': name, 'object': names.get(source_mesh.name) or source_mesh.stem,
                   'path': str(visual_path.relative_to(output)), 'sha256': file_hash(visual_path)}
            manifest['geometry_meshes'].append(row)
            node.set('filename', str(visual_path.relative_to(output))); node.attrib.pop('scale', None)
            if origin is not None: visual.remove(origin)
            obj = mesh_prim('/Asset/'+name+'/visuals/'+visual_names[source_mesh.stem]+'/mesh', mesh)
            color = visual.find('material/color')
            rgb = [float(v) for v in color.get('rgba').split()[:3]] if color is not None else [.65]*3
            obj.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
            if not authored:
                collision_sources.append((row['object'], mesh))
        if authored:
            collision_sources = [(r['object'], r['mesh']) for r in authored if r['link'] == name]
        for i, (object_name, mesh) in enumerate(collision_sources):
            rejected = asset_error or {}
            if (rejected.get('stage') == 'collision_cook' and link_names.get(rejected.get('link'), rejected.get('link')) == name
                    and rejected.get('part') == object_name):
                parts, audit = [_box(mesh)], {'method': 'native_cook_rejected_part_box', 'evidence': rejected}
            else:
                parts, audit = collision_parts(mesh)
            changes.append({'link': name, 'object': object_name, **audit})
            volume = max(sum(abs(p.volume) for p in parts), 1e-12)
            part_parameters = parameters.get('objects', {}).get(object_name, {})
            def positive(value): return isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            if positive(part_parameters.get('mass_kg')):
                mass_terms.append(part_parameters['mass_kg'])
            else:
                density = part_parameters.get('density_kg_m3', 1000.)
                mass_terms.append((density if positive(density) else 1000.)*volume)
            for j, part in enumerate(parts):
                all_parts.append(part)
                path = mesh_dir/f'{name}_{i}_{j}_collision.obj'; part.export(path)
                manifest['collision_meshes'].append({'link': name, 'object': object_name,
                    'path': str(path.relative_to(output)), 'sha256': file_hash(path)})
                shape = mesh_prim('/Asset/'+name+f'/collision_{i}_{j}', part)
                shape.GetPrim().CreateAttribute('neoworld:sourceObject', Sdf.ValueTypeNames.String).Set(object_name)
                shape.CreateVisibilityAttr('invisible')
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
                UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(body_material, materialPurpose='physics')
                element = ET.SubElement(link, 'collision'); geometry = ET.SubElement(element, 'geometry')
                ET.SubElement(geometry, 'mesh', filename=str(path.relative_to(output)))
        if not all_parts: raise ValueError('Export link has no recoverable geometry: '+name)
        merged = trimesh.util.concatenate(all_parts); box = _box(merged)
        volume = max(sum(abs(p.volume) for p in all_parts), 1e-12)
        mass = sum(mass_terms)
        if positive(parameters.get('mass_kg')): mass = parameters['mass_kg']
        elif positive(parameters.get('density_kg_m3')): mass = parameters['density_kg_m3']*volume
        dims = box.extents; center = box.bounds.mean(0)
        diagonal = mass/12 * np.array([dims[1]**2+dims[2]**2, dims[0]**2+dims[2]**2, dims[0]**2+dims[1]**2])
        manifest['link_properties'][name] = {'mass_kg': mass, 'volume_m3': volume,
            'center_of_mass_m': center.tolist(), 'inertia_kg_m2': np.diag(diagonal).tolist(),
            'principal_inertias_kg_m2': diagonal.tolist(), 'mass_basis': 'retained_parameters_with_explicit_missing_defaults',
            'inertia_basis': 'proxy_box_estimate', 'material': contact}
        ma = UsdPhysics.MassAPI.Apply(body.GetPrim()); ma.CreateMassAttr(float(mass))
        ma.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, center))); ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, diagonal)))
        inertial = ET.SubElement(link, 'inertial'); ET.SubElement(inertial, 'origin', xyz=fmt(center), rpy='0 0 0')
        ET.SubElement(inertial, 'mass', value=str(mass))
        ET.SubElement(inertial, 'inertia', ixx=str(diagonal[0]), iyy=str(diagonal[1]), izz=str(diagonal[2]), ixy='0', ixz='0', iyz='0')
    for j in spec['joints']:
        kind = j['type']; path = '/Asset/joints/'+j['name']
        joint = (UsdPhysics.FixedJoint.Define(stage, path) if kind == 'fixed' else
                 UsdPhysics.PrismaticJoint.Define(stage, path) if kind == 'prismatic' else UsdPhysics.RevoluteJoint.Define(stage, path))
        joint.CreateBody0Rel().SetTargets(['/Asset/'+j['parent']]); joint.CreateBody1Rel().SetTargets(['/Asset/'+j['child']])
        axis_frame = np.eye(3)
        if kind != 'fixed':
            axis_frame = Rotation.align_vectors([j['axis']], [[1, 0, 0]])[0].as_matrix()
            joint.CreateAxisAttr('X')
        origin = _pose(j['origin'])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, origin[:3, 3]))); joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        for attr, rot in ((joint.CreateLocalRot0Attr, origin[:3, :3] @ axis_frame), (joint.CreateLocalRot1Attr, axis_frame)):
            quat = Rotation.from_matrix(rot).as_quat(); attr(Gf.Quatf(float(quat[3]), Gf.Vec3f(*map(float, quat[:3]))))
        if kind != 'fixed':
            unit = 1. if kind == 'prismatic' else 180./math.pi
            if kind != 'continuous':
                joint.CreateLowerLimitAttr(float(j['limit']['lower']*unit)); joint.CreateUpperLimitAttr(float(j['limit']['upper']*unit))
            scope = 'linear' if kind == 'prismatic' else 'angular'
            joint.GetPrim().CreateAttribute(f'state:{scope}:physics:position', Sdf.ValueTypeNames.Float, custom=False).Set(q[j['name']]*unit)
            joint.GetPrim().CreateAttribute(f'state:{scope}:physics:velocity', Sdf.ValueTypeNames.Float, custom=False).Set(qd[j['name']]*unit)
    tree.write(output/'object.urdf', encoding='utf-8', xml_declaration=True)
    stage.Export(str(output/'object.usd'))
    write_json(output/'kinematics.json', spec); write_json(output/'asset_manifest.json', manifest)
    write_json(output/'physics_profile.json', {'policy': POLICY, 'estimated': True, 'density_kg_m3': 1000.,
        'material': {'static_friction': .5, 'dynamic_friction': .5, 'restitution': 0.},
        'initial_joint_positions': q, 'initial_joint_velocities': qd, 'joint_state_sources': q_sources,
        'source_USD_state_error': source_state_error,
        'retained_parameters': physical_parameters})
    from .asset_delivery import validate_files
    validate_files(output, spec)
    write_json(receipt_path, {'binding': binding, 'changes': changes, 'quality_passed': False,
        'state_provenance': 'constructed_explicit_initial_state', 'native_evaluation_complete': False,
        'artifacts': {str(p.relative_to(output)): file_hash(p) for p in output.rglob('*') if p.is_file()}})
    return output


def finish_physics(root, base, state, reason, *, asset_error=None):
    from .asset_delivery import publish, verify
    root, base = Path(root), Path(base)
    proposal_path = root/'physics_initial_proposal.json'
    proposal = json.loads(proposal_path.read_text()) if proposal_path.is_file() else {}
    folder = build_dynamic(base, root/'degraded_physical_asset', reason=reason, asset_error=asset_error,
                           physical_parameters=proposal.get('profile', {}))
    spec = json.loads((folder/'kinematics.json').read_text())
    publish(root, folder, folder/'object.usd', spec, candidate_id='degraded_physical', validation_status='incomplete',
            findings=[{'policy': POLICY, 'reason': reason, 'degradation': json.loads((folder/'degradation.json').read_text())}],
            terminal=True, evaluation_complete=False)
    state.data.update(status='completed', delivery_status='completed', validation_status='incomplete',
        evaluation_complete=False, termination_reason=reason, delivery_mode='degraded_dynamic')
    state.save()
    write_json(root/'validation_report.json', {'passed': False, 'delivery_status': 'completed',
        'validation_status': 'incomplete', 'evaluation_complete': False, 'delivery_mode': 'degraded_dynamic', 'reason': reason})
    ledger_path = root/'delivery_research.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    ledger.update({'status': 'completed', 'delivery_mode': 'degraded_dynamic',
        'degradation_policy': POLICY, 'validation_status': 'incomplete', 'termination_reason': reason,
        'delivery_manifest_sha256': file_hash(root/'final/delivery_manifest.json')})
    write_json(ledger_path, ledger)
    return verify(root)


def rigid_generation(args, api, state, reason):
    from .orchestration import _runtime
    from .generation_delivery import record, finalize
    root = Path(args.out_dir)
    inventory = json.loads((root/'geometry_inventory/inventory.json').read_text())
    spec = {'schema_version': 1, 'root': 'body',
            'links': [{'name': 'body', 'objects': [o['name'] for o in inventory['objects']]}], 'joints': []}
    folder = root/'degraded_articulation'; folder.mkdir(exist_ok=True)
    write_json(folder/'spec.json', spec)
    _runtime(args, api, 'build', folder, folder/'spec.json', render=False)
    write_json(folder/'degradation.json', {'policy': POLICY, 'reason': reason, 'topology_replaced': True,
                                         'quality_passed': False, 'visual_geometry_preserved': True})
    record(root, folder, blender={'passed': False, 'reason': 'Explicit rigid articulation fallback'})
    finalize(root, state, reason)
