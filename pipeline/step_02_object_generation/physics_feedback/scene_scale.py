"""Bake the parsing size into a physical input before parameter generation."""
import copy
import json
from pathlib import Path

import numpy as np

from urdf_core.run_state import file_hash, fingerprint, write_json

POLICY = 'object_scene_scale.v1'
FILE = 'scene_scale.json'


def target(context):
    if context is None:
        return None
    size = np.asarray(context['target']['obb']['size_m'], float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('Scene scale requires three finite positive target lengths')
    return {'policy': POLICY, 'object_id': context['target']['object_id'],
            'layout_sha256': context['layout']['sha256'], 'target_size_m': size.tolist()}


def fit(inventory, size):
    boxes = [o['bbox_world'] for o in inventory['objects']]
    low = np.min([b['min'] for b in boxes], axis=0)
    high = np.max([b['max'] for b in boxes], axis=0)
    dimensions = high - low
    if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError('Scene scale requires valid three-dimensional source bounds')
    factor = float(dimensions @ np.asarray(size) / (dimensions @ dimensions))
    return factor, dimensions.tolist()


def scale_spec(spec, factor):
    value = copy.deepcopy(spec)
    for joint in value['joints']:
        joint['origin']['xyz'] = (np.asarray(joint['origin']['xyz']) * factor).tolist()
        if joint['type'] == 'prismatic':
            for key in ('lower', 'upper', 'velocity'):
                joint['limit'][key] *= factor
            if 'mimic' in joint and 'offset' in joint['mimic']:
                joint['mimic']['offset'] *= factor
    return value


def load(base):
    path = Path(base) / FILE
    return json.loads(path.read_text())['initialization'] if path.is_file() else None


def verify(base, context):
    base = Path(base).resolve()
    receipt = json.loads((base/FILE).read_text())
    if receipt['initialization']['target'] != target(context):
        raise ValueError('Physical scale differs from the parsing target')
    for name, digest in receipt['source'].items():
        if file_hash(name) != digest:
            raise ValueError('Scene scale source changed')
    for name, digest in receipt['files'].items():
        path = (base/name).resolve()
        if not path.is_relative_to(base) or file_hash(path) != digest:
            raise ValueError('Scaled physical input changed')
    return receipt['initialization']


def prepare(base, spec, context, *, args, api, out_dir):
    """Re-export a uniformly scaled copy; never edit the generation delivery."""
    from urdf_core.storage import copy_asset
    from urdf_core.modeling import run_blender
    from urdf_core.orchestration import _runtime
    from urdf_core.articulation_groups import FILE as GROUPS, load as load_groups, validate
    from urdf_core.execution import TechnicalExecutionError
    base, out_dir = Path(base).resolve(), Path(out_dir).resolve()
    binding = target(context)
    if binding is None:
        return base, spec
    existing = load(base)
    if existing:
        if existing['target'] != binding:
            raise ValueError('Scaled physical input belongs to a different scene target')
        return base, spec
    sources = [base/'geometry_baseline.blend', base/'geometry_inventory/inventory.json',
               base/'kinematics.json', base/'validation_report.json']
    if (base/GROUPS).is_file():
        sources.append(base/GROUPS)
    source = {str(p): file_hash(p) for p in sources}
    receipt_path = out_dir/FILE
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt['source'] != source or receipt['initialization']['target'] != binding:
            raise ValueError('Scene scale input changed; use a new run')
        if any(file_hash(out_dir/p) != h for p, h in receipt['files'].items()):
            raise ValueError('Scaled physical input changed')
        return out_dir, json.loads((out_dir/'kinematics.json').read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((base/'geometry_inventory/inventory.json').read_text())
    factor, dimensions = fit(inventory, binding['target_size_m'])
    scaled = scale_spec(spec, factor)
    write_json(out_dir/'kinematics.json', scaled)
    script = out_dir/'scale_input.py'
    script.write_text('import bpy\nfrom mathutils import Matrix\n'
        + 'bpy.ops.wm.open_mainfile(filepath=' + repr(str(base/'geometry_baseline.blend')) + ')\n'
        + 'scale=Matrix.Scale(' + repr(factor) + ',4)\n'
        + 'roots=[(o,o.matrix_world.copy()) for o in bpy.context.scene.objects if o.parent is None]\n'
        + 'for obj, matrix in roots: obj.matrix_world=scale @ matrix\n'
        + 'bpy.context.view_layer.update()\nbpy.context.preferences.filepaths.save_version=0\n'
        + 'bpy.ops.wm.save_as_mainfile(filepath=' + repr(str(out_dir/'geometry_baseline.blend')) + ')\n')
    result = run_blender([args.blender, '--background', '--factory-startup', '--disable-autoexec',
                         '--threads', '4', '--python-exit-code', '1', '--python', str(script)],
                        cwd=out_dir, log_path=out_dir/'scale.log', api=api)
    if result.returncode:
        raise TechnicalExecutionError('Blender could not export the scaled physical input')
    child = copy.copy(args)
    child.out_dir = out_dir
    _runtime(child, api, 'inspect', out_dir/'geometry_inventory')
    _runtime(child, api, 'build', out_dir, out_dir/'kinematics.json', render=False)
    # Preserve the upstream quality verdict separately from structural re-export.
    copy_asset(out_dir/'validation_report.json', out_dir/'scale_export_validation.json')
    copy_asset(base/'validation_report.json', out_dir/'validation_report.json')
    for name in ('Res1_3D.png', 'geometry_degradation.json'):
        if (base/name).is_file():
            copy_asset(base/name, out_dir/name)
    if (base/GROUPS).is_file():
        groups = copy.deepcopy(load_groups(base/GROUPS, spec))
        groups['spec_fingerprint'] = fingerprint(scaled)
        groups['source'] = {'spec': {'path': str(out_dir/'kinematics.json'),
                                   'sha256': file_hash(out_dir/'kinematics.json')},
                            'urdf': {'path': str(out_dir/'urdf_export/object.urdf'),
                                     'sha256': file_hash(out_dir/'urdf_export/object.urdf')},
                            'artifacts': {str(p.relative_to(out_dir/'urdf_export')): file_hash(p)
                                          for p in (out_dir/'urdf_export').rglob('*') if p.is_file()}}
        validate(groups, scaled)
        write_json(out_dir/GROUPS, groups)
    actual = json.loads((out_dir/'geometry_inventory/inventory.json').read_text())
    _, actual_dimensions = fit(actual, binding['target_size_m'])
    if not np.allclose(actual_dimensions, np.asarray(dimensions)*factor, rtol=1e-5, atol=1e-7):
        raise ValueError('Scaled export does not preserve the uniform source dimensions')
    initialization = {'target': binding, 'uniform_scale': factor, 'source_size_m': dimensions,
                      'initialized_size_m': actual_dimensions,
                      'size_residual_m': (np.asarray(actual_dimensions)-binding['target_size_m']).tolist(),
                      'mass_policy': 'recompute_from_scaled_geometry_and_proposed_parameters'}
    write_json(receipt_path, {'initialization': initialization, 'source': source,
        'files': {str(p.relative_to(out_dir)): file_hash(p) for p in out_dir.rglob('*')
                  if p.is_file() and p != receipt_path}})
    return out_dir, scaled
