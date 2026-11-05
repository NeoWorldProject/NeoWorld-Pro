"""Recover bound authored exports, then explicit rigid/basic input derivatives."""
import copy
import json
from pathlib import Path
import shutil

from urdf_core.run_state import file_hash, write_json
from urdf_core.storage import copy_asset
from .mesh_preparation import prepare, GeometryInputError


def prepare_input(base, spec, *, args, api, budget, ask, images):
    from urdf_core.orchestration import _runtime, CandidateRejected
    from urdf_core.modeling import run_blender
    from urdf_core.degraded_asset import proxy_script
    from urdf_core.articulation_groups import FILE
    from .boolean_repair import source_binding
    base = Path(base)
    quality_path = base/'validation_report.json'
    blend = base/'geometry_baseline.blend'
    binding = {'export': source_binding(base/'urdf_export', spec), 'kinematics': spec,
        'visual_source_sha256': file_hash(blend) if blend.is_file() else None,
        'generation_report_sha256': file_hash(quality_path) if quality_path.is_file() else None}
    root = base.parent/'geometry_input_recovery'
    root.mkdir(exist_ok=True)
    receipt_path = root/'selection.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['source'] != binding:
            raise RuntimeError('Geometry input recovery source changed')
        selected = (root/receipt['directory']).resolve()
        if not selected.is_relative_to(root.resolve()):
            raise RuntimeError('Geometry recovery selection escapes its run')
        for relative, digest in receipt['files'].items():
            artifact = (selected/relative).resolve()
            if not artifact.is_relative_to(selected) or file_hash(artifact) != digest:
                raise RuntimeError('Geometry input recovery artifact changed')
        current = json.loads((selected/'kinematics.json').read_text())
        return selected, current, prepare(selected,current,budget=budget,api=api,ask=ask,images=images,blender=args.blender)
    try:
        return base, spec, prepare(base,spec,budget=budget,api=api,ask=ask,images=images,blender=args.blender)
    except Exception as error:
        from urdf_core.error_recovery import external_failure, cancelled
        if cancelled(error) or external_failure(error):
            raise
        if not isinstance(error, GeometryInputError) and getattr(args, '_error_recovery', None):
            args._error_recovery.disposition('physical_geometry', error, evidence=binding)
        reason = str(error)
    write_json(root/'source_failure.json', {'reason':reason,'source':binding})
    for method in ('reexport','rigid','basic'):
        target = root/method
        target.mkdir(exist_ok=True)
        child = copy.copy(args); child.out_dir=target
        if method != 'basic' and not blend.is_file():
            continue
        try:
            if method == 'basic':
                script = target/'proxy.py'
                script.write_text(proxy_script(args) + '\nbpy.ops.wm.save_as_mainfile(filepath=' + repr(str(target/'geometry_baseline.blend')) + ')\n')
                result=run_blender([args.blender,'--background','--factory-startup','--threads','4',
                    '--python-exit-code','1','--python',str(script)],cwd=target,log_path=target/'proxy.log',api=api)
                if result.returncode: raise RuntimeError('Deterministic basic proxy construction failed')
            else:
                copy_asset(blend,target/'geometry_baseline.blend')
            _runtime(child,api,'inspect',target/'geometry_inventory')
            inventory=json.loads((target/'geometry_inventory/inventory.json').read_text())
            current = spec if method=='reexport' else {'schema_version':1,'root':'body',
                'links':[{'name':'body','objects':[o['name'] for o in inventory['objects']]}],'joints':[]}
            write_json(target/'kinematics.json', current)
            _runtime(child,api,'build',target,target/'kinematics.json',render=False)
            structural_report = target/'validation_report.json'
            if structural_report.is_file():
                copy_asset(structural_report,target/'export_validation_report.json')
            quality = json.loads(quality_path.read_text()) if quality_path.is_file() else {
                'passed':False, 'evaluation_complete':False, 'reason':'upstream_quality_unavailable'}
            quality['geometry_recovery'] = {'method':method, 'source':str(quality_path),
                'sha256':binding['generation_report_sha256']}
            if method != 'reexport':
                quality.update(passed=False, validation_status='failed',
                    reason='geometry_recovery_replaced_topology_or_visuals')
            write_json(structural_report,quality)
            if method=='reexport' and (base/FILE).is_file(): copy_asset(base/FILE,target/FILE)
            if method!='basic' and (base/'Res1_3D.png').is_file(): copy_asset(base/'Res1_3D.png',target/'Res1_3D.png')
            write_json(target/'geometry_degradation.json', {'policy':'geometry_input_recovery.v1',
                'method':method,'reason':reason,'visual_geometry_preserved':method!='basic',
                'topology_replaced':method!='reexport','quality_passed':False,'source':binding})
            prepared=prepare(target,current,budget=budget,api=api,ask=ask,images=images,blender=args.blender)
        except Exception as error:
            if cancelled(error) or external_failure(error):
                raise
            if not isinstance(error, (GeometryInputError, CandidateRejected)) and getattr(args, '_error_recovery', None):
                args._error_recovery.disposition('geometry_' + method, error, evidence=binding)
            write_json(target/'rejection.json', {'reason':str(error)})
            continue
        write_json(receipt_path, {'source':binding,'directory':method,'files':{
            str(p.relative_to(target)):file_hash(p) for p in target.rglob('*') if p.is_file()}})
        return target,current,prepared
    raise RuntimeError('Basic geometry derivative could not be constructed; see geometry_input_recovery')
