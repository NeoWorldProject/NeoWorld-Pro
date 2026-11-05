"""One input-bound degradation pass before material restoration and scene search."""
from pathlib import Path

from .protocol import read_json, write_json, file_hash, digest

POLICY = 'scene_dynamic_derivative.v1'


def verify_derivative(record):
    ref = record['dynamic_derivative']
    if file_hash(ref['path']) != ref['sha256']:
        raise ValueError('Scene derivative receipt changed')
    receipt = read_json(ref['path'])
    if receipt.get('policy') != POLICY or receipt['object_id'] != record['object_id']:
        raise ValueError('Invalid scene derivative identity')
    for path, expected in {**receipt['sources'], **receipt['files']}.items():
        if file_hash(path) != expected:
            raise ValueError('Scene derivative input or output binding changed: '+path)
    checked = receipt['checked']
    if Path(checked['usd_path']).resolve() != Path(record['usd_path']).resolve():
        raise ValueError('Scene derivative USD binding changed')
    return checked


def prepare(path, layout_path, directory, runtime):
    from .assets import load_published_asset
    from .frozen_assets import verify_record
    from urdf_core.degraded_asset import build_dynamic_chain
    from urdf_core.error_recovery import external_failure, cancelled
    from physics_feedback.limited_evaluation import run as assess
    path, directory = Path(path).resolve(), Path(directory).resolve()
    manifest = read_json(path)
    if manifest.get('schema_version') != 'object_pipeline.v1':return path
    binding = {'source':str(path),'sha256':file_hash(path),'layout':file_hash(layout_path),
               'implementation':file_hash(__file__)}
    from physics_feedback.verify_physics_isaac import source_hashes
    binding['runtime_sources'] = source_hashes()
    target = directory/digest(binding)
    receipt_path = target/'receipt.json'; effective_path = target/'assets.json'
    if receipt_path.is_file():
        receipt=read_json(receipt_path)
        if receipt['binding']!=binding or file_hash(effective_path)!=receipt['sha256']:
            raise ValueError('Scene handoff receipt changed')
        for record in read_json(effective_path)['objects']:
            if record.get('dynamic_derivative'):verify_derivative(record)
        return effective_path
    objects = {o['object_id']:o for o in read_json(layout_path)['objects_data'] if not o['is_environment_support']}
    records = {o['object_id']:o for o in manifest['objects'] if not o.get('is_environment_support')}
    if len(records) != sum(not o.get('is_environment_support') for o in manifest['objects']):
        raise ValueError('Duplicate object identities in scene asset manifest')
    updated=[]
    for oid, obj in objects.items():
        record=records.get(oid, {'object_id':oid,'name':obj['name']})
        checked=None
        if record.get('object_run_dir') and record.get('usd_path'):
            # Integrity verification is outside the content fallback boundary.
            checked=verify_record(record)
            try:
                load_published_asset(record)
                updated.append(record)
                continue
            except Exception as error:
                if external_failure(error) or cancelled(error):raise
                from urdf_core.dynamic_asset import AssetBindingError
                if isinstance(error,AssetBindingError):raise
                reason=f'{type(error).__name__}: {error}'
        else:
            if record.get('failure_kind') == 'external_failure':
                from urdf_core.error_recovery import VerifiedExternalError
                raise VerifiedExternalError('Upstream object is technically paused',kind='upstream',evidence=record)
            reason='No usable object publication'
        root=target/oid
        base=root/'source'
        sources={str(path):file_hash(path),str(Path(layout_path).resolve()):file_hash(layout_path)}
        if checked:
            from urdf_core.storage import copy_asset_tree, copy_asset
            original=Path(checked['usd_path']).parent
            copy_asset_tree(original,base/'urdf_export')
            copy_asset(checked['kinematics_path'],base/'kinematics.json')
            sources.update({str(p.resolve()):file_hash(p) for p in original.rglob('*') if p.is_file()})
            for key, name in (('physics_profile_path','retained_profile.json'),
                              ('physics_contract_path','physics_test_contract.json')):
                if checked.get(key):
                    copy_asset(checked[key],root/name)
                    sources[str(Path(checked[key]).resolve())]=file_hash(checked[key])
        folder=build_dynamic_chain(base,root/'dynamic',reason=reason,dimensions=obj.get('obb',{}).get('size_m'))
        report_path=assess(folder,root,isaac_python=runtime.get('isaac_python'),ffmpeg=runtime.get('ffmpeg'))
        report=read_json(report_path)
        # Native import may materialize a distinct USD hierarchy. Bind this exact asset.
        result={'publication_schema':POLICY,'acceptance_policy':report['acceptance_policy'],'usd_path':str(report_path.parent/'object.usd'),
            'asset_manifest_path':str(folder/'asset_manifest.json'),'kinematics_path':str(folder/'kinematics.json'),
            'physics_profile_path':str(root/'limited_evaluation/profile.json'),
            'physics_contract_path':str(root/'limited_evaluation/contract.json'),
            'physics_report_path':str(report_path),'support_rest_pose_path':
                report.get('support_rest_pose',{}).get('path') if report.get('support_rest_pose',{}).get('verified') else None,
            'technical_available':True,'upstream_quality':(checked or {}).get('upstream_quality',{}),
            'degradation_reason':reason,'limited_passed':report.get('limited_passed'), 'passed':False}
        result['upstream_quality']={**result['upstream_quality'],'scene_derivative':True,
                                   'unassessed_checks':report.get('unassessed_checks',[])}
        derivative=root/'derivative.json'
        write_json(derivative,{'policy':POLICY,'object_id':oid,'sources':sources,'checked':result,
            'files':{str(p.resolve()):file_hash(p) for p in [*folder.rglob('*'), report_path,
                report_path.parent/'object.usd',root/'limited_evaluation/profile.json',root/'limited_evaluation/contract.json'] if p.is_file()}})
        new={k:v for k,v in record.items() if k not in ('verification_receipt_path','verification_receipt_sha256','publication_runtime')}
        new.update(object_run_dir=str(root),usd_path=result['usd_path'],status='delivered_unaccepted',
                   dynamic_derivative={'path':str(derivative),'sha256':file_hash(derivative)})
        load_published_asset(new)
        updated.append(new)
    write_json(effective_path,{**manifest,'objects':updated})
    write_json(receipt_path,{'binding':binding,'sha256':file_hash(effective_path)})
    return effective_path
