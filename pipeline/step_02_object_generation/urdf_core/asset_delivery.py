"""Atomic, loadable asset snapshots with independent quality status."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import time
import uuid
from .run_state import file_hash, write_json
from .delivery_policy import POLICY

DELIVERY_SCHEMA = 'best_effort_delivery.v1'

def validate_files(folder, spec):
    from .urdf_validation import validate_urdf
    from pxr import Ar, Usd, UsdUtils, UsdGeom
    folder=Path(folder);result=validate_urdf(folder/'object.urdf',spec)
    if not result['passed']:raise ValueError('Delivery URDF is structurally invalid: '+str(result['errors']))
    from .environment import isaac_mdl_paths
    mdl_paths=isaac_mdl_paths()
    with Ar.ResolverContextBinder(Ar.DefaultResolverContext(list(map(str, mdl_paths)))):
        stage=Usd.Stage.Open(str(folder/'object.usd'))
        if not stage:raise ValueError('Delivery USD cannot be opened')
        if not any(prim.IsA(UsdGeom.Gprim) for prim in stage.Traverse()):
            raise ValueError('Delivery USD contains no geometry')
        layers,assets,missing=UsdUtils.ComputeAllDependencies(str(folder/'object.usd'))
    if missing:raise ValueError('Delivery USD resources missing: '+str(missing))
    for resource in [*[layer.realPath for layer in layers],*assets]:
        resource=Path(resource).resolve()
        if not resource.is_relative_to(folder.resolve()) and not any(resource.is_relative_to(p) for p in mdl_paths):
            raise ValueError('USD resource escapes delivery package: '+str(resource))
    # Mesh dependencies must remain inside the deliverable; built-in MDL is the sole SDK dependency.
    import xml.etree.ElementTree as ET
    for mesh in ET.parse(folder/'object.urdf').iter('mesh'):
        name=mesh.get('filename','');p=(folder/name).resolve()
        if not p.is_relative_to(folder.resolve()) or not p.is_file():
            raise ValueError('URDF mesh escapes the delivery package')
    return {'passed':True,'links':result.get('link_count'),'joints':result.get('joint_count'),
            'usd_layers':len(layers),'usd_assets':len(assets),'missing_resources':[]}


def publish(root, asset_dir, usd_path, spec, *, candidate_id, validation_status, report=None,
            findings=(), lineage=(), terminal=False, evaluation_complete=None):
    if validation_status not in ('passed','failed','incomplete'):raise ValueError('Unknown validation status')
    root=Path(root);asset_dir=Path(asset_dir);usd_path=Path(usd_path)
    if not (asset_dir/'object.urdf').is_file() or not usd_path.is_file():raise ValueError('No valid export candidate')
    store=root/'deliveries';store.mkdir(parents=True,exist_ok=True)
    target=store/(candidate_id+'_'+uuid.uuid4().hex[:10]);pending=store/('.pending_'+uuid.uuid4().hex)
    shutil.copytree(asset_dir,pending)
    try:
        shutil.copy2(usd_path,pending/'object.usd')
        write_json(pending/'kinematics.json',spec)
        technical=validate_files(pending,spec)
        # Independent source identities bind the whole candidate; never join pieces from different candidates.
        binding={'urdf':{'path':str((asset_dir/'object.urdf').resolve()),'sha256':file_hash(asset_dir/'object.urdf')},
                 'usd':{'path':str(usd_path.resolve()),'sha256':file_hash(usd_path)}}
        report_ref=None
        source_summary = {}
        if report:
            report=Path(report);report_ref={'path':str(report.resolve()),'sha256':file_hash(report)}
            shutil.copy2(report,pending/'source_validation_report.json')
            source_summary = json.loads(report.read_text())
        if evaluation_complete is None:
            evaluation_complete = bool(source_summary.get('evaluation_complete') and
                (source_summary.get('passed') or source_summary.get('trial_evaluation_policy') == 'all_independent_trials.v1'))
        status={'schema_version':DELIVERY_SCHEMA,'harness_policy':POLICY,
                'delivery_status':'completed','validation_status':validation_status,
                'evaluation_complete':bool(evaluation_complete),'assessment_status':'complete' if evaluation_complete else 'incomplete',
                'passed':validation_status=='passed','candidate_id':candidate_id,'terminal':terminal,
                'technical_validation':technical,'source_report':report_ref,'findings':list(findings),
                'lineage':list(lineage),'source_assets':binding,'created_epoch':time.time()}
        write_json(pending/'validation_report.json',status)
        hashes={str(p.relative_to(pending)):file_hash(p) for p in pending.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}
        write_json(pending/'delivery_manifest.json',dict(status,files=hashes))
        pending.rename(target)
        link=root/('.final_'+uuid.uuid4().hex);link.symlink_to(target.relative_to(root),target_is_directory=True)
        if (root/'final').exists() and not (root/'final').is_symlink():raise ValueError('Refuse to overwrite unrelated final directory')
        os.replace(link,root/'final')
        write_json(root/'delivery_status.json',dict(status,final=str((root/'final').resolve())))
        return target
    except BaseException:
        # Previous committed delivery remains intact on any failed publication.
        if pending.exists():shutil.rmtree(pending)
        raise


def verify(root, *, require_pass=False):
    root=Path(root);folder=(root/'final').resolve();manifest=json.loads((folder/'delivery_manifest.json').read_text())
    if (manifest.get('schema_version') != DELIVERY_SCHEMA or manifest.get('harness_policy') != POLICY
            or manifest.get('delivery_status') != 'completed'):raise ValueError('Invalid delivery')
    for name,digest in manifest['files'].items():
        p=(folder/name).resolve()
        if not p.is_relative_to(folder) or not p.is_file() or file_hash(p)!=digest:raise ValueError('Delivered file changed')
    summary=json.loads((folder/'validation_report.json').read_text())
    if any(manifest.get(k)!=summary.get(k) for k in summary):raise ValueError('Delivery status/evidence mismatch')
    for name,key in [('object.urdf','urdf'),('object.usd','usd')]:
        if file_hash(folder/name)!=manifest['source_assets'][key]['sha256']:
            raise ValueError('Delivered file differs from tested candidate')
    if manifest.get('source_report') and file_hash(folder/'source_validation_report.json')!=manifest['source_report']['sha256']:
        raise ValueError('Source report binding changed')
    if require_pass and manifest['validation_status']!='passed':raise ValueError('Asset delivered but quality not accepted')
    validate_files(folder,json.loads((folder/'kinematics.json').read_text()))
    return manifest


def finish_existing(root, state, reason, *, reason_kind='technical_error'):
    """Retain the best existing asset and distinguish budget exhaustion from interruption."""
    root = Path(root)
    previous = verify(root)
    folder = (root / 'final').resolve()
    publish(root, folder, folder / 'object.usd', json.loads((folder / 'kinematics.json').read_text()),
            candidate_id=previous['candidate_id'] + ('_budget_exhausted' if reason_kind == 'refinement_budget_exhausted' else '_interrupted'),
            validation_status='incomplete',
            findings=[*previous.get('findings', []), {reason_kind: reason}],
            lineage=[{'previous_delivery': str(folder), 'sha256': file_hash(folder / 'delivery_manifest.json')}],
            terminal=True, evaluation_complete=False)
    delivery = verify(root)
    status = json.loads((root / 'delivery_status.json').read_text())
    status.update(termination_reason=reason, termination_kind=reason_kind)
    write_json(root / 'delivery_status.json', status)
    state.data.update(status='completed', delivery_status='completed', validation_status='incomplete',
                      assessment_status='incomplete', evaluation_complete=False, termination_reason=reason,
                      termination_kind=reason_kind,
                      delivery_manifest=str((root / 'final/delivery_manifest.json').resolve()))
    state.save()
    ledger_path = root / 'delivery_research.json'
    if ledger_path.exists():
        from .delivery_policy import RunTiming
        ledger = json.loads(ledger_path.read_text())
        ledger.update(status='completed', termination_reason=reason, delivery_status='completed',
            termination_kind=reason_kind,
            validation_status='incomplete', assessment_status='incomplete', evaluation_complete=False,
            finished_epoch=time.time(), elapsed_seconds=RunTiming(root).elapsed)
        write_json(ledger_path, ledger)
    write_json(root / 'validation_report.json', {'delivery_status':'completed',
        'validation_status':'incomplete', 'assessment_status':'incomplete', 'evaluation_complete':False,
        'passed':False, 'termination_reason':reason, 'termination_kind':reason_kind, 'candidate_id':delivery['candidate_id']})
    return delivery


def physical_stage_result(root, checked):
    """Project a verified modern publication without changing its original quality verdict."""
    if (checked.get('publication_schema') != DELIVERY_SCHEMA
            or not str(checked.get('acceptance_policy', '')).startswith('automatic_object_physics_')):
        return checked
    result = dict(checked)
    result.setdefault('source_quality', {key: checked.get(key) for key in
        ('physics_passed', 'passed', 'validation_status', 'assessment_status', 'evaluation_complete', 'terminal')})
    folder = (Path(root)/'final').resolve()
    manifest = json.loads((folder/'delivery_manifest.json').read_text())
    ref = manifest.get('source_report')
    result['physics_passed'] = False
    if ref:
        path = Path(ref['path']).resolve()
        if (not checked.get('physics_report_path') or path != Path(checked['physics_report_path']).resolve()
                or file_hash(path) != ref['sha256']):
            raise ValueError('Physical stage report differs from verified delivery')
        native = json.loads(path.read_text())
        if native.get('delivered_usd_sha256') != file_hash(folder/'object.usd'):
            raise ValueError('Physical stage report refers to another delivered USD')
        result['physics_passed'] = (native.get('passed') is True
            and native.get('evaluation_complete', native.get('acceptance_complete')) is True)
    return result


def verify_run(root,expected_config=None,*,require_pass=False):
    """Explicit delivery gate; accepting callers must request require_pass."""
    root=Path(root).resolve();run=json.loads((root/'run_manifest.json').read_text())
    if run.get('config',{}).get('delivery_policy')!='best-effort':raise ValueError('Not a delivery-policy run')
    if run.get('config', {}).get('harness_policy') != POLICY:
        raise ValueError('Unsupported historical run; use a new output with this release')
    from .runtime_snapshot import verified_runtime, verify_frozen_run
    runtime = run.get('config', {}).get('runtime_snapshot')
    if runtime:
        package = verified_runtime(runtime)
        if package != Path(__file__).resolve().parents[1]:
            return physical_stage_result(root, verify_frozen_run(root, runtime, expected_config, require_pass=require_pass))
    from .run_validation import reject_assisted_provenance
    reject_assisted_provenance(root,run)
    if expected_config and any(run['config'].get(k)!=v for k,v in expected_config.items()):
        raise ValueError('Delivery does not match requested inputs/configuration')
    value=verify(root,require_pass=require_pass);folder=(root/'final').resolve()
    if require_pass and (run.get('status') != 'completed' or not value.get('terminal')
                         or not value.get('evaluation_complete')):
        raise ValueError('Passing acceptance requires terminal delivery and completed checks')
    physics=run['config']['acceptance_policy'].startswith('automatic_object_physics_')
    terminal = run.get('status') == 'completed' and bool(value.get('terminal'))
    accepted = terminal and bool(value.get('evaluation_complete')) and value['validation_status'] == 'passed'
    result={'delivery_status':'completed','validation_status':value['validation_status'],
        'evaluation_complete':terminal and value.get('evaluation_complete',False),
        'assessment_status':value.get('assessment_status','incomplete') if terminal else 'incomplete',
        'terminal':terminal,'run_status':run.get('status'),
        'acceptance_policy':run['config']['acceptance_policy'], 'publication_schema': value['schema_version'],
        'delivery_manifest_path': str(folder / 'delivery_manifest.json'),
        'asset_manifest_path': str(folder / 'asset_manifest.json'), 'kinematics_path': str(folder / 'kinematics.json'),
        'physics_passed':False,
        'passed':accepted,'urdf_path':str(folder/'object.urdf'),
        'usd_path':str(folder/'object.usd'),'meshes_dir':str(folder/'meshes'),
        'physics_profile_path':str(folder/'physics_profile.json') if physics and (folder/'physics_profile.json').is_file() else None,
        'physics_contract_path':None,'physics_report_path':None,'support_rest_pose_path':None}
    if physics and value.get('source_report'):
        from physics_feedback.verification import verify_simulation_report
        report_ref=value['source_report'];path=Path(report_ref['path'])
        if file_hash(path)!=report_ref['sha256']:raise ValueError('Final native report changed')
        native=json.loads(path.read_text());contract=Path(native['inputs']['contract']['path'])
        asset=Path(value['source_assets']['urdf']['path']).parent
        from physics_feedback.contract_lineage import verify_contract_lineage
        verify_contract_lineage(root, run, value, contract, asset)
        verify_simulation_report(path,root=root,contract_path=contract,asset_dir=asset,
            require_pass=value['validation_status']=='passed')
        if native['delivered_usd_sha256']!=file_hash(folder/'object.usd'):
            raise ValueError('Delivery does not match reloaded native USD')
        frozen = json.loads(contract.read_text())
        result.update(physics_contract_path=str(contract),physics_report_path=str(path),
                      physics_passed=(native.get('passed') is True and
                          native.get('evaluation_complete', native.get('acceptance_complete')) is True),
                      stable_required=frozen.get('stable_required'), physics_acceptance_policy=native['acceptance_policy'],
                      support_rest_pose_path=(native.get('support_rest_pose') or {}).get('path'),
                      free_base_policy=native.get('free_base_policy'), motion_scope_policy=native.get('motion_scope_policy'),
                      collision_observation_policy=native.get('collision_observation_policy'))
    elif physics and value['validation_status']=='passed':raise ValueError('Passed physics requires native evidence')
    elif not physics:
        from .generation_delivery import verify_source
        verify_source(root)
    return result
