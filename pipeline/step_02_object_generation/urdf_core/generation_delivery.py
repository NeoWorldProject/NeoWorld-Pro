"""Generation best-candidate retention and explicit unaccepted physics handoff."""
import json
from pathlib import Path
import shutil
from .run_state import file_hash,write_json
from .asset_delivery import publish,verify,validate_files
from .critic_policy import accepted as critic_accepted


def record(root, folder, *, blender=None, isaac=None):
    root,folder=Path(root),Path(folder);path=root/'generation_delivery_candidates.json'
    data=json.loads(path.read_text()) if path.exists() else {'candidates':{},'best':None}
    spec=json.loads((folder/'kinematic_spec.json').read_text())
    validate_files(folder/'urdf_export',spec)
    key=str(folder.relative_to(root));entry=data['candidates'].setdefault(key,{'folder':key,'blender':None,'isaac':None})
    geometry_path = root / 'geometry_acceptance.json'
    if geometry_path.is_file():
        entry['geometry'] = json.loads(geometry_path.read_text())
    if blender is not None:entry['blender']=blender
    if isaac is not None:entry['isaac']=isaac
    entry['files']={str(p.relative_to(root)):file_hash(p) for p in [folder/'kinematic_spec.json',folder/'pipeline_scene.blend',*sorted((folder/'urdf_export').rglob('*'))] if p.is_file()}
    if geometry_path.is_file():
        entry['files']['geometry_acceptance.json'] = file_hash(geometry_path)
    for degradation in (root/'geometry_degradation.json', folder/'degradation.json'):
        if degradation.is_file(): entry['files'][str(degradation.relative_to(root))] = file_hash(degradation)
    def quality(e):
        reviews=[e['blender'],e['isaac']]
        if 'geometry' in e: reviews.append(e['geometry'])
        unknown=sum(v is None for v in reviews)
        failures=sum(v is not None and not critic_accepted(v) for v in reviews)
        critical=sum(len(v.get('critical_errors',[]))+sum(f.get('issue_type')=='asset_error' for f in v.get('findings',[])) for v in reviews if v)
        return (failures+unknown,critical,unknown)
    def passed(e,k):return e[k] is not None and critic_accepted(e[k])
    previous=data.get('best')
    eligible=[k for k,e in data['candidates'].items() if previous is None or
              all(not passed(data['candidates'][previous],stage) or passed(e,stage) for stage in ('blender','isaac'))]
    best=min(eligible,key=lambda k:quality(data['candidates'][k]));data['best']=best
    write_json(path,data)
    selected=data['candidates'][best];selected_folder=root/best
    status='passed' if quality(selected)[0]==0 else 'incomplete' if any(selected[k] is None for k in ('blender','isaac')) else 'failed'
    publish(root,selected_folder/'urdf_export',selected_folder/'urdf_export/object.usd',
        json.loads((selected_folder/'kinematic_spec.json').read_text()),candidate_id='generation_'+selected_folder.name,
        validation_status=status,findings=[selected],lineage=[{'path':str(path.resolve()),'candidate':best}],
        evaluation_complete=all(selected[k] is not None for k in ('blender','isaac')))
    return data


def finalize(root,state,reason):
    root=Path(root);path=root/'generation_delivery_candidates.json'
    if not path.exists():return False
    data=json.loads(path.read_text());best=data['candidates'][data['best']];folder=root/best['folder']
    for name,digest in best['files'].items():
        if file_hash(root/name)!=digest:raise ValueError('Saved generation candidate changed')
    for name in ('pipeline_scene.blend','Res2_Articulation.mp4'):
        if (folder/name).exists():shutil.copy2(folder/name,root/name)
    shutil.copy2(folder/'kinematic_spec.json',root/'kinematics.json')
    target=root/'urdf_export'
    if target.exists():shutil.rmtree(target)
    shutil.copytree(folder/'urdf_export',target)
    delivery=verify(root)
    publish(root,folder/'urdf_export',folder/'urdf_export/object.usd',
        json.loads((folder/'kinematic_spec.json').read_text()),candidate_id='generation_'+folder.name,
        validation_status=delivery['validation_status'],findings=[best],terminal=True,
        evaluation_complete=delivery.get('evaluation_complete',False))
    write_json(root/'validation_report.json',{'passed':delivery['validation_status']=='passed',
        'delivery_status':'completed','validation_status':delivery['validation_status'],
        'evaluation_complete':delivery.get('evaluation_complete',False),'reason':reason,
        'candidate':best,'delivery_manifest':str((root/'final/delivery_manifest.json').resolve())})
    state.data.update(status='completed',delivery_status='completed',validation_status=delivery['validation_status'],
        evaluation_complete=delivery.get('evaluation_complete',False),
        delivery_manifest=str((root/'final/delivery_manifest.json').resolve()),
        isaac_verification='passed' if delivery['validation_status']=='passed' else delivery['validation_status'])
    for stage in state.data.get('stages',{}).values():
        artifacts=stage.get('artifacts',{})
        if 'validation_report.json' in artifacts:
            artifacts['validation_report.json']=file_hash(root/'validation_report.json')
    state.data.pop('error',None);state.save()
    status=json.loads((root/'delivery_status.json').read_text());status.update(terminal=True,termination_reason=reason);write_json(root/'delivery_status.json',status)
    return True


def verify_source(root):
    root=Path(root);result=verify(root)
    data=json.loads((root/'generation_delivery_candidates.json').read_text());best=data['candidates'][data['best']]
    for name,digest in best['files'].items():
        if file_hash(root/name)!=digest:raise ValueError('Generation source evidence changed')
    if file_hash(root/'kinematics.json')!=file_hash(root/best['folder']/'kinematic_spec.json'):
        raise ValueError('Canonical source kinematics differ from selected candidate')
    for path in (root/best['folder']/'urdf_export').rglob('*'):
        if path.is_file() and file_hash(root/'urdf_export'/path.relative_to(root/best['folder']/'urdf_export'))!=file_hash(path):
            raise ValueError('Canonical export differs from selected candidate')
    return result


def import_source(source,target):
    verify_source(source)
    shutil.copytree(source,target)
    # Relative final symlinks and immutable evidence survive relocation.
    verify_source(target)
