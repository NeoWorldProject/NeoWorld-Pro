"""Prepare one immutable mass/collision source per authored part before parameters."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import json
import time
import numpy as np
import trimesh

from urdf_core.run_state import file_hash, fingerprint, write_json
from urdf_core.paths import RESOURCE_ROOT
from . import coacd_worker, native_geometry

POLICY = 'prepared_physical_mesh.v1'
FOLDER = 'prepared_geometry'


class GeometryInputError(ValueError):
    """Authored placement/topology needs re-export or whole-object replacement."""


def input_arrays(vertices, faces):
    v, f = np.asarray(vertices), np.asarray(faces)
    if v.ndim != 2 or v.shape[1:] != (3,) or not len(v):
        raise ValueError('No three-dimensional vertex array')
    if f.ndim != 2 or f.shape[1:] != (3,) or not len(f) or not np.issubdtype(f.dtype, np.integer):
        raise ValueError('No integer triangle array')
    if f.min() < 0 or f.max() >= len(v) or f.max() > np.iinfo(np.int32).max:
        raise ValueError('Triangle indices exceed the vertex array or native integer range')
    return np.asarray(v, dtype=np.float64), f.astype(np.int64)


def clean(mesh):
    v, f = input_arrays(mesh.vertices, mesh.faces)
    result = trimesh.Trimesh(vertices=v.copy(), faces=f.copy(), process=False)
    result.remove_infinite_values()
    result.process(validate=False)
    result.update_faces(result.unique_faces() & result.nondegenerate_faces())
    # Preserve coherent inward cavity walls rather than turning every shell outward.
    trimesh.repair.fix_winding(result)
    trimesh.repair.fix_inversion(result, multibody=False)
    result.remove_unreferenced_vertices()
    if (result.is_empty or not np.isfinite(result.vertices).all()
            or not np.isfinite(result.area) or result.area <= 0
            or not np.isfinite(result.extents).all() or max(result.extents) <= 0):
        raise ValueError('Standard cleanup left no usable surface')
    return result


def valid_solid(mesh):
    from .physicalize import validate_inertia
    input_arrays(mesh.vertices, mesh.faces)
    if not np.isfinite(mesh.vertices).all() or not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError('Prepared mesh is not a finite oriented closed surface')
    props = mesh.mass_properties
    validate_inertia(float(props['volume']), props['center_mass'], props['inertia'])
    return mesh


def native(mesh, operation, **options):
    result = coacd_worker.run_coacd(mesh.vertices, mesh.faces, operation=operation, **options)
    output = trimesh.Trimesh(vertices=result['vertices'], faces=result['faces'], process=False)
    return output, {k:v for k,v in result.items() if k not in {'vertices', 'faces'}}


def prepare_part(mesh, *, resolution=50, force=False, meshfix_first=False):
    started = time.monotonic()
    cleaned = clean(mesh)
    audit = {'cleanup_seconds': time.monotonic()-started, 'source_vertices': len(mesh.vertices),
             'source_faces': len(mesh.faces), 'clean_vertices': len(cleaned.vertices),
             'clean_faces': len(cleaned.faces), 'resolution': resolution, 'attempts': []}
    last = None
    with coacd_worker.isolated_worker(2):
        for fix in ([True] if meshfix_first else [False, True]):
            attempt = {'meshfix': fix}; tick = time.monotonic()
            try:
                source = cleaned
                if fix:
                    source, _ = native(cleaned, 'meshfix')
                    attempt['meshfix_seconds'] = time.monotonic()-tick
                    source = clean(source)
                tick = time.monotonic()
                result, detail = native(source, 'preprocess', resolution=resolution, force=force)
                attempt.update(detail, preprocess_seconds=time.monotonic()-tick)
                valid_solid(result)
                attempt['status'] = 'passed'; audit['attempts'].append(attempt)
                audit.update(seconds=time.monotonic()-started, geometry_degraded=False,
                             mass_basis='prepared_closed_volume', output_faces=len(result.faces))
                return result, audit
            except coacd_worker.CoacdSystemError as error:
                if not coacd_worker.repeated_native_crash(error.diagnostics):
                    raise
                last = str(error); attempt.update(status='native_failed', error=last, evidence=error.diagnostics)
            except (ValueError, TypeError) as error:
                last = str(error); attempt.update(status='invalid_geometry', error=last)
            audit['attempts'].append(attempt)
    error = ValueError('Mesh preparation failed: ' + str(last))
    error.diagnostics = audit
    raise error


def proxy(mesh):
    points = np.asarray(mesh.vertices, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise GeometryInputError('No trustworthy part placement for a proxy')
    try:
        result = trimesh.convex.convex_hull(points)
        valid_solid(result)
        method = 'part_convex_hull'
    except (ValueError, RuntimeError):
        dims = np.maximum(np.ptp(points, axis=0), 1e-4)
        result = trimesh.creation.box(extents=dims)
        result.apply_translation((points.min(0)+points.max(0))/2)
        valid_solid(result); method = 'part_box'
    return result, {'geometry_degraded': True, 'method': method, 'mass_basis': 'prepared_proxy_volume',
                    'minimum_box_extent_m': 1e-4}


def implementation():
    from . import boolean_repair, boolean_worker, visual_text, collision_batch, physicalize
    from urdf_core import urdf_validation
    library, headers = native_geometry.native_identity()
    versions = {n: version(n) for n in ('trimesh', 'numpy', 'scipy', 'coacd', 'pymeshfix')}
    try:
        versions['networkx'] = version('networkx')
    except PackageNotFoundError:
        versions['networkx'] = None  # Trimesh can use SciPy for graph operations.
    return {'policy': POLICY, 'versions': versions,
            'native': headers, 'library_sha256': file_hash(library),
            'sources': {Path(p).name:file_hash(p) for p in (__file__,native_geometry.__file__,coacd_worker.__file__,
                boolean_repair.__file__, boolean_worker.__file__, visual_text.__file__, collision_batch.__file__,
                physicalize.__file__, urdf_validation.__file__)}}


def load(base, spec):
    from .boolean_repair import source_binding
    folder = Path(base)/FOLDER
    receipt = json.loads((folder/'receipt.json').read_text())
    if receipt['policy'] != POLICY or receipt['source'] != source_binding(Path(base)/'urdf_export', spec):
        raise RuntimeError('Prepared geometry source binding changed')
    if receipt['implementation'] != implementation():
        raise RuntimeError('Prepared geometry runtime changed; use a new run')
    entries = []
    for row in receipt['parts']:
        path = folder/row['path']
        if not path.resolve().is_relative_to(folder.resolve()) or file_hash(path) != row['sha256']:
            raise RuntimeError('Prepared geometry content changed')
        with np.load(path, allow_pickle=False) as arrays:
            mesh = trimesh.Trimesh(vertices=arrays['vertices'], faces=arrays['faces'], process=False)
        if not row.get('visual_only_surface'): valid_solid(mesh)
        entries.append({**row, 'mesh': mesh, 'sources': row['source_files'], 'cleanup': row['audit'],
                        'prepared_physical': True})
    expected = {(r['name'],n) for r in spec['links'] for n in r['objects']}
    if {(r['link'],r['object']) for r in entries} != expected:
        raise RuntimeError('Prepared geometry does not cover every authored part')
    return entries


def prepare(base, spec, *, budget, api, ask, images=(), blender=None):
    from .physicalize import reconstruct_objects
    from .boolean_repair import source_binding, discover, render_evidence, run_worker, validate_result
    from .collision_batch import translation_match
    from .visual_text import collision_components
    base = Path(base); folder = base/FOLDER; folder.mkdir(exist_ok=True)
    if (folder/'receipt.json').exists():
        load(base,spec); return folder/'receipt.json'
    binding = {'source':source_binding(base/'urdf_export',spec), 'implementation':implementation()}
    binding_path = folder/'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise RuntimeError('Interrupted geometry preparation inputs changed')
    write_json(binding_path,binding)
    try:
        objects = reconstruct_objects(base/'urdf_export',spec,clean=False)
    except (ValueError, IndexError, KeyError) as error:
        raise GeometryInputError(str(error)) from error
    if (base/'physical_geometry/receipt.json').exists():
        from .boolean_repair import apply_saved
        objects = apply_saved(objects,base/'urdf_export',spec)
    records, pending, representatives = [], [], []

    def save(entry, mesh, audit, index, visual_only=False):
        path = folder/f'part_{index:04d}.npz'
        temporary = path.with_suffix('.pending.npz')
        np.savez_compressed(temporary,vertices=mesh.vertices,faces=mesh.faces)
        temporary.replace(path)
        row = {'link':entry['link'],'object':entry['object'],'path':path.name,'sha256':file_hash(path),
               'audit':audit,'source_files':entry['sources'],'visual_only_surface':visual_only}
        write_json(folder/f'part_{index:04d}.json',row)
        records.append(row)

    for index,entry in enumerate(objects):
        existing = folder/f'part_{index:04d}.json'
        if existing.exists():
            row=json.loads(existing.read_text())
            if row['link']!=entry['link'] or row['object']!=entry['object'] or file_hash(folder/row['path'])!=row['sha256']:
                raise RuntimeError('Prepared part checkpoint changed')
            records.append(row); continue
        mesh = entry['mesh']
        try:
            cleaned=clean(mesh)
            parts=list(cleaned.split(only_watertight=False,repair=False))
            indices,_=collision_components(entry,parts,return_indices=True)
            if not indices:
                save(entry,cleaned,{'method':'visual_only_surface','geometry_degraded':False},index,True);continue
            contact=trimesh.util.concatenate([parts[i] for i in indices])
            reused=False
            for original,result,audit in representatives:
                match=translation_match([original],[contact])
                if match is not None:
                    shift,error=match;result=result.copy();result.apply_translation(shift)
                    save(entry,result,{**audit,'reuse_translation_m':shift.tolist(),'reuse_error_m':error},index)
                    reused=True;break
            if reused:continue
            issues=discover([{**entry,'mesh':contact}])
            if issues:
                pending.append({'entry':{**entry,'mesh':contact},'index':index,
                    'error':'Intersecting closed shells require union/difference selection','boolean_issues':issues})
                continue
            tick=time.monotonic();result,audit=prepare_part(contact)
            audit['total_seconds']=time.monotonic()-tick
            representatives.append((contact,result,audit));save(entry,result,audit,index)
        except (ValueError,TypeError) as error:
            pending.append({'entry':entry,'index':index,'error':str(error)})
            write_json(folder/f'failure_{index:04d}.json',{'error':str(error),'details':getattr(error,'diagnostics',{})})

    while pending:
        number=len(budget.data['entries'])
        stage='mesh_geometry_repair_'+str(number)
        # Resume an interrupted model request under its original reservation.
        incomplete=[k for k,v in budget.data['entries'].items() if k.startswith('mesh_geometry_repair_') and v['status']=='pending']
        if incomplete: stage=incomplete[0]
        elif not budget.remaining: break
        groups=[]
        for item in pending:
            for group in groups:
                match=translation_match([group[0]['entry']['mesh']],[item['entry']['mesh']])
                if match is not None and bool(group[0].get('boolean_issues'))==bool(item.get('boolean_issues')):
                    group.append(item);break
            else:groups.append([item])
        typical=[group[0] for group in groups]
        public=[{'id':str(p['index']),'link':p['entry']['link'],'part':p['entry']['object'],'error':p['error'],
                 'equivalent_parts':len(group),
                 'boolean_operand_counts':[len(i['operands']) for i in p.get('boolean_issues',[])],
                 'vertices':len(p['entry']['mesh'].vertices),'faces':len(p['entry']['mesh'].faces)} for group in groups for p in [group[0]]]
        request_path = folder/(stage+'_request.json')
        request = {'issues': public, 'members': {str(member['index']): str(group[0]['index'])
                   for group in groups for member in group}}
        if request_path.exists():
            request = json.loads(request_path.read_text())
            public = request['issues']
            # Restore the original grouping even when part saves preceded interruption.
            by_representative = {}
            for item in pending:
                by_representative.setdefault(request['members'][str(item['index'])], []).append(item)
            groups = list(by_representative.values())
            typical = [group[0] for group in groups]
        else:
            write_json(request_path, request)
        budget.reserve(stage,'physical_geometry',{'binding':binding,'request':request})
        prompt=(RESOURCE_ROOT/'prompts/isaac/physics_geometry_repair.md').read_text()
        content=[api.make_text_input(prompt),api.make_text_input(json.dumps(public))]
        content.extend(api.make_image_input(Path(p)) for p in images if Path(p).is_file())
        for item in typical:
            for i,issue in enumerate(item.get('boolean_issues',[])):
                picture=folder/f"boolean_{item['index']}_{i}.png"
                if not picture.exists():render_evidence({'group_id':str(item['index'])+'/'+str(i),'_issue':issue},picture)
                content.append(api.make_image_input(picture))
        response_path=folder/(stage+'.json')
        raw_path=folder/(stage+'.txt')
        if raw_path.exists():raw=raw_path.read_text()
        else:
            raw=ask(stage,content)
            temporary=raw_path.with_suffix('.pending.txt')
            temporary.write_text(raw)
            temporary.replace(raw_path)
        try:
            value=api.extract_json(raw)
            choices=value['repairs']
            if not isinstance(choices,list) or len(choices)!=len(public):raise ValueError('One repair per failed geometry group is required')
            decisions={r['id']:r for r in choices}
            if len(decisions)!=len(public) or set(decisions)!={p['id'] for p in public}:raise ValueError('Repair identities differ')
            for row in choices:
                if set(row)-{'id','operation','resolution','reason','base_operand'} or row['operation'] not in {'coacd_remesh','meshfix','convex_hull','box','boolean_union','boolean_difference'}:raise ValueError('Unsupported geometry operation')
                if type(row.get('resolution',50)) is not int or not 20<=row.get('resolution',50)<=100:raise ValueError('Resolution must be 20..100')
                item=next((p for p in typical if request['members'][str(p['index'])]==row['id']), None)
                if item is None: continue
                if item.get('boolean_issues'):
                    if row['operation'] not in {'boolean_union','boolean_difference','convex_hull','box'}:raise ValueError('Intersecting shells require explicit solid semantics or a proxy')
                    if row['operation'].startswith('boolean_') and (type(row.get('base_operand',0)) is not int or any(not 0<=row.get('base_operand',0)<len(i['operands']) for i in item['boolean_issues'])):raise ValueError('Invalid boolean operand')
                elif row['operation'].startswith('boolean_'):raise ValueError('No bound boolean operands')
            write_json(response_path,value)
        except (ValueError,KeyError,TypeError) as error:
            budget.complete(stage,status='rejected',error=str(error));continue
        remaining=[]
        for group in groups:
            item=group[0]
            choice=decisions[request['members'][str(item['index'])]];mesh=item['entry']['mesh']
            try:
                if choice['operation'].startswith('boolean_'):
                    if blender is None:raise RuntimeError('Blender is required for physical boolean repair')
                    issues=item['boolean_issues'];decision={**choice,'base_operand':choice.get('base_operand',0)}
                    jobs=[{'id':str(i),**decision,'operands':[{'vertices':m.vertices.tolist(),'faces':m.faces.tolist()} for m in issue['operands']]} for i,issue in enumerate(issues)]
                    for i,job in enumerate(jobs):job['id']=str(i)
                    rows=run_worker(blender,jobs,folder/stage/str(item['index']))
                    pieces=[validate_result(row,issue,decision) for row,issue in zip(rows,issues)]
                    removed={c for issue in issues for c in issue['component_indices']}
                    pieces.extend(m for i,m in enumerate(issues[0]['parts']) if i not in removed)
                    result,audit=prepare_part(trimesh.util.concatenate(pieces))
                elif choice['operation'] in {'convex_hull','box'}:
                    result,audit=proxy(mesh)
                    if choice['operation']=='box':
                        result=trimesh.creation.box(extents=np.maximum(result.extents,1e-4),transform=trimesh.transformations.translation_matrix(result.bounds.mean(0)))
                        audit['method']='part_box'
                else:
                    result,audit=prepare_part(mesh,resolution=choice.get('resolution',50),force=True,
                                             meshfix_first=choice['operation']=='meshfix')
                for member in group:
                    shift,_=translation_match([mesh],[member['entry']['mesh']])
                    derived=result.copy();derived.apply_translation(shift)
                    save(member['entry'],derived,{**audit,'repair':choice,'representative':item['index']},member['index'])
            except (ValueError,TypeError) as error:
                remaining.extend({**member,'error':str(error)} for member in group)
        budget.complete(stage,status='evaluated',remaining=[p['index'] for p in remaining]);pending=remaining
    for item in pending:
        result,audit=proxy(item['entry']['mesh'])
        save(item['entry'],result,{**audit,'reason':item['error'],'repair_budget_exhausted':True},item['index'])
    for key, value in budget.data['entries'].items():
        if value['status'] == 'pending': budget.complete(key, status='evaluated', remaining=[])
    records.sort(key=lambda r:(r['link'],r['object']))
    write_json(folder/'receipt.json',{'policy':POLICY,**binding,'parts':records,
        'status':'prepared','geometry_degraded':any(r['audit'].get('geometry_degraded',False) for r in records)})
    load(base,spec)
    return folder/'receipt.json'
