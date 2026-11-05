"""Blender exact boolean worker for explicitly selected physical operands."""
import json
from pathlib import Path
import sys


def execute(job):
    import bpy
    from mathutils import Vector
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    points = [v for part in job['operands'] for v in part['vertices']]
    low = [min(v[i] for v in points) for i in range(3)]
    high = [max(v[i] for v in points) for i in range(3)]
    center = [(a+b)/2 for a,b in zip(low,high)]
    scale = max(b-a for a,b in zip(low,high))
    objects=[]
    for i,part in enumerate(job['operands']):
        mesh=bpy.data.meshes.new('operand_'+str(i))
        mesh.from_pydata([[(v[k]-center[k])/scale for k in range(3)] for v in part['vertices']],[],part['faces'])
        mesh.update()
        obj=bpy.data.objects.new(mesh.name,mesh);bpy.context.collection.objects.link(obj);objects.append(obj)
    base=objects[job.get('base_operand',0)]
    bpy.context.view_layer.objects.active=base;base.select_set(True)
    for other in objects:
        if other==base:continue
        modifier=base.modifiers.new('PhysicalBoolean','BOOLEAN')
        modifier.operation='UNION' if job['operation']=='boolean_union' else 'DIFFERENCE'
        modifier.solver='EXACT';modifier.object=other
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    base.data.calc_loop_triangles()
    return {'id':job['id'],'status':'completed',
        'vertices':[[float(v.co[k])*scale+center[k] for k in range(3)] for v in base.data.vertices],
        'faces':[list(t.vertices) for t in base.data.loop_triangles]}


def main():
    request,result=map(Path,sys.argv[sys.argv.index('--')+1:])
    data=json.loads(request.read_text());rows=[]
    for job in data['jobs']:
        try:rows.append(execute(job))
        except (RuntimeError,ValueError) as error:
            rows.append({'id':job['id'],'status':'rejected','reason':str(error)})
        result.write_text(json.dumps({'schema':1,'results':rows}))
        print('BOOLEAN_JOB_COMPLETED',job['id'],flush=True)

if __name__=='__main__':main()
