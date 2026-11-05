"""Deterministic object comparisons derived only from the three existing images.

Crops preserve orientation and context. Relative size is judged from full frames.
"""
import math
from pathlib import Path
import numpy as np

from .protocol import digest, file_hash, read_json, write_json

RULES = {'schema_version':'scene_object_comparison_crops.v1', 'context_fraction':.12,
         'cell_width':300, 'cell_height':260, 'heading_height':36,
         'orientation':'unaltered', 'scale_comparison':'full_frames_only'}


def pixel_box(normalized, size, padding=.12):
    values=np.asarray(normalized,dtype=float)
    if values.shape!=(4,) or not np.isfinite(values).all():
        raise ValueError('Invalid object crop box')
    width,height=size
    x0,y0,x1,y1=values*np.array([width,height,width,height])
    if x1<=x0 or y1<=y0:
        raise ValueError('Empty object crop box')
    margin=max(x1-x0,y1-y0)*padding
    box=(max(0,math.floor(x0-margin)),max(0,math.floor(y0-margin)),
         min(width,math.ceil(x1+margin)),min(height,math.ceil(y1+margin)))
    if box[2]<=box[0] or box[3]<=box[1]:
        raise ValueError('Object is outside the source frame')
    return box


def projected_box(points,camera):
    points=np.asarray(points,dtype=float)
    eye=np.asarray(camera['position'],dtype=float)
    forward=np.asarray(camera['look_at'],dtype=float)-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    delta=points-eye;depth=delta@forward
    if not np.isfinite(delta).all() or np.any(depth<=camera['clipping_range'][0]):
        raise ValueError('Cannot project object crossing the near plane')
    half_x=camera['horizontal_aperture']/(2*camera['focal_length'])
    half_y=half_x*camera['resolution'][1]/camera['resolution'][0]
    xy=np.column_stack([.5+.5*(delta@right)/(depth*half_x),
                        .5-.5*(delta@up)/(depth*half_y)])
    return [*xy.min(0),*xy.max(0)]


def comparisons(candidate, original_image, output_dir, *, original_layout_path=None):
    from PIL import Image,ImageDraw
    from pxr import Usd,UsdGeom
    original_image=Path(original_image).resolve();output_dir=Path(output_dir).resolve()
    images=candidate.get('images',candidate.get('physics',{}).get('images',[]))
    if len(images)!=2:
        raise ValueError('Object comparisons require the two existing fixed views')
    usd_digest=file_hash(candidate['usd_path'])
    for item in images:
        if file_hash(item['path'])!=item['sha256'] or item.get('source_usd_sha256')!=usd_digest:
            raise ValueError('Crop source render does not bind the unchanged candidate USD')
    layout=read_json(candidate['layout_path']);manifest=read_json(candidate['scene_manifest_path'])
    objects=[o for o in layout['objects_data'] if not o['is_environment_support']]
    reference_path=original_layout_path or candidate['layout_path']
    references={o['object_id']:o for o in read_json(reference_path)['objects_data']}
    binding={'rules':RULES,'implementation_sha256':file_hash(__file__),
             'original_sha256':file_hash(original_image),'usd_sha256':usd_digest,
             'layout_sha256':file_hash(candidate['layout_path']),
             'original_layout_sha256':file_hash(reference_path),
             'manifest_sha256':file_hash(candidate['scene_manifest_path']),
             'views':[{'sha256':i['sha256'],'camera':i['camera']} for i in images]}
    folder=output_dir/digest(binding);receipt_path=folder/'crop_receipt.json'
    if receipt_path.exists():
        prior=read_json(receipt_path)
        if prior['binding']!=binding or file_hash(prior['image']['path'])!=prior['image']['sha256']:
            raise ValueError('Derived comparison evidence changed')
        return prior
    folder.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.Open(candidate['usd_path'])
    bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render])
    sources=[Image.open(p).convert('RGB') for p in [original_image,*[i['path'] for i in images]]]
    cw,ch,heading=RULES['cell_width'],RULES['cell_height'],RULES['heading_height']
    panel=Image.new('RGB',(3*cw,heading+len(objects)*(ch+heading)),'#f2f2f2')
    draw=ImageDraw.Draw(panel)
    for col,title in enumerate(('Original RGB','Candidate fixed view 1','Candidate fixed view 2')):
        draw.text((col*cw+8,10),title,fill='#111111')
    rows=[]
    for index,obj in enumerate(objects):
        oid=obj['object_id'];row=manifest['objects'][oid]
        prim=stage.GetPrimAtPath(row['prim_path'])
        if not prim.IsValid():raise ValueError('Missing rendered object prim: '+oid)
        bound=bounds.ComputeWorldBound(prim).ComputeAlignedRange()
        points=np.array([bound.GetCorner(i) for i in range(8)])
        boxes=[references[oid]['2d_bbox_normalized'],*[projected_box(points,i['camera']) for i in images]]
        y=heading+index*(ch+heading)
        draw.text((8,y+8),oid+' '+obj.get('name',oid),fill='#111111')
        row_receipt={'object_id':oid,'sources':[]}
        for col,(source,box) in enumerate(zip(sources,boxes)):
            rectangle=pixel_box(box,source.size,RULES['context_fraction'])
            crop=source.crop(rectangle)
            ratio=min((cw-16)/crop.width,(ch-16)/crop.height)
            resized=crop.resize((max(1,round(crop.width*ratio)),max(1,round(crop.height*ratio))),Image.Resampling.LANCZOS)
            panel.paste(resized,(col*cw+(cw-resized.width)//2,y+heading+(ch-resized.height)//2))
            row_receipt['sources'].append({'source_index':col,'pixel_box':rectangle,
                'source_sha256':binding['original_sha256'] if col==0 else images[col-1]['sha256']})
        rows.append(row_receipt)
    target=folder/'object_comparisons.png';panel.save(target)
    receipt={'binding':binding,'rows':rows,'image':{'path':str(target),'sha256':file_hash(target),
        'source':'derived_object_comparison','rules_version':RULES['schema_version']},
        'labels_transmitted':False,'camera_count_added':0}
    write_json(receipt_path,receipt)
    for source in sources:source.close()
    return receipt
