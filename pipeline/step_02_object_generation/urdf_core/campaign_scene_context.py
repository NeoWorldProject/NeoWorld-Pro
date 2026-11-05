"""Bind fresh scene evidence to an unchanged generated object's identity."""
from pathlib import Path
import json

from .run_state import file_hash


def _iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return intersection / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)


def bind_scene_context(obj, path):
    from physics_feedback.scene_context import load_context
    path = Path(path).resolve()
    context = load_context(path)
    source = obj['source']
    if context['scene_image']['sha256'] != file_hash(source['image_path']):
        raise ValueError('Scene context belongs to a different original image')
    if file_hash(obj['crop_path']) != source['crop_sha256']:
        raise ValueError('Original generated-object crop changed')
    layout = json.loads(Path(context['layout']['path']).read_text())
    footprint = source['bbox_2d_normalized']
    score = _iou(footprint, context['target']['2d_bbox_normalized'])
    rivals = [_iou(footprint, o['2d_bbox_normalized']) for o in layout['objects_data']
              if o['object_id'] != context['target_id'] and not o.get('is_environment_support', False)
              and o.get('2d_bbox_normalized')]
    margin = score - max(rivals, default=0.)
    if score < .8 or margin < .2:
        raise ValueError('Scene context target cannot be unambiguously mapped to the generated asset')
    return {'path': str(path), 'sha256': file_hash(path), 'context_fingerprint': context['fingerprint'],
            'source_receipt_sha256': obj['source_receipt_sha256'], 'source_crop_sha256': source['crop_sha256'],
            'target_id': context['target_id'], 'target_crop_sha256': context['target_crop']['sha256'],
            'scene_image_sha256': context['scene_image']['sha256'], 'bbox_iou': score, 'alternative_margin': margin}


def verify_scene_context(obj, binding):
    if bind_scene_context(obj, binding['path']) != binding:
        raise ValueError('Frozen campaign scene context binding changed')
    from physics_feedback.scene_context import load_context
    return load_context(binding['path'])


def physics_scene_source_manifest(obj, binding):
    """Keep generation identity separate from a mapped, newer judgment crop."""
    context = verify_scene_context(obj, binding)
    source = Path(obj['generation']['run_dir']).resolve()
    receipt = Path(obj['source_receipt']).resolve()
    if file_hash(receipt) != obj['source_receipt_sha256'] or json.loads(receipt.read_text()) != obj['source']:
        raise ValueError('Generated-object source receipt changed')
    images = json.loads((source/'run_manifest.json').read_text())['config']['source_images']
    if images != {str(Path(obj['crop_path']).resolve()): obj['source']['crop_sha256']}:
        raise ValueError('Generation source image differs from its object receipt')
    paths = [source/'run_manifest.json', receipt, Path(obj['crop_path']), Path(binding['path']),
             *[Path(context[k]['path']) for k in ('scene_image','layout','target_crop')]]
    return {'schema_version':1, 'source_kind':'verified_kinematics_run', 'run_dir':str(source),
        'file_hashes':{str(p.resolve()):file_hash(p) for p in paths},
        'generation_source_images':images, 'source_receipt':str(receipt),
        'scene_context_binding':binding}


def verify_physics_scene_source(source, manifest_path, context_path, requested_images):
    """Verify fresh judgment evidence without relabeling a generated asset."""
    if not manifest_path or not context_path:
        raise ValueError('Different physics crop requires a bound scene context source manifest')
    path = Path(manifest_path).resolve(); value = json.loads(path.read_text())
    source = Path(source).resolve()
    if (value.get('schema_version') != 1 or value.get('source_kind') != 'verified_kinematics_run'
            or Path(value['run_dir']).resolve() != source):
        raise ValueError('Scene-context source manifest belongs to another generated run')
    hashes=value['file_hashes']
    if any(not Path(p).is_absolute() or file_hash(p)!=h for p,h in hashes.items()):
        raise ValueError('Scene-context source evidence changed')
    receipt=Path(value['source_receipt']).resolve(); original=json.loads(receipt.read_text())
    images=json.loads((source/'run_manifest.json').read_text())['config']['source_images']
    if len(images)!=1 or images!=value['generation_source_images']:
        raise ValueError('Generated asset source images changed')
    crop=next(iter(images))
    if images[crop]!=original['crop_sha256'] or hashes.get(str(receipt))!=file_hash(receipt):
        raise ValueError('Object receipt does not bind the original generation crop')
    obj={'source':original,'crop_path':crop,'source_receipt':str(receipt),
         'source_receipt_sha256':file_hash(receipt),'generation':{'run_dir':str(source)}}
    binding=value['scene_context_binding']
    if Path(binding['path']).resolve()!=Path(context_path).resolve():
        raise ValueError('Scene-context source manifest binds a different context')
    if physics_scene_source_manifest(obj,binding)!=value:
        raise ValueError('Scene-context source manifest changed or has incomplete evidence')
    context=verify_scene_context(obj,binding)
    if requested_images!={context['target_crop']['path']:context['target_crop']['sha256']}:
        raise ValueError('Requested physics image is not the mapped scene crop')
    return {'path':str(path),'sha256':file_hash(path),'generation_source_images':images,
            'scene_context_binding':binding}


def source_manifest_image(manifest_path, context_path=None):
    """Resolve the judgment image only from the import's hash-bound sources."""
    path = Path(manifest_path).resolve()
    value = json.loads(path.read_text())
    if value.get('schema_version') != 1 or value.get('source_kind') != 'verified_kinematics_run':
        raise ValueError('Physics source must declare a verified kinematics run')
    source = Path(value['run_dir'])
    hashes = value.get('file_hashes', {})
    if (not source.is_absolute() or not hashes
            or any(not Path(p).is_absolute() or file_hash(p) != h for p, h in hashes.items())):
        raise ValueError('Physics source evidence changed or is incomplete')
    manifest = source / 'run_manifest.json'
    if hashes.get(str(manifest.resolve())) != file_hash(manifest):
        raise ValueError('Physics source does not bind its generation manifest')
    binding = value.get('scene_context_binding')
    if binding:
        from physics_feedback.scene_context import load_context
        context_path = Path(context_path or binding['path']).resolve()
        context = load_context(context_path)
        image = Path(context['target_crop']['path'])
        if value.get('image_path') and Path(value['image_path']).resolve() != image:
            raise ValueError('Explicit physics image differs from the mapped crop')
        verify_physics_scene_source(source, path, context_path,
                                    {str(image): file_hash(image)})
        return image, context_path
    images = json.loads(manifest.read_text())['config']['source_images']
    if len(images) != 1:
        raise ValueError('Multiple source images require --image-dir')
    image = Path(value.get('image_path') or next(iter(images)))
    if (not image.is_absolute() or str(image.resolve()) not in images
            or file_hash(image) != images[str(image.resolve())]):
        raise ValueError('Physics image differs from the generation source')
    return image.resolve(), context_path
