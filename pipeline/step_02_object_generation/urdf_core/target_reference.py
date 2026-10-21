"""One target identity and image scope shared by generation and review."""
import io
import json
import math
from pathlib import Path

from .paths import RESOURCE_ROOT
from .run_state import file_hash, fingerprint, write_json


def load_reference(context_path, prompts_dir=None):
    if context_path is None:
        return None
    from physics_feedback.scene_context import load_context
    context = load_context(context_path)
    layout = json.loads(Path(context['layout']['path']).read_text())
    other = [{'object_id': obj['object_id'], 'name': obj['name']}
             for obj in layout['objects_data']
             if obj['object_id'] != context['target_id'] and not obj.get('is_environment_support', False)]
    prompt = Path(prompts_dir or RESOURCE_ROOT / 'prompts/urdf') / 'target_object.md'
    text = prompt.read_text().format(target_id=context['target_id'], target_name=context['target']['name'],
        other_objects='; '.join(f"{obj['object_id']} — {obj['name']}" for obj in other) or 'none')
    return {'policy': 'target_object_reference.v1', 'context': context, 'other_objects': other,
            'prompt_path': str(prompt.resolve()), 'prompt_sha256': file_hash(prompt), 'text': text,
            'implementation_sha256': file_hash(Path(__file__))}


def reference_content(reference, api, output):
    from PIL import Image, ImageDraw
    context = reference['context']
    if load_reference(context['context_file']['path'], Path(reference['prompt_path']).parent) != reference:
        raise ValueError('Target reference changed after binding')
    output = Path(output) / 'input_evidence'
    output.mkdir(parents=True, exist_ok=True)
    key = fingerprint(reference)
    scene = output / f'target_scene_{key}.png'
    with Image.open(context['scene_image']['path']) as source:
        picture = source.convert('RGB')
    width, height = picture.size
    box = context['target']['2d_bbox_normalized']
    pixels = [math.floor(box[0] * width), math.floor(box[1] * height),
              min(width - 1, math.ceil(box[2] * width)), min(height - 1, math.ceil(box[3] * height))]
    draw = ImageDraw.Draw(picture)
    draw.rectangle(pixels, outline=(255, 80, 0), width=max(2, round(max(width, height) / 400)))
    draw.text((pixels[0] + 3, max(0, pixels[1] - 14)), context['target_id'], fill=(255, 80, 0))
    buffer = io.BytesIO()
    picture.save(buffer, format='PNG')
    data = buffer.getvalue()
    if scene.exists() and scene.read_bytes() != data:
        raise ValueError('Annotated target scene changed')
    if not scene.exists():
        scene.write_bytes(data)
    write_json(output / f'target_reference_{key}.json', {
        'reference': reference, 'annotated_scene': {'path': str(scene.resolve()), 'sha256': file_hash(scene)}})
    return [api.make_text_input(reference['text']),
            api.make_text_input('Scene context with the target bounding box; this is not an object mask.'),
            api.make_image_input(scene), api.make_text_input('Unmodified target crop:'),
            api.make_image_input(Path(context['target_crop']['path']))]


def modeling_reference(args, api, images):
    reference = load_reference(getattr(args, 'scene_context', None), args.prompts_dir)
    verify_source_images(reference, args.image_paths)
    return reference, reference_content(reference, api, args.out_dir) if reference else images


def verify_source_images(reference, images):
    if reference and reference['context']['target_crop']['sha256'] not in {file_hash(Path(p)) for p in images}:
        raise ValueError('Target reference crop does not belong to this object')
