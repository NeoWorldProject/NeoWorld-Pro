"""RGB-derived review panels and fast-v1 feedback contracts."""
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from .provider import digest_bytes, write_json

WARNING_TYPES = {'mask_interior_gap', 'environment_coverage', 'unobservable_detail', 'estimated_geometry'}


def validate_feedback(data, inventory):
    ids = {o['object_id'] for o in inventory['objects_data']} | {'Global', 'global'}
    for field in ('issues', 'warnings'):
        if not isinstance(data.get(field), list):
            raise ValueError(f'{field} must be an explicit array')
        for issue in data[field]:
            if (not isinstance(issue, dict) or issue.get('object_id') not in ids
                    or any(not isinstance(issue.get(k), str) or not issue[k].strip()
                           for k in ('issue_type', 'description', 'suggestion'))):
                raise ValueError('Feedback needs a valid object_id, issue_type, description and suggestion')
            if field == 'issues' and issue.get('severity') not in {'minor', 'major'}:
                raise ValueError('Blocking issues need minor/major severity')
            if field == 'warnings' and (issue['issue_type'] not in WARNING_TYPES or any(
                    issue.get(k) is not False for k in ('affects_extent', 'affects_identity', 'affects_geometry'))):
                raise ValueError('Warnings must explicitly have no effect on extent, identity or geometry')


def _valid_normalized_evidence_region(region):
    return (isinstance(region, list) and len(region) == 4
            and all(type(v) in (int, float) for v in region)
            and 0 <= region[0] < region[2] <= 1 and 0 <= region[1] < region[3] <= 1)


def register_images(out, image_path, fingerprint, paths, *, source='rgb_derived_perception'):
    path = out / 'perception_input_manifest.json'
    data = json.loads(path.read_text()) if path.exists() else {
        'schema_version': 'perception_inputs.v1', 'original_sha256': digest_bytes(image_path.read_bytes()),
        'fingerprint': fingerprint, 'images': {}}
    for image in paths:
        image = Path(image).resolve()
        if not image.is_relative_to(out.resolve()) or not image.is_file():
            raise ValueError('Review image must be generated inside this run')
        relative = str(image.relative_to(out.resolve()))
        sha = digest_bytes(image.read_bytes())
        if relative in data['images'] and data['images'][relative]['sha256'] != sha:
            raise ValueError('A registered review image was modified')
        data['images'][relative] = {'sha256': sha, 'source': source}
    write_json(path, data)


def _font(size=16):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def _fit(image, width, height):
    return ImageOps.contain(image, (width, height), method=Image.Resampling.LANCZOS)


def render_review_panels(image_path, rows, directory, prefix):
    """Four RGB rows/page; context expands the current crop by 25% per side."""
    directory.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        original = source.convert('RGB')
    width, height = original.size
    cards, records = [], []
    for row in rows:
        ident = row.get('object_id', row.get('candidate_id', 'unknown'))
        box = row.get('2d_bbox_normalized')
        if box is None:
            continue
        x0, y0, x1, y1 = [box[0]*width, box[1]*height, box[2]*width, box[3]*height]
        bounds = [math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)]
        dx, dy = (x1-x0)*.25, (y1-y0)*.25
        context = [max(0, math.floor(x0-dx)), max(0, math.floor(y0-dy)),
                   min(width, math.ceil(x1+dx)), min(height, math.ceil(y1+dy))]
        crop = original.crop(bounds)
        from .recovery import identity_stem
        crop_path = directory / f'{prefix}_{identity_stem(ident, row.get("name", ""))}_crop.png'
        crop.save(crop_path)
        context_image = original.crop(context)
        drawing = ImageDraw.Draw(context_image)
        drawing.rectangle([bounds[0]-context[0], bounds[1]-context[1], bounds[2]-context[0]-1,
                           bounds[3]-context[1]-1], outline='#ff3300', width=2)
        card = Image.new('RGB', (1024, 288), 'white')
        draw = ImageDraw.Draw(card)
        label = f'{ident}: {row.get("name", row.get("label", ""))}'
        draw.text((8, 3), label[:110], fill='black', font=_font(17))
        for offset, picture, title in [(0, context_image, 'CONTEXT / red = current box'),
                                       (342, crop, 'ACTUAL RECTANGULAR CROP'),
                                       (684, context_image, 'CONTEXT REPEATED (no mask)')]:
            draw.text((offset+6, 27), title, fill='black', font=_font(13))
            fitted = _fit(picture.convert('RGB'), 330, 238)
            card.paste(fitted, (offset+6, 48))
        cards.append(card)
        records.append({'id': ident, 'bbox_pixels': bounds, 'context_pixels': context,
                        'crop': crop_path.name, 'mask_sha256': None})
    paths = []
    for start in range(0, len(cards), 4):
        page = Image.new('RGB', (1024, 28+288*min(4, len(cards)-start)), 'white')
        ImageDraw.Draw(page).text((8, 3), f'{prefix}: isolated instances; columns retain different zoom scales', fill='black', font=_font())
        for offset, card in enumerate(cards[start:start+4]):
            page.paste(card, (0, 28+offset*288))
        path = directory / f'{prefix}_page_{start//4+1:02d}.png'
        page.save(path)
        paths.append(path)
    write_json(directory / f'{prefix}_panels.json', {'schema_version': 'review_panels.v1', 'objects': records,
               'pages': [p.name for p in paths], 'image_sha256': digest_bytes(image_path.read_bytes())})
    return paths


def _boundary_box_pixels(box, width, height):
    """Use the actual crop's outward pixel rounding, including subpixel boxes."""
    if not _valid_normalized_evidence_region(box):
        raise ValueError('Boundary evidence requires a valid normalized bbox')
    return [math.floor(box[0]*width), math.floor(box[1]*height),
            math.ceil(box[2]*width), math.ceil(box[3]*height)]


def _overlapping_boundary_intervals(start, end, tile_length=256, overlap=64):
    """Cover a half-open image interval with bounded, overlapping detail tiles."""
    if not 0 < overlap < tile_length or end <= start:
        raise ValueError('Boundary tile interval and overlap must be positive')
    if end-start <= tile_length:
        return [(start, end)]
    starts = list(range(start, end-tile_length+1, tile_length-overlap))
    if starts[-1] != end-tile_length:
        starts.append(end-tile_length)
    return [(offset, offset+tile_length) for offset in starts]


def _boundary_evidence_tiles(box, width, height):
    """Continuous four-edge coverage; contexts never become replacement boxes."""
    bounds = _boundary_box_pixels(box, width, height)
    left, top, right, bottom = bounds
    radius = max(24, min(64, math.ceil(.09*max(right-left, bottom-top))))
    patches = []
    for edge in ('top', 'right', 'bottom', 'left'):
        horizontal = edge in ('top', 'bottom')
        edge_start, edge_end = (left, right) if horizontal else (top, bottom)
        axis_limit, normal_limit = (width, height) if horizontal else (height, width)
        # Last included pixel belongs to right/bottom; PIL crop ends are exclusive.
        normal = {'top': top, 'right': right-1, 'bottom': bottom-1, 'left': left}[edge]
        normal_start, normal_end = max(0, normal-radius), min(normal_limit, normal+radius+1)
        intervals = _overlapping_boundary_intervals(max(0, edge_start-radius), min(axis_limit, edge_end+radius))
        for index, (start, end) in enumerate(intervals, 1):
            context = ([start, normal_start, end, normal_end] if horizontal else
                       [normal_start, start, normal_end, end])
            patches.append({'edge': edge, 'edge_tile': index, 'context_pixels': context,
                            'evidence_region_norm': [context[0]/width, context[1]/height,
                                                     context[2]/width, context[3]/height],
                            'reviewed_edge_interval_pixels': [max(edge_start, start), min(edge_end, end)]})
    return bounds, patches


def render_edge_panels(image_path, rows, directory, prefix='edges'):
    """Show every visible crop-edge pixel in overlapping original-RGB tiles.

    Tile bounds describe context, not a revised object extent. Long edges are
    split instead of downscaling an entire edge until small components vanish.
    The source image, input rows, and crop coordinates are never modified.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        original = source.convert('RGB')
    width, height = original.size
    cards, records = [], []
    for row in rows:
        box = row.get('2d_bbox_normalized')
        if row.get('is_environment_support') or box is None:
            continue
        bbox_pixels, patches = _boundary_evidence_tiles(box, width, height)
        left, top, right, bottom = bbox_pixels
        for patch_record in patches:
            bounds = patch_record['context_pixels']
            patch = original.crop(bounds)
            ImageDraw.Draw(patch).rectangle([left-bounds[0], top-bounds[1], right-bounds[0]-1,
                                           bottom-bounds[1]-1], outline='#ff3300', width=1)
            card = Image.new('RGB', (256, 224), 'white')
            draw = ImageDraw.Draw(card)
            draw.text((5, 3), f'{row["object_id"]}: {patch_record["edge"]} #{patch_record["edge_tile"]}',
                      fill='black', font=_font(14))
            draw.text((5, 23), f'RGB px {bounds[0]},{bounds[1]} - {bounds[2]},{bounds[3]}',
                      fill='black', font=_font(11))
            card.paste(_fit(patch, 246, 180), (5, 41))
            patch_record['page'] = f'{prefix}_page_{len(cards)//16+1:02d}.png'
            patch_record['tile_index_1based'] = len(cards)%16+1
            cards.append(card)
        records.append({'object_id': row['object_id'], 'original_box': list(box),
                        'bbox_pixels': bbox_pixels, 'patches': patches})
    paths=[]
    for start in range(0, len(cards), 16):
        page = Image.new('RGB', (1024, 30+224*math.ceil(min(16, len(cards)-start)/4)), 'white')
        ImageDraw.Draw(page).text((8, 3), 'CONTINUOUS EDGE CONTEXT / RED = actual crop boundary / tiles overlap',
                                 fill='black', font=_font(16))
        for index, card in enumerate(cards[start:start+16]):
            page.paste(card, ((index%4)*256, 30+(index//4)*224))
        path = directory/f'{prefix}_page_{start//16+1:02d}.png'
        page.save(path)
        paths.append(path)
    write_json(directory/f'{prefix}_panels.json', {'schema_version': 'boundary_review.v2', 'objects': records,
               'image_size_pixels': [width, height], 'bbox_pixel_convention': 'floor_min_ceil_max_half_open',
               'tile_length_pixels': 256, 'tile_overlap_pixels': 64,
               'image_sha256': digest_bytes(image_path.read_bytes()), 'pages': [p.name for p in paths]})
    return paths
