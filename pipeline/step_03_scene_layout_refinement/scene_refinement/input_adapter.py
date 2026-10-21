"""Bind current object publications and their optional material overlays."""
from pathlib import Path

from .protocol import read_json, file_hash




def referenced_files(path):
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get('schema_version') == 'scene_assets.visual_overlay.v1':
        base = Path(manifest['base_manifest']).resolve()
        if read_json(base).get('schema_version') == 'scene_assets.visual_overlay.v1':
            raise ValueError('Nested visual overlays are not supported')
        result = [base, *referenced_files(base)]
        for entry in manifest['visual_materials'].values():
            receipt = Path(entry['path']).resolve()
            package = read_json(receipt)
            result.extend([receipt, Path(package['usd_path']), Path(package['bake_receipt']),
                           *map(Path, package['images']),
                           *map(Path, package.get('source_evidence', {}))])
        return result
    if manifest.get('schema_version') != 'object_pipeline.v1':
        raise ValueError('Unsupported asset manifest; use the current image pipeline')
    return []


def load_scene_assets(path):
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get('schema_version') == 'scene_assets.visual_overlay.v1':
        base = Path(manifest['base_manifest']).resolve()
        if read_json(base).get('schema_version') == 'scene_assets.visual_overlay.v1' or file_hash(base) != manifest['base_manifest_sha256']:
            raise ValueError('Invalid visual overlay base manifest binding')
        bound = load_scene_assets(base)
        ids = {r['object_id'] for r in bound['objects']}
        if not set(manifest['visual_materials']) <= ids:
            raise ValueError('Material overlay refers to an unknown object')
        for record in bound['objects']:
            if record['object_id'] in manifest['visual_materials']:
                record['visual_material_package'] = manifest['visual_materials'][record['object_id']]
        bound.setdefault('input_bindings', {}).update({str(p):file_hash(p) for p in [path, *referenced_files(path)]})
        return bound
    if manifest.get('schema_version') == 'object_pipeline.v1':
        from .assets import SceneAssetUnavailable
        # Environment supports come from the accepted layout, not object publications.
        records = [r for r in manifest['objects'] if r.get('is_environment_support') is not True]
        for record in records:
            for field in ('object_run_dir', 'usd_path'):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise SceneAssetUnavailable(
                        f"scene_asset_unavailable: object {record.get('object_id', '<missing id>')} "
                        f"(status={record.get('status', 'unknown')}) has no {field}; "
                        "scene composition requires a published physical asset")
        return {**manifest, 'objects':records}
    raise ValueError('Unsupported asset manifest; use the current image pipeline')
