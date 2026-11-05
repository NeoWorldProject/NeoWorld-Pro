"""Freeze one diagnostic projection before reserving its repair attempt."""
import json
from pathlib import Path

from urdf_core.run_state import file_hash, fingerprint, write_json

POLICY = 'immutable_physics_diagnostic.v1'


def _file(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('Diagnostic source is missing or outside its run: ' + str(path))
    return {'path': str(path), 'sha256': file_hash(path)}


def binding(*, root, repair_id, source, best, source_content, prompt_path):
    def candidate(item):
        folder = Path(item['folder'])
        return {'id': item['id'], 'parameters': _file(folder/'parameters.json', root),
                'evidence': _file(item['evidence'], root), 'report': _file(item['report'], root)}
    return {'repair_id': repair_id, 'source': candidate(source), 'best': candidate(best),
            'source_content_sha256': fingerprint(source_content),
            'prompt_sha256': file_hash(prompt_path)}


def freeze(path, *, root, inputs, build):
    path = Path(path)
    if path.is_file():
        value = json.loads(path.read_text())
        expected = value.pop('content_sha256', None)
        if (value.get('policy') != POLICY or value.get('binding') != inputs
                or fingerprint(value) != expected):
            raise ValueError('Frozen diagnostic request or its source binding changed')
    else:
        projected, frames, scope = build()
        value = {'policy': POLICY, 'binding': inputs, 'projected': projected,
                 'frames': frames, 'scope': scope}
    for frame in value['frames']:
        measured = _file(frame['path'], root)
        if measured['sha256'] != frame['sha256']:
            raise ValueError('Frozen diagnostic frame changed: ' + measured['path'])
        if frame['trial_id'] not in value['scope']['trial_ids']:
            raise ValueError('Frozen diagnostic frame belongs to another trial')
    if not path.exists():
        write_json(path, {**value, 'content_sha256': fingerprint(value)})
    return value['projected'], value['frames'], value['scope']
