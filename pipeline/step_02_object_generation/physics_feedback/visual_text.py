"""Keep disconnected surface lettering visible without making contact proxies."""
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

POLICY = 'surface_text_visual_only.v1'
_TEXT = re.compile(r'\b(?:legends?|lettering|typography|branding|printed[ _-]+text|text[ _-]+labels?)\b', re.I)


def material_is_text(material):
    role = material.get('collision_role')
    if role is not None:
        return role == 'visual_only_text'
    return bool(_TEXT.search(material.get('name', '').replace('_', ' ')))


def inventory_materials(export_dir):
    path = Path(export_dir).parent / 'geometry_inventory' / 'inventory.json'
    if not path.is_file():
        return {}
    return {o['name']: o.get('materials', []) for o in json.loads(path.read_text())['objects']}


def source_text_vertices(record):
    """Shared text/structural vertices remain physical; no size-based exclusion."""
    text = set(v for f in record.get('text_faces', []) for v in f)
    solid = set(v for f in record.get('contact_faces', []) for v in f)
    return [record['vertices'][v] for v in sorted(text - solid)]


def collision_components(entry, components, *, return_indices=False):
    vertices = entry.get('visual_text_vertices', [])
    excluded = []
    if vertices:
        tree = cKDTree(np.asarray(vertices))
        for i, part in enumerate(components):
            # Export/repaired OBJ formatting may differ below a nanometre.
            if len(part.vertices) and np.max(tree.query(part.vertices)[0]) <= 1e-9:
                excluded.append(i)
    indices = [i for i in range(len(components)) if i not in excluded]
    audit = {'policy': POLICY, 'excluded_components': len(excluded),
             'excluded_component_indices': excluded,
             'excluded_source_faces': sum(len(components[i].faces) for i in excluded),
             'material_evidence': entry.get('visual_text_materials', []),
             'visual_geometry_changed': False, 'mass_model_changed': False}
    return (indices if return_indices else [components[i] for i in indices]), audit
