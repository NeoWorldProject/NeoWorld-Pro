"""Verify a parser delivery independently of its quality acceptance."""
import json
from pathlib import Path

from .run_state import file_hash


def layout_quality(summary):
    accepted = summary.get('accepted') is True
    limit = summary.get('max_refine_iters')
    used = summary.get('refinements_used')
    selected = summary.get('selected_round')
    degraded = (summary.get('delivery_mode') in ('degraded_layout', 'environment_only')
                and summary.get('degradation_receipt') == 'degradation_receipt.json' and selected is None)
    exhausted = (summary.get('status') == 'unaccepted'
        and summary.get('termination_reason') == 'refinement_budget_exhausted'
        and type(limit) is int and 0 <= limit <= 3
        and type(used) is int and used == limit
        and (type(selected) is int and selected >= 1 or degraded))
    ready = summary.get('status') != 'technical_failure' and (accepted or exhausted)
    return {'accepted': accepted, 'ready_for_next_step': ready,
        'delivery_mode': ('accepted' if accepted else summary.get('delivery_mode', 'best_available')
                          if exhausted else 'unavailable'),
        'status': summary.get('status'), 'termination_reason': summary.get('termination_reason'),
        'selected_round': selected, 'refinements_used': used, 'max_refine_iters': limit,
        'remaining_issues': summary.get('remaining_issues', [])}


def verify_layout_delivery(layout_path, image_path=None):
    path = Path(layout_path).resolve()
    root = path.parent
    def read(relative):
        value = json.loads((root / relative).read_text())
        if not isinstance(value, dict):
            raise ValueError('Parser evidence must be a JSON object: ' + relative)
        return value
    summary, manifest = read('refine_summary.json'), read('run_manifest.json')
    quality = layout_quality(summary)
    if manifest.get('status') != 'completed' or not quality['ready_for_next_step']:
        raise ValueError('Stage-one layout is neither accepted nor a completed budget-exhausted delivery')
    artifacts = manifest.get('artifacts', {})
    if not isinstance(artifacts, dict) or not {'final_layout.json', 'refine_summary.json'} <= artifacts.keys():
        raise ValueError('Parser delivery has no bound layout and summary')
    for relative, expected in artifacts.items():
        artifact = (root / relative).resolve()
        if (not relative or not artifact.is_relative_to(root) or not artifact.is_file()
                or file_hash(artifact) != expected):
            raise ValueError('Layout or delivery evidence differs from the stage-one manifest: ' + relative)
    if file_hash(path) != artifacts['final_layout.json']:
        raise ValueError('Selected input differs from the delivered final_layout.json')
    if image_path is not None and manifest.get('binding', {}).get('image_sha256') != file_hash(Path(image_path)):
        raise ValueError('Layout delivery belongs to a different original RGB image')
    layout = read(path.name)
    selected = quality['selected_round']
    if selected is not None:
        relative = f'layout_round_{selected}.json'
        if relative not in artifacts or read(relative) != layout:
            raise ValueError('Delivered layout differs from its selected candidate')
    if not quality['accepted']:
        binding = manifest.get('binding', {})
        policy = binding.get('fast_policy', {})
        limit, used = quality['max_refine_iters'], quality['refinements_used']
        if (policy.get('policy') != 'scene_refinement_budget.v2'
                or binding.get('max_refine_iters') != limit or policy.get('max_refinements') != limit
                or summary.get('fast_policy') != policy):
            raise ValueError('Budget-exhausted delivery differs from the frozen refinement limit')
        ledger_path = 'fast/refinements.json'
        ledger = read(ledger_path) if ledger_path in artifacts else {'limit': limit, 'repairs': []}
        repairs = ledger.get('repairs')
        if (ledger.get('limit') != limit or not isinstance(repairs, list) or len(repairs) != used
                or [r.get('refinement') for r in repairs if isinstance(r, dict)] != list(range(1, used + 1))):
            raise ValueError('Budget-exhausted delivery differs from the refinement ledger')
        if 'input_audit.json' not in artifacts or read('input_audit.json').get('passed') is not True:
            raise ValueError('Best-layout delivery has no passing provenance audit')
        if quality['delivery_mode'] in ('degraded_layout', 'environment_only'):
            if ('degradation_receipt.json' not in artifacts or
                    read('degradation_receipt.json').get('policy') != 'layout_degraded_delivery.v1'):
                raise ValueError('Degraded layout has no bound construction receipt')
            if 'degraded_layout.json' not in artifacts or read('degraded_layout.json') != layout:
                raise ValueError('Delivered layout differs from its degraded construction')
    if not isinstance(layout.get('objects_data'), list) or not layout['objects_data']:
        raise ValueError('Layout delivery has no usable object inventory')
    return layout, quality
