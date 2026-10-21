"""Bind current physical candidates to their pre-test decisions and shared budget."""
from __future__ import annotations

import json
from pathlib import Path

from urdf_core.delivery_policy import POLICY, REPAIR_LIMITS
from urdf_core.run_state import file_hash, fingerprint
from .critic import bound_file
from .diagnosis import validate_parameter_bounds
from .protocol import make_test_contract


def _read(path):
    return json.loads(Path(path).read_text())


def _stage(root, manifest, name, key):
    stage = manifest.get('stages', {}).get(name, {})
    if stage.get('status') != 'passed' or stage.get('key') != key or not stage.get('artifacts'):
        raise ValueError('Frozen physics stage is incomplete or changed: ' + name)
    files = {bound_file(root / name, digest, root, allow_empty=True)
             for name, digest in stage['artifacts'].items()}
    result = (root / stage.get('result_path', '')).resolve()
    if result not in files:
        raise ValueError('Frozen physics result is not bound by its stage: ' + name)
    return _read(result), files


def verify_contract_lineage(root, manifest, delivery, contract_path, asset_dir):
    root, contract_path, asset_dir = (Path(p).resolve() for p in (root, contract_path, asset_dir))
    config = manifest['config']
    if config.get('harness_policy') != POLICY or config.get('acceptance_policy') != 'automatic_object_physics_v11':
        raise ValueError('Current physics lineage requires the current harness and frozen v11 rules')
    for path, digest in config['source_images'].items():
        if file_hash(path) != digest:
            raise ValueError('Frozen object reference image changed')
    initial, _ = _stage(root, manifest, 'initial_physics_profile',
        fingerprint({'task': fingerprint(config), 'stage': 'initial_physics_profile'}))
    from .test_modes import free_base_policy, verify_selection
    mode = free_base_policy(config['physics_free_base'])
    _, mode_files = _stage(root, manifest, 'free_base_test_mode', fingerprint(mode))
    context = config.get('scene_context')
    if context:
        from .scene_context import RULE_VERSION, load_context, verify_classification
        if config.get('support_applicability_rule') != RULE_VERSION or load_context(context['context_file']['path']) != context:
            raise ValueError('Frozen scene support context changed')
        support, _ = _stage(root, manifest, 'scene_support_applicability',
            fingerprint({'context': context['fingerprint'], 'rule': RULE_VERSION}))
        if (support != verify_classification(root / 'scene_support_applicability.json', context)
                or support.get('status') != 'classified'
                or initial['profile']['stable_required'] is not support['decision']['stable_required']):
            raise ValueError('Physical applicability differs from its pre-test scene decision')
    base = root / 'object_revisions/kinematics'
    spec = _read(base / 'kinematics.json')
    if spec != _read(asset_dir / 'kinematics.json'):
        raise ValueError('Physical refinement changed the generated object kinematics')
    from .loop import _validate_initial
    if _validate_initial(initial, spec, base, require_current=True, require_orientation=True,
                         free_base_policy=mode) != initial:
        raise ValueError('Frozen initial physics profile is not normalized')
    diagonal = _read(base / 'geometry_inventory/inventory.json')['scene_diagonal']
    ledger = _read(root / 'delivery_research.json')
    if ledger.get('schema_version') != POLICY or ledger.get('identity') != fingerprint({'config': config, 'initial': initial, 'spec': spec}):
        raise ValueError('Physics search is not bound to its frozen initial inputs')
    budget = _read(root / 'delivery_budgets/physics.json')
    if (budget.get('schema_version') != POLICY or budget.get('domain') != 'physics'
            or budget.get('limit') != config['shared_physics_repairs']
            or type(budget.get('limit')) is not int
            or not 0 <= len(budget['entries']) <= budget['limit'] <= REPAIR_LIMITS['physics']):
        raise ValueError('Physics shared repair budget changed or was exceeded')
    candidates = ledger['candidates']
    selected = [item for item in candidates if item['id'] == delivery['candidate_id']]
    if len(selected) != 1 or selected[0]['status'] != 'measured':
        raise ValueError('Delivery does not select one measured physical candidate')
    groups = None
    scope = config.get('joint_test_scope')
    preflight = make_test_contract(spec, initial['effort_budgets'], config['physics_config'], diagonal,
        profile=initial['profile'], acceptance_policy='object_functional_physics.v11',
        orientation_policy=initial['orientation_policy'], free_base_policy=mode)
    controls = {name: {key: values[key] for key in ('kp', 'kd', 'speed_fraction', 'initial_position')}
                for name, values in preflight['joints'].items()}
    if scope == 'representative':
        from .joint_groups import load_groups, initialize_from_representatives
        if _read(root / 'joint_groups/physics_test_contract.json') != preflight:
            raise ValueError('Grouping changed the frozen initial test contract')
        groups = load_groups(root / 'joint_groups', preflight)
        if groups is None:
            raise ValueError('Representative selection has no frozen joint groups')
        applied = initialize_from_representatives(groups=groups, profile=initial['profile'],
            initial_profile=initial['profile'], controller=controls, contract=preflight, spec=spec)
        if _read(root / 'joint_groups/initialization.json') != applied:
            raise ValueError('Frozen representative initialization changed')
    elif scope != 'all':
        raise ValueError('Missing frozen joint test scope')
    from .joint_groups import inconsistencies
    from .representative_motion import INPUT_FILE, selected_joints
    snapshots = {}
    for index, item in enumerate(candidates):
        if item.get('index') != index or item.get('id') != f'candidate_{index:03d}':
            raise ValueError('Physical candidate identity or ordering changed')
        folder = (root / 'delivery_candidates' / item['id']).resolve()
        if Path(item['folder']).resolve() != folder:
            raise ValueError('Physical candidate points outside its own snapshot')
        for name, digest in item['files'].items():
            bound_file(name, digest, root, allow_empty=True)
        snapshot_path = folder / 'parameters.json'
        if str(snapshot_path) not in item['files'] or str(folder / 'physics_test_contract.json') not in item['files']:
            raise ValueError('Physical candidate parameters lack immutable bindings')
        snapshot = _read(snapshot_path)
        if snapshot['candidate_id'] != item['id']:
            raise ValueError('Physical candidate snapshot identity changed')
        profile, controller, contract = (snapshot[key] for key in ('profile', 'controller', 'contract'))
        parent_index = snapshot['parent']
        parent = snapshots.get(parent_index)
        if index and (type(parent_index) is not int or parent is None):
            raise ValueError('Physical repair lacks an earlier immutable parent')
        if not index:
            expected_profile = applied['profile'] if groups is not None else initial['profile']
            if parent_index is not None or profile != expected_profile or snapshot.get('repair_id') is not None:
                raise ValueError('Initial candidate changed pre-test physical parameters')
            if controller != (applied['controller'] if groups is not None else controls):
                raise ValueError('Initial candidate changed frozen initial controls')
        elif snapshot.get('repair_id') not in budget['entries']:
            raise ValueError('Physical repair was not reserved in the shared budget')
        rebuilt = make_test_contract(spec, initial['effort_budgets'], config['physics_config'], diagonal,
            previous=parent['contract'] if parent else None, profile=profile, controller=controller,
            acceptance_policy='object_functional_physics.v11', orientation_policy=initial['orientation_policy'],
            free_base_policy=mode)
        if contract != rebuilt or _read(folder / 'physics_test_contract.json') != contract:
            raise ValueError('Physical repair changed its frozen test contract')
        if profile['stable_required'] is not initial['profile']['stable_required']:
            raise ValueError('Physical repair changed frozen support applicability')
        validate_parameter_bounds(profile, initial['profile'], controller, contract)
        verify_selection(root, config, contract, mode_files)
        if groups is not None:
            if inconsistencies(groups, profile, controller):
                raise ValueError('Physical repair broke frozen shared group parameters')
            request = _read(folder / INPUT_FILE)
            if request.get('candidate_id') != item['id']:
                raise ValueError('Representative selection belongs to another candidate')
            selected_joints(contract, folder / INPUT_FILE)
        elif (folder / INPUT_FILE).exists():
            raise ValueError('All-joint execution cannot introduce representative sampling')
        snapshots[index] = snapshot
    chosen = selected[0]
    folder = Path(chosen['folder'])
    if contract_path != folder / 'physics_test_contract.json' or asset_dir != folder / 'physical_asset':
        raise ValueError('Delivery uses another candidate contract or physical asset')
    if _read(asset_dir / 'physics_profile.json') != snapshots[chosen['index']]['profile']:
        raise ValueError('Delivered physical parameters differ from their selected snapshot')
    return {'candidate_id': chosen['id'], 'joint_test_scope': scope, 'repairs_reserved': len(budget['entries'])}
