"""CLI-owned test selection, frozen before physical inference and measurement."""
from pathlib import Path

from urdf_core.run_state import write_json

RUN_POLICY = 'automatic_object_physics_v11'
CONTRACT_POLICY = 'object_functional_physics.v11'
RUN_POLICIES = {RUN_POLICY}
FREE_BASE_VERSION = 'free_base_cli.v1'
FIXED_BASE_POLICY = 'elevated_full_range.v1'


def free_base_policy(enabled):
    if type(enabled) is not bool:
        raise ValueError('Free-base CLI selection must be boolean')
    return {'required': enabled, 'reason': (
        'Enabled explicitly by the command-line free-base option; scene support applicability still applies.'
        if enabled else 'Free-base joint tests disabled by command-line selection (default).')}


def resolve_free_base(args, selected_policy):
    """Omission means false for new runs, or the frozen value on resume."""
    import json
    requested = getattr(args, 'physics_free_base', None)
    if requested is not None and type(requested) is not bool:
        raise ValueError('Free-base CLI selection must be boolean')
    if selected_policy not in RUN_POLICIES:
        raise ValueError('Only the current v11 policy is supported')
    path = Path(args.out_dir) / 'run_manifest.json'
    previous = json.loads(path.read_text())['config'] if path.is_file() else None
    if previous is not None:
        if previous.get('acceptance_policy') not in RUN_POLICIES or type(previous.get('physics_free_base')) is not bool:
            raise ValueError('Changing historical test selection requires a new physics experiment')
        frozen = previous['physics_free_base']
        if requested is not None and requested != frozen:
            raise ValueError('Free-base CLI selection is frozen; changing it requires a new physics experiment')
        return frozen
    return False if requested is None else requested


def freeze_selection(root, enabled):
    """Persist actual CLI provenance instead of asking a model to select tests."""
    import json
    root = Path(root)
    path = root / 'free_base_test_mode.json'
    value = {'schema_version': FREE_BASE_VERSION, 'source': 'command_line',
             'physics_free_base': enabled, 'free_base_policy': free_base_policy(enabled),
             'fixed_base_policy': FIXED_BASE_POLICY}
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Frozen CLI test mode changed')
    else:
        write_json(path, value)
    return path


def verify_selection(root, config, contract, hashed_artifacts):
    import json
    root = Path(root)
    path = root / 'free_base_test_mode.json'
    enabled = config.get('physics_free_base')
    expected = {'schema_version': FREE_BASE_VERSION, 'source': 'command_line',
                'physics_free_base': enabled, 'free_base_policy': free_base_policy(enabled),
                'fixed_base_policy': FIXED_BASE_POLICY}
    if (config.get('free_base_policy_version') != FREE_BASE_VERSION
            or config.get('fixed_base_policy') != FIXED_BASE_POLICY
            or path.resolve() not in {Path(p).resolve() for p in hashed_artifacts}
            or json.loads(path.read_text()) != expected
            or contract.get('free_base_policy') != expected['free_base_policy']
            or contract.get('fixed_base_policy') != FIXED_BASE_POLICY):
        raise ValueError('Accepted tests differ from the frozen command-line selection')
    return expected['free_base_policy']
