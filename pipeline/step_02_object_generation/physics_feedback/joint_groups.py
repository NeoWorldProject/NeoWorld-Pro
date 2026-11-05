"""Frozen shared-parameter groups and one persistent random representative."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from urdf_core.run_state import file_hash, write_json
from urdf_core.urdf_validation import read_urdf, urdf_fk
from .protocol import json_hash, free_base_required

POLICY = 'shared_joint_groups.v2'
SUPPORTED_POLICIES = {POLICY}
AUTH_FILE = 'joint_group_authorization.json'
GROUP_FILE = 'joint_group_contract.json'
PASSIVE = ('static_friction_effort', 'dynamic_friction_effort', 'damping')
CONTROL = ('kp', 'kd', 'speed_fraction')
MATERIAL = ('static_friction', 'dynamic_friction')  # Restitution remains frozen zero.
SOURCE_ROLES = {'spec', 'profile', 'contract', 'manifest', 'urdf', 'cooked'}


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_hash(path)}


def read_bound(row):
    if set(row) != {'path', 'sha256'} or file_hash(row['path']) != row['sha256']:
        raise ValueError('Joint grouping source changed')
    return Path(row['path'])


def _close(a, b):
    return np.allclose(a, b, rtol=1e-10, atol=1e-10)


def _frozen(j):
    return {k: j[k] for k in ('type', 'lower', 'upper', 'initial_position', 'max_effort')}


def _settings(name, joint, profile, contract):
    return {'frozen': _frozen(contract['joints'][name]),
            'passive': profile['joints'][name],
            'control': {k: contract['joints'][name][k] for k in CONTROL},
            'material': profile['links'][joint['child']]['material']}


def _scope_key(scope):
    # Distinct ground-obstructed intervals are never covered by one operation.
    return (scope['free_status'], round(scope['free_lower'], 10), round(scope['free_upper'], 10))


def classify(spec, profile, contract, manifest, cooked, parsed, seed, *, policy=POLICY):
    """Classify by either independent slides OR matching repeated subtrees."""
    from .motion_scope import plan_joint_motion
    from .representative_motion import _clouds, _shape_equal
    if policy not in SUPPORTED_POLICIES:
        raise ValueError('Unsupported shared grouping policy')
    from .joint_group_geometry import SourceGeometry, coordinates_close, rotation_close
    source = SourceGeometry(parsed, manifest)
    joints = {j['name']: j for j in spec['joints']}
    active = sorted((n for n, j in joints.items() if j['type'] != 'fixed'))
    if set(active) != set(contract['joints']):
        raise ValueError('Joint grouping lost an active DOF')
    children = {}
    for j in joints.values():
        children.setdefault(j['parent'], []).append(j)
    q0 = {n: j['initial_position'] for n, j in contract['joints'].items()}
    frames = urdf_fk(copy.deepcopy(parsed), q0)
    if set(cooked) != set(frames):
        raise ValueError('Grouping requires native geometry for all links')
    points = np.concatenate([np.asarray(mesh['vertices']) @ frames[n][:3, :3].T + frames[n][:3, 3] for n, meshes in cooked.items() for mesh in meshes])
    root = np.eye(4)
    root[2, 3] = -float(points[:, 2].min())
    scopes = {n: plan_joint_motion(parsed, cooked, q0, root, n, contract['joints'][n]['lower'], contract['joints'][n]['upper']) for n in active}
    from urdf_core.articulation_groups import partition
    def compatible(a, b, kind):
        return (_scope_key(scopes[a]) == _scope_key(scopes[b]) if kind == 'slide'
                else source.same_ground_obstruction(scopes[a], scopes[b], frames, root))
    groups = partition(spec, source, frames, lambda n: _frozen(contract['joints'][n]), compatible)
    for group in groups:
        random_seed = int(hashlib.sha256(f"{seed}:{contract['spec_sha256']}:{group['id']}".encode()).hexdigest(), 16)
        group['representative'] = random.Random(random_seed).choice(group['members'])
        group['ground_scope'] = {n: {'status': scopes[n]['free_status'],
            'lower': scopes[n]['free_lower'], 'upper': scopes[n]['free_upper']} for n in group['members']}
    for group in groups:
        group['grouping_evidence'] = source.evidence(group['members'])
    return groups


def load_groups(root, contract=None):
    """Reproduce the frozen grouping from its source geometry and parameters."""
    auth_path = Path(root) / AUTH_FILE
    if not auth_path.is_file():
        return None
    auth = json.loads(auth_path.read_text())
    owner = {1: 'user', 2: 'object_pipeline.v1'}.get(auth.get('schema_version'))
    if (set(auth) != {'schema_version', 'policy', 'authorized_by', 'authorization_text', 'seed', 'sources'}
            or owner is None or auth['policy'] not in SUPPORTED_POLICIES or auth['authorized_by'] != owner
            or not isinstance(auth['authorization_text'], str) or not auth['authorization_text'].strip()
            or type(auth['seed']) is not int or set(auth['sources']) not in (SOURCE_ROLES, SOURCE_ROLES | {'articulation_groups'})):
        raise ValueError('Invalid shared joint group authorization')
    paths = {k: read_bound(v) for k, v in auth['sources'].items()}
    values = {k: json.loads(p.read_text()) for k, p in paths.items() if k != 'urdf'}
    original = values['contract']
    if contract is not None and ({n: _frozen(j) for n, j in contract['joints'].items()} !=
               {n: _frozen(j) for n, j in original['joints'].items()}):
        raise ValueError('Shared joint groups cannot change joint identities, types, initial states, ranges or caps')
    if 'articulation_groups' in values:
        from urdf_core.articulation_groups import validate
        early = validate(values['articulation_groups'], values['spec'])
        groups = copy.deepcopy(early['groups'])
        for group in groups:
            caps = {original['joints'][n]['max_effort'] for n in group['members']}
            if len(caps) != 1:
                raise ValueError('A frozen articulation group requires one shared effort cap')
    else:
        groups = classify(values['spec'], values['profile'], original, values['manifest'], values['cooked'],
                          read_urdf(paths['urdf']), auth['seed'], policy=auth['policy'])
    record = {'schema_version': 1, 'policy': auth['policy'], 'authorization': bind(auth_path),
              'sources': auth['sources'], 'seed': auth['seed'], 'groups': groups}
    path = Path(root) / GROUP_FILE
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError('Frozen shared group membership or random representative changed')
    return record


def freeze_runtime_groups(runtime):
    """Cook once before testing and freeze the default representative scope."""
    root = Path(runtime.args.contract).parent
    request = json.loads((root / 'grouping_request.json').read_text())
    if request.get('schema_version') != 'object_pipeline.v1' or type(request.get('seed')) is not int:
        raise ValueError('Invalid initial grouping request')
    runtime.create_trial(fixed=False)
    cooked_path = root / 'cooked_geometry.json'
    cooked = {name: [{key: value.tolist() if isinstance(value, np.ndarray) else value
                     for key, value in mesh.items()} for mesh in meshes]
              for name, meshes in runtime.cooked.items()}
    if cooked_path.exists():
        from .representative_motion import same_cooked_geometry
        if not same_cooked_geometry(json.loads(cooked_path.read_text()), cooked):
            raise ValueError('Initial grouping geometry changed')
    else:
        write_json(cooked_path, cooked)
    sources = {'spec': runtime.args.spec, 'profile': runtime.args.profile,
               'contract': runtime.args.contract, 'manifest': runtime.args.asset_dir / 'asset_manifest.json',
               'urdf': runtime.args.asset_dir / 'object.urdf', 'cooked': cooked_path}
    if request.get('articulation_groups') is not None:
        sources['articulation_groups'] = read_bound(request['articulation_groups'])
    auth = {'schema_version': 2, 'policy': POLICY, 'authorized_by': 'object_pipeline.v1',
            'authorization_text': 'Frozen command-line joint_test_scope=representative',
            'seed': request['seed'], 'sources': {key: bind(path) for key, path in sources.items()}}
    auth_path = root / AUTH_FILE
    if auth_path.exists() and json.loads(auth_path.read_text()) != auth:
        raise ValueError('Initial grouping inputs changed')
    if not auth_path.exists():
        write_json(auth_path, auth)
    groups = load_groups(root, runtime.contract)
    write_json(root / GROUP_FILE, groups)
    return bind(root / GROUP_FILE)


def group_paths(group):
    """Each list is one shared scalar; mass/COM/inertia are never included."""
    result = []
    for key in PASSIVE:
        result.append([f'profile.joints.{n}.{key}' for n in group['members']])
    for key in CONTROL:
        result.append([f'controller.{n}.{key}' for n in group['members']])
    for key in MATERIAL:
        result.append(list(dict.fromkeys(f'profile.links.{group["child_links"][n]}.material.{key}'
                                        for n in group['members'])))
    return result


def expand_patches(patches, groups):
    if groups is None:
        return patches
    aliases = {path: paths for group in groups['groups'] for paths in group_paths(group) for path in paths}
    output, explicit = {}, set()
    for patch in patches:
        if not isinstance(patch, dict) or set(patch) != {'path', 'value'}:
            raise ValueError('Each patch has exactly path/value')
        path, value = patch['path'], patch['value']
        if path in explicit:
            raise ValueError('Duplicate explicit group patch')
        explicit.add(path)
        for target in aliases.get(path, [path]):
            if target in output and output[target] != value:
                raise ValueError('Conflicting values for a shared joint group parameter')
            output[target] = value
    return [{'path': p, 'value': v} for p, v in output.items()]


def inconsistencies(groups, profile, controller):
    from .diagnosis import _path_locations
    locations = _path_locations(profile, controller)
    result = []
    for group in groups['groups']:
        if len(group['members']) < 2:
            continue
        for paths in group_paths(group):
            values = [locations[p][0][locations[p][1]] for p in paths]
            if any(v != values[0] for v in values[1:]):
                result.append({'group': group['id'], 'paths': paths, 'values': values})
    return result


def public_context(groups):
    """Code-owned grouping facts only; no authorization text/operator notes."""
    if groups is None:
        return None
    return {'policy': groups['policy'], 'groups': [
        {k: copy.deepcopy(g[k]) for k in ('id', 'members', 'representative', 'rules', 'child_links')}
        for g in groups['groups']], 'shared_joint_fields': list(PASSIVE), 'shared_controller_fields': list(CONTROL),
        'shared_child_material_fields': list(MATERIAL),
        'patch_semantics': 'Changing a shared field through ANY member expands atomically to ALL group members. '
                           'Use one value per group and field. Mass/density/COM/inertia are not shared. '
                           'All original phase permissions and frozen caps remain in force. '
                           'The fixed and free tests use the same saved random representative.'}


def initialize_from_representatives(*, groups, profile, initial_profile, controller, contract, spec):
    """User-authorized deterministic initialization; no model or operator values."""
    from .diagnosis import _path_locations, validate_parameter_bounds
    from .protocol import validate_profile
    updated, controls = copy.deepcopy(profile), copy.deepcopy(controller)
    locations = _path_locations(updated, controls)
    patches = []
    for group in groups['groups']:
        if len(group['members']) < 2:
            continue
        representative = group['representative']
        for paths in group_paths(group):
            source = next(p for p in paths if p.startswith('profile.joints.' + representative + '.')
                or p.startswith('controller.' + representative + '.')
                or p.startswith('profile.links.' + group['child_links'][representative] + '.material.'))
            data, key = locations[source]; value = data[key]
            if any(locations[p][0][locations[p][1]] != value for p in paths):
                for target in paths:
                    owner, field = locations[target]; owner[field] = value
                    patches.append({'path': target, 'value': value})
    updated = validate_profile(updated, spec)
    validate_parameter_bounds(updated, initial_profile, controls, contract)
    if inconsistencies(groups, updated, controls):
        raise ValueError('Representative copying did not synchronize shared parameters')
    proposal = None if not patches else {'kind': 'representative_group_initialization',
        'reason': 'User-authorized code rule: copy shared values from each saved random representative; '
                  'leave mass, density, COM, inertia, ranges, initial states and force caps unchanged.',
        'patches': patches}
    return {'profile': updated, 'controller': controls, 'proposal': proposal, 'budget_family': None}


def runtime_plan(spec, profile, contract, manifest, cooked, request):
    from .representative_motion import cooked_digest, same_cooked_geometry
    path = read_bound(request['joint_group_contract'])
    saved = json.loads(path.read_text()); groups = load_groups(path.parent, contract)
    if groups != saved:
        raise ValueError('Shared group contract is not independently reproducible')
    original_cooked = json.loads(read_bound(groups['sources']['cooked']).read_text())
    if (cooked_digest(original_cooked) != cooked_digest(cooked)
            and not same_cooked_geometry(original_cooked, cooked)):
        raise ValueError('Shared grouping geometry changed; cannot silently regroup')
    controls = {n: {k: j[k] for k in (*CONTROL, 'initial_position')} for n, j in contract['joints'].items()}
    if inconsistencies(groups, profile, controls):
        raise ValueError('Native grouped testing requires synchronized group parameters')
    selected = [g['representative'] for g in groups['groups']]
    free = selected if free_base_required(contract) else []
    coverage = {n: {'group': g['id'], **{phase: {'status':
        'not_applicable' if not names else 'individually_required' if n in names else 'covered_by_representatives',
        'representatives': [g['representative']] if names else []}
        for phase, names in [('fixed_joint', selected), ('free_joint', free)]}}
        for g in groups['groups'] for n in g['members']}
    return {'schema_version': 2, 'policy': groups['policy'], 'request_sha256': json_hash(request),
        'contract_sha256': json_hash(contract), 'profile_sha256': json_hash(profile),
        'manifest_sha256': json_hash(manifest), 'cooked_geometry_sha256': cooked_digest(cooked),
        'groups': groups['groups'], 'selected': {'fixed_joint': sorted(selected), 'free_joint': sorted(free)},
        'coverage': coverage, 'all_joint_ids': sorted(contract['joints']),
        'observed_and_controlled_joint_ids': sorted(contract['joints']),
        'exhaustive_individual_motion_claimed': len(selected) == len(contract['joints'])}
