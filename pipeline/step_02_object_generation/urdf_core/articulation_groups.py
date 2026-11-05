"""Structural joint groups shared by preview, motion review and physics."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from .run_state import file_hash, fingerprint, write_json
from .urdf_validation import read_urdf, urdf_fk

POLICY = 'articulation_groups.v1'
FILE = 'articulation_groups.json'


def partition(spec, source, frames, frozen, compatible):
    """Independent slides or geometrically matching independent subtrees."""
    from physics_feedback.joint_group_geometry import coordinates_close, rotation_close
    from scipy.spatial.transform import Rotation
    joints = {j['name']: j for j in spec['joints']}
    active = sorted(n for n, j in joints.items() if j['type'] != 'fixed')
    children = {}
    for j in joints.values():
        children.setdefault(j['parent'], []).append(j)
    owner, reasons = {n: n for n in active}, {n: set() for n in active}

    def head(n):
        while owner[n] != n:
            n = owner[n]
        return n

    def merge(a, b, reason):
        aa, bb = head(a), head(b)
        if aa != bb:
            first, second = sorted((aa, bb))
            owner[second] = first
            reasons[first] |= reasons[second]
        reasons[head(a)].add(reason)

    def descendant_active(link):
        return any(j['type'] != 'fixed' or descendant_active(j['child'])
                   for j in children.get(link, []))

    leaves = [n for n in active if joints[n]['type'] == 'prismatic'
              and not descendant_active(joints[n]['child']) and not joints[n].get('mimic')]
    for i, a in enumerate(leaves):
        ja = joints[a]
        axis_a = Rotation.from_euler('xyz', ja['origin']['rpy']).apply(ja['axis'])
        for b in leaves[i + 1:]:
            jb = joints[b]
            axis_b = Rotation.from_euler('xyz', jb['origin']['rpy']).apply(jb['axis'])
            if (ja['parent'] == jb['parent']
                    and np.allclose(axis_a, axis_b, rtol=1e-10, atol=1e-10)
                    and frozen(a) == frozen(b) and compatible(a, b, 'slide')):
                merge(a, b, 'independent_prismatic_leaves')

    def subtree(link):
        return {link} | {x for j in children.get(link, []) for x in subtree(j['child'])}

    matches = {}

    def match(a, b, top=False):
        key = (a['name'], b['name'], top)
        if key not in matches:
            matches[key] = match_uncached(a, b, top)
        return matches[key]

    def match_uncached(a, b, top=False):
        if a['type'] != b['type'] or a.get('mimic') or b.get('mimic'):
            return None
        if not coordinates_close(a.get('axis', [0, 0, 0]), b.get('axis', [0, 0, 0])):
            return None
        if not top and (not coordinates_close(a['origin']['xyz'], b['origin']['xyz'])
                or not rotation_close(Rotation.from_euler('xyz', a['origin']['rpy']).as_matrix(),
                                      Rotation.from_euler('xyz', b['origin']['rpy']).as_matrix())):
            return None
        if top:
            rotation = frames[b['child']][:3, :3] @ frames[a['child']][:3, :3].T
            if not coordinates_close(rotation @ [0., 0., 1.], [0., 0., 1.]):
                return None
        if a['type'] != 'fixed' and frozen(a['name']) != frozen(b['name']):
            return None
        if not source.equivalent(a['child'], b['child']):
            return None
        aa = sorted(children.get(a['child'], []), key=lambda j: j['name'])
        bb = sorted(children.get(b['child'], []), key=lambda j: j['name'])
        if len(aa) != len(bb):
            return None
        mapping = [] if a['type'] == 'fixed' else [(a['name'], b['name'])]
        unused = set(range(len(bb)))
        for child in aa:
            options = [(i, match(child, bb[i])) for i in sorted(unused)]
            options = [(i, result) for i, result in options if result is not None]
            if len(options) != 1:
                return None
            i, result = options[0]
            unused.remove(i)
            mapping += result
        return mapping

    for siblings in children.values():
        clusters = []
        for branch in sorted(siblings, key=lambda j: j['name']):
            cluster = next((g for g in clusters if all(
                not subtree(branch['child']) & subtree(other['child'])
                and match(other, branch, top=True) is not None for other in g)), None)
            if cluster is None:
                clusters.append([branch])
                continue
            for a, b in match(cluster[0], branch, top=True):
                roles = [dict(match(other, branch, top=True)) for other in cluster]
                peers = [next(n for n, target in mapping.items() if target == b) for mapping in roles]
                if all(compatible(n, b, 'subtree') for n in peers):
                    merge(a, b, 'repeated_independent_components')
            cluster.append(branch)
    members = {}
    for n in active:
        members.setdefault(head(n), []).append(n)
    return [{'id': 'group_' + fingerprint(ns)[:12], 'members': ns,
             'rules': sorted(reasons[head(ns[0])]) if len(ns) > 1 else ['individual'],
             'child_links': {n: joints[n]['child'] for n in ns}}
            for ns in sorted(members.values(), key=lambda ns: ns[0])]


def representative(members, seed):
    number = int(hashlib.sha256(f'{seed}:{fingerprint(sorted(members))}'.encode()).hexdigest(), 16)
    return random.Random(number).choice(sorted(members))


def validate(plan, spec):
    if (not isinstance(plan, dict) or plan.get('policy') != POLICY
            or plan.get('spec_fingerprint') != fingerprint(spec) or type(plan.get('seed')) is not int):
        raise ValueError('Articulation groups do not bind the current specification')
    active = {j['name']: j for j in spec['joints'] if j['type'] != 'fixed'}
    seen = []
    for group in plan['groups']:
        members = group['members']
        if (not members or members != sorted(set(members)) or not set(members) <= set(active)
                or group['id'] != 'group_' + fingerprint(members)[:12]
                or group['representative'] != representative(members, plan['seed'])
                or group['child_links'] != {n: active[n]['child'] for n in members}):
            raise ValueError('Invalid articulation group membership or representative')
        seen.extend(members)
    if len(seen) != len(active) or set(seen) != set(active):
        raise ValueError('Articulation groups must cover every active joint exactly once')
    return plan


def freeze(spec_path, urdf_path, target, *, seed=0, scope='representative'):
    from physics_feedback.joint_group_geometry import SourceGeometry
    spec_path, urdf_path, target = map(Path, (spec_path, urdf_path, target))
    spec = json.loads(spec_path.read_text())
    parsed = read_urdf(urdf_path)
    artifacts = {str(p.relative_to(urdf_path.parent)): file_hash(p)
                 for p in sorted(urdf_path.parent.rglob('*')) if p.is_file()}
    source = SourceGeometry(parsed, {'artifacts': artifacts})
    by_name = {j['name']: j for j in spec['joints']}
    def frozen(n):
        joint = by_name[n]
        return {'type': joint['type'], 'limit': joint.get('limit'), 'initial_position': 0.}
    if scope not in {'all', 'representative'}:
        raise ValueError('Unknown articulation group scope')
    groups = partition(spec, source, urdf_fk(parsed, {}), frozen,
                       lambda a, b, kind: scope == 'representative')
    for group in groups:
        group['representative'] = representative(group['members'], seed)
    plan = {'policy': POLICY, 'spec_fingerprint': fingerprint(spec), 'seed': seed, 'scope': scope,
            'groups': groups, 'source': {'spec': {'path': str(spec_path.resolve()), 'sha256': file_hash(spec_path)},
             'urdf': {'path': str(urdf_path.resolve()), 'sha256': file_hash(urdf_path)}, 'artifacts': artifacts},
            'coverage': 'Representatives are rendered/reviewed; all joints retain structural and numeric validation.'}
    validate(plan, spec)
    if target.exists() and json.loads(target.read_text()) != plan:
        raise ValueError('Frozen articulation groups changed; use a new evaluation')
    write_json(target, plan)
    return plan


def load(path, spec):
    return validate(json.loads(Path(path).read_text()), spec)


def effort_budgets(proposal, plan):
    """Expand one model-authored force/torque ceiling per frozen group."""
    import copy
    budgets = proposal.get('group_effort_budgets')
    expected = {g['id'] for g in plan['groups']}
    if not isinstance(budgets, dict) or set(budgets) != expected:
        raise ValueError('group_effort_budgets must cover every articulation group exactly once')
    expanded = {name: copy.deepcopy(budgets[g['id']])
                for g in plan['groups'] for name in g['members']}
    if 'effort_budgets' in proposal and proposal['effort_budgets'] != expanded:
        raise ValueError('Per-joint effort budgets differ from their group ceiling')
    return expanded


def targets(spec, plan=None):
    if plan is not None:
        validate(plan, spec)
        names = {g['representative'] for g in plan['groups']}
        return [j for j in spec['joints'] if j['name'] in names]
    return [j for j in spec['joints'] if j['type'] != 'fixed']


def evidence_plan(manifest, spec, remember):
    binding = manifest.get('articulation_groups')
    if binding is None:
        return None
    path = remember(binding['path'], binding['sha256'])
    return load(path, spec)
