"""Explicitly authorized representative motion, with replayable all-DOF coverage.

This is sampling, not proof of identical dynamics at every installation. No
name/category recognition; unsupported mechanisms remain individual tests.
"""
from __future__ import annotations
import copy
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from .protocol import json_hash, free_base_required
from urdf_core.run_state import file_hash, write_json

INPUT_FILE = "representative_motion_input.json"
# Geometry equality allows float32 import noise, not dimensional simplification.
GEOMETRY_EPS_M = 1e-6
# Small label-dependent COM/inertia differences may be sampled, never discarded
# from native dynamics. These explicit equivalence tolerances are not pass gates.


def prepare_group_input(path, group_root, contract, candidate_id):
    from . import joint_groups
    groups = joint_groups.load_groups(group_root, contract)
    if groups is None:
        raise ValueError('Representative scope requires frozen groups')
    value = {'schema_version': 2, 'policy': groups['policy'],
             'contract_sha256': json_hash(contract), 'candidate_id': candidate_id,
             'joint_group_contract': joint_groups.bind(Path(group_root) / joint_groups.GROUP_FILE)}
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Representative scope changed on resume')
    if not path.exists():
        write_json(path, value)
    return path


def selected_joints(contract, request_path=None):
    """The supervisor and workers share the same frozen selected joint set."""
    names = list(contract['joints'])
    if request_path is not None:
        from . import joint_groups
        request = json.loads(Path(request_path).read_text())
        verify_input(request, contract)
        if request.get('policy') not in joint_groups.SUPPORTED_POLICIES:
            raise ValueError('New parallel execution requires frozen joint groups')
        path = joint_groups.read_bound(request['joint_group_contract'])
        groups = joint_groups.load_groups(path.parent, contract)
        names = sorted(group['representative'] for group in groups['groups'])
    return {'fixed_joint': names, 'free_joint': names if free_base_required(contract) else []}


def runtime_selection(runtime, report, contract):
    path = getattr(getattr(runtime, 'args', None), 'representative_motion', None)
    if path is None:
        return selected_joints(contract)
    if 'representative_motion' not in report:
        request = json.loads(Path(path).read_text())
        verify_input(request, contract)
        from .trial_checkpoint import for_runtime
        cache = for_runtime(runtime)
        if cache is not None:
            cache.restore_delivery(runtime)
        runtime.create_trial(fixed=False)
        report['representative_motion'] = write_runtime_plan(runtime, request)
    return report['representative_motion']['plan']['selected']




def verify_input(value, contract):
    from . import joint_groups
    if value.get('policy') in joint_groups.SUPPORTED_POLICIES:
        if (set(value) != {'schema_version', 'policy', 'contract_sha256', 'candidate_id', 'joint_group_contract'}
                or value['schema_version'] != 2 or value['contract_sha256'] != json_hash(contract)):
            raise ValueError('Invalid shared group input binding')
        path = joint_groups.read_bound(value['joint_group_contract'])
        if joint_groups.load_groups(path.parent, contract) != json.loads(path.read_text()):
            raise ValueError('Unreproducible shared grouping input')
        return set()  # History never changes the saved random representative.
    raise ValueError('Unsupported representative policy; current shared groups are required')


def cooked_digest(cooked):
    digest = hashlib.sha256()
    for name, meshes in sorted(cooked.items()):
        digest.update(name.encode())
        for mesh in meshes:
            digest.update(np.asarray(mesh['vertices'], dtype=np.float64).tobytes())
            digest.update(np.asarray(mesh['faces'], dtype=np.int64).tobytes())
    return digest.hexdigest()


def same_cooked_geometry(left, right):
    """Same native topology/points, allowing only float64 frame roundoff.

    This compares repeated observations of one collider, not visual fidelity.
    Keep exact component ordering and triangle indices; do not merge hulls or
    accept a geometrically close but independently simplified proxy.
    """
    if set(left) != set(right):
        return False
    for name, meshes in left.items():
        if len(meshes) != len(right[name]):
            return False
        for a, b in zip(meshes, right[name]):
            x, y = np.asarray(a['vertices'], dtype=float), np.asarray(b['vertices'], dtype=float)
            if (x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or not len(x)
                    or not np.isfinite(x).all() or not np.isfinite(y).all()
                    or not np.array_equal(a['faces'], b['faces'])):
                return False
            scale = max(1., float(np.abs(x).max()), float(np.abs(y).max()))
            if not np.allclose(x, y, rtol=0., atol=64 * np.finfo(np.float64).eps * scale):
                return False
    return True


def _clouds(cooked, name):
    clouds = []
    for mesh in cooked.get(name, []):
        v = np.asarray(mesh['vertices'], dtype=float)
        if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
            raise ValueError('Missing/invalid cooked collider vertices')
        # A native convex polytope is uniquely determined by its vertex set.
        clouds.append(v)
    return clouds


def _shape_equal(a, b):
    if len(a) != len(b):
        return False
    from scipy.optimize import linear_sum_assignment
    costs = np.full((len(a), len(b)), 1e6)
    for i, u in enumerate(a):
        for j, v in enumerate(b):
            if u.shape != v.shape:
                continue
            d = np.linalg.norm(u[:, None] - v[None, :], axis=2)
            rows, cols = linear_sum_assignment(d)
            costs[i, j] = float(d[rows, cols].max())
    rows, cols = linear_sum_assignment(costs)
    return bool(np.all(costs[rows, cols] <= GEOMETRY_EPS_M))






def build_plan(spec, profile, contract, manifest, cooked, parsed, request):
    from . import joint_groups
    verify_input(request, contract)
    return joint_groups.runtime_plan(spec, profile, contract, manifest, cooked, request)


def write_runtime_plan(runtime, request):
    from .verify_physics_isaac import sha256
    cooked = {n:[{k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in mesh.items()}
                 for mesh in meshes] for n,meshes in runtime.cooked.items()}
    path = runtime.args.out_dir / 'representative_cooked_geometry.json'
    write_json(path, cooked)
    plan = build_plan(runtime.spec, runtime.profile, runtime.contract, runtime.manifest,
                      cooked, runtime.parsed, request)
    return {'plan': plan, 'geometry': {'path': str(path), 'sha256': sha256(path)}}


def verify_report_plan(report, contract, asset_dir):
    record = report.get('representative_motion')
    binding = report.get('inputs', {}).get('representative_motion')
    if record is None and binding is None:
        return None, set()
    if record is None or binding is None or file_hash(binding['path']) != binding['sha256']:
        raise ValueError('Unbound representative motion coverage')
    request = json.loads(Path(binding['path']).read_text())
    verify_input(request, contract)
    geometry = record['geometry']; path = Path(geometry['path'])
    if file_hash(path) != geometry['sha256']:
        raise ValueError('Representative native geometry changed')
    cooked = json.loads(path.read_text())
    from urdf_core.urdf_validation import read_urdf
    asset = Path(asset_dir)
    plan = build_plan(json.loads((asset/'kinematics.json').read_text()),
        json.loads((asset/'physics_profile.json').read_text()), contract,
        json.loads((asset/'asset_manifest.json').read_text()), cooked,
        read_urdf(asset/'object.urdf'), request)
    if plan != record['plan']:
        raise ValueError('Representative proof or all-DOF coverage does not reproduce')
    # All native trials must have used the same cooked collisions as the proof.
    for trial in [*report['trials'], *report.get('inapplicable_ground_operations', [])]:
        if trial['setup'].get('cooked_geometry_sha256') != plan['cooked_geometry_sha256']:
            native = trial['setup'].get('motion_scope_geometry')
            if not native or file_hash(native['path']) != native['sha256']:
                raise ValueError('Representative proof differs from native trial geometry')
            observed = json.loads(Path(native['path']).read_text())
            if (cooked_digest(observed) != trial['setup']['cooked_geometry_sha256']
                    or not same_cooked_geometry(cooked, observed)):
                raise ValueError('Representative proof differs from native trial geometry')
    files = {path.resolve(), Path(binding['path']).resolve()}
    from . import joint_groups
    gp = joint_groups.read_bound(request['joint_group_contract'])
    record = json.loads(gp.read_text())
    files.update({gp.resolve(), Path(record['authorization']['path']).resolve()})
    files.update(Path(r['path']).resolve() for r in record['sources'].values())
    return plan, files
