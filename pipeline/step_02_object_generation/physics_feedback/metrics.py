"""Hard numerical acceptance of complete physics traces, independent of API review."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_CONFIG = {
    "physics_dt_s": 1 / 240,
    "settle_duration_s": 8.0,
    "settle_window_s": 1.0,
    "orientation_tolerance_deg": 1.0,
    "tilt_limit_deg": 10.0,
    "coverage_min": 0.98,
    "endpoint_tolerance_fraction": 0.02,
    "endpoint_hold_s": 0.5,
}


def _joint_mapping(joints):
    if joints is None:
        return {}
    if isinstance(joints, list):
        return {joint["name"]: joint for joint in joints if joint.get("type") != "fixed"}
    return {name: joint for name, joint in joints.items() if joint.get("type") != "fixed"}


def _rotation(wxyz):
    q = np.asarray(wxyz, dtype=float)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise ValueError("root quaternion data are missing/nonfinite")
    if np.any(np.abs(np.linalg.norm(q, axis=1) - 1) > 1e-3):
        raise ValueError("root quaternions must be normalized")
    return Rotation.from_quat(q[:, [1, 2, 3, 0]])


def _ground_contact(contact):
    if contact.get("ground_contact") is True:
        return True
    for pair in contact.get("pairs", []):
        if isinstance(pair, dict):
            values = [pair.get("body0", ""), pair.get("body1", ""), *pair.get("links", [])]
        else:
            values = pair if isinstance(pair, (list, tuple)) else [pair]
        if any("ground" in str(value).lower() for value in values):
            return True
    return False


def _holding_intervals(times, values, target, tolerance, after):
    """Observed contiguous in-tolerance intervals, without interpolating missing samples."""
    selected = (times >= after - 1e-8) & (np.abs(values - target) <= tolerance + 1e-8)
    intervals, start = [], None
    for i, inside in enumerate(selected):
        if inside and start is None:
            start = i
        if start is not None and (not inside or i == len(selected) - 1):
            end = i if inside else i - 1
            intervals.append((float(times[start]), float(times[end])))
            start = None
    return intervals


def _passive_velocity_diagnostic_only(contract):
    """A versioned policy, never a per-object threshold or silent old-run change."""
    return (contract.get("acceptance_policy") in {'object_functional_physics.v11'}
            and contract.get("passive_tail_velocity_policy") == "diagnostic_only")


def _upright_required(contract):
    """Only a new, explicitly classified contract may remove the tilt gate."""
    if contract.get("acceptance_policy") not in {'object_functional_physics.v11'}:
        return True
    from .protocol import validate_orientation_policy
    return validate_orientation_policy(contract.get("orientation_policy"))["upright_required"]


def _passive_tail(trace, times, settings, contract, joint_positions, joint_velocities, errors, metrics):
    """Full-window pose gates; native speed is diagnostic under explicit v6/v7."""
    window = float(settings["zero_effort_tail_s"])
    tail_mask = times >= times[-1] - window - 1e-8
    selected = [step for step, keep in zip(trace, tail_mask) if keep]
    pos_tol = float(settings["position_tolerance_m"])
    angle_tol = float(settings["orientation_tolerance_deg"])
    linear_limit, angular_limit = pos_tol / window, math.radians(angle_tol) / window
    velocity_gate = not _passive_velocity_diagnostic_only(contract)
    required = set(contract.get("required_links", []))
    values = {}
    if (not required or any(set(step.get("link_transforms", {})) != required
                           or set(step.get("link_velocities", {})) != required for step in selected)):
        errors.append("passive tail requires every rigid link pose and native velocity")
    else:
        for name in sorted(required):
            poses = np.asarray([step["link_transforms"][name] for step in selected], dtype=float)
            linear = np.asarray([step["link_velocities"][name].get("linear_m_s") for step in selected], dtype=float)
            angular = np.asarray([step["link_velocities"][name].get("angular_rad_s") for step in selected], dtype=float)
            if (poses.shape != (len(selected), 4, 4) or linear.shape != (len(selected), 3)
                    or angular.shape != (len(selected), 3) or not all(np.isfinite(a).all() for a in (poses, linear, angular))):
                errors.append("missing/nonfinite passive tail rigid-link state: " + name)
                continue
            movement = float(np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1).max())
            rotations = Rotation.from_matrix(poses[:, :3, :3])
            angle = float(np.degrees((rotations[0].inv() * rotations).magnitude()).max())
            peak_linear = float(np.linalg.norm(linear, axis=1).max())
            peak_angular = float(np.linalg.norm(angular, axis=1).max())
            values[name] = {"position_span_m": movement, "orientation_span_deg": angle,
                           "peak_native_linear_velocity_m_s": peak_linear,
                           "peak_native_angular_velocity_rad_s": peak_angular}
            if movement > pos_tol + 1e-8 or angle > angle_tol + 1e-6:
                errors.append("passive tail rigid link did not remain stationary: " + name)
            if velocity_gate and (peak_linear > linear_limit + 1e-8 or peak_angular > angular_limit + 1e-8):
                errors.append("passive tail rigid link retains native velocity: " + name)
    metrics["passive_tail"] = {"window_s": window, "links": values,
                               "linear_velocity_limit_m_s": linear_limit,
                               "angular_velocity_limit_rad_s": angular_limit}
    if not velocity_gate:
        metrics["passive_tail"]["native_velocity_acceptance_gate"] = False
        metrics["passive_tail"]["native_velocity_policy"] = "diagnostic_only"
    for name, velocity in joint_velocities.items():
        limit = linear_limit if contract["joints"][name]["type"] == "prismatic" else angular_limit
        if velocity_gate and np.max(np.abs(velocity[tail_mask])) > limit + 1e-8:
            errors.append("passive tail joint retains native velocity: " + name)
        if name in joint_positions:
            span = float(np.ptp(joint_positions[name][tail_mask]))
            tolerance = pos_tol if contract["joints"][name]["type"] == "prismatic" else math.radians(angle_tol)
            if span > tolerance + 1e-8:
                errors.append("passive tail joint did not remain stationary: " + name)


def _initialization_evidence(sample, contract, errors, metrics):
    """Check recorded native t=0 state, independently of later natural settling."""
    from .support_rest_pose import joint_initial_tolerance
    expected = sample.get("initialization", {})
    required = {"expected_root_position", "expected_root_quaternion_wxyz", "expected_joint_positions",
                "mode", "ground_alignment_required", "tilt_reference_quaternion_wxyz"}
    modes = {"cooked_ground", "support_rest_pose"}
    if contract.get("motion_scope_policy"):
        modes.add("fixed_motion_fixture")
    if not required <= set(expected) or expected.get("mode") not in modes:
        errors.append("missing or invalid native initialization evidence")
        return
    numeric_position = max(1e-6, float(contract["diagonal_m"]) * 1e-6)
    try:
        target = np.asarray(expected["expected_root_position"], dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("invalid initialization root reference")
        delta = float(np.linalg.norm(np.asarray(sample["root_position"]) - target))
        angular = float((_rotation([expected["expected_root_quaternion_wxyz"]]).inv()
                         * _rotation([sample["root_quaternion_wxyz"]])).magnitude()[0])
        metrics["initialization"] = {"mode": expected["mode"], "root_position_error_m": delta,
                                     "root_orientation_error_rad": angular,
                                     "ground_min_z_m": sample.get("ground_min_z_m")}
        if delta > numeric_position or angular > 1e-5:
            errors.append("native initialization root pose differs from its requested pose")
        names = set(contract["joints"])
        if set(expected["expected_joint_positions"]) != names:
            raise ValueError("initialization joint reference coverage differs")
        for name, joint in contract["joints"].items():
            value = float(expected["expected_joint_positions"][name])
            if not math.isfinite(value) or abs(value - joint["initial_position"]) > joint_initial_tolerance(joint, contract) + 1e-8:
                raise ValueError("initialization q reference differs from frozen q_initial tolerance")
            if expected["mode"] in {"cooked_ground", "fixed_motion_fixture"} and abs(value - joint["initial_position"]) > 1e-10:
                raise ValueError("fresh trial must reset the frozen q_initial")
            if abs(float(sample["joint_positions"][name]) - value) > 1e-6:
                errors.append("native initialization joint differs from requested q: " + name)
            if abs(float(sample["joint_velocities"][name])) > 1e-6 or abs(float(sample["joint_efforts"][name])) > 1e-8:
                errors.append("native initialization must have zero joint velocity and active effort: " + name)
        references, poses = sample.get("initial_link_transforms", {}), sample.get("link_transforms", {})
        if set(references) != set(contract["required_links"]) or set(poses) != set(references):
            raise ValueError("initialization lacks every link initial pose reference")
        for name, reference in references.items():
            ref, actual = np.asarray(reference, dtype=float), np.asarray(poses[name], dtype=float)
            if ref.shape != (4, 4) or actual.shape != (4, 4) or not np.isfinite(ref).all() or not np.isfinite(actual).all():
                raise ValueError("invalid initialization link pose")
            angle = float(Rotation.from_matrix(ref[:3, :3].T @ actual[:3, :3]).magnitude())
            if np.linalg.norm(ref[:3, 3] - actual[:3, 3]) > numeric_position or angle > 1e-5:
                errors.append("native initialization differs from the requested link initial pose: " + name)
            native = sample["link_velocities"][name]
            velocities = [np.asarray(native[key], dtype=float) for key in ("linear_m_s", "angular_rad_s")]
            if any(value.shape != (3,) or not np.isfinite(value).all() for value in velocities):
                raise ValueError("invalid initialization native link velocity")
            if any(np.linalg.norm(value) > 1e-6 for value in velocities):
                errors.append("native initialization rigid link velocity is not zero: " + name)
        alignment = expected["ground_alignment_required"]
        if type(alignment) is not bool or alignment != (expected["mode"] == "cooked_ground"):
            raise ValueError("invalid initialization alignment policy")
        bottom = float(sample["ground_min_z_m"])
        if not math.isfinite(bottom) or (alignment and abs(bottom) > numeric_position):
            errors.append("native initialization is not aligned to its actual cooked ground support")
        _rotation([expected["tilt_reference_quaternion_wxyz"]])
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        errors.append("invalid native initialization evidence: " + str(exc))


def _pre_motion_reference(trace, times, position, rotations, contract, errors, metrics, *, kind):
    """A motion starts only after the full recorded, settled warmup window."""
    from .support_rest_pose import joint_initial_tolerance
    start = float(contract["config"]["settle_duration_s"])
    indices = np.flatnonzero(times >= start - 1e-8)
    if not len(indices) or abs(float(times[indices[0]]) - start) > float(contract["config"]["physics_dt_s"]) * 1.01:
        errors.append("missing measured pre-motion reference at operation start")
        return None
    index = int(indices[0])
    before = trace[:index + 1]
    q = {name: np.asarray([s["joint_positions"][name] for s in before], dtype=float) for name in contract["joints"]}
    qd = {name: np.asarray([s["joint_velocities"][name] for s in before], dtype=float) for name in contract["joints"]}
    local_errors, local_metrics = [], {}
    _passive_tail(before, times[:index + 1], contract["config"], contract, q, qd, local_errors, local_metrics)
    from .motion_acceptance import enabled, hold_tolerance
    for name, value in q.items():
        tolerance = hold_tolerance(contract['joints'][name], contract) if enabled(contract, kind) else joint_initial_tolerance(contract['joints'][name], contract)
        if np.max(np.abs(value - contract["joints"][name]["initial_position"])) > tolerance + 1e-8:
            local_errors.append("pre-motion joint did not hold frozen q_initial: " + name)
    # A world-fixed root is supported by its fixture, and PhysX need not emit
    # a ground contact for it. Only free motion needs measured ground support.
    if kind == "free_joint" and contract["stable_required"]:
        tail = [s for s in before if s["time_s"] >= start - float(contract["config"]["settle_window_s"]) - 1e-8]
        if not any(_ground_contact(s.get("contacts", {})) for s in tail):
            local_errors.append("pre-motion stable window lacks observed ground support contact")
    metrics["pre_motion"] = {"passed": not local_errors, "errors": local_errors,
                              "stability": local_metrics.get("passive_tail"),
                              "reference": {"time_s": float(times[index]), "root_position": position[index].tolist(),
                                            "root_quaternion_wxyz": trace[index]["root_quaternion_wxyz"]}}
    errors.extend("pre-motion: " + value for value in local_errors)
    return index


def planned_free_operation_scope(contract, plan):
    """Resolve a proven interval; do not certify motion below observed precision."""
    from .motion_scope import validate_motion_plan
    from .control import scoped_motion_joint
    validate_motion_plan(plan)
    if plan["free_status"] != "operable":
        raise ValueError("A ground-blocked joint cannot claim a motion trace")
    operation = {"joint_name": plan["joint_name"], "lower": plan["free_lower"], "upper": plan["free_upper"]}
    scoped_motion_joint(contract, plan["joint_name"], operation)
    # Native initialization already uses 1e-6 SI joint-coordinate consistency.
    # A requested endpoint tolerance below this cannot establish the claimed
    # motion. Report an observation limitation, not successful tiny travel or
    # a geometric reason to omit a free trial.
    if (operation["upper"] - operation["lower"]) * contract["config"]["endpoint_tolerance_fraction"] < 1e-6:
        raise ValueError("Ground operation interval is below native measurement resolution")
    return operation


def _motion_scope(sample, contract, kind, joint_name):
    """Recompute the geometry-derived operation; never change physical limits.

    Native cooking is the measurement source. The receipt contains the analytic
    height curves so an independent reader can reproduce the selected interval.
    Publication additionally binds the cooked geometry and source URDF hashes.
    """
    if not contract.get("motion_scope_policy") or kind not in {"fixed_joint", "free_joint"}:
        if sample.get("initialization", {}).get("mode") == "fixed_motion_fixture":
            raise ValueError("Fixed motion fixture is only valid for an opted-in fixed trial")
        return None
    from .motion_scope import validate_motion_plan
    from .control import scoped_motion_joint
    plan = validate_motion_plan(sample.get("motion_scope_plan"))
    joint = contract["joints"][joint_name]
    for key, expected in (("joint_name", joint_name), ("full_lower", joint["lower"]),
                          ("full_upper", joint["upper"]), ("initial_position", joint["initial_position"])):
        if plan.get(key) != expected:
            raise ValueError("Motion scope changed a frozen joint field: " + key)
    binding = sample.get("motion_scope_binding", {})
    for key in ("urdf_sha256", "cooked_geometry_sha256"):
        digest = binding.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Motion scope lacks its native source binding")
    expected_position = np.asarray(plan["root_transform"], dtype=float)[:3, 3].copy()
    mode = sample.get("initialization", {}).get("mode")
    if kind == "fixed_joint":
        clearance = max(.005, .01 * contract["diagonal_m"])
        if binding.get("fixture_clearance_m") != clearance or mode != "fixed_motion_fixture":
            raise ValueError("Fixed fixture clearance differs from its deterministic policy")
        expected_position[2] += max(0., clearance - plan["fixed_min_z_m"])
        operation = None
    else:
        if binding.get("fixture_clearance_m") is not None or mode not in {"cooked_ground", "support_rest_pose"}:
            raise ValueError("Free ground motion cannot use a fixed or elevated fixture")
        operation = planned_free_operation_scope(contract, plan)
    actual = np.asarray(sample["initialization"]["expected_root_position"], dtype=float)
    if not np.allclose(actual, expected_position, rtol=0, atol=1e-10):
        raise ValueError("Motion fixture placement differs from the frozen geometry plan")
    expected_rotation = np.asarray(plan["root_transform"], dtype=float)[:3, :3]
    actual_rotation = _rotation([sample['initialization']['expected_root_quaternion_wxyz']]).as_matrix()[0]
    if not np.allclose(actual_rotation, expected_rotation, rtol=0, atol=1e-10):
        raise ValueError('Motion scope orientation differs from its initialization')
    return operation


def evaluate_trial(trace, *, kind, scene_diagonal=1.0, joint=None, joints=None, effort_caps=None, config=None, contract=None, tail_seconds=None, max_duration=None, joint_name=None, **_):
    """Reject incomplete traces; free-root translation is reported, never a tilt proxy.

    Step fields: time_s, root_position, root_quaternion_wxyz, joint_positions,
    joint_efforts, geometry and contacts. ``geometry.passed`` includes source and
    cooked-collider checks when the worker uses two checkers.
    """
    from .motion_acceptance import enabled as motion_v9, hold_tolerance, StructurePlan
    relaxed = motion_v9(contract, kind)
    from .protocol import validate_current_contract
    validate_current_contract(contract)
    modern = True
    natural = True
    settings = dict(DEFAULT_CONFIG)
    if contract:
        settings.update(contract.get('config', {}))
        joints = joints if joints is not None else contract.get('joints', {})
        scene_diagonal = contract.get('diagonal_m', contract.get('scene_diagonal', contract.get('scene_diagonal_m', scene_diagonal)))
    settings.update(config or {})
    if tail_seconds is not None:
        settings['settle_window_s'] = tail_seconds
    if max_duration is not None:
        settings['settle_duration_s'] = max_duration
    mapping = _joint_mapping(joints)
    if joint is not None:
        joint_name = joint_name or joint.get('name')
        if joint_name:
            mapping[joint_name] = joint
    result = {'kind': kind, 'passed': False, 'status': 'failed', 'errors': [], 'metrics': {}}
    errors, metrics = (result['errors'], result['metrics'])
    try:
        from .protocol import validate_current_contract
        validate_current_contract(contract)
        if kind not in {'fall', 'ground_settle', 'fixed_joint', 'free_joint', 'reload_supported'}:
            raise ValueError(f'unknown trial kind: {kind}')
        if not isinstance(trace, list) or len(trace) < 2:
            raise ValueError('trace must include at least two physics steps')
        times = np.asarray([step['time_s'] for step in trace], dtype=float)
        position = np.asarray([step['root_position'] for step in trace], dtype=float)
        rotations = _rotation([step['root_quaternion_wxyz'] for step in trace])
        if times.shape != (len(trace),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or (position.shape != (len(trace), 3)) or (not np.isfinite(position).all()):
            raise ValueError('root pose/time is invalid or timestamps are not strictly increasing')
        if abs(times[0]) > 1e-08:
            errors.append('schema 2 trace must include the initialization pose at time zero')
        dt = float(settings['physics_dt_s'])
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('physics_dt_s must be positive and finite')
        for name in ('settle_duration_s', 'settle_window_s', 'orientation_tolerance_deg', 'tilt_limit_deg', 'coverage_min', 'endpoint_tolerance_fraction'):
            if not math.isfinite(float(settings[name])) or float(settings[name]) <= 0:
                raise ValueError(f'invalid frozen threshold: {name}')
        if not math.isfinite(float(scene_diagonal)) or float(scene_diagonal) <= 0:
            raise ValueError('scene diagonal must be positive and finite')
        if np.any(np.diff(times) > dt * 1.05 + 1e-08):
            errors.append('trace omits physics steps')
        duration = float(times[-1] - times[0])
        metrics['duration_s'] = duration
        metrics['physics_steps'] = len(trace)
        _initialization_evidence(trace[0], contract, errors, metrics)
        if relaxed:
            plan = StructurePlan(contract)
            poses = np.asarray([[step['link_transforms'][n] for n in plan.names] for step in trace])
            q_structure = np.asarray([[step['joint_positions'].get(j['name'], 0.0) for j in plan.joints] for step in trace])
            distances, angles = plan.arrays(poses, q_structure)
            metrics['structural_joints'] = {j['name']: {'max_translation_error_m': float(distances[:, i].max()), 'translation_tolerance_m': float(plan.tolerances[i]), 'max_rotation_error_deg': float(angles[:, i].max()), 'rotation_tolerance_deg': 10.0} for i, j in enumerate(plan.joints)}
            if not plan.passed(distances, angles):
                errors.append('parent-child structural error exceeds frozen motion tolerance')
        operation_scope = _motion_scope(trace[0], contract, kind, joint_name)
        if kind == 'fixed_joint' and (contract or {}).get('acceptance_policy') in {'object_functional_physics.v11'}:
            clearance = np.asarray([step['ground_min_z_m'] for step in trace], dtype=float)
            if not np.isfinite(clearance).all():
                raise ValueError('Nonfinite elevated-fixture ground clearance')
            metrics['fixed_base_fixture'] = {'policy': contract['fixed_base_policy'], 'planned_clearance_m': trace[0]['motion_scope_binding']['fixture_clearance_m'], 'minimum_measured_clearance_m': float(clearance.min()), 'ground_support_allowed': False}
            if np.any(clearance <= 0.0):
                errors.append('fixed-base joint motion must remain clear of the ground')
        if contract.get('motion_scope_policy') and kind in {'fixed_joint', 'free_joint'}:
            metrics['motion_scope'] = {'policy': contract['motion_scope_policy'], 'full_lower': contract['joints'][joint_name]['lower'], 'full_upper': contract['joints'][joint_name]['upper'], 'tested_lower': (operation_scope or contract['joints'][joint_name])['lower'], 'tested_upper': (operation_scope or contract['joints'][joint_name])['upper'], 'full_range_claimed': operation_scope is None or (operation_scope['lower'] == contract['joints'][joint_name]['lower'] and operation_scope['upper'] == contract['joints'][joint_name]['upper'])}
        depths = [-float(s.get('ground_min_z_m', 0.0)) for s in trace]
        native_depths = [-float(s['min_ground_separation_m']) for s in trace if s.get('min_ground_separation_m') is not None]
        if not all((math.isfinite(v) for v in depths + native_depths)):
            errors.append('nonfinite ground penetration diagnostic')
        else:
            metrics['penetration_diagnostics'] = {'acceptance_gate': False, 'max_cooked_depth_m': max([0.0, *depths]), 'max_native_depth_m': max([0.0, *native_depths])}
        for index, step in enumerate(trace):
            geometry = step.get('geometry', {})
            diagnostic_only = bool(geometry.get('issues')) and all((issue.get('type') in {'ground_penetration', 'ground_contact_penetration'} for issue in geometry['issues']))
            if geometry.get('complete', True) is not True or (not diagnostic_only and (geometry.get('passed') is not True or geometry.get('status', 'passed') != 'passed')):
                errors.append(f'geometry failed or uncertain at physics step {index}')
                break
        for index, step in enumerate(trace):
            if step.get('contacts', {}).get('observed') is not True:
                errors.append(f'contact observation missing at physics step {index}')
                break
        up = rotations.apply(np.array([0.0, 0.0, 1.0]))
        reference_up = up[0]
        reference_up = _rotation([trace[0].get('initialization', {}).get('tilt_reference_quaternion_wxyz', trace[0]['root_quaternion_wxyz'])]).apply([0.0, 0.0, 1.0])[0]
        tilt = np.degrees(np.arccos(np.clip(up @ reference_up, -1, 1)))
        metrics['max_tilt_deg'] = float(tilt.max())
        upright_required = _upright_required(contract or {})
        if (contract or {}).get('acceptance_policy') in {'object_functional_physics.v11'}:
            metrics['tilt_acceptance_gate'] = upright_required
            metrics['initial_reference_max_tilt_deg'] = metrics['max_tilt_deg']
        metrics['horizontal_slip_m'] = float(np.linalg.norm(position[-1, :2] - position[0, :2]))
        metrics['max_horizontal_displacement_m'] = float(np.linalg.norm(position[:, :2] - position[0, :2], axis=1).max())
        motion_index = None
        if kind in {'fixed_joint', 'free_joint'}:
            motion_index = _pre_motion_reference(trace, times, position, rotations, contract, errors, metrics, kind=kind)
            if motion_index is not None and kind == 'free_joint':
                metrics['pre_motion_max_tilt_deg'] = float(tilt[:motion_index + 1].max())
                if upright_required and metrics['pre_motion_max_tilt_deg'] > float(settings['tilt_limit_deg']) + 1e-06:
                    errors.append('object tipped during pre-motion settling')
                motion_tilt = np.degrees(np.arccos(np.clip(up[motion_index:] @ up[motion_index], -1, 1)))
                metrics['max_tilt_deg'] = float(motion_tilt.max())
                metrics['horizontal_slip_m'] = float(np.linalg.norm(position[-1, :2] - position[motion_index, :2]))
                metrics['max_horizontal_displacement_m'] = float(np.linalg.norm(position[motion_index:, :2] - position[motion_index, :2], axis=1).max())
        if upright_required and kind in {'fall', 'ground_settle', 'free_joint'} and (metrics['max_tilt_deg'] > float(settings['tilt_limit_deg']) + 1e-06):
            errors.append('object tipped beyond frozen tilt limit')
        if kind == 'free_joint':
            slip_tolerance = float(settings.get('free_slip_tolerance_m') or max(0.005, 0.02 * float(scene_diagonal)))
            metrics['free_slip_tolerance_m'] = slip_tolerance
            if metrics['max_horizontal_displacement_m'] > slip_tolerance + 1e-08:
                errors.append('free root exceeded the frozen horizontal slip limit')
        joint_metrics = {}
        joint_positions = {}
        joint_velocities = {}
        for name, definition in mapping.items():
            q = np.asarray([step.get('joint_positions', {}).get(name, np.nan) for step in trace], dtype=float)
            velocity = np.asarray([step.get('joint_velocities', {}).get(name, np.nan) for step in trace], dtype=float)
            effort = np.asarray([step.get('joint_efforts', {}).get(name, np.nan) for step in trace], dtype=float)
            if q.shape != times.shape or effort.shape != times.shape or (not np.isfinite(q).all()) or (not np.isfinite(effort).all()):
                errors.append(f'missing/nonfinite joint position or effort: {name}')
                continue
            if velocity.shape != times.shape or not np.isfinite(velocity).all():
                errors.append(f'missing/nonfinite joint velocity: {name}')
                continue
            joint_velocities[name] = velocity
            velocity_cap = float(definition.get('max_velocity', definition.get('limit', {}).get('velocity', np.nan)))
            if not math.isfinite(velocity_cap) or velocity_cap <= 0:
                errors.append(f'joint has no positive frozen velocity cap: {name}')
                continue
            velocity_epsilon = max(1e-08, velocity_cap * 1e-06)
            cap = float((effort_caps or {}).get(name, definition.get('max_effort', definition.get('limit', {}).get('effort', np.nan))))
            if not math.isfinite(cap) or cap <= 0:
                errors.append(f'joint has no positive frozen effort cap: {name}')
                continue
            if np.any(np.abs(effort) > cap + max(1e-08, cap * 1e-06)):
                errors.append(f'joint active effort exceeded frozen cap: {name}')
            limit_diagnostics = {}
            if definition.get('type') == 'continuous':
                q = np.unwrap(q)
            else:
                limits = definition.get('limit', definition)
                lower, upper = (float(limits['lower']), float(limits['upper']))
                if not math.isfinite(lower + upper) or lower >= upper:
                    raise ValueError(f'invalid joint limits: {name}')
                limit_fraction = 0.2 if relaxed else float(settings['endpoint_tolerance_fraction'])
                limit_slack = max(1e-05, (upper - lower) * limit_fraction)
                limit_diagnostics = {'physical_limit_tolerance_policy': 'motion_joint_limit_20pct.v1' if relaxed else 'frozen_endpoint_fraction.v1', 'physical_limit_tolerance': limit_slack, 'physical_limit_excursion': max(0.0, lower - float(q.min()), float(q.max()) - upper)}
                if q.min() < lower - limit_slack or q.max() > upper + limit_slack:
                    errors.append(f'joint violated its physical limits: {name}')
            joint_positions[name] = q
            joint_metrics[name] = {'min_q': float(q.min()), 'max_q': float(q.max()), 'peak_active_effort': float(np.abs(effort).max()), 'rms_active_effort': float(np.sqrt(np.mean(effort ** 2))), 'saturated_fraction': float(np.mean(np.abs(effort) >= 0.999 * cap)), 'saturated_duration_s': float(np.diff(times)[np.abs(effort[1:]) >= 0.999 * cap].sum()), 'peak_velocity': float(np.abs(velocity).max()), 'max_velocity': velocity_cap, 'velocity_numerical_epsilon': velocity_epsilon, **{'peak_velocity_acceptance_gate': False, 'target_velocity_limit': velocity_cap, 'peak_velocity_above_target_limit': bool(np.any(np.abs(velocity) > velocity_cap + velocity_epsilon))}, 'max_effort': cap, **limit_diagnostics}
        metrics['joints'] = joint_metrics
        if kind in {'fall', 'ground_settle'}:
            window = float(settings['settle_window_s'])
            maximum = float(settings['settle_duration_s'])
            if duration < window - dt or duration > maximum + 2 * dt:
                errors.append('fall duration does not contain its tail window or exceeds the time budget')
            if kind != 'ground_settle':
                errors.append('schema 2 requires ground_settle, not historical fall')
            if duration < maximum - dt / 2:
                errors.append('ground settle did not observe the full frozen duration')
            pos_tol_initial = float(settings.get('position_tolerance_m') or max(0.001, 0.001 * float(scene_diagonal)))
            expected_names = set(contract.get('required_links', []))
            references = trace[0].get('initial_link_transforms', {})
            if set(references) != expected_names:
                errors.append('ground settle lacks authored initial link pose references')
            if not expected_names or any((set(step.get('link_transforms', {})) != expected_names for step in trace)):
                errors.append('ground settle requires every link initial and measured pose')
            else:
                initial_metrics = {}
                for name in sorted(expected_names):
                    poses = np.asarray([step['link_transforms'][name] for step in trace], dtype=float)
                    if poses.shape != (len(trace), 4, 4) or not np.isfinite(poses).all():
                        errors.append('invalid ground link pose: ' + name)
                        continue
                    reference = np.asarray(references.get(name, poses[0]), dtype=float)
                    if reference.shape != (4, 4) or not np.isfinite(reference).all():
                        errors.append('invalid initial link reference: ' + name)
                        continue
                    observed_poses = poses[:1]
                    movement = float(np.linalg.norm(observed_poses[:, :3, 3] - reference[:3, 3], axis=1).max())
                    orientations = Rotation.from_matrix(observed_poses[:, :3, :3])
                    angle = float(np.degrees((Rotation.from_matrix(reference[:3, :3]).inv() * orientations).magnitude()).max())
                    initial_metrics[name] = {'max_initial_position_deviation_m': movement, 'max_initial_orientation_deviation_deg': angle}
                    if movement > pos_tol_initial + 1e-08 or angle > float(settings['orientation_tolerance_deg']) + 1e-06:
                        errors.append('ground settle changed a link initial pose: ' + name)
                metrics['initial_pose_links'] = initial_metrics
            if any((abs(value) > 1e-08 for step in trace for value in step.get('joint_efforts', {}).values())):
                errors.append('ground settle requires zero active effort throughout')
            tail_mask = times >= times[-1] - window - 1e-08
            tail = position[tail_mask]
            tail_rotations = rotations[tail_mask]
            if len(tail) < max(2, int(window / dt) - 1):
                errors.append('fall tail is missing physics samples')
            span = float(np.linalg.norm(tail - tail[0], axis=1).max())
            angular_span = float(np.degrees((tail_rotations[0].inv() * tail_rotations).magnitude()).max())
            pos_tol = float(settings.get('position_tolerance_m', max(0.001, 0.001 * float(scene_diagonal))))
            if not math.isfinite(pos_tol) or pos_tol <= 0:
                raise ValueError('position_tolerance_m must be positive and finite')
            metrics.update(tail_position_span_m=span, tail_orientation_span_deg=angular_span, position_tolerance_m=pos_tol, settle_window_s=window)
            if span > pos_tol or angular_span > float(settings['orientation_tolerance_deg']):
                errors.append('fall did not settle within the frozen tail tolerances')
            if not any((_ground_contact(step.get('contacts', {})) for step, selected in zip(trace, tail_mask) if selected)):
                errors.append('fall tail has no observed ground support contact')
            for name, q in joint_positions.items():
                tolerance = pos_tol if mapping[name].get('type') == 'prismatic' else math.radians(float(settings['orientation_tolerance_deg']))
                initial_deviation = float(np.max(np.abs(q - mapping[name]['initial_position'])))
                joint_metrics[name]['max_initial_q_deviation'] = initial_deviation
                from .support_rest_pose import joint_initial_tolerance
                initial_tolerance = joint_initial_tolerance(mapping[name], contract)
                if contract.get('acceptance_policy') == 'object_functional_physics.v11':
                    joint_metrics[name]['initial_q_tolerance'] = initial_tolerance
                if initial_deviation > initial_tolerance + 1e-08:
                    errors.append('ground settle changed an initial joint position: ' + name)
                q_span = float(np.ptp(q[tail_mask]))
                joint_metrics[name]['tail_q_span'] = q_span
                if q_span > tolerance:
                    errors.append(f'fall joint did not settle: {name}')
                peak_velocity = float(np.abs(joint_velocities[name][tail_mask]).max())
                joint_metrics[name]['tail_peak_velocity'] = peak_velocity
                if not _passive_velocity_diagnostic_only(contract or {}) and peak_velocity > tolerance / window:
                    errors.append(f'fall joint retains tail velocity: {name}')
            if any(('link_transforms' in step for step in trace)):
                names = set(trace[0].get('link_transforms', {}))
                if not names or any((set(step.get('link_transforms', {})) != names for step in trace)):
                    errors.append('fall link pose coverage is incomplete')
                else:
                    link_metrics = {}
                    for name in names:
                        transforms = np.asarray([step['link_transforms'][name] for step in trace], dtype=float)
                        if transforms.shape != (len(trace), 4, 4) or not np.isfinite(transforms).all():
                            errors.append(f'missing/nonfinite fall link pose: {name}')
                            continue
                        tail_transforms = transforms[tail_mask]
                        movement = float(np.linalg.norm(tail_transforms[:, :3, 3] - tail_transforms[0, :3, 3], axis=1).max())
                        orientations = Rotation.from_matrix(tail_transforms[:, :3, :3])
                        angle = float(np.degrees((orientations[0].inv() * orientations).magnitude()).max())
                        link_metrics[name] = {'tail_position_span_m': movement, 'tail_orientation_span_deg': angle}
                        if movement > pos_tol or angle > float(settings['orientation_tolerance_deg']):
                            errors.append(f'fall rigid link did not settle: {name}')
                    metrics['links'] = link_metrics
        elif kind != 'reload_supported':
            if joint_name not in mapping or joint_name not in joint_positions:
                errors.append('joint trial has no valid target joint trace')
            else:
                definition, q = (mapping[joint_name], joint_positions[joint_name])
                if operation_scope is not None:
                    from .control import scoped_motion_joint
                    definition = scoped_motion_joint(contract, joint_name, operation_scope)
                    lower, upper = (operation_scope['lower'], operation_scope['upper'])
                elif definition.get('type') == 'continuous':
                    q_initial = definition.get('initial_position', q[0])
                    lower, upper = (q_initial - 2 * math.pi, q_initial + 2 * math.pi)
                else:
                    limits = definition.get('limit', definition)
                    lower, upper = (float(limits['lower']), float(limits['upper']))
                span = upper - lower
                tolerance = span * (0.8 if relaxed else float(settings['endpoint_tolerance_fraction']))
                coverage = min(1.0, max(0.0, min(q.max(), upper) - max(q.min(), lower)) / span)
                endpoint_errors = {'lower': float(abs(q.min() - lower)), 'upper': float(abs(q.max() - upper)), 'return': float(abs(q[-1] - definition.get('initial_position', q[0])))}
                metrics.update(joint_name=joint_name, coverage=coverage, endpoint_errors=endpoint_errors)
                if relaxed:
                    metrics.update(motion_acceptance=contract['motion_acceptance'], coverage_min=0.8, endpoint_tolerance=tolerance)
                for held_name, held_q in joint_positions.items():
                    if held_name != joint_name:
                        held_initial = mapping[held_name].get('initial_position', held_q[0])
                        deviation = float(np.abs(held_q - held_initial).max())
                        joint_metrics[held_name]['max_hold_deviation'] = deviation
                        hold_tol = float(settings.get('position_tolerance_m') or max(0.001, 0.001 * float(scene_diagonal))) if mapping[held_name]['type'] == 'prismatic' else math.radians(float(settings['orientation_tolerance_deg']))
                        if relaxed:
                            hold_tol = hold_tolerance(mapping[held_name], contract)
                        joint_metrics[held_name]['hold_tolerance'] = hold_tol
                        if deviation > hold_tol + 1e-08:
                            errors.append('non-tested joint exceeded its initial hold tolerance: ' + held_name)
                if coverage + 1e-09 < (0.8 if relaxed else float(settings['coverage_min'])):
                    errors.append('joint did not traverse the required range')
                if any((error > tolerance + 1e-08 for error in endpoint_errors.values())):
                    errors.append('joint did not reach both endpoints and return to its initial position')
                low_hits = np.flatnonzero(np.abs(q - lower) <= tolerance + 1e-08)
                high_hits = np.flatnonzero(np.abs(q - upper) <= tolerance + 1e-08)
                if not len(low_hits) or not len(high_hits) or low_hits[0] >= high_hits[-1]:
                    errors.append('joint did not execute the ordered lower/upper/return trajectory')
                from .control import target_schedule
                _, required_motion_duration = target_schedule(definition, settings, contract=contract)
                required_duration = float(settings['settle_duration_s']) + required_motion_duration + (0.0 if relaxed else float(settings['zero_effort_tail_s']))
                metrics['required_duration_s'] = required_duration
                if duration < required_duration - dt / 2:
                    errors.append('joint trial lacks its full frozen trajectory' if relaxed else 'joint trial lacks its full frozen trajectory and zero-effort tail')
                q0 = float(definition['initial_position'])
                initial_reference = trace[0].get('initialization', {}).get('expected_joint_positions', {}).get(joint_name, q0)
                if abs(q[0] - initial_reference) > max(1e-06, tolerance * 0.001):
                    errors.append('joint trial did not initialize at the frozen initial position')
                if not relaxed:
                    tail_s = float(settings.get('zero_effort_tail_s', 1.0))
                    tail = times > times[-1] - tail_s + 1e-08
                    if np.count_nonzero(tail) < max(1, int(tail_s / dt) - 1):
                        errors.append('zero effort tail is incomplete')
                    for name in mapping:
                        tail_effort = [step.get('joint_efforts', {}).get(name, math.nan) for step, selected in zip(trace, tail) if selected]
                        if any((not math.isfinite(v) or abs(v) > 1e-08 for v in tail_effort)):
                            errors.append('final one-second tail contains active effort: ' + name)
                    hold = float(settings['endpoint_hold_s'])
                    lower_intervals = _holding_intervals(times, q, lower, tolerance, float(settings['settle_duration_s']))
                    lower_holds = [(a, b) for a, b in lower_intervals if b - a >= hold - 1e-08]
                    lower_ready = lower_holds[0][0] + hold if lower_holds else float('inf')
                    upper_intervals = _holding_intervals(times, q, upper, tolerance, lower_ready)
                    upper_holds = [(a, b) for a, b in upper_intervals if b - a >= hold - 1e-08]
                    metrics['endpoint_actual_holds_s'] = {'lower': max((b - a for a, b in lower_intervals), default=0.0), 'upper_after_lower': max((b - a for a, b in upper_intervals), default=0.0)}
                    if not lower_holds or not upper_holds:
                        errors.append('joint did not actually hold both ordered endpoints for the frozen duration')
                    elif not np.any((times >= upper_holds[0][0] + hold - 1e-08) & (np.abs(q - q0) <= tolerance + 1e-08)):
                        errors.append('no return after the completed upper endpoint hold')
        if kind in {'ground_settle', 'fixed_joint', 'free_joint'} and (not relaxed):
            _passive_tail(trace, times, settings, contract, joint_positions, joint_velocities, errors, metrics)
        errors[:] = list(dict.fromkeys(errors))
        result.update(passed=not errors, status='passed' if not errors else 'failed')
    except (ValueError, TypeError, KeyError, IndexError, FloatingPointError) as error:
        errors.append(f'invalid/incomplete physics evidence: {error}')
    return result
