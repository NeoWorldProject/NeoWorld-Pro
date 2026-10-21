"""Small deterministic controllers; all joint units are metres/radians and SI effort.

Passive damping is reported separately: Isaac already applies it in the joint.
Only ``active`` is an actuator command. There is no teleportation in this module.
"""
from __future__ import annotations

import math

UNIFORM_PID_POLICY = "uniform_inertia_pid.v1"
INTEGRAL_POLICIES = {"initial_discrete_pid_observed_bias.v1", UNIFORM_PID_POLICY}
BIAS_POLICIES = {"initial_discrete_pd_observed_bias.v1", *INTEGRAL_POLICIES}
INERTIA_POLICIES = {"initial_discrete_pd_risk.v1", *BIAS_POLICIES}
# Engineering default of the versioned PID algorithm, not a per-asset patch.
INTEGRAL_TIME_CONSTANT_S = 1.0


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


MOTION_V9_RULES = {
    'policy': 'joint_motion_acceptance.v1',
    'trial_kinds': ['fixed_joint', 'free_joint'],
    'coverage_min': .8,
    'endpoint_tolerance_fraction': .8,
    'joint_limit_tolerance_fraction': .2,
    'minimum_translation_tolerance_m': 1e-5,
    'non_target_rotation_tolerance_deg': 10.,
    'non_target_prismatic_tolerance_fraction': .2,
    'endpoint_hold_required': False,
    'post_motion_zero_effort_required': False,
    'structural_rotation_tolerance_deg': 10.,
    'structural_prismatic_tolerance_fraction': .2,
    'structural_reference': 'parent_child_relative_pose',
    'other_structural_translation': 'original_position_tolerance',
}

def trial_resource_budget(duration, config):
    """Account for a complete planned trajectory, without truncating it.

    trial_timeout_s is a legacy workload floor, not an acceptance limit.
    The supervisor detects startup failures and stalls without a runtime cap.
    """
    duration = _finite(duration, "planned trial duration")
    floor = _finite(config["trial_timeout_s"], "trial allocation floor")
    dt = _finite(config["physics_dt_s"], "physics time step")
    if min(duration, floor, dt) <= 0:
        raise ValueError("Trial workload and time step must be positive")
    steps = math.ceil(duration / dt)
    return {"policy": "complete_planned_trial_workload.v1",
            "planned_duration_s": duration, "expected_physics_steps": steps,
            "legacy_allocation_floor_s": floor,
            "effective_simulated_allocation_s": max(floor, steps * dt),
            "trajectory_or_acceptance_changed": False}


def joint_waypoints(joint, q0=0.0):
    """Return q0 -> lower -> upper -> q0 (continuous: one turn each way)."""
    q0 = _finite(q0, "q0")
    if joint.get("type") == "fixed":
        return [q0]
    if joint.get("type") == "continuous":
        return [q0, q0 - 2 * math.pi, q0 + 2 * math.pi, q0]
    limits = joint.get("limit", joint)
    lower, upper = (_finite(limits[k], k) for k in ("lower", "upper"))
    if lower >= upper or not lower <= q0 <= upper:
        raise ValueError("joint requires lower < upper and q0 inside limits")
    return [q0, lower, upper, q0]


def trajectory_duration(joint, *, q0=0.0, endpoint_hold_s=0.5, speed_fraction=0.25):
    """Duration keeps the quintic target's peak speed below the slow speed cap."""
    velocity = _finite(joint.get("max_velocity", joint.get("limit", {}).get("velocity")), "max_velocity")
    fraction = _finite(speed_fraction, "speed_fraction")
    hold = _finite(endpoint_hold_s, "endpoint_hold_s")
    if velocity <= 0 or not 0 < fraction <= 1 or hold < 0:
        raise ValueError("velocity/fraction must be positive and hold nonnegative")
    waypoints = joint_waypoints(joint, q0)
    return sum(1.875 * abs(b - a) / (velocity * fraction) + hold
               for a, b in zip(waypoints, waypoints[1:]))


def joint_target(joint, t, duration, q0=0.0, endpoint_hold_s=0.0):
    """Quintic ease-in/out, distance-weighted segments and explicit endpoint holds."""
    t, duration = _finite(t, "t"), _finite(duration, "duration")
    hold = _finite(endpoint_hold_s, "endpoint_hold_s")
    points = joint_waypoints(joint, q0)
    if len(points) == 1:
        return points[0]
    moving_time = duration - hold * (len(points) - 1)
    if duration <= 0 or hold < 0 or moving_time <= 0:
        raise ValueError("duration must exceed endpoint holds")
    distances = [abs(b - a) for a, b in zip(points, points[1:])]
    total = sum(distances)
    if t <= 0:
        return points[0]
    if t >= duration:
        return points[-1]
    cursor = 0.0
    for a, b, distance in zip(points, points[1:], distances):
        travel = moving_time * distance / total
        if t < cursor + travel and travel > 0:
            u = (t - cursor) / travel
            return a + (b - a) * (10 * u**3 - 15 * u**4 + 6 * u**5)
        cursor += travel
        if t <= cursor + hold:
            return b
        cursor += hold
    return points[-1]


def bounded_pd(q, qd, target, max_effort, kp, kd, passive_damping=0.0,
               passive_friction=0.0):
    """Clip active PD independently of passive resistance; never add the latter."""
    q, qd, target, cap, kp, kd, damping, friction = (
        _finite(value, name) for value, name in
        zip((q, qd, target, max_effort, kp, kd, passive_damping, passive_friction),
            ("q", "qd", "target", "max_effort", "kp", "kd", "passive_damping", "passive_friction")))
    if cap <= 0 or min(kp, kd, damping, friction) < 0:
        raise ValueError("effort cap must be positive; gains/resistance nonnegative")
    requested = kp * (target - q) - kd * qd
    active = max(-cap, min(cap, requested))
    passive = -damping * qd - (math.copysign(friction, qd) if abs(qd) > 1e-12 else 0.0)
    return {"active": active, "requested": requested, "passive_estimate": passive,
            "saturated": abs(requested) > cap, "max_effort": cap}


def native_joint_axis_values(parameters, kind):
    """USD axis damping uses degrees; SI static/dynamic effort needs no conversion."""
    if kind not in {"angular", "linear"}:
        raise ValueError("native joint axis must be angular or linear")
    static, dynamic, damping = (_finite(parameters[key], key) for key in
        ("static_friction_effort", "dynamic_friction_effort", "damping"))
    if min(static, dynamic, damping) < 0 or dynamic > static:
        raise ValueError("invalid static/dynamic/viscous joint friction")
    return {"staticFrictionEffort": static, "dynamicFrictionEffort": dynamic,
            "viscousFrictionCoefficient": damping * math.pi / 180 if kind == "angular" else damping}


def _validated_operation_scope(contract, joint_name, operation_scope):
    """Check an explicit command interval, without claiming geometric clearance.

    The native motion-scope adapter owns geometry/evidence verification. This
    standalone executor only accepts its three-field normalized command interval.
    """
    if contract.get("motion_scope_policy") != "fixed_full_ground_operable.v1":
        raise ValueError("Operation scope requires fixed_full_ground_operable.v1")
    if (not isinstance(operation_scope, dict)
            or set(operation_scope) != {"joint_name", "lower", "upper"}):
        raise ValueError("Operation scope requires exactly joint_name, lower and upper")
    if operation_scope["joint_name"] != joint_name:
        raise ValueError("Operation scope belongs to a different joint")
    joint = contract["joints"][joint_name]
    values = {}
    for name, value in (("lower", operation_scope["lower"]),
                        ("upper", operation_scope["upper"]),
                        ("physical lower", joint["lower"]),
                        ("physical upper", joint["upper"]),
                        ("q_initial", joint["initial_position"])):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Operation scope " + name + " must be a finite number")
        values[name] = _finite(value, "Operation scope " + name)
    lower, upper, q0 = values["lower"], values["upper"], values["q_initial"]
    if not lower < upper:
        raise ValueError("Operation scope must have a nonempty lower < upper interval")
    if not values["physical lower"] <= lower <= q0 <= upper <= values["physical upper"]:
        raise ValueError("Operation scope must contain q_initial and remain inside physical limits")
    return {"joint_name": joint_name, "lower": lower, "upper": upper}


def scoped_motion_joint(contract, joint_name, operation_scope=None):
    """Copy a joint for scheduling; retain its complete physical limits/caps.

    With an explicit, policy-enabled scope, only the target waypoints become
    q_initial -> scope lower -> scope upper -> q_initial. This is a command shape
    check, not a proof of collision clearance or ground operability. A missing
    scope always schedules the original full operation.
    """
    import copy
    if (not isinstance(contract, dict) or not isinstance(contract.get("joints"), dict)
            or not isinstance(joint_name, str) or joint_name not in contract["joints"]):
        raise ValueError("Unknown operation joint: " + str(joint_name))
    joint = copy.deepcopy(contract["joints"][joint_name])
    if operation_scope is not None:
        scope = _validated_operation_scope(contract, joint_name, operation_scope)
        q0 = joint["initial_position"]
        joint["waypoints"] = [q0, scope["lower"], scope["upper"], q0]
    return joint


def target_schedule(joint, config, *, contract=None):
    """Frozen q_initial -> lower -> upper -> q_initial with bounded cosine ramps."""
    current = float(joint.get("initial_position", joint["waypoints"][0]))
    fraction = _finite(joint.get("speed_fraction", .8), "speed_fraction")
    if not 0 < fraction <= 1:
        raise ValueError("speed_fraction must be in (0, 1]")
    relaxed = bool(contract and contract.get("acceptance_policy") in {'object_functional_physics.v11'})
    hold = 0. if relaxed else float(config["endpoint_hold_s"]) + float(config.get("endpoint_tracking_allowance_s", 0.0))
    cursor = hold
    segments = [(0.0, cursor, current, current)]
    for target in joint["waypoints"][1:]:
        target = float(target)
        duration = abs(target - current) * math.pi / (2 * joint["max_velocity"] * fraction)
        if duration:
            segments.append((cursor, cursor + duration, current, target))
            cursor += duration
        segments.append((cursor, cursor + hold, target, target))
        cursor += hold
        current = target
    return segments, cursor


def schedule_target(segments, t):
    for start, end, a, b in segments:
        if t <= end:
            u = max(0.0, min(1.0, (t - start) / max(end - start, 1e-12)))
            return a + (b - a) * (1 - math.cos(math.pi * u)) / 2
    return float(segments[-1][3])


def controller_policy(contract, effort_policy="explicit_bounded_pd.v1"):
    """Self-contained delivery policy; importing/loading never activates it."""
    import copy
    import hashlib
    import json
    frozen = copy.deepcopy(contract)
    encoded = (json.dumps(frozen, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if effort_policy not in {"explicit_bounded_pd.v1", *INERTIA_POLICIES}:
        raise ValueError("Unknown bounded effort policy")
    result = {"schema_version": 2, "default_active": False, "mode": "bounded_joint_effort",
            "contract": frozen, "contract_sha256": hashlib.sha256(encoded).hexdigest(),
            "joints": {name: {key: joint[key] for key in ("kp", "kd", "speed_fraction", "initial_position")}
                       for name, joint in frozen["joints"].items()}}
    if effort_policy != "explicit_bounded_pd.v1":
        result.update(schema_version=3, effort_policy=effort_policy)
    return result


def post_operation_observation_s(config, *, contract=None):
    """Allow passive settling before applying the final stability window.

    The configured zero-effort tail remains the measurement window, not a
    requirement to be motionless immediately when a supporting effort ends.
    Targets, active control duration and acceptance tolerances are unchanged.
    """
    if contract and contract.get("acceptance_policy") in {'object_functional_physics.v11'}:
        return 0.
    window = _finite(config["zero_effort_tail_s"], "zero_effort_tail_s")
    settle = _finite(config["settle_duration_s"], "settle_duration_s")
    if min(window, settle) <= 0:
        raise ValueError("Passive observation durations must be positive")
    return max(window, settle)


class BoundedJointExecutor:
    """Reusable name-based effort controller, initially inactive.

    ``start(name)`` begins one finite operation at command time zero, independent
    of simulation initialization. ``start()`` explicitly holds all initial joint
    positions. ``stop()`` disables all active effort; apply/step must still be
    called to send those zero commands to a simulator. No method teleports bodies
    or changes passive resistance. All observed q/qd and efforts use SI units.

    ``step(q, qd, t=seconds)`` uses absolute seconds since the latest start;
    alternatively ``dt=seconds`` advances the command clock before evaluation.
    The explicit path uses the standard library; the inertia-based policy and
    articulation adapter require NumPy. The copied controller_executor.py is
    standalone and binds the policy used during native acceptance.

    ``operation_scope={"joint_name": name, "lower": q_min, "upper": q_max}``
    explicitly narrows one command trajectory under the contract's opt-in scope
    policy. It never changes physical limits or the delivered full contract, and
    does not itself verify geometric operability. Without a scope, start(name)
    retains the full frozen operation. Loading either form remains inactive.
    """
    def __init__(self, contract, operation_scope=None, *, effort_policy="explicit_bounded_pd.v1"):
        import copy
        import json
        # Reject nonfinite values even in fields not consumed by this executor.
        json.dumps(contract, allow_nan=False)
        if not isinstance(contract, dict) or contract.get("schema_version") not in (2, 3):
            raise ValueError("Executor requires a frozen schema 2 or 3 contract")
        if effort_policy not in {"explicit_bounded_pd.v1", *INERTIA_POLICIES}:
            raise ValueError("Unknown bounded effort policy")
        self.effort_policy = effort_policy
        self.contract = copy.deepcopy(contract)
        self.motion_v9 = contract.get("acceptance_policy") in {'object_functional_physics.v11'}
        if self.motion_v9:
            if contract.get('motion_acceptance') != MOTION_V9_RULES:
                raise ValueError('v9 requires its frozen motion acceptance rules')
        self.joints = self.contract.get("joints")
        if not isinstance(self.joints, dict):
            raise ValueError("Contract joints must be a name-keyed mapping")
        config = self.contract.get("config", {})
        self.tail_s = self._number(config.get("zero_effort_tail_s"), "zero_effort_tail_s")
        if self.tail_s < 0:
            raise ValueError("zero_effort_tail_s must be nonnegative")
        for name, joint in self.joints.items():
            if not isinstance(name, str) or not name or not isinstance(joint, dict):
                raise ValueError("Invalid joint name or policy")
            values = {key: self._number(joint.get(key), name + "." + key) for key in
                      ("initial_position", "lower", "upper", "max_effort", "max_velocity", "kp", "kd", "speed_fraction")}
            if (joint.get("type") not in {"revolute", "prismatic", "continuous"}
                    or not values["lower"] < values["upper"]
                    or not values["lower"] <= values["initial_position"] <= values["upper"]
                    or min(values["max_effort"], values["max_velocity"], values["kp"]) <= 0
                    or values["kd"] < 0 or not .05 <= values["speed_fraction"] <= .8):
                raise ValueError("Invalid frozen controller values: " + name)
            expected = [values["initial_position"], values["lower"], values["upper"], values["initial_position"]]
            if joint.get("waypoints") != expected:
                raise ValueError("Controller waypoints differ from the frozen operation: " + name)
            target_schedule(joint, config, contract=self.contract)
        for key in ("endpoint_hold_s", "endpoint_tracking_allowance_s"):
            if self._number(config.get(key, 0.), key) < 0:
                raise ValueError(key + " must be nonnegative")
        self.names = tuple(self.joints)
        self.execution_optimized = True
        self._native_previous, self._native_unwrapped = {}, {}
        self.initial = {name: joint["initial_position"] for name, joint in self.joints.items()}
        self.last_targets = dict(self.initial)
        self.operation_scope = None
        if operation_scope is not None:
            name = operation_scope.get("joint_name") if isinstance(operation_scope, dict) else None
            # Check the named joint before validating the normalized interval.
            scoped_motion_joint(self.contract, name, operation_scope)
            self.operation_scope = _validated_operation_scope(self.contract, name, operation_scope)
        self.stop()

    @staticmethod
    def _number(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(name + " must be a finite number")
        return _finite(value, name)

    @classmethod
    def from_files(cls, policy_path, contract_path=None, operation_scope=None):
        """Load the full policy, optionally with an explicit command-only scope."""
        import hashlib
        import json
        from pathlib import Path
        policy = json.loads(Path(policy_path).read_text())
        if (policy.get("schema_version") not in (2, 3) or policy.get("default_active") is not False
                or policy.get("mode") != "bounded_joint_effort"):
            raise ValueError("Invalid or automatically active controller policy")
        contract = policy.get("contract")
        effort_policy = "explicit_bounded_pd.v1" if policy["schema_version"] == 2 else policy.get("effort_policy")
        if policy["schema_version"] == 3 and effort_policy not in INERTIA_POLICIES:
            raise ValueError("Invalid schema 3 effort policy")
        expected = controller_policy(contract, effort_policy)
        if policy != expected:
            raise ValueError("Controller policy does not bind its full frozen contract")
        if contract_path is not None:
            raw = Path(contract_path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != policy["contract_sha256"] or json.loads(raw) != contract:
                raise ValueError("External contract differs from the delivered controller")
        return cls(contract, operation_scope=operation_scope, effort_policy=effort_policy)

    def start(self, joint_name=None):
        """Explicitly start a joint operation, or hold every initial pose."""
        if joint_name is not None and joint_name not in self.joints:
            raise ValueError("Unknown operation joint: " + str(joint_name))
        segments, duration = None, None
        if joint_name is not None:
            joint = scoped_motion_joint(self.contract, joint_name, self.operation_scope)
            segments, duration = target_schedule(joint, self.contract["config"], contract=self.contract)
        self.controller_selection = None
        self._previous_dynamics = None
        self._integral_effort = {name: 0. for name in self.names}
        self.active = True
        self.target_joint = joint_name
        self.elapsed_s = 0.0
        self.last_targets = dict(self.initial)
        self.segments, self.motion_duration_s = segments, duration
        self.phase = "holding" if joint_name is None else "motion"
        return self

    def stop(self):
        """Disable effort. The returned zeros can be sent immediately by a caller."""
        self.controller_selection = None
        self._previous_dynamics = None
        self._integral_effort = {name: 0. for name in self.names}
        self.active = False
        self.target_joint = None
        self.elapsed_s = 0.0
        self.segments, self.motion_duration_s = None, None
        self.last_targets = dict(self.initial)
        self.phase = "inactive"
        return {name: 0.0 for name in self.names}

    def step(self, q=None, qd=None, *, dt=None, t=None, mass_matrix=None, physics_dt=None):
        """Return name -> bounded active effort, never passive force or position.

        The optional inertia-based policy requires matrix coordinates in names
        order and a physics_dt independent of the command clock. Selection is
        frozen at the first active sample of each start(), without retuning any
        API gain or effort cap. Missing inertia cannot silently change methods.
        """
        zeros = {name: 0.0 for name in self.names}
        if not self.active:
            return zeros
        if (dt is None) == (t is None):
            raise ValueError("Supply exactly one of dt or t while active")
        if dt is not None:
            increment = self._number(dt, "dt")
            if increment <= 0:
                raise ValueError("dt must be positive")
            now = self.elapsed_s + increment
        else:
            now = self._number(t, "t")
            if now < self.elapsed_s:
                raise ValueError("Command time must not go backwards; call start to reset")
        self.elapsed_s = now
        self.last_targets = dict(self.initial)
        if self.target_joint is not None:
            self.last_targets[self.target_joint] = schedule_target(self.segments, now)
            if not self.motion_v9 and now >= self.motion_duration_s - 1e-8:
                self._previous_dynamics = None
                self._integral_effort = {name: 0. for name in self.names}
                self.phase = "tail" if now < self.motion_duration_s + self.tail_s - 1e-8 else "complete"
                if self.phase == "complete":
                    self.active = False
                return zeros
        if not isinstance(q, dict) or not isinstance(qd, dict) or set(q) != set(self.names) or set(qd) != set(self.names):
            raise ValueError("q and qd must cover every controller joint exactly once")
        self.phase = "holding" if self.target_joint is None else "motion"
        if self.effort_policy in INERTIA_POLICIES and self.names:
            gain = lambda key: [self.joints[name][key] for name in self.names]
            solver_dt = self._number(physics_dt, "physics_dt")
            if solver_dt <= 0:
                raise ValueError("physics_dt must be positive")
            if self.controller_selection is None:
                self.controller_selection = select_pd_method(mass_matrix, gain("kp"), gain("kd"), solver_dt)
                if self.effort_policy == UNIFORM_PID_POLICY:
                    # A stable explicit PD linearization says nothing about
                    # steady load rejection. Keep bounded integral action for
                    # every joint under this opt-in version, including those
                    # that did not require implicit integration for stiffness.
                    self.controller_selection.update(
                        linearized_pd_method=self.controller_selection["method"], method="implicit")
                if self.effort_policy in BIAS_POLICIES:
                    self.controller_selection.update(selection_policy=self.effort_policy,
                        prediction_bias="Previous native velocity increment minus predicted commanded acceleration; reset on start/stop.")
                if self.effort_policy in INTEGRAL_POLICIES:
                    self.controller_selection.update(
                        integral_time_constant_s=INTEGRAL_TIME_CONSTANT_S,
                        integral_applicability="implicit_path_only",
                        integral_antiwindup="freeze_outward_integration_at_saturation",
                        integral_effort_bound="frozen_joint_effort_cap")
            if self.controller_selection["method"] == "implicit":
                import numpy as np
                columns = lambda values: [self._number(values[name], name) for name in self.names]
                velocity = np.asarray(columns(qd))
                external = None
                if self.effort_policy in BIAS_POLICIES:
                    external = np.zeros(len(self.names))
                    if self._previous_dynamics is not None:
                        previous_v, previous_b, previous_u, previous_t, previous_dt = self._previous_dynamics
                        if not math.isclose(now-previous_t, solver_dt, rel_tol=1e-6, abs_tol=1e-9) or previous_dt != solver_dt:
                            raise ValueError("Observed-bias controller requires one command per frozen physics step")
                        external = (velocity-previous_v)/solver_dt - previous_b @ previous_u
                efforts, _ = bounded_implicit_pd(columns(q), columns(qd), columns(self.last_targets),
                    gain("kp"), gain("kd"), gain("max_effort"), mass_matrix, solver_dt,
                    external_acceleration=external,
                    feedforward_effort=columns(self._integral_effort) if self.effort_policy in INTEGRAL_POLICIES else None)
                if external is not None:
                    self._previous_dynamics = (velocity.copy(), joint_mobility(mass_matrix, len(self.names)),
                                               efforts.copy(), now, solver_dt)
                if self.effort_policy in INTEGRAL_POLICIES:
                    # Universal one-second integral time constant. Integral
                    # contribution and TOTAL applied effort share the frozen
                    # actuator bound. Freeze integration driving saturation;
                    # allow integration opposing it (anti-windup).
                    for index, name in enumerate(self.names):
                        joint = self.joints[name]
                        change = joint['kp'] * (self.last_targets[name] - q[name]) * solver_dt / INTEGRAL_TIME_CONSTANT_S
                        cap = joint['max_effort']
                        if abs(efforts[index]) < cap * (1 - 1e-8) or change * efforts[index] <= 0:
                            self._integral_effort[name] = max(-cap, min(cap, self._integral_effort[name] + change))
                return dict(zip(self.names, map(float, efforts)))
        if self.execution_optimized:
            result = {}
            for name, joint in self.joints.items():
                position = self._number(q[name], name + '.q')
                velocity = self._number(qd[name], name + '.qd')
                requested = joint['kp'] * (self.last_targets[name] - position) - joint['kd'] * velocity
                if not math.isfinite(requested):
                    raise ValueError('Nonfinite commanded joint effort')
                cap = joint['max_effort']
                result[name] = max(-cap, min(cap, requested))
            return result
        return {name: bounded_pd(self._number(q[name], name + ".q"), self._number(qd[name], name + ".qd"),
                                 self.last_targets[name], joint["max_effort"], joint["kp"], joint["kd"])["active"]
                for name, joint in self.joints.items()}

    @property
    def needs_native_inertia(self):
        """Inactive and exact explicit paths need no optional inertia query."""
        return (self.active and bool(self.names)
                and self.effort_policy in INERTIA_POLICIES
                and (self.controller_selection is None or self.controller_selection["method"] == "implicit"))

    def apply_to_articulation(self, articulation, dof_names, *, dt=None, t=None):
        """Isaac tensor adapter with explicit native DOF order, without guessing.

        Requires NumPy and a single articulation exposing get_joint_positions,
        get_joint_velocities and set_joint_efforts. Handles [n] or [1,n] state;
        Continuous native angles are unwrapped incrementally from the first
        observed state, matching the physics worker; sample frequently enough
        that the angular change between calls stays below pi. Direct step()
        callers supply already-unwrapped positions for continuous joints.
        Callers with another simulator can apply the named step result instead.
        """
        import numpy as np
        names = tuple(dof_names)
        if len(names) != len(set(names)) or set(names) != set(self.names):
            raise ValueError("Native DOF names must exactly match controller joint names")
        q, qd = {}, {}
        if self.active:
            for method, destination in (("get_joint_positions", q), ("get_joint_velocities", qd)):
                values = np.asarray(getattr(articulation, method)())
                if values.shape not in {(len(names),), (1, len(names))}:
                    raise ValueError("Expected a single articulation state matching DOF order")
                destination.update(zip(names, map(float, values.reshape(-1))))
            for name in names:
                if self.joints[name]["type"] != "continuous":
                    continue
                raw = q[name]
                if name in self._native_previous:
                    delta = (raw - self._native_previous[name] + math.pi) % (2 * math.pi) - math.pi
                    self._native_unwrapped[name] += delta
                else:
                    self._native_unwrapped[name] = raw
                self._native_previous[name] = raw
                q[name] = self._native_unwrapped[name]
        matrix = None
        if self.needs_native_inertia:
            matrix = ordered_native_inertia(articulation, names, self.names)
        efforts = self.step(q, qd, dt=dt, t=t, mass_matrix=matrix,
                            physics_dt=self.contract["config"].get("physics_dt_s"))
        articulation.set_joint_efforts(np.asarray([[efforts[name] for name in names]], dtype=np.float32))
        return efforts


def joint_mobility(mass_matrix, n):
    import numpy as np
    matrix = np.asarray(mass_matrix, dtype=float)
    if (matrix.shape not in {(n, n), (n + 6, n + 6)}
            or not np.isfinite(matrix).all()):
        raise ValueError('Invalid native generalized mass matrix')
    scale = max(float(np.linalg.norm(matrix)), np.finfo(float).tiny)
    if np.linalg.norm(matrix - matrix.T) > 1e-5 * scale:
        raise ValueError('Non-symmetric native generalized mass matrix')
    matrix = .5 * (matrix + matrix.T)
    np.linalg.cholesky(matrix)
    # A floating base has six unactuated coordinates. Use the joint block of
    # inverse mass, not the inverse of the joint principal mass block.
    mobility = np.linalg.solve(matrix, np.eye(len(matrix)))[-n:, -n:]
    return .5 * (mobility + mobility.T)


def bounded_implicit_pd(q, qd, target, kp, kd, caps, mass_matrix, dt, *, external_acceleration=None, feedforward_effort=None):
    """Solve the box-constrained next-step PD effort using native inertia.

    v_next = v + dt B u; q_next = q + dt v_next
    u = Kp(target-q_next)-Kd v_next + feedforward (if unsaturated).
    Minimize .5 u^T (B+D^-1) u - u^T D^-1 f, |u|<=caps,
    where D=dt Kd+dt^2 Kp and f=Kp(target-q-dt*v)-Kd*v.
    Saturation is solved jointly, including coupling, rather than clipping an
    unconstrained coupled solution and mispredicting other joint efforts.
    Optional feedforward is included INSIDE the same total effort bounds.
    """
    import numpy as np
    arrays = [np.asarray(x, dtype=float) for x in (q, qd, target, kp, kd, caps)]
    q, qd, target, kp, kd, caps = arrays
    if (q.ndim != 1 or not len(q) or any(x.shape != q.shape for x in arrays)
            or not all(np.isfinite(x).all() for x in arrays)
            or np.any(kp <= 0) or np.any(kd < 0) or np.any(caps <= 0)
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Invalid bounded controller inputs')
    mobility = joint_mobility(mass_matrix, len(q))
    d = dt * kd + dt * dt * kp
    requested = kp * (target - q - dt * qd) - kd * qd
    if feedforward_effort is not None:
        feedforward = np.asarray(feedforward_effort, dtype=float)
        if feedforward.shape != q.shape or not np.isfinite(feedforward).all():
            raise ValueError('Invalid integral effort')
        requested = requested + feedforward
    if external_acceleration is not None:
        external = np.asarray(external_acceleration, dtype=float)
        if external.shape != q.shape or not np.isfinite(external).all():
            raise ValueError('Invalid observed external acceleration')
        # Predict v_next = v + dt (B u + a_external). This corrects the
        # no-external-force integration model without increasing actuator caps
        # or changing native passive parameters. First observation uses zero.
        requested = requested - d * external
    hessian = mobility + np.diag(1 / d)
    rhs = requested / d
    # Transform to normalized effort bounds [-1,1], so SI effort magnitudes
    # do not determine convergence tolerance or stopping behavior.
    h = caps[:, None] * hessian * caps[None, :]
    b = caps * rhs
    magnitude = max(np.max(abs(h)), np.max(abs(b)), np.finfo(float).tiny)
    h, b = h / magnitude, b / magnitude
    y = np.clip(np.linalg.solve(h, b), -1., 1.)
    # Strictly convex coordinate minimization, with an explicit KKT check.
    residual = float('inf')
    for sweep in range(1000):
        for i in range(len(y)):
            y[i] = np.clip(y[i] - (h[i] @ y - b[i]) / h[i, i], -1., 1.)
        gradient = h @ y - b
        violation = gradient.copy()
        violation[y <= -1. + 1e-12] = np.minimum(gradient[y <= -1. + 1e-12], 0.)
        violation[y >= 1. - 1e-12] = np.maximum(gradient[y >= 1. - 1e-12], 0.)
        residual = float(np.max(abs(violation)))
        if residual <= 1e-10:
            break
    else:
        raise RuntimeError('Bounded predictive controller did not converge')
    effort = caps * y
    if not np.isfinite(effort).all() or np.any(abs(effort) > caps * (1 + 1e-12)):
        raise RuntimeError('Invalid or unbounded predictive effort')
    return effort, {'method': 'bounded_implicit_pd.v1',
                    'normalized_kkt_residual': residual, 'sweeps': sweep + 1}


def select_pd_method(mass_matrix, kp, kd, dt):
    """Freeze a controller choice from the initial unconstrained PD model.

    This is a numerical risk condition, not a prediction of native contact or
    friction stability. It contains no asset names, thresholds from acceptance,
    failure history or parameter recommendations. Stable explicit models keep
    the original exact effort calculation for the complete operation.
    """
    import numpy as np
    kp, kd = np.asarray(kp, dtype=float), np.asarray(kd, dtype=float)
    if (kp.ndim != 1 or not len(kp) or kd.shape != kp.shape
            or not np.isfinite(kp).all() or not np.isfinite(kd).all()
            or np.any(kp <= 0) or np.any(kd < 0)
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Invalid discrete controller selection inputs')
    mobility = joint_mobility(mass_matrix, len(kp))
    stiffness = dt * dt * mobility @ np.diag(kp)
    damping = dt * mobility @ np.diag(kd)
    identity = np.eye(len(kp))
    transition = np.block([[identity - stiffness, dt * (identity - damping)],
                           [-stiffness / dt, identity - damping]])
    poles = np.linalg.eigvals(transition)
    if not np.isfinite(poles).all():
        raise ValueError('Nonfinite discrete controller selection model')
    radius = float(np.max(abs(poles)))
    return {'method': 'implicit' if radius > 1. + 1e-6 else 'explicit',
            'spectral_radius': radius, 'selection_policy': 'initial_discrete_pd_risk.v1',
            'numeric_comparison_epsilon': 1e-6,
            'model_scope': 'Unsaturated semi-implicit Euler without contact, friction or external forces.',
            'selection_frozen_before_motion': True, 'parameters_modified': False,
            'acceptance_gate': False}


def ordered_native_inertia(articulation, native_names, controller_names):
    """Observe one native inertia and explicitly permute its joint coordinates.

    A floating base keeps its six unactuated coordinates first. Invalid or
    missing required observations are implementation errors, never silent force
    fallbacks or physical rejection. The standalone adapter shares this path.
    """
    import numpy as np
    native_names, controller_names = tuple(native_names), tuple(controller_names)
    n = len(controller_names)
    if (not n or len(native_names) != len(set(native_names))
            or len(controller_names) != len(set(controller_names))
            or set(native_names) != set(controller_names)):
        raise ValueError("Native inertia names must exactly cover controller joints")
    matrix = np.asarray(articulation.get_mass_matrices(), dtype=float)
    if matrix.ndim != 3 or matrix.shape[0] != 1:
        raise ValueError("Cannot observe one native generalized inertia")
    matrix = matrix[0]
    if matrix.shape not in {(n, n), (n + 6, n + 6)}:
        raise ValueError("Native inertia does not cover all required coordinates")
    extra = len(matrix) - n
    order = list(range(extra)) + [extra + native_names.index(name) for name in controller_names]
    return matrix[np.ix_(order, order)]
