"""Analytic, fixed-root ground clearance planning for one commanded joint.

This plans fixture height and a ground-feasible operation interval; it is not a
penetration or motion acceptance test. Other joints stay at their initial state.
Lifting, repositioning, tilting the root, and simultaneous joint actions are not
searched. Every cooked polygon's height extremum occurs at one of its vertices.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from urdf_core.urdf_validation import urdf_fk

SCHEMA = "joint_ground_motion_scope.v1"
_EPS = float(np.finfo(np.float64).eps)
_TAU = 2 * math.pi


def _number(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _json(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Source keys must be strings")
        return {key: _json(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("Source contains an unsupported or nonfinite value")


def _hash(value):
    raw = json.dumps(_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _matrix(value):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (4, 4):
        raise ValueError("root_transform must be a 4x4 rigid transform")
    matrix = np.asarray([[_number(x, "root transform") for x in row] for row in raw])
    if (not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-10, rtol=0)
            or not math.isclose(float(np.linalg.det(matrix[:3, :3])), 1., abs_tol=1e-10)):
        raise ValueError("root_transform must be a finite rigid transform without scale")
    return matrix


def _digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a SHA-256 digest")


def _curve_arrays(curves):
    if not isinstance(curves, list) or not curves:
        raise ValueError("Complete cooked vertex curves are required")
    arrays, identities = [], set()
    for row in curves:
        if not isinstance(row, dict) or set(row) != {"link", "mesh_index", "moving", "source_mesh_sha256", "coefficients"}:
            raise ValueError("Unexpected curve fields")
        if not isinstance(row["link"], str) or not row["link"]:
            raise ValueError("Invalid curve link")
        index = row["mesh_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Invalid mesh index")
        identity = (row["link"], index)
        if identity in identities or type(row["moving"]) is not bool:
            raise ValueError("Duplicate mesh or invalid moving flag")
        identities.add(identity)
        _digest(row["source_mesh_sha256"], "source_mesh_sha256")
        data = row["coefficients"]
        if (not isinstance(data, list) or not data or any(
                not isinstance(point, list) or len(point) != 3 for point in data)):
            raise ValueError("Each cooked vertex requires three height coefficients")
        array = np.asarray([[_number(x, "height coefficient") for x in point] for point in data])
        if not row["moving"] and np.any(array[:, :2] != 0):
            raise ValueError("Stationary geometry has a moving height curve")
        arrays.append(array)
    return np.concatenate(arrays)


def _derive(curves, kind, initial, lower, upper):
    coefficients = _curve_arrays(curves)
    linear = kind == "prismatic"
    if linear and np.any(coefficients[:, 0] != 0):
        raise ValueError("Prismatic curves must be linear")
    scale = max(1., float(np.max(np.abs(coefficients))))
    tolerance = 128 * _EPS * scale
    q_tolerance = 128 * _EPS * max(1., abs(initial), abs(lower), abs(upper))
    lo, hi = lower - initial, upper - initial
    if not all(math.isfinite(x) for x in (lo, hi, hi - lo)):
        raise ValueError("Unrepresentable joint range")
    starts = coefficients[:, 2] + (0 if linear else coefficients[:, 0])
    initial_min = float(np.min(starts))
    # Preserve existing seating depth; this is no additional ground intrusion.
    floor = min(0., initial_min)
    free_lo, free_hi, minimum = lo, hi, math.inf
    for a, b, c in coefficients:
        if linear:
            minimum = min(minimum, float(b * lo + c), float(b * hi + c))
            if abs(b) * (hi - lo) <= tolerance:
                continue
            boundary = float((floor - c) / b)
            if b > 0:
                free_lo = max(free_lo, boundary)
            else:
                free_hi = min(free_hi, boundary)
            continue
        radius = math.hypot(a, b)
        minimum = min(minimum, a * math.cos(lo) + b * math.sin(lo) + c,
                      a * math.cos(hi) + b * math.sin(hi) + c)
        if radius == 0:
            continue
        phase = math.atan2(b, a)
        critical = phase + math.pi
        critical += _TAU * math.ceil((lo - critical) / _TAU)
        if critical <= hi:
            minimum = min(minimum, c - radius)
        if a + c == floor and abs(b) > tolerance:
            # At exact initial contact, factor sin(delta/2). This preserves a
            # small nonzero interval even when hypot(a,b) rounds to abs(a).
            other_root = 2 * math.atan2(b, a)
            if b > 0:
                free_lo, free_hi = max(free_lo, 0.), min(free_hi, other_root)
            else:
                free_lo, free_hi = max(free_lo, other_root), min(free_hi, 0.)
            continue
        if c - radius >= floor - tolerance:
            continue  # Includes tangency at a minimum: no forbidden gap.
        if c + radius <= floor:
            if c + radius < floor:
                raise ValueError("Numerically unresolved maximum below the initial coordinate")
            # An isolated maximum on the plane is not an operation interval.
            free_lo, free_hi = max(free_lo, 0.), min(free_hi, 0.)
            continue
        angle = 2 * math.atan2(math.sqrt(c + radius - floor), math.sqrt(floor - (c - radius)))
        left, right = phase - angle, phase + angle
        if left > q_tolerance or right < -q_tolerance:
            raise ValueError("Numerically unresolved feasible component at the initial coordinate")
        else:
            free_lo, free_hi = max(free_lo, min(0., left)), min(free_hi, max(0., right))
    free_lower = max(lower, min(initial, initial + free_lo))
    free_upper = min(upper, max(initial, initial + free_hi))
    blocked = free_upper - free_lower <= q_tolerance
    if blocked:
        free_lower = free_upper = initial
    result = {
        "fixed_min_z_m": float(minimum), "fixed_clearance_lift_m": max(0., -float(minimum)),
        "initial_min_z_m": initial_min, "free_floor_z_m": floor,
        "numerical_tolerance_m": tolerance, "parameter_tolerance": q_tolerance,
        "free_lower": free_lower, "free_upper": free_upper,
        "free_status": "ground_blocked" if blocked else "operable",
        "free_range_fraction": (free_upper - free_lower) / (upper - lower),
        "blocked_lower": free_lower > lower + q_tolerance,
        "blocked_upper": free_upper < upper - q_tolerance,
        "range_resolution": "analytic",
    }
    if any(isinstance(v, float) and not math.isfinite(v) for v in result.values()):
        raise ValueError("Height calculation overflowed")
    return result


def plan_joint_motion(parsed, cooked, initial_positions, root_transform, joint_name, lower, upper):
    """Return analytic geometric scope; never modify a URDF, mesh or contract.

    ``cooked`` contains link-local PhysX vertices and faces, including empty lists
    for coordinate-only links. ``root_transform`` is the actual seated world/root
    transform. Its fixed-root minimum is an absolute world height; add the returned
    clearance lift to that placement for a fixture, plus caller-configured margin.
    """
    root = _matrix(root_transform)
    joints = {joint["name"]: joint for joint in parsed["joints"]}
    if len(joints) != len(parsed["joints"]) or joint_name not in joints:
        raise ValueError("Unknown or duplicate joint")
    joint = joints[joint_name]
    kind = joint["type"]
    if kind not in {"revolute", "continuous", "prismatic"}:
        raise ValueError("Only an active single joint can be planned")
    if set(initial_positions) - set(joints):
        raise ValueError("Unknown initial coordinate")
    positions = {name: _number(initial_positions.get(name, 0.), "initial position") for name in joints}
    lower, upper = _number(lower, "lower"), _number(upper, "upper")
    initial = positions[joint_name]
    if not lower <= initial <= upper or not lower < upper:
        raise ValueError("Full range must contain the initial position and have positive span")
    if kind != "continuous" and (lower < joint["limit"]["lower"] or upper > joint["limit"]["upper"]):
        raise ValueError("Planning must not extend source joint limits")
    # urdf_fk normalizes np.asarray(axis) in place; keep caller arrays read-only.
    frames = urdf_fk(copy.deepcopy(parsed), positions)
    if set(frames) != set(parsed["links"]) or set(cooked) != set(frames):
        raise ValueError("Incomplete cooked link coverage")
    world = {name: root @ pose for name, pose in frames.items()}
    moving = {joint["child"]}
    while True:
        added = {j["child"] for j in joints.values() if j["parent"] in moving} - moving
        if not added:
            break
        moving.update(added)
    axis = np.asarray(joint["axis"], dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or not math.isclose(float(np.linalg.norm(axis)), 1., abs_tol=1e-6):
        raise ValueError("Joint axis must be finite and normalized")
    axis = world[joint["child"]][:3, :3] @ (axis / np.linalg.norm(axis))
    pivot = world[joint["child"]][:3, 3]
    curves = []
    for link in sorted(cooked):
        for index, mesh in enumerate(cooked[link]):
            points = np.asarray(mesh["vertices"], dtype=float)
            faces = np.asarray(mesh["faces"])
            if (points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all()
                    or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                    or not np.issubdtype(faces.dtype, np.integer) or np.min(faces) < 0 or np.max(faces) >= len(points)):
                raise ValueError("Invalid cooked mesh")
            points = points @ world[link][:3, :3].T + world[link][:3, 3]
            coefficients = np.zeros_like(points)
            coefficients[:, 2] = points[:, 2]
            if link in moving:
                if kind == "prismatic":
                    coefficients[:, 1] = axis[2]
                else:
                    relative = points - pivot
                    parallel = np.outer(relative @ axis, axis)
                    coefficients[:, 0] = (relative - parallel)[:, 2]
                    coefficients[:, 1] = np.cross(axis, relative)[:, 2]
                    coefficients[:, 2] = pivot[2] + parallel[:, 2]
            curves.append({"link": link, "mesh_index": index, "moving": link in moving,
                           "source_mesh_sha256": _hash(mesh), "coefficients": coefficients.tolist()})
    source = {"parsed": parsed, "cooked": cooked, "initial_positions": positions,
              "root_transform": root, "joint_name": joint_name, "lower": lower, "upper": upper}
    plan = {"schema_version": SCHEMA, "planning_only": True,
            "scope": "fixed_root_single_joint_other_coordinates_at_initial",
            "source_input_sha256": _hash(source), "joint_name": joint_name, "joint_type": kind,
            "full_lower": lower, "full_upper": upper, "initial_position": initial,
            "initial_positions": positions, "root_transform": root.tolist(),
            "moving_links": sorted(moving), "curves": curves}
    plan.update(_derive(curves, kind, initial, lower, upper))
    plan["plan_sha256"] = _hash(plan)
    return validate_motion_plan(plan)


def validate_motion_plan(plan):
    """Recompute all declared limits from receipt curves, not from sampled q.

    Source/geometry provenance requires the caller's original-input artifact
    binding (or regenerating a plan from those inputs). A self-contained digest
    alone is not a proof of the origin of arbitrary supplied coefficients.
    """
    if not isinstance(plan, dict) or plan.get("schema_version") != SCHEMA or plan.get("planning_only") is not True:
        raise ValueError("Invalid motion planning receipt")
    _digest(plan.get("source_input_sha256"), "source_input_sha256")
    _digest(plan.get("plan_sha256"), "plan_sha256")
    if plan["plan_sha256"] != _hash({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("Motion planning receipt hash changed")
    if plan.get("scope") != "fixed_root_single_joint_other_coordinates_at_initial":
        raise ValueError("Unsupported operation scope")
    _matrix(plan["root_transform"])
    kind = plan["joint_type"]
    if kind not in {"revolute", "continuous", "prismatic"}:
        raise ValueError("Invalid joint type")
    name = plan["joint_name"]
    if not isinstance(name, str) or not name or not isinstance(plan["initial_positions"], dict):
        raise ValueError("Invalid joint identity")
    positions = {key: _number(value, "initial position") for key, value in plan["initial_positions"].items()}
    initial = _number(plan["initial_position"], "initial position")
    lower, upper = _number(plan["full_lower"], "full lower"), _number(plan["full_upper"], "full upper")
    if positions.get(name) != initial or not lower <= initial <= upper or not lower < upper:
        raise ValueError("Invalid frozen full range or initial position")
    moving = plan["moving_links"]
    if not isinstance(moving, list) or not moving or any(not isinstance(v, str) or not v for v in moving) or moving != sorted(set(moving)):
        raise ValueError("Invalid moving-link identities")
    derived = _derive(plan["curves"], kind, initial, lower, upper)
    if any(row["moving"] != (row["link"] in moving) for row in plan["curves"]):
        raise ValueError("Curve ownership differs from moving-link identities")
    expected = {"schema_version", "planning_only", "scope", "source_input_sha256", "joint_name", "joint_type",
                "full_lower", "full_upper", "initial_position", "initial_positions", "root_transform",
                "moving_links", "curves", "plan_sha256"} | set(derived)
    if set(plan) != expected:
        raise ValueError("Unexpected motion planning receipt fields")
    for key, value in derived.items():
        if isinstance(value, float):
            _number(plan[key], key)
        if type(plan[key]) is not type(value) or plan[key] != value:
            raise ValueError("Motion planning derivation differs: " + key)
    return plan
