"""Validated, geometry-independent articulation contract (metres and radians).

Joint origins are parent-link-local, axes are joint-local, and the root frame
is identity.  The input mesh assembly represents the all-zero joint pose.
"""
from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping

import numpy as np

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_TYPES = {"fixed", "revolute", "continuous", "prismatic"}


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _vector(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain three numbers")
    return [_number(v, label) for v in value]


def _name(value, label):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"{label} must be a URDF-safe name")
    return value


def _object_name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty object name")
    return value


def validate_spec(spec, inventory=None):
    """Return a deep-copied, normalized spec; reject ambiguous or invalid data.

    Inventory, when supplied, requires exact ownership of all source objects
    after optional polygon splits. Splits must partition every source polygon.
    """
    if not isinstance(spec, Mapping):
        raise ValueError("spec must be an object")
    result = copy.deepcopy(dict(spec))
    if type(result.get("schema_version")) is not int or result.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    root = _name(result.get("root"), "root")
    links = result.get("links")
    joints = result.get("joints")
    if not isinstance(links, list) or not links or not isinstance(joints, list):
        raise ValueError("links must be nonempty and joints must be a list")
    names, owners = set(), {}
    for link in links:
        if not isinstance(link, dict):
            raise ValueError("every link must be an object")
        name = _name(link.get("name"), "link.name")
        if name in names:
            raise ValueError(f"duplicate link name: {name}")
        names.add(name)
        objects = link.get("objects")
        if not isinstance(objects, list) or not objects:
            raise ValueError(f"link {name} must own at least one object")
        for obj in objects:
            _object_name(obj, f"{name}.objects")
            if obj in owners:
                raise ValueError(f"object {obj!r} is owned more than once")
            owners[obj] = name
    if root not in names:
        raise ValueError("root must name a link")
    joint_names, children = set(), {}
    for joint in joints:
        if not isinstance(joint, dict):
            raise ValueError("every joint must be an object")
        name = _name(joint.get("name"), "joint.name")
        if name in joint_names:
            raise ValueError(f"duplicate joint name: {name}")
        joint_names.add(name)
        kind = joint.get("type")
        if not isinstance(kind, str) or kind not in _TYPES:
            raise ValueError(f"unsupported joint type on {name}: {kind}")
        parent, child = joint.get("parent"), joint.get("child")
        if not isinstance(parent, str) or not isinstance(child, str) or parent not in names or child not in names or parent == child:
            raise ValueError(f"invalid parent/child on joint {name}")
        if child == root or child in children:
            raise ValueError(f"root cannot have a parent and each child needs one parent: {child}")
        children[child] = joint
        origin = joint.get("origin")
        if not isinstance(origin, dict):
            raise ValueError(f"joint {name} requires origin.xyz and origin.rpy")
        origin["xyz"] = _vector(origin.get("xyz"), f"{name}.origin.xyz")
        origin["rpy"] = _vector(origin.get("rpy"), f"{name}.origin.rpy")
        if kind == "fixed" and ("axis" in joint or "limit" in joint):
            raise ValueError(f"fixed joint {name} must not define axis or limit")
        if kind != "fixed":
            axis = np.asarray(_vector(joint.get("axis"), f"{name}.axis"))
            magnitude = float(np.max(np.abs(axis)))
            if magnitude == 0:
                raise ValueError(f"joint {name} axis must be nonzero")
            # Normalize after scaling to avoid overflow/underflow for finite
            # nonzero directions, without imposing a geometry-dependent cutoff.
            axis = axis / magnitude
            joint["axis"] = (axis / np.linalg.norm(axis)).tolist()
        if kind in {"revolute", "prismatic"}:
            limit = joint.get("limit")
            if not isinstance(limit, dict):
                raise ValueError(f"joint {name} requires lower/upper/effort/velocity limits")
            if set(limit) != {"lower", "upper", "effort", "velocity"}:
                raise ValueError(f"joint {name} requires exactly lower/upper/effort/velocity limits")
            for key in ("lower", "upper", "effort", "velocity"):
                limit[key] = _number(limit.get(key), f"{name}.limit.{key}")
            if not limit["lower"] <= 0 <= limit["upper"] or limit["lower"] >= limit["upper"]:
                raise ValueError(f"joint {name} limits must have positive span and include zero")
            if limit["effort"] <= 0 or limit["velocity"] <= 0:
                raise ValueError(f"joint {name} effort and velocity must be positive")
        elif "limit" in joint:
            limit = joint["limit"]
            if not isinstance(limit, dict):
                raise ValueError(f"joint {name} limit must be an object")
            if set(limit) - {"effort", "velocity"}:
                raise ValueError(f"continuous joint {name} only supports effort/velocity limits")
            for key, value in limit.items():
                limit[key] = _number(value, f"{name}.limit.{key}")
                if key in {"effort", "velocity"} and limit[key] <= 0:
                    raise ValueError(f"joint {name} {key} must be positive")
    if set(children) != names - {root}:
        raise ValueError("every non-root link must have exactly one parent joint")
    for name in names:
        seen, cursor = set(), name
        while cursor != root:
            if cursor in seen:
                raise ValueError(f"kinematic tree contains a cycle at {cursor}")
            seen.add(cursor)
            cursor = children[cursor]["parent"]

    source_objects = None
    if inventory is not None:
        if not isinstance(inventory, Mapping) or not isinstance(inventory.get("objects"), list):
            raise ValueError("inventory.objects must be a list")
        source_objects = {}
        for item in inventory["objects"]:
            if not isinstance(item, Mapping):
                raise ValueError("inventory object must be an object")
            name = _object_name(item.get("name"), "inventory name")
            if name in source_objects:
                raise ValueError(f"duplicate inventory object: {name}")
            source_objects[name] = item
    splits = result.get("splits", [])
    if not isinstance(splits, list):
        raise ValueError("splits must be a list")
    split_sources, split_names = set(), set()
    for split in splits:
        if not isinstance(split, dict):
            raise ValueError("split must be an object")
        source = _object_name(split.get("object"), "split.object")
        if source in split_sources or source in owners:
            raise ValueError(f"split source {source!r} is repeated or still directly owned")
        split_sources.add(source)
        parts = split.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError(f"split {source} requires nonempty parts")
        used = set()
        for part in parts:
            if not isinstance(part, dict):
                raise ValueError("split part must be an object")
            name = _name(part.get("name"), "split part.name")
            if name in split_names or (source_objects is not None and name in source_objects):
                raise ValueError(f"split part name collides: {name}")
            split_names.add(name)
            if name not in owners:
                raise ValueError(f"split part {name} has no owning link")
            indices = part.get("polygon_indices")
            if not isinstance(indices, list) or not indices:
                raise ValueError(f"split part {name} requires polygon_indices")
            for index in indices:
                if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in used:
                    raise ValueError(f"invalid or repeated polygon index on {source}: {index}")
                used.add(index)
        if source_objects is not None:
            if source not in source_objects:
                raise ValueError(f"split source absent from inventory: {source}")
            count = source_objects[source].get("polygon_count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"split source {source} needs a positive polygon_count")
            if used != set(range(count)):
                raise ValueError(f"split {source} must cover every polygon exactly once")
    if source_objects is not None:
        expected = (set(source_objects) - split_sources) | split_names
        if set(owners) != expected:
            raise ValueError(f"object coverage mismatch: missing={sorted(expected-set(owners))}, unknown={sorted(set(owners)-expected)}")
    return result


def validate_motion_refinement(previous_spec, proposed_spec, inventory=None):
    """Validate a proposal without allowing refinement to remove motion coverage.

    Joint names retain their protocol identity across semantic refinement.
    Existing angular joints may switch between revolute and continuous; linear
    joints remain prismatic. Origins, axes, limits, topology, object ownership
    and polygon splits may change under the normal spec contract. New joints
    are allowed and become part of the next round's required coverage.
    """
    previous = validate_spec(previous_spec)
    proposed = validate_spec(proposed_spec, inventory)
    by_name = {joint["name"]: joint for joint in proposed["joints"]}

    def motion_kind(joint):
        return "angular" if joint["type"] in {"revolute", "continuous"} else "linear" if joint["type"] == "prismatic" else "fixed"

    errors = []
    for joint in previous["joints"]:
        if joint["type"] == "fixed":
            continue
        replacement = by_name.get(joint["name"])
        if replacement is None:
            errors.append(f"existing movable joint {joint['name']!r} was removed or renamed")
        elif motion_kind(replacement) != motion_kind(joint):
            errors.append(f"existing movable joint {joint['name']!r} must retain {motion_kind(joint)} motion; got {replacement['type']}")
    if errors:
        raise ValueError("Refinement motion coverage violation: " + "; ".join(errors) +
                         ". Preserve every existing movable joint name and motion kind; repair its origin, axis, limits or topology instead.")
    return proposed


def _pose(origin):
    r, p, y = origin["rpy"]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    out = np.eye(4)
    out[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
    out[:3, 3] = origin["xyz"]
    return out


def fk(spec, positions=None):
    """Forward kinematics for explicit joint coordinates, defaulting to zero."""
    spec = validate_spec(spec)
    positions = dict(positions or {})
    known = {j["name"]: j for j in spec["joints"]}
    if set(positions) - set(known):
        raise ValueError(f"unknown joint coordinates: {sorted(set(positions)-set(known))}")
    frames = {spec["root"]: np.eye(4)}
    pending = list(spec["joints"])
    while pending:
        for joint in pending[:]:
            if joint["parent"] not in frames:
                continue
            q = _number(positions.get(joint["name"], 0.0), f"coordinate {joint['name']}")
            kind = joint["type"]
            if kind == "fixed" and q != 0:
                raise ValueError(f"fixed joint {joint['name']} cannot move")
            if kind in {"revolute", "prismatic"} and not joint["limit"]["lower"]-1e-10 <= q <= joint["limit"]["upper"]+1e-10:
                raise ValueError(f"coordinate outside limits: {joint['name']}")
            motion = np.eye(4)
            if kind == "prismatic":
                motion[:3, 3] = np.asarray(joint["axis"]) * q
            elif kind in {"revolute", "continuous"}:
                x, y, z = joint["axis"]
                cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                motion[:3, :3] = np.eye(3) + math.sin(q)*cross + (1-math.cos(q))*(cross@cross)
            frames[joint["child"]] = frames[joint["parent"]] @ _pose(joint["origin"]) @ motion
            pending.remove(joint)
    return frames


def pose_samples(spec):
    """Rest, isolated limits/midpoints, and coupled poses for nested joints."""
    spec = validate_spec(spec)
    samples = [{"name": "rest", "positions": {}}]
    movable = [j for j in spec["joints"] if j["type"] != "fixed"]
    combined_low, combined_mid, combined_high = {}, {}, {}
    for joint in movable:
        if joint["type"] == "continuous":
            values = [("quarter_turn", math.pi/2), ("half_turn", math.pi)]
            low, mid, high = 0.0, math.pi/2, math.pi
        else:
            low, high = joint["limit"]["lower"], joint["limit"]["upper"]
            mid = (low+high)/2
            values = [("lower", low), ("mid", mid), ("upper", high)]
        samples.extend({"name": f"{joint['name']}:{label}", "positions": {joint["name"]: value}} for label, value in values)
        combined_low[joint["name"]], combined_mid[joint["name"]], combined_high[joint["name"]] = low, mid, high
    if len(movable) > 1:
        samples.extend({"name": name, "positions": positions} for name, positions in [("combined_lower", combined_low), ("combined_mid", combined_mid), ("combined_upper", combined_high)])
    return samples
