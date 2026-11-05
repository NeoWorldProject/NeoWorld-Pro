"""Strict scene contracts. Normalization never invents scene semantics or geometry."""

from copy import deepcopy
import math
import re

from .geometry import obb_to_aabb


SCHEMA_VERSION = "scene_layout.v2"
FRAME_CONSTANTS = {"units": "meters", "handedness": "right", "up_axis": "+Z",
                   "x_axis": "view_right_horizontal", "y_axis": "scene_depth"}
RELATIONS = {"on", "inside", "attached", "unknown", "root"}
_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _vector(value, length):
    return isinstance(value, list) and len(value) == length and all(_number(v) for v in value)


def _nonfinite(data, path="layout"):
    errors = []
    if isinstance(data, float) and not math.isfinite(data):
        errors.append(f"{path}: non-finite numeric value")
    elif isinstance(data, dict):
        for key, value in data.items():
            errors.extend(_nonfinite(value, f"{path}.{key}"))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            errors.extend(_nonfinite(value, f"{path}[{i}]"))
    return errors


def _graph_errors(data, objects):
    errors = []
    tree = data.get("scene_tree")
    if not isinstance(tree, dict):
        return ["scene_tree must be an object mapping names to {children: [...]}."]
    names = {o["name"] for o in objects if isinstance(o, dict) and _text(o.get("name"))}
    parents = {}
    represented = set()
    graph = {}
    for name, node in tree.items():
        if name not in names:
            errors.append(f"scene_tree: unknown parent name {name!r}")
        represented.add(name)
        if not isinstance(node, dict) or not isinstance(node.get("children"), list):
            errors.append(f"scene_tree.{name}: must contain a children list")
            continue
        graph[name] = []
        seen_children = set()
        for child in node["children"]:
            if not _text(child):
                errors.append(f"scene_tree.{name}: child references must be nonempty names")
                continue
            if child not in names:
                errors.append(f"scene_tree.{name}: unknown child name {child!r}")
            if child in seen_children:
                errors.append(f"scene_tree.{name}: duplicate child {child!r}")
            seen_children.add(child)
            represented.add(child)
            if child in parents and parents[child] != name:
                errors.append(f"scene_tree: multiple parents for {child!r}")
            parents[child] = name
            graph[name].append(child)
    for name in sorted(names - represented):
        errors.append(f"scene_tree: object {name!r} is missing; represent standalone roots with an empty children list")
    # Iterative DFS avoids recursion limits on malformed long trees.
    visited, active = set(), set()
    for start in names:
        if start in visited:
            continue
        stack = [(start, False)]
        while stack:
            name, leaving = stack.pop()
            if leaving:
                active.discard(name)
                visited.add(name)
            elif name in active:
                errors.append(f"scene_tree: cycle involving {name!r}")
            elif name not in visited:
                active.add(name)
                stack.append((name, True))
                stack.extend((child, False) for child in graph.get(name, []))
    for obj in objects:
        if not isinstance(obj, dict) or not _text(obj.get("name")):
            continue
        relation, name = obj.get("support_relation"), obj["name"]
        if name in parents and relation == "root":
            errors.append(f"{name}: root support_relation cannot have a parent")
        elif name not in parents and isinstance(relation, str) and relation in {"on", "inside", "attached"}:
            errors.append(f"{name}: support_relation {relation!r} requires a parent")
    return errors


def validate_inventory(data):
    """Validate semantics before assigning IDs; do not require OBBs or scene_frame."""
    if not isinstance(data, dict):
        return ["inventory must be a JSON object"]
    errors = _nonfinite(data)
    objects = data.get("objects_data")
    if not isinstance(objects, list) or not objects:
        return errors + ["objects_data must be a nonempty list"]
    names, ids = set(), set()
    for i, obj in enumerate(objects):
        at = f"objects_data[{i}]"
        if not isinstance(obj, dict):
            errors.append(f"{at} must be an object")
            continue
        name = obj.get("name")
        if not _text(name):
            errors.append(f"{at}.name must be a nonempty string")
        elif name in names:
            errors.append(f"{at}: duplicate name {name!r}")
        else:
            names.add(name)
        if "object_id" in obj:
            ident = obj["object_id"]
            if not _text(ident) or not _ID_RE.fullmatch(ident):
                errors.append(f"{at}.object_id must be a nonempty portable identifier")
            elif ident in ids:
                errors.append(f"{at}: duplicate object_id {ident!r}")
            else:
                ids.add(ident)
        if not _text(obj.get("description")):
            errors.append(f"{at}.description must be a nonempty string")
        for flag in ("is_environment_support", "is_occluded"):
            if not isinstance(obj.get(flag), bool):
                errors.append(f"{at}.{flag} must be an explicit boolean")
        if not isinstance(obj.get("occlusion_reasoning"), str):
            errors.append(f"{at}.occlusion_reasoning must be a string")
        elif obj.get("is_occluded") and not _text(obj["occlusion_reasoning"]):
            errors.append(f"{at}: an occluded object needs occlusion_reasoning")
        relation = obj.get("support_relation")
        if not isinstance(relation, str) or relation not in RELATIONS:
            errors.append(f"{at}.support_relation must be one of {sorted(RELATIONS)}")
        bbox = obj.get("2d_bbox_normalized")
        if "2d_bbox_normalized" not in obj:
            errors.append(f"{at}.2d_bbox_normalized is required (support objects may use null)")
        elif bbox is None and obj.get("is_environment_support") is True:
            pass
        elif not _vector(bbox, 4):
            errors.append(f"{at}.2d_bbox_normalized must be four finite numbers")
        elif not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1):
            errors.append(f"{at}.2d_bbox_normalized must satisfy 0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1")
    return errors + _graph_errors(data, objects)


def _identity_errors(data, previous):
    errors = []
    current_objects = [o for o in data.get("objects_data", []) if isinstance(o, dict)]
    old_objects = [o for o in previous.get("objects_data", []) if isinstance(o, dict)]
    old_by_name = {o["name"]: o.get("object_id") for o in old_objects if _text(o.get("name"))}
    old_by_id = {o["object_id"]: o.get("name") for o in old_objects if _text(o.get("object_id"))}
    registry = previous.get("identity_registry", {})
    new_by_name = {o["name"]: o.get("object_id") if _text(o.get("object_id")) else None
                   for o in current_objects if _text(o.get("name"))}
    for obj in current_objects:
        name, ident = obj.get("name"), obj.get("object_id")
        if _text(name) and name in old_by_name and ident != old_by_name[name]:
            errors.append(f"{name}: existing object_id must remain {old_by_name[name]!r}")
        if _text(ident) and ident in old_by_id and name != old_by_id[ident]:
            errors.append(f"{ident}: existing name must remain {old_by_id[ident]!r}")
        if _text(ident) and ident in registry and ident not in old_by_id:
            errors.append(f"{ident}: retired object_id cannot be reused, including for a reintroduced object")
    expected = {("add", ident, name) for name, ident in new_by_name.items() if name not in old_by_name}
    expected |= {("remove", ident, name) for name, ident in old_by_name.items() if name not in new_by_name}
    changes = data.get("inventory_changes", [])
    if not isinstance(changes, list):
        return errors + ["inventory_changes must be a list"]
    declared = set()
    for i, change in enumerate(changes):
        if not isinstance(change, dict) or not all(_text(change.get(k)) for k in ("action", "object_id", "name", "reason")):
            errors.append(f"inventory_changes[{i}] requires action, object_id, name and nonempty reason")
            continue
        key = (change["action"], change["object_id"], change["name"])
        if key in declared:
            errors.append(f"inventory_changes[{i}]: duplicate change")
        declared.add(key)
        if key not in expected:
            errors.append(f"inventory_changes[{i}]: change does not match an actual addition/removal")
    for action, ident, name in sorted(expected - declared, key=str):
        errors.append(f"inventory_changes: missing explicit {action} record for {name!r} ({ident})")
    return errors


def assign_object_ids(data, previous=None):
    """Copy data and fill absent IDs deterministically; never replace supplied IDs."""
    result = deepcopy(data)
    objects = result.get("objects_data") if isinstance(result, dict) else None
    if not isinstance(objects, list) or any(not isinstance(o, dict) or not _text(o.get("name")) for o in objects):
        raise ValueError("Cannot assign IDs without an objects_data list of named objects")
    names = [o["name"] for o in objects]
    if len(set(names)) != len(names):
        raise ValueError("Cannot assign IDs to duplicate object names")
    old = previous.get("objects_data", []) if isinstance(previous, dict) else []
    by_name = {o["name"]: o["object_id"] for o in old}
    by_id = {o["object_id"]: o["name"] for o in old}
    registry = previous.get("identity_registry", {}) if isinstance(previous, dict) else {}
    used = set(by_id) | set(registry)
    supplied = set()
    for obj in objects:
        if "object_id" in obj:
            ident = obj["object_id"]
            if not _text(ident) or not _ID_RE.fullmatch(ident) or ident in supplied:
                raise ValueError("Invalid or duplicate supplied object_id")
            if obj["name"] in by_name and ident != by_name[obj["name"]]:
                raise ValueError(f"Existing object_id changed for {obj['name']}")
            if ident in by_id and obj["name"] != by_id[ident]:
                raise ValueError(f"Existing name changed for {ident}")
            if ident in registry and ident not in by_id:
                raise ValueError(f"Retired object_id cannot be reused: {ident}")
            supplied.add(ident)
            used.add(ident)
    next_id = 1
    for obj in sorted(objects, key=lambda o: o["name"]):
        if "object_id" in obj:
            continue
        if obj["name"] in by_name:
            obj["object_id"] = by_name[obj["name"]]
        else:
            while f"obj_{next_id:03d}" in used:
                next_id += 1
            obj["object_id"] = f"obj_{next_id:03d}"
            used.add(obj["object_id"])
            next_id += 1
    return result


def validate_layout(data, previous=None):
    """Validate full model output; AABB is optional because code derives it."""
    errors = validate_inventory(data)
    if not isinstance(data, dict):
        return errors
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    frame = data.get("scene_frame")
    if not isinstance(frame, dict):
        errors.append("scene_frame must explicitly describe the shared coordinate frame")
    else:
        for key, value in FRAME_CONSTANTS.items():
            if frame.get(key) != value:
                errors.append(f"scene_frame.{key} must be {value!r}")
        for key in ("origin_description", "scale_reasoning"):
            if not _text(frame.get(key)):
                errors.append(f"scene_frame.{key} must be a nonempty string")
    objects = data.get("objects_data")
    if not isinstance(objects, list):
        return errors
    for i, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        at = f"objects_data[{i}]"
        if "object_id" not in obj:
            errors.append(f"{at}.object_id is required")
        for key in ("geometry_reasoning", "uncertainty"):
            if not _text(obj.get(key)):
                errors.append(f"{at}.{key} must be a nonempty string")
        obb = obj.get("obb")
        if not isinstance(obb, dict):
            errors.append(f"{at}.obb is required, including for environment supports")
            continue
        for key in ("center_m", "size_m", "rotation_xyz_deg"):
            if not _vector(obb.get(key), 3):
                errors.append(f"{at}.obb.{key} must be three finite numbers")
        if _vector(obb.get("size_m"), 3) and any(v <= 0 for v in obb["size_m"]):
            errors.append(f"{at}.obb.size_m values must be positive")
        if "3d_bbox_meters" in obj and not _vector(obj["3d_bbox_meters"], 6):
            errors.append(f"{at}.3d_bbox_meters, if supplied, must be six finite numbers")
    if previous is not None:
        errors.extend(_identity_errors(data, previous))
    return errors


def normalize_layout(data, previous=None):
    """Add IDs/version and derive AABBs; reject invalid content instead of repairing it."""
    result = assign_object_ids(data, previous)
    result.setdefault("schema_version", SCHEMA_VERSION)
    errors = validate_layout(result, previous)
    if errors:
        raise ValueError("Invalid scene layout: " + "; ".join(errors))
    for obj in result["objects_data"]:
        obj["obb"]["rotation_xyz_deg"] = [(angle + 180.0) % 360.0 - 180.0 for angle in obj["obb"]["rotation_xyz_deg"]]
        obj["3d_bbox_meters"] = obb_to_aabb(obj["obb"])
        if not _vector(obj["3d_bbox_meters"], 6):
            raise ValueError(f"Derived AABB is not finite for {obj['object_id']}")
    return result
