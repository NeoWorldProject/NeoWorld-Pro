"""Independent URDF/OBJ parsing and forward-kinematics verification.

This module deliberately reconstructs transforms from XML without calling the
articulation-spec FK implementation, so export errors can be detected.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _values(text, count, label, default=None):
    if text is None:
        if default is None:
            raise ValueError(f"missing {label}")
        return list(default)
    try:
        values = [float(value) for value in text.split()]
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if len(values) != count or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{label} requires {count} finite values")
    return values


def _name(value, label):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _single(parent, tag, required=False):
    nodes = parent.findall(tag)
    if len(nodes) > 1 or (required and not nodes):
        raise ValueError(f"expected {'exactly' if required else 'at most'} one {tag}")
    return nodes[0] if nodes else None


def _origin(element):
    node = _single(element, "origin")
    return {
        "xyz": _values(node.get("xyz") if node is not None else None, 3, "origin xyz", [0.0]*3),
        "rpy": _values(node.get("rpy") if node is not None else None, 3, "origin rpy", [0.0]*3),
    }


def _transform(origin):
    """URDF fixed-axis roll, pitch, yaw: Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = origin["rpy"]
    rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    result = np.eye(4)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = origin["xyz"]
    return result


def read_obj(path):
    """Read finite OBJ vertices and polygon faces (returned indices are zero-based)."""
    path = Path(path)
    vertices, faces, pending, texcoords, normals = [], [], [], 0, 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        tokens = line.split("#", 1)[0].split()
        if not tokens:
            continue
        kind, values = tokens[0], tokens[1:]
        if kind in {"v", "vt", "vn"}:
            try:
                numbers = [float(value) for value in values]
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_number}: nonnumeric {kind}") from exc
            if not all(math.isfinite(v) for v in numbers):
                raise ValueError(f"{path.name}:{line_number}: nonfinite {kind}")
            if kind == "v":
                if len(numbers) not in {3, 4, 6, 7} or (len(numbers) == 4 and numbers[3] != 1):
                    raise ValueError(f"{path.name}:{line_number}: invalid vertex")
                vertices.append(numbers[:3])
            elif kind == "vt":
                if not 1 <= len(numbers) <= 3:
                    raise ValueError(f"{path.name}:{line_number}: invalid texture coordinate")
                texcoords += 1
            else:
                if len(numbers) != 3:
                    raise ValueError(f"{path.name}:{line_number}: invalid normal")
                normals += 1
        elif kind == "f":
            if len(values) < 3:
                raise ValueError(f"{path.name}:{line_number}: face needs three vertices")
            pending.append((line_number, values, (len(vertices), texcoords, normals)))
    final_counts = (len(vertices), texcoords, normals)
    for line_number, tokens, running_counts in pending:
        face = []
        for token in tokens:
            fields = token.split("/")
            if len(fields) > 3 or not fields[0]:
                raise ValueError(f"{path.name}:{line_number}: invalid face reference")
            for field_index, field in enumerate(fields):
                if not field:
                    continue
                try:
                    value = int(field)
                except ValueError as exc:
                    raise ValueError(f"{path.name}:{line_number}: invalid face index") from exc
                resolved = value-1 if value > 0 else running_counts[field_index]+value
                if value == 0 or not 0 <= resolved < final_counts[field_index]:
                    raise ValueError(f"{path.name}:{line_number}: face index out of range")
                if field_index == 0:
                    face.append(resolved)
        if len(set(face)) < 3:
            raise ValueError(f"{path.name}:{line_number}: degenerate face indices")
        faces.append(face)
    if len(vertices) < 3 or not faces:
        raise ValueError(f"{path.name} must contain vertices and faces")
    return {"vertices": np.asarray(vertices, dtype=float), "faces": faces}


def _geometry(element, directory):
    origin = _origin(element)
    geometry = _single(element, "geometry", required=True)
    kinds = [child for child in geometry if child.tag in {"mesh", "box", "sphere", "cylinder"}]
    if len(kinds) != 1 or len(geometry) != 1:
        raise ValueError("geometry must contain exactly one supported shape")
    shape = kinds[0]
    output = {"origin": _transform(origin), "geometry": shape.tag}
    if shape.tag == "mesh":
        filename = shape.get("filename", "")
        source = Path(filename)
        if not filename or source.is_absolute() or ":" in filename or "\\" in filename or ".." in source.parts:
            raise ValueError(f"unsafe mesh path: {filename!r}")
        target = (directory / source).resolve()
        if not target.is_relative_to(directory) or not target.is_file():
            raise ValueError(f"mesh is missing or outside URDF directory: {filename}")
        if target.suffix.lower() != ".obj":
            raise ValueError(f"unsupported mesh format for independent validation: {filename}")
        scale = _values(shape.get("scale"), 3, "mesh scale", [1.0]*3)
        if any(v <= 0 for v in scale):
            raise ValueError("mesh scales must be positive")
        output.update(mesh=target, scale=np.asarray(scale))
    else:
        fields = {"box": [("size", 3)], "sphere": [("radius", 1)], "cylinder": [("radius", 1), ("length", 1)]}
        for name, count in fields[shape.tag]:
            values = _values(shape.get(name), count, f"{shape.tag}.{name}")
            if any(value <= 0 for value in values):
                raise ValueError(f"{shape.tag}.{name} must be positive")
            output[name] = values if count > 1 else values[0]
    return output


def read_urdf(urdf_path):
    """Parse and validate XML, tree, joint fields and local mesh references."""
    path = Path(urdf_path).resolve()
    source = path.read_text(encoding="utf-8")
    if "<!DOCTYPE" in source or "<!ENTITY" in source:
        raise ValueError("URDF document type/entity declarations are not supported")
    robot = ET.fromstring(source)
    if robot.tag != "robot":
        raise ValueError("URDF root element must be robot")
    links, joints = {}, []
    for node in robot.findall("link"):
        name = _name(node.get("name"), "link name")
        if name in links:
            raise ValueError(f"duplicate link: {name}")
        links[name] = {"name": name, "visuals": [_geometry(v, path.parent) for v in node.findall("visual")], "collisions": [_geometry(v, path.parent) for v in node.findall("collision")]}
        inertial = _single(node, "inertial")
        if inertial is not None:
            _origin(inertial)
            mass = _single(inertial, "mass", required=True)
            if _values(mass.get("value"), 1, "mass")[0] <= 0:
                raise ValueError("inertial mass must be positive")
            inertia = _single(inertial, "inertia", required=True)
            values = {key: _values(inertia.get(key), 1, f"inertia {key}")[0] for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
            matrix = np.array([[values["ixx"], values["ixy"], values["ixz"]], [values["ixy"], values["iyy"], values["iyz"]], [values["ixz"], values["iyz"], values["izz"]]])
            if np.min(np.linalg.eigvalsh(matrix)) <= 0:
                raise ValueError("inertia must be positive definite")
    if not links:
        raise ValueError("URDF has no links")
    children, joint_names = {}, set()
    for node in robot.findall("joint"):
        name = _name(node.get("name"), "joint name")
        kind = node.get("type")
        if name in joint_names or kind not in {"fixed", "revolute", "continuous", "prismatic"}:
            raise ValueError(f"duplicate or unsupported joint: {name}")
        joint_names.add(name)
        if node.find("mimic") is not None:
            raise ValueError("mimic joints are outside this articulation contract")
        parent = _single(node, "parent", required=True).get("link")
        child = _single(node, "child", required=True).get("link")
        if parent not in links or child not in links or parent == child or child in children:
            raise ValueError(f"invalid joint parent/child: {name}")
        children[child] = parent
        joint = {"name": name, "type": kind, "parent": parent, "child": child, "origin": _origin(node)}
        if kind == "fixed" and (node.find("axis") is not None or node.find("limit") is not None):
            raise ValueError(f"fixed joint {name} must not define axis or limit")
        if kind != "fixed":
            axis = _values(_single(node, "axis", required=True).get("xyz"), 3, f"{name} axis")
            norm = float(np.linalg.norm(axis))
            if not math.isfinite(norm) or abs(norm-1) > 1e-6:
                raise ValueError(f"joint {name} axis must be nonzero and normalized")
            joint["axis"] = axis
        limit = _single(node, "limit", required=kind in {"revolute", "prismatic"})
        if limit is not None:
            data = {key: _values(value, 1, f"{name} limit {key}")[0] for key, value in limit.attrib.items()}
            if kind in {"revolute", "prismatic"}:
                if set(data) != {"lower", "upper", "effort", "velocity"}:
                    raise ValueError(f"joint {name} requires complete limits")
                if not data["lower"] <= 0 <= data["upper"] or data["lower"] >= data["upper"]:
                    raise ValueError(f"joint {name} limits must include zero and have positive span")
            elif kind == "continuous" and set(data) - {"effort", "velocity"}:
                raise ValueError(f"continuous joint {name} only supports effort/velocity limits")
            if any(data[key] <= 0 for key in ("effort", "velocity") if key in data):
                raise ValueError(f"joint {name} effort/velocity must be positive")
            joint["limit"] = data
        joints.append(joint)
    roots = set(links) - set(children)
    if len(roots) != 1:
        raise ValueError("URDF must have exactly one root")
    root = next(iter(roots))
    for name in links:
        cursor, seen = name, set()
        while cursor != root:
            if cursor in seen or cursor not in children:
                raise ValueError(f"URDF cycle or disconnected link: {name}")
            seen.add(cursor)
            cursor = children[cursor]
    return {"path": path, "root": root, "links": links, "joints": joints}


def urdf_fk(parsed, positions=None):
    """Reconstruct URDF transforms independently using axis-angle quaternions."""
    positions = dict(positions or {})
    if set(positions) - {j["name"] for j in parsed["joints"]}:
        raise ValueError("unknown URDF joint coordinate")
    frames, pending = {parsed["root"]: np.eye(4)}, list(parsed["joints"])
    while pending:
        progressed = False
        for joint in pending[:]:
            if joint["parent"] not in frames:
                continue
            raw_q = positions.get(joint["name"], 0.0)
            if isinstance(raw_q, bool) or not isinstance(raw_q, (int, float)) or not math.isfinite(raw_q):
                raise ValueError("joint coordinate must be finite")
            q = float(raw_q)
            kind = joint["type"]
            if kind == "fixed" and q != 0:
                raise ValueError("fixed joint cannot move")
            if kind in {"revolute", "prismatic"} and not joint["limit"]["lower"]-1e-10 <= q <= joint["limit"]["upper"]+1e-10:
                raise ValueError("joint coordinate outside URDF limits")
            motion = np.eye(4)
            if kind == "prismatic":
                motion[:3, 3] = np.asarray(joint["axis"]) * q
            elif kind in {"continuous", "revolute"}:
                axis = np.asarray(joint["axis"], dtype=float)
                axis /= np.linalg.norm(axis)
                x, y, z = axis * math.sin(q/2)
                w = math.cos(q/2)
                motion[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
            frames[joint["child"]] = frames[joint["parent"]] @ _transform(joint["origin"]) @ motion
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("URDF joints do not form a rooted tree")
    return frames


def validate_export_geometry_contract(parsed, exports):
    """Check the fixed link-local export contract against parsed URDF geometry.

    Unlike general URDF validation, this exporter requires identical visual and
    collision mesh multisets, identity geometry origins and unit mesh scales.
    ``exports`` entries contain ``link`` and ``path``; other fields are ignored.
    """
    errors = []
    links = parsed.get("links") if isinstance(parsed, dict) else None
    if not isinstance(links, dict) or not links:
        return {"passed": False, "errors": ["parsed URDF must contain links"]}
    if not isinstance(exports, list) or not exports:
        return {"passed": False, "errors": ["exports must be a nonempty list"]}
    expected = {link: Counter() for link in links}
    for index, entry in enumerate(exports):
        if not isinstance(entry, dict):
            errors.append(f"export entry {index} must be an object")
            continue
        link, path = entry.get("link"), entry.get("path")
        if not isinstance(link, str) or link not in expected:
            errors.append(f"export entry {index} references an unknown link")
            continue
        if not isinstance(path, (str, Path)) or not str(path):
            errors.append(f"export entry {index} has no mesh path")
            continue
        expected[link][str(Path(path).resolve())] += 1
    for link_name, link in links.items():
        if not isinstance(link, dict):
            errors.append(f"invalid parsed link: {link_name}")
            continue
        for role in ("visuals", "collisions"):
            actual = Counter()
            entries = link.get(role)
            if not isinstance(entries, list):
                errors.append(f"{link_name}.{role} must be a list")
                continue
            for index, entry in enumerate(entries):
                label = f"{link_name}.{role}[{index}]"
                if not isinstance(entry, dict) or entry.get("geometry") != "mesh" or "mesh" not in entry:
                    errors.append(f"{label} must be a mesh")
                    continue
                try:
                    actual[str(Path(entry["mesh"]).resolve())] += 1
                    origin = np.asarray(entry.get("origin"), dtype=float)
                    scale = np.asarray(entry.get("scale"), dtype=float)
                    if origin.shape != (4, 4) or not np.allclose(origin, np.eye(4), rtol=0, atol=1e-9):
                        errors.append(f"{label} origin must be identity for link-local export")
                    if scale.shape != (3,) or not np.allclose(scale, np.ones(3), rtol=0, atol=1e-9):
                        errors.append(f"{label} mesh scale must be one")
                except (ValueError, TypeError, OSError) as exc:
                    errors.append(f"{label} has malformed geometry data: {exc}")
            if actual != expected[link_name]:
                missing = dict(expected[link_name] - actual)
                extra = dict(actual - expected[link_name])
                errors.append(f"{link_name}.{role} mesh multiset differs from export records: missing={missing}, extra={extra}")
    return {"passed": not errors, "errors": errors}


def validate_urdf(urdf_path, spec=None):
    """Return a JSON-serializable report, including optional protocol agreement."""
    report = {"passed": False, "errors": [], "warnings": [], "urdf_path": str(Path(urdf_path).resolve())}
    try:
        parsed = read_urdf(urdf_path)
        meshes = {entry["mesh"] for link in parsed["links"].values() for key in ("visuals", "collisions") for entry in link[key] if "mesh" in entry}
        mesh_reports = []
        for path in sorted(meshes):
            mesh = read_obj(path)
            mesh_reports.append({"path": str(path), "vertex_count": len(mesh["vertices"]), "face_count": len(mesh["faces"])})
        if not any(link["visuals"] for link in parsed["links"].values()):
            raise ValueError("URDF has no visual geometry")
        report.update(root=parsed["root"], link_count=len(parsed["links"]), joint_count=len(parsed["joints"]), meshes=mesh_reports)
        if spec is not None:
            from .kinematics import validate_spec
            normalized = validate_spec(spec)
            if parsed["root"] != normalized["root"] or set(parsed["links"]) != {link["name"] for link in normalized["links"]}:
                raise ValueError("URDF links/root differ from articulation spec")
            actual = {joint["name"]: joint for joint in parsed["joints"]}
            if set(actual) != {joint["name"] for joint in normalized["joints"]}:
                raise ValueError("URDF joint names differ from articulation spec")
            for expected in normalized["joints"]:
                joint = actual[expected["name"]]
                for key in ("type", "parent", "child"):
                    if joint[key] != expected[key]:
                        raise ValueError(f"{joint['name']} {key} differs from articulation spec")
                if not np.allclose(_transform(joint["origin"]), _transform(expected["origin"]), atol=1e-7, rtol=0):
                    raise ValueError(f"{joint['name']} origin differs from articulation spec")
                if expected["type"] != "fixed" and not np.allclose(joint["axis"], expected["axis"], atol=1e-7, rtol=0):
                    raise ValueError(f"{joint['name']} axis differs from articulation spec")
                for key, value in expected.get("limit", {}).items():
                    if key not in joint.get("limit", {}) or not math.isclose(joint["limit"][key], value, abs_tol=1e-7, rel_tol=0):
                        raise ValueError(f"{joint['name']} limit.{key} differs from articulation spec")
        urdf_fk(parsed)
        report["passed"] = True
    except (ValueError, TypeError, KeyError, OSError, ET.ParseError) as exc:
        report["errors"].append(str(exc))
    return report
