"""Auxiliary, source-bound native execution facts; never a semantic verdict.

A missing certificate does not change native acceptance. Vertex coverage verifies
imported geometry association, not topology/material identity or mechanical intent.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shlex
import xml.etree.ElementTree as ET

EXECUTION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return artifact(path)


def _read(path):
    value = json.loads(Path(path).read_text())
    json.dumps(value, allow_nan=False)
    return value


def _matrix(value):
    import numpy as np
    a = np.asarray(value, dtype=float)
    if (a.shape != (4, 4) or not np.isfinite(a).all() or
            not np.allclose(a[3], [0, 0, 0, 1], atol=1e-7) or
            not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(a[:3, :3]), 1, atol=1e-5)):
        raise ValueError("Invalid actual rigid transform")
    return a


def create_input_binding(urdf, spec, path, run_id):
    """Take a source ledger BEFORE queue/import, including material dependencies.

    Unsupported/missing dependencies produce an honest incomplete ledger. Existing
    native input validation remains responsible for actual import failures.
    """
    urdf, spec = Path(urdf).resolve(), Path(spec).resolve()
    records, reasons = {}, []
    def bind(p):
        p = Path(p).resolve()
        if not p.is_file():
            reasons.append("missing_source:" + str(p))
            return None
        records[str(p)] = artifact(p)
        return p
    bind(urdf); bind(spec)
    visuals = []
    try:
        from urdf_core.urdf_validation import read_urdf
        parsed = read_urdf(urdf)
        for name, link in parsed["links"].items():
            for index, visual in enumerate(link["visuals"]):
                if "mesh" not in visual:
                    reasons.append("non_mesh_visual:" + name)
                    continue
                mesh = bind(visual["mesh"])
                visuals.append({"visual_id": f"{name}::{index}", "link": name,
                                "source_mesh": artifact(mesh),
                                "origin": visual["origin"].tolist(),
                                "scale": visual["scale"].tolist()})
        # Include collision dependencies too: the importer reads the whole URDF.
        tree = ET.parse(urdf)
        for node in tree.findall(".//mesh") + tree.findall(".//texture"):
            filename = node.get("filename", "")
            if filename:
                bind(urdf.parent / filename)
        for mesh in list(records):
            if Path(mesh).suffix.lower() != ".obj":
                continue
            for line in Path(mesh).read_text().splitlines():
                tokens = shlex.split(line, comments=True)
                if not tokens or tokens[0] != "mtllib":
                    continue
                for filename in tokens[1:]:
                    material = bind(Path(mesh).parent / filename)
                    if material is None:
                        continue
                    for material_line in material.read_text().splitlines():
                        fields = shlex.split(material_line, comments=True)
                        if fields and (fields[0].lower().startswith("map_") or fields[0].lower() in {"bump", "disp", "decal", "refl"}):
                            if len(fields) < 2:
                                reasons.append("invalid_material_dependency:" + str(material))
                            else:
                                bind(material.parent / fields[-1])
        export = urdf.parent / "export_manifest.json"
        if export.is_file():
            bind(export)
        labels = {}
        if export.is_file():
            ownership = {link["name"]: link.get("objects", []) for link in _read(spec)["links"]}
            for item in _read(export).get("meshes", []):
                mesh = Path(item["path"]).resolve()
                key = (item["link"], str(mesh))
                if key in labels or item.get("object") not in ownership.get(item["link"], []):
                    raise ValueError("Ambiguous authored export ownership")
                labels[key] = item.get("object")
        for visual in visuals:
            visual["authored_object_label"] = labels.get((visual["link"], visual["source_mesh"]["path"]))
    except Exception as exc:
        reasons.append("source_binding_unavailable:" + type(exc).__name__ + ":" + str(exc))
    result = {"schema_version": "isaac_capture_input_binding.v1", "run_id": run_id,
              "status": "verified" if not reasons else "unavailable", "reason_codes": reasons,
              "created_before_import": True, "execution_module_sha256": EXECUTION_SOURCE_SHA256,
              "urdf": records.get(str(urdf)), "spec": records.get(str(spec)),
              "artifacts": list(records.values()), "visuals": visuals}
    return save(path, result)


def verify_input_binding(reference, run_id):
    """Verify the immutable pre-import ledger; callers treat failure as auxiliary."""
    if not isinstance(reference, dict) or sha(reference["path"]) != reference["sha256"]:
        raise ValueError("Capture input binding changed")
    binding = _read(reference["path"])
    if (binding.get("schema_version") != "isaac_capture_input_binding.v1" or
            binding.get("run_id") != run_id or binding.get("created_before_import") is not True):
        raise ValueError("Capture input binding has wrong identity")
    for item in binding["artifacts"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Bound capture source changed: " + item["path"])
    return binding


def match_imported_visuals(binding, expected, imported, body_paths, tolerance):
    """Associate complete imported Mesh points with individual source visuals.

    expected: {visual_id: N x 3 link-local points}; imported: {link: [{path,
    points}]}. Multiple imported prims may represent one source (material splits).
    Ambiguous coincident sources intentionally yield unavailable evidence.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Invalid geometry tolerance")
    by_link = {}
    for row in binding["visuals"]:
        by_link.setdefault(row["link"], []).append(row)
    output, seen_paths = [], set()
    for link, body_path in body_paths.items():
        sources = by_link.get(link, [])
        clouds = {}
        for source in sources:
            points = np.asarray(expected[source["visual_id"]], dtype=float)
            if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
                raise ValueError("Invalid complete source visual points")
            clouds[source["visual_id"]] = points
        assigned = {source["visual_id"]: [] for source in sources}
        for prim in imported.get(link, []):
            p = str(prim["path"])
            if p in seen_paths or not p.startswith(body_path.rstrip("/") + "/"):
                raise ValueError("Imported visual is duplicated or has the wrong rigid body")
            seen_paths.add(p)
            points = np.asarray(prim["points"], dtype=float)
            if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
                raise ValueError("Invalid complete imported visual points")
            candidates = [key for key, source_points in clouds.items()
                          if float(cKDTree(source_points).query(points)[0].max()) <= tolerance]
            if len(candidates) != 1:
                raise ValueError("Imported visual source association is absent or ambiguous: " + p)
            assigned[candidates[0]].append((p, points))
        mesh_rows, all_points = [], []
        for source in sources:
            key = source["visual_id"]
            if not assigned[key]:
                raise ValueError("Source visual omitted from imported rigid body: " + key)
            actual = np.concatenate([points for _, points in assigned[key]])
            source_points = clouds[key]
            distance = max(float(cKDTree(source_points).query(actual)[0].max()),
                           float(cKDTree(actual).query(source_points)[0].max()))
            if distance > tolerance:
                raise ValueError("Imported visual point coverage is incomplete: " + key)
            all_points.append(actual)
            mesh_rows.append({"visual_id": key, "source_mesh": source["source_mesh"],
                              "authored_object_label": source.get("authored_object_label"),
                              "imported_prim_paths": [p for p, _ in assigned[key]],
                              "expected_vertices": len(source_points), "imported_vertices": len(actual),
                              "max_distance_m": distance, "sampled": False,
                              "bound_link_local_aabb_m": [actual.min(axis=0).tolist(), actual.max(axis=0).tolist()]})
        cloud = np.concatenate(all_points) if all_points else None
        output.append({"link": link, "body_path": body_path,
                       "authored_object_labels": sorted({m["authored_object_label"] for m in mesh_rows if m["authored_object_label"]}),
                       "labels_are_not_semantic_ground_truth": True, "visual_meshes": mesh_rows,
                       "bound_link_local_aabb_m": None if cloud is None else [cloud.min(axis=0).tolist(), cloud.max(axis=0).tolist()]})
    if set(by_link) - set(body_paths) or set(imported) - set(body_paths):
        raise ValueError("Imported/source body coverage differs")
    return output


def _frames(manifest):
    rows, refs = [], set()
    for joint in manifest.get("joints", []):
        for view in joint.get("views", []):
            for frame in view.get("frames", []):
                ref = f"{joint['name']}::{view['name']}::{frame['frame_index']}"
                if ref in refs:
                    raise ValueError("Duplicate captured frame reference")
                refs.add(ref)
                rows.append({"frame_ref": ref, "sha256": frame["sha256"],
                             "time_seconds": frame["time_seconds"], "numeric_pose_name": frame["numeric_pose_name"],
                             "numeric_passed": frame["numeric_passed"], "measured_positions": frame["measured_positions"],
                             "link_poses": frame["link_poses"]})
    return rows


def measurement_digest(manifest):
    return hashlib.sha256(json.dumps(_frames(manifest), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_execution_evidence(binding_reference, report, manifest, links, reasons=()):
    """Validate raw measured state against native report and publish no verdict."""
    problems = list(reasons)
    artifacts = []
    binding = None
    published_binding = None
    if binding_reference and Path(binding_reference["path"]).is_file():
        artifacts.append(binding_reference)
        published_binding = binding_reference
    try:
        binding = verify_input_binding(binding_reference, report["run_id"])
        artifacts.extend(binding["artifacts"])
        if binding["status"] != "verified":
            problems.extend(binding["reason_codes"])
    except Exception as exc:
        problems.append("input_binding_unavailable:" + str(exc))
    result = {"schema_version": "isaac_execution_evidence.v1", "status": "unavailable",
              "reason_codes": problems, "run_id": report["run_id"], "mode": "kinematic_playback",
              "execution_module_sha256": EXECUTION_SOURCE_SHA256, "input_binding": published_binding,
              "artifacts": artifacts, "links": links, "frame_state_sha256": None, "frame_count": 0}
    try:
        if not manifest or manifest.get("passed") is not True or report.get("numerical_passed") is not True:
            raise ValueError("Native motion measurement is incomplete")
        for name in ("urdf", "spec"):
            if not binding or manifest[name]["sha256"] != binding[name]["sha256"]:
                raise ValueError("Native capture snapshot differs from pre-import source")
        frames = _frames(manifest)
        if not frames:
            raise ValueError("No measured native frames")
        names = {row["link"] for row in links}
        if names != set(report["body_names"]) or len(names) != len(links):
            raise ValueError("Incomplete imported visual binding")
        poses = {p["name"]: p for p in report["poses"]}
        for frame in frames:
            native = poses[frame["numeric_pose_name"]]
            if (frame["numeric_passed"] is not True or native["passed"] is not True or
                    frame["measured_positions"] != native["measured_positions"] or set(frame["link_poses"]) != names):
                raise ValueError("Captured measurement/report mismatch")
            for name, value in frame["link_poses"].items():
                _matrix(value)
                if native["links"][name]["passed"] is not True or value != native["links"][name]["actual"]:
                    raise ValueError("Actual link pose/report mismatch")
        result.update(frame_state_sha256=measurement_digest(manifest), frame_count=len(frames))
        if not problems:
            result["status"] = "verified"
    except Exception as exc:
        problems.append("measured_execution_unavailable:" + str(exc))
    return result


def review_summary(evidence, manifest, joint_name, frame_refs):
    """Select already-bound measured facts, never commanded positions or FK."""
    if not evidence or evidence.get("status") == "unavailable":
        return None
    if evidence.get("schema_version") != "isaac_execution_evidence.v1" or evidence.get("status") != "verified":
        raise ValueError("Invalid execution evidence declaration")
    if evidence.get("frame_state_sha256") != measurement_digest(manifest):
        raise ValueError("Execution evidence measurement digest differs")
    rows = _frames(manifest)
    if evidence.get("frame_count") != len(rows):
        raise ValueError("Execution frame coverage differs")
    selected = []
    captured = {row["frame_ref"]: row for row in rows}
    names = {link["link"] for link in evidence["links"]}
    if len(names) != len(evidence["links"]):
        raise ValueError("Duplicate evidence body")
    for ref in frame_refs:
        if not isinstance(ref, str) or not ref.startswith(joint_name + "::") or ref not in captured:
            raise ValueError("Selected frame does not belong to this joint")
        frame = captured[ref]
        if frame["numeric_passed"] is not True or set(frame["link_poses"]) != names:
            raise ValueError("Invalid selected measured frame")
        for value in frame["link_poses"].values():
            _matrix(value)
        selected.append({"frame_ref": ref, "actual_joint_positions": frame["measured_positions"],
                         "actual_world_from_link": frame["link_poses"]})
    return {"scope": "Source-bound imported visual point geometry and actual native link execution; no semantic verdict.",
            "not_proven": ["Authored labels identify the intended real-world part", "The proposed axis, pivot and parent are mechanically correct", "All required functional structures/followers belong to the moving body", "Point coverage independently certifies surface topology or material identity"],
            "geometry_evidence_method": "Complete per-source visual points associated with actual imported Mesh prims under their rigid body; full bidirectional point coverage without sampling. Native link transforms are measured at captured frames.",
            "links": evidence["links"], "frames": selected}
