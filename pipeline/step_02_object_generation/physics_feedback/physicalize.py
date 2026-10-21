"""Derive physical URDFs from immutable, validated Blender export packages.

The export manifest reconstructs authored object identity across material files.
Added reverse faces are discarded by provenance, not by geometric guessing.
There is deliberately no whole-link convex hull or bounding-box mass fallback.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from urdf_core.kinematics import validate_spec
from urdf_core.urdf_validation import read_obj
from .protocol import validate_profile, load_config


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _fmt(values):
    return " ".join(format(float(value), ".17g") for value in values)


def _write_obj(path, mesh):
    with Path(path).open("w") as stream:
        stream.write("# Physical geometry, original link-local metre frame.\n")
        for vertex in mesh.vertices:
            stream.write("v " + _fmt(vertex) + "\n")
        for face in mesh.faces:
            stream.write("f " + " ".join(str(int(index) + 1) for index in face) + "\n")


def _coherent_winding(mesh):
    """Orient manifold face neighbors without trimesh's optional networkx."""
    edge_uses = defaultdict(list)
    for index, (a, b, c) in enumerate(np.asarray(mesh.faces)):
        for start, end in ((a, b), (b, c), (c, a)):
            edge_uses[(min(start, end), max(start, end))].append((index, start < end))
    adjacency = defaultdict(list)
    for uses in edge_uses.values():
        if len(uses) == 2:
            (left, direction_a), (right, direction_b) = uses
            different = direction_a == direction_b
            adjacency[left].append((right, different))
            adjacency[right].append((left, different))
    flip = {}
    for start in range(len(mesh.faces)):
        if start in flip:
            continue
        flip[start], queue = False, [start]
        while queue:
            here = queue.pop()
            for neighbor, different in adjacency[here]:
                required = flip[here] != different
                if neighbor in flip:
                    if flip[neighbor] != required:
                        raise ValueError("nonorientable source surface requires geometry repair")
                else:
                    flip[neighbor] = required
                    queue.append(neighbor)
    faces = np.asarray(mesh.faces).copy()
    selected = np.array([flip[i] for i in range(len(faces))], dtype=bool)
    faces[selected] = faces[selected, ::-1]
    mesh.faces = faces
    return int(selected.sum())


def _clean_authored_mesh(vertices, faces):
    """Keep authored solids distinct while removing only surface-neutral defects.

    Adjacent authored solids often have exactly coincident vertices. Global
    welding would turn their valid closed surfaces into nonmanifold four-face
    edges. Source vertex connectivity therefore bounds every permitted weld.
    """
    source = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    keep = source.area_faces != 0
    labels = trimesh.graph.connected_component_labels(source.edges, node_count=len(vertices))
    face_labels = labels[faces[:, 0]]
    parts, welded, repaired_components, untouched_closed, skipped, flipped = [], 0, 0, 0, 0, 0
    for component in np.unique(face_labels):
        selected = faces[(face_labels == component) & keep]
        if not len(selected):
            continue
        used, remap = np.unique(selected, return_inverse=True)
        part = trimesh.Trimesh(vertices=vertices[used], faces=remap.reshape((-1, 3)), process=False)
        if part.is_watertight:
            untouched_closed += 1
        else:
            unique, inverse = np.unique(part.vertices, axis=0, return_inverse=True)
            if len(unique) < len(part.vertices):
                candidate = trimesh.Trimesh(vertices=unique, faces=inverse[part.faces], process=False)
                _, counts = np.unique(candidate.edges_sorted, axis=0, return_counts=True)
                if counts.max() <= 2:
                    welded += len(part.vertices) - len(unique)
                    repaired_components += 1
                    part = candidate
                else:
                    skipped += 1
        # Coherent face winding is a representation repair only; no vertex,
        # triangle support, hole or disconnected solid is changed.
        if part.is_watertight and not part.is_winding_consistent:
            flipped += _coherent_winding(part)
        parts.append(part)
    if not parts:
        raise ValueError("authored object contains no positive-area geometry")
    return trimesh.util.concatenate(parts), {
        "policy": "exact_zero_area_only_and_source_component_preserving_cleanup",
        "removed_zero_area_triangle_indices": np.flatnonzero(~keep).tolist(),
        "welded_coincident_vertex_count": welded, "source_connected_components": int(len(np.unique(face_labels))),
        "closed_components_preserved_without_welding": untouched_closed,
        "open_components_welded": repaired_components, "unsafe_component_welds_skipped": skipped,
        "reoriented_faces": flipped, "positive_area_triangles_removed": 0, "holes_filled": 0}


def reconstruct_objects(export_dir, spec, *, clean=True):
    """Recover one unmodified authored triangle mesh per object from provenance."""
    export_dir = Path(export_dir).resolve()
    manifest = json.loads((export_dir / "export_manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("mesh_frame") != "link_local" or manifest.get("pose") != "q_zero":
        raise ValueError("physicalization requires a link-local q_zero export manifest")
    expected = {(link["name"], name) for link in spec["links"] for name in link["objects"]}
    records = {}
    for entry in manifest.get("meshes", []):
        key = (entry.get("link"), entry.get("object"))
        if key not in expected:
            raise ValueError("export manifest has unknown object ownership")
        original_path = Path(entry["path"])
        # Canonical packages retain original absolute paths after publication.
        # Resolve by package-relative mesh name, never use historical outside data.
        path = export_dir / "meshes" / original_path.name
        if not path.resolve().is_relative_to(export_dir) or not path.is_file():
            raise ValueError("missing or escaping source mesh: " + str(path))
        parsed = read_obj(path)
        vertices, faces = parsed["vertices"], parsed["faces"]
        indices = entry.get("source_vertex_indices")
        triangles = entry.get("source_triangle_vertices")
        reverse = entry.get("backface_source_triangle_indices", [])
        if (not isinstance(indices, list) or len(indices) != len(vertices) or len(set(indices)) != len(indices)
                or not isinstance(triangles, list) or not triangles
                or not isinstance(reverse, list) or len(faces) != len(triangles) + len(reverse)):
            raise ValueError("source mesh reconstruction provenance is incomplete: " + path.name)
        if any(type(i) is not int or not 0 <= i < len(triangles) for i in reverse):
            raise ValueError("invalid reverse-face provenance")
        for face, triangle in zip(faces, triangles):
            if len(face) != 3 or [indices[i] for i in face] != triangle:
                raise ValueError("authored triangle differs from export provenance: " + path.name)
        for face, index in zip(faces[len(triangles):], reverse):
            if [indices[i] for i in face] != list(reversed(triangles[index])):
                raise ValueError("added backface differs from export provenance")
        record = records.setdefault(key, {"vertices": {}, "faces": [], "sources": []})
        for index, vertex in zip(indices, vertices):
            if index in record["vertices"] and not np.allclose(record["vertices"][index], vertex, rtol=0, atol=1e-10):
                raise ValueError("material splits disagree on shared authored vertex")
            record["vertices"][index] = vertex
        record["faces"].extend(triangles)
        record["sources"].append({"path": str(path), "sha256": file_hash(path),
                                  "removed_render_backfaces": len(reverse)})
    if set(records) != expected:
        raise ValueError("export manifest does not cover every source object")
    result = []
    for (link, name), record in records.items():
        ordered = sorted(record["vertices"])
        remap = {index: target for target, index in enumerate(ordered)}
        vertices = np.asarray([record["vertices"][i] for i in ordered], dtype=float)
        faces = np.asarray([[remap[i] for i in face] for face in record["faces"]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if not np.isfinite(vertices).all() or not np.isfinite(mesh.area) or clean and mesh.area <= 0:
            raise ValueError("invalid authored geometry on " + name)
        if clean:
            mesh, cleanup = _clean_authored_mesh(vertices, faces)
        else:
            cleanup = {'policy': 'authored_geometry_without_cleanup'}
        result.append({"link": link, "object": name, "mesh": mesh, "sources": record["sources"],
                       "cleanup": cleanup})
    return result


def validate_profile_geometry(profile, spec, objects):
    """Profile-format validation with measured surfaces, for API debug retries.

    `objects` is the output of reconstruct_objects. This check never makes an
    object-specific physical assumption; a missing shell thickness goes back to
    the model as a contract error before any simulation or semantic refinement.
    """
    result = validate_profile(profile, spec)
    for entry in objects:
        link_params = result["links"][entry["link"]]
        params = link_params["objects"][entry["object"]] if result["schema_version"] == 3 else link_params
        decision = params if result["schema_version"] == 3 else params.get("objects", {}).get(entry["object"], {})
        if decision.get("role") == "decoration" and (result["schema_version"] != 3 or "mass_kg" in params):
            continue
        if not entry["mesh"].is_watertight:
            if ("thickness_m" not in params or not isinstance(params.get("approximation_reason"), str)
                    or not params["approximation_reason"].strip()):
                raise ValueError("open geometry requires API thickness_m and approximation_reason: " + entry["object"])
        if params.get("mass_model") == "surface_shell":
            # A closed envelope can legitimately use a shell mass approximation.
            # Check that the *declared approximation* can be constructed before
            # freezing the profile; do not mistake a vessel opening for an open
            # material surface or silently replace the model's thickness.
            for index, component in enumerate(entry["mesh"].split(only_watertight=False, repair=False)):
                if not component.is_watertight:
                    continue
                try:
                    shell = _thicken_surface(component, params["thickness_m"])
                    props = shell.mass_properties
                    validate_inertia(props['volume'], props['center_mass'], props['inertia'])
                except ValueError as error:
                    raise ProfileGeometryContractError(
                        "Declared surface_shell cannot form a valid material volume; "
                        "the measured source is already watertight. Correct the mass_model/thickness "
                        "using the existing geometry (auto retains its authored volume)",
                        {"code": "invalid_closed_surface_shell_profile", "link": entry["link"],
                         "object": entry["object"], "component_index": index,
                         "source_watertight": bool(entry["mesh"].is_watertight),
                         "component_watertight": True, "thickness_m": params["thickness_m"],
                         "source_export_files": entry["sources"],
                         "surface_error": getattr(error, "diagnostics", {"reason": str(error)})}) from error
        elif params.get('mass_model', 'auto') == 'auto':
            components = list(entry['mesh'].split(only_watertight=False, repair=False))
            if components and all(part.is_watertight for part in components):
                _orient_closed_components(components)
                props = trimesh.util.concatenate(components).mass_properties
                try:
                    validate_inertia(props['volume'], props['center_mass'], props['inertia'])
                except ValueError as error:
                    raise ProfileGeometryContractError(
                        'The auto signed-solid mass assumption does not produce a valid inertia. '
                        'Choose a justified valid surface_shell mass approximation with explicit '
                        'thickness, or report a geometry problem; do not change geometry, mass, '
                        'q_initial or effort limits to hide this representation failure.',
                        {'code': 'invalid_closed_solid_mass_profile', 'link': entry['link'],
                         'object': entry['object'], 'source_watertight': True,
                         'signed_volume_m3': float(props['volume']),
                         'principal_inertias_unit_density': np.linalg.eigvalsh(props['inertia']).tolist(),
                         'source_export_files': entry['sources'], 'reason': str(error)}) from error
    return result


def _surface_mass_info(mesh, params):
    """Integrate a visual-only lamina without inventing normals or a collider.

    An explicitly supplied mass suffices for an ideal thin sheet. Density needs
    an explicit thickness; thickness contributes normal variance to inertia.
    This also supports disconnected/folded print geometry with ambiguous vertex
    normals, because integration uses actual individual triangles.
    """
    triangles = np.asarray(mesh.triangles)
    area = np.asarray(mesh.area_faces)
    total_area = float(area.sum())
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("decorative surface has no finite positive area")
    center = np.average(triangles.mean(axis=1), axis=0, weights=area)
    points = triangles - center
    sums = points.sum(axis=1)
    second = (np.einsum("nki,nkj->nij", points, points) + np.einsum("ni,nj->nij", sums, sums)) / 12
    thickness = params.get("thickness_m")
    if thickness is not None:
        normals = np.asarray(mesh.face_normals)
        second += np.einsum("ni,nj->nij", normals, normals) * float(thickness) ** 2 / 12
    covariance = np.einsum("n,nij->ij", area / total_area, second)
    unit_mass_inertia = np.eye(3) * np.trace(covariance) - covariance
    volume = total_area * float(thickness) if thickness is not None else None
    if "mass_kg" in params:
        mass = float(params["mass_kg"])
    elif volume is not None:
        mass = float(params["density_kg_m3"]) * volume
    else:
        raise ValueError("decorative surface density requires explicit thickness")
    inertia = mass * unit_mass_inertia
    validate_inertia(mass, center, inertia)
    return {"volume_m3": volume, "center": center, "part_mass_kg": mass,
            "part_density_kg_m3": mass / volume if volume is not None else None,
            "part_inertia_kg_m2": inertia, "mass_basis": "authored_triangle_surface_lamina",
            "surface_area_m2": total_area}


def _inside_closed(mesh, point):
    """Solid-angle containment without optional spatial-index dependencies."""
    vectors = np.asarray(mesh.triangles) - np.asarray(point)
    lengths = np.linalg.norm(vectors, axis=2)
    if np.any(lengths < 1e-12):
        raise ValueError("touching disconnected shells have ambiguous mass topology")
    a, b, c = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = (lengths.prod(axis=1)
                   + np.einsum("ij,ij->i", a, b) * lengths[:, 2]
                   + np.einsum("ij,ij->i", b, c) * lengths[:, 0]
                   + np.einsum("ij,ij->i", c, a) * lengths[:, 1])
    return abs(float(np.arctan2(numerator, denominator).sum() * 2)) > 2 * np.pi


def _orient_closed_components(parts):
    """Retain nested cavity shells; orient them by containment parity."""
    for mesh in parts:
        if not mesh.is_winding_consistent:
            _coherent_winding(mesh)
        if mesh.volume < 0:
            mesh.invert()
    nested = False
    for index, mesh in enumerate(parts):
        depth = 0
        for other_index, other in enumerate(parts):
            if index == other_index:
                continue
            if np.all(mesh.bounds[0] >= other.bounds[0]) and np.all(mesh.bounds[1] <= other.bounds[1]):
                # A bbox alone never establishes containment or a cavity.
                sample = mesh.vertices[np.linspace(0, len(mesh.vertices) - 1, min(32, len(mesh.vertices)), dtype=int)]
                inside = [_inside_closed(other, point) for point in sample]
                if any(inside) and not all(inside):
                    raise ValueError("intersecting closed shells need explicit geometry repair")
                depth += int(all(inside))
        if depth % 2:
            mesh.invert()
            nested = True
    return nested


class PhysicalizationGeometryError(ValueError):
    """A physical representation failure with bounded, source-grounded evidence."""
    def __init__(self, message, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(message + "; geometry_diagnostics=" + json.dumps(
            diagnostics, sort_keys=True, allow_nan=False, separators=(",", ":")))


class ProfileGeometryContractError(PhysicalizationGeometryError):
    """An API representation assumption contradicts measured source geometry."""


def physicalization_failure_record(error):
    """Keep representation/contract failures distinct from bad source assets."""
    diagnostics = getattr(error, "diagnostics", None)
    status = "upstream_asset_error"
    if isinstance(error, ProfileGeometryContractError):
        status = "profile_contract_error"
    elif isinstance(diagnostics, dict) and diagnostics.get("code") == "collision_planning_failed":
        status = "collision_generation_error"
    return {"status": status, "passed": False, "error": str(error),
            "geometry_diagnostics": diagnostics, "physics_trial_executed": False}


def _undefined_normal_evidence(source, bad_vertices):
    """Keep component-local indices and the actual adjacent authored surfaces."""
    def finite(values):
        return [float(v) if np.isfinite(v) else None for v in np.asarray(values).reshape(-1)]
    evidence = []
    for vertex_index in bad_vertices[:16]:
        incident = np.flatnonzero((source.faces == vertex_index).any(axis=1))
        faces, weighted = [], np.zeros(3)
        for face_index in incident:
            ids = np.asarray(source.faces[face_index])
            points = np.asarray(source.vertices[ids])
            raw = np.cross(points[1] - points[0], points[2] - points[0])
            length = float(np.linalg.norm(raw))
            normal = raw / length if length else np.zeros(3)
            corner = int(np.flatnonzero(ids == vertex_index)[0])
            a = points[(corner + 1) % 3] - points[corner]
            b = points[(corner + 2) % 3] - points[corner]
            angle = float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))
            weighted += normal * angle
            if len(faces) < 16:
                faces.append({"component_face_index": int(face_index),
                              "component_vertex_indices": ids.tolist(),
                              "vertices_link_local_m": [finite(point) for point in points],
                              "unit_face_normal": finite(normal), "area_m2": length / 2,
                              "corner_angle_rad": angle})
        evidence.append({"component_vertex_index": int(vertex_index),
                         "position_link_local_m": finite(source.vertices[vertex_index]),
                         "computed_vertex_normal": finite(source.vertex_normals[vertex_index]),
                         "independent_angle_weighted_normal_sum": finite(weighted),
                         "incident_face_count": len(incident), "incident_faces": faces,
                         "incident_faces_truncated": len(incident) > len(faces)})
    return {"code": "undefined_surface_shell_vertex_normal",
            "invalid_vertex_count": len(bad_vertices),
            "invalid_component_vertex_indices": bad_vertices[:16].tolist(),
            "vertices": evidence, "vertices_truncated": len(bad_vertices) > len(evidence),
            "unused_vertex_count": len(source.vertices) - len(np.unique(source.faces)),
            "minimum_triangle_area_m2": float(source.area_faces.min())}


def _split_boundary_vertex_fans(source):
    """Separate disconnected surface fans at a pinched boundary vertex.

    Triangle positions and support stay identical. Sharing a vertex alone does
    not join two sheet boundaries into a manifold material wall.
    """
    edges = source.edges_sorted
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    boundary = source.edges[counts[inverse] == 1]
    vertices, degrees = np.unique(boundary, return_counts=True)
    suspects = vertices[degrees > 2]
    if not len(suspects):
        return source, 0
    faces = np.asarray(source.faces).copy()
    points = source.vertices.tolist()
    added = 0
    for vertex in suspects:
        incident = np.flatnonzero((faces == vertex).any(axis=1))
        edge_faces = defaultdict(list)
        for face_index in incident:
            for neighbor in faces[face_index]:
                if neighbor != vertex:
                    edge_faces[int(neighbor)].append(int(face_index))
        adjacent = defaultdict(set)
        for uses in edge_faces.values():
            if len(uses) == 2:
                a, b = uses
                adjacent[a].add(b)
                adjacent[b].add(a)
        remaining = set(map(int, incident))
        first = True
        while remaining:
            start = min(remaining)
            group, todo = set(), [start]
            while todo:
                here = todo.pop()
                if here in group:
                    continue
                group.add(here)
                todo.extend(adjacent[here] - group)
            remaining -= group
            if first:
                first = False
                continue
            replacement = len(points)
            points.append(source.vertices[vertex].tolist())
            for face_index in group:
                faces[face_index, faces[face_index] == vertex] = replacement
            added += 1
    return trimesh.Trimesh(vertices=points, faces=faces, process=False), added


def _thicken_surface(mesh, thickness):
    """Explicit centered surface-shell approximation with closed boundary walls."""
    source = mesh.copy()
    if not source.is_winding_consistent:
        _coherent_winding(source)
    if source.is_watertight and source.volume < 0:
        source.invert()
    edges = source.edges_sorted
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("nonmanifold open geometry requires geometry repair")
    source, split_vertices = _split_boundary_vertex_fans(source)
    # The fan split changes indices only; recompute boundary incidence before
    # creating walls and normals for each separate sheet fan.
    edges = source.edges_sorted
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    normals = np.asarray(source.vertex_normals)
    bad = np.flatnonzero(~np.isfinite(normals).all(axis=1) | (np.linalg.norm(normals, axis=1) < .9))
    if len(bad):
        raise PhysicalizationGeometryError("surface-shell normals are undefined",
                                            _undefined_normal_evidence(source, bad))
    n = len(source.vertices)
    vertices = np.vstack((source.vertices + normals * thickness / 2,
                          source.vertices - normals * thickness / 2))
    if source.is_watertight:
        # A valid closed shell is an outward solid minus an inward solid, using
        # the same source orientation. Positive total shell volume alone is not
        # sufficient: an over-thick inward offset can invert and be added as
        # positive material instead. This is representation validity, not a
        # proof of self-intersection freedom or a functional-fidelity test.
        outer = trimesh.Trimesh(vertices=vertices[:n], faces=source.faces, process=False)
        inner = trimesh.Trimesh(vertices=vertices[n:], faces=source.faces, process=False)
        volumes = [float(inner.volume), float(source.volume), float(outer.volume)]
        if not np.isfinite(volumes).all() or not 0 < volumes[0] < volumes[1] < volumes[2]:
            raise PhysicalizationGeometryError("closed surface-shell offset volume inverted or collapsed", {
                "code": "closed_shell_offset_volume_invalid", "thickness_m": float(thickness),
                "inner_offset_signed_volume_m3": volumes[0] if np.isfinite(volumes[0]) else None,
                "source_signed_volume_m3": volumes[1] if np.isfinite(volumes[1]) else None,
                "outer_offset_signed_volume_m3": volumes[2] if np.isfinite(volumes[2]) else None,
                "required_order": "0 < inner_offset_volume < source_volume < outer_offset_volume"})
    faces = [*source.faces.tolist(), *(source.faces[:, ::-1] + n).tolist()]
    for edge in source.edges[counts[inverse] == 1]:
        a, b = map(int, edge)
        faces.extend([[b, a, a + n], [b, a + n, b + n]])
    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not result.is_watertight or not result.is_winding_consistent:
        raise ValueError("surface-shell construction did not produce a closed oriented volume")
    if result.volume < 0:
        result.invert()
    if not np.isfinite(result.volume) or result.volume <= 1e-18:
        raise ValueError("surface-shell approximation has no physical volume")
    result.metadata['surface_shell_representation'] = {
        'boundary_vertex_fan_splits': split_vertices,
        'authored_triangle_positions_changed': False,
        'authored_triangles_removed': 0,
        'holes_filled': 0}
    return result


def validate_inertia(mass, center, inertia):
    matrix = np.asarray(inertia, dtype=float)
    if (not np.isfinite(mass) or mass <= 0 or np.asarray(center).shape != (3,)
            or not np.isfinite(center).all() or matrix.shape != (3, 3)
            or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-14)):
        raise ValueError("invalid mass, center of mass, or inertia tensor")
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = max(float(eigenvalues[-1]) * 1e-9, 1e-18)
    if eigenvalues[0] <= 0 or eigenvalues[-1] > eigenvalues[:2].sum() + tolerance:
        raise ValueError("inertia must be positive definite and obey principal triangle inequalities")
    return eigenvalues.tolist()


def physicalize_asset(base_run: Path, profile: dict, out_dir: Path, *, config=None, collision_cache_root=None) -> dict:
    """Write a new self-contained package and its content-bound lineage manifest."""
    base_run, out_dir = Path(base_run).resolve(), Path(out_dir).resolve()
    export_dir = base_run / "urdf_export"
    spec_path = base_run / "kinematics.json"
    source_urdf = export_dir / "object.urdf"
    if out_dir == base_run or out_dir.is_relative_to(export_dir) or base_run.is_relative_to(out_dir):
        raise ValueError("derived asset directory must not replace or nest within the base export")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("physical asset output must be empty; prior revisions are immutable")
    spec = validate_spec(json.loads(spec_path.read_text()))
    objects = reconstruct_objects(export_dir, spec)
    profile = validate_profile_geometry(profile, spec, objects)
    settings = load_config(config)
    lightweight = profile["schema_version"] == 3
    adaptive = lightweight or profile["collision"]["approximation"] == "adaptive"
    root = ET.parse(source_urdf).getroot()
    if {node.get("name") for node in root.findall("link")} != {link["name"] for link in spec["links"]}:
        raise ValueError("URDF link identities differ from the shared kinematic protocol")
    # All physical calculations finish before a package is created.
    link_parts, prepared = defaultdict(list), []
    for entry in objects:
        source_mesh = entry["mesh"]
        link_params = profile["links"][entry["link"]]
        params = link_params["objects"][entry["object"]] if lightweight else link_params
        decision = params if lightweight else params.get("objects", {}).get(entry["object"], {"role": "contact"})
        if decision["role"] == "decoration" and not lightweight:
            prepared.append({**entry, "physical_parts": [], "collision_parts": [], "proxies": [],
                             "mass_basis": "included_in_explicit_total_link_mass", "decision": decision})
            continue
        # trimesh defaults to repair=True and may fill small holes even when
        # only_watertight=False. Such silent geometry changes are forbidden.
        components = list(source_mesh.split(only_watertight=False, repair=False))
        if not components:
            raise ValueError("source object has no connected geometry")
        closed = all(part.is_watertight for part in components)
        if lightweight and decision["role"] == "decoration" and not closed:
            info = {**entry, **_surface_mass_info(source_mesh, params), "physical_parts": [],
                    "collision_parts": [], "proxies": [], "decision": decision,
                    "nested_cavity_shells": False}
            link_parts[entry["link"]].append(info)
            prepared.append(info)
            continue
        shell = params.get("mass_model", "auto") == "surface_shell" or not closed
        if shell:
            if "thickness_m" not in params or not isinstance(params.get("approximation_reason"), str) or not params["approximation_reason"].strip():
                raise ValueError("open geometry requires API thickness_m and approximation_reason: " + entry["object"])
            physical_parts = []
            for component_index, part in enumerate(components):
                try:
                    physical_parts.append(_thicken_surface(part, params["thickness_m"]))
                except ValueError as error:
                    context = {"code": "surface_shell_construction_failed",
                               "link": entry["link"], "object": entry["object"],
                               "component_index": component_index,
                               "component_indexing": "zero_based_after_source_component_split",
                               "coordinate_frame": "link_local_metres",
                               "component_vertex_count": len(part.vertices),
                               "component_triangle_count": len(part.faces),
                               "open_surface": not bool(part.is_watertight),
                               "mass_model": params["mass_model"], "thickness_m": params["thickness_m"],
                               "auto_policy": "auto uses closed solids; open surfaces still require explicit thickness and surface-shell construction",
                               "source_export_files": entry["sources"],
                               "surface_error": getattr(error, "diagnostics", {"reason": str(error)})}
                    message = (f"surface-shell physicalization failed for link={entry['link']} "
                               f"object={entry['object']} component={component_index}: "
                               + str(error).split("; geometry_diagnostics=", 1)[0])
                    raise PhysicalizationGeometryError(message, context) from error
            mass_mesh = trimesh.util.concatenate(physical_parts)
            nested = False
            basis = "explicit_centered_surface_shell"
        else:
            nested = _orient_closed_components(components)
            mass_mesh = trimesh.util.concatenate(components)
            physical_parts = [mass_mesh] if nested else components
            basis = "closed_authored_volume_with_cavity_parity"
        props = mass_mesh.mass_properties
        volume, center, inertia = float(props["volume"]), np.asarray(props["center_mass"]), np.asarray(props["inertia"])
        validate_inertia(volume, center, inertia)
        info = {**entry, "physical_parts": physical_parts, "volume_m3": volume,
                "center": center, "unit_density_inertia": inertia, "mass_basis": basis,
                "nested_cavity_shells": nested, "decision": decision}
        if shell:
            info['surface_shell_representation'] = [
                part.metadata['surface_shell_representation'] for part in physical_parts]
        if lightweight:
            # mass_model is a mass approximation, not permission to expand an
            # already closed contact surface. Open surfaces still need the
            # declared thickness to produce a finite collision solid.
            closed_reference = [part.copy() for part in components if part.is_watertight]
            if closed_reference:
                collision_reference = closed_reference
                collision_nested = _orient_closed_components(collision_reference)
                if collision_nested:
                    collision_reference = [trimesh.util.concatenate(collision_reference)]
                collision_reference.extend(physical_parts[index] for index, part in enumerate(components)
                                           if not part.is_watertight)
                collision_basis = ("closed_authored_volume_with_cavity_parity" if closed else
                                   "closed_authored_components_and_explicit_open_shells")
            else:
                collision_reference = physical_parts
                collision_basis = "explicit_centered_surface_shell"
            info["collision_basis"] = collision_basis
            density = float(params.get("density_kg_m3", params.get("mass_kg", 0) / volume))
            info.update(part_mass_kg=density * volume, part_density_kg_m3=density,
                        part_inertia_kg_m2=inertia * density)
            if decision["role"] == "decoration":
                info.update(collision_parts=[], proxies=[])
            else:
                from .lightweight_collision import plan_lightweight_collisions
                try:
                    if collision_cache_root is None:
                        refs, proxies, audit = plan_lightweight_collisions(collision_reference, config=settings)
                    else:
                        from .collision_reuse import plan_cached_collisions
                        refs, proxies, audit = plan_cached_collisions(collision_reference, config=settings,
                            cache_root=collision_cache_root, identity={'link': entry['link'], 'object': entry['object']})
                except (ValueError, RuntimeError) as error:
                    raise PhysicalizationGeometryError("lightweight collision planning failed: " + str(error),
                        {"link": entry["link"], "object": entry["object"], "code": "collision_planning_failed",
                         "collision_error": getattr(error, "diagnostics", {"reason": str(error)})}) from error
                info.update(collision_parts=refs, proxies=proxies, collision_audit=audit)
        elif adaptive:
            from .collision_planning import plan_collisions
            try:
                refs, proxies, audit = plan_collisions(physical_parts, decision,
                    settings["penetration_tolerance_m"], settings["geometry_tolerance_m"])
            except ValueError as error:
                raise PhysicalizationGeometryError("collision planning failed: " + str(error),
                    {"link": entry["link"], "object": entry["object"], "code": "collision_planning_failed"}) from error
            info.update(collision_parts=refs, proxies=proxies, collision_audit=audit)
        else:
            info["collision_parts"] = physical_parts
        link_parts[entry["link"]].append(info)
        prepared.append(info)
    if set(link_parts) != {link["name"] for link in spec["links"]}:
        raise ValueError("every rigid link must retain physical mass")
    if lightweight and not any(p["collision_parts"] for p in prepared):
        raise ValueError("physical asset needs at least one functional contact part")
    for entry in prepared:
        if lightweight or entry["decision"]["role"] != "decoration":
            continue
        from .collision_planning import check_decoration_support
        host = next(p for p in prepared if p["link"] == entry["link"] and p["object"] == entry["decision"]["represented_by"])
        try:
            entry["support_check"] = check_decoration_support(entry["mesh"], host["physical_parts"],
                settings["penetration_tolerance_m"], settings["geometry_tolerance_m"])
        except ValueError as error:
            raise PhysicalizationGeometryError("decoration support check failed: " + str(error),
                {"link": entry["link"], "object": entry["object"], "host": host["object"],
                 "code": "decoration_host_coverage_failed"}) from error
    link_properties = {}
    for name, parts in link_parts.items():
        if lightweight:
            mass = sum(p["part_mass_kg"] for p in parts)
            center = sum(p["center"] * p["part_mass_kg"] for p in parts) / mass
            inertia = np.zeros((3, 3))
            for part in parts:
                delta = part["center"] - center
                inertia += part["part_inertia_kg_m2"] + part["part_mass_kg"] * (
                    np.eye(3) * np.dot(delta, delta) - np.outer(delta, delta))
            eigenvalues = validate_inertia(mass, center, inertia)
            volumes_known = all(p["volume_m3"] is not None for p in parts)
            volume = sum(p["volume_m3"] for p in parts) if volumes_known else None
            link_properties[name] = {"mass_kg": mass,
                "density_kg_m3": mass / volume if volume is not None else None,
                "volume_m3": volume, "center_of_mass_m": center.tolist(),
                "inertia_kg_m2": inertia.tolist(), "principal_inertias_kg_m2": eigenvalues,
                "aggregation": "per_authored_part_mass_weighted_parallel_axis",
                "objects": [{"name": p["object"], "mass_kg": p["part_mass_kg"],
                             "density_kg_m3": p["part_density_kg_m3"], "volume_m3": p["volume_m3"],
                             "center_of_mass_m": p["center"].tolist(),
                             "inertia_kg_m2": p["part_inertia_kg_m2"].tolist(),
                             "mass_basis": p["mass_basis"], "role": p["decision"]["role"]} for p in parts]}
            continue
        volume = sum(part["volume_m3"] for part in parts)
        params = profile["links"][name]
        density = params.get("density_kg_m3", params.get("mass_kg", 0) / volume)
        mass = density * volume
        center = sum(part["center"] * part["volume_m3"] for part in parts) / volume
        inertia = np.zeros((3, 3))
        for part in parts:
            delta = part["center"] - center
            inertia += density * (part["unit_density_inertia"] + part["volume_m3"] * (np.eye(3) * np.dot(delta, delta) - np.outer(delta, delta)))
        eigenvalues = validate_inertia(mass, center, inertia)
        link_properties[name] = {"mass_kg": mass, "density_kg_m3": density,
                                  "volume_m3": volume, "center_of_mass_m": center.tolist(),
                                  "inertia_kg_m2": inertia.tolist(), "principal_inertias_kg_m2": eigenvalues,
                                  "aggregation": "additive_authored_objects_parallel_axis",
                                  "objects": [{"name": p["object"], "volume_m3": p["volume_m3"], "mass_basis": p["mass_basis"]} for p in parts]}
        decorations = [p["object"] for p in prepared if p["link"] == name and p["decision"]["role"] == "decoration"]
        if decorations:
            link_properties[name]["decoration_mass_approximation"] = {
                "objects": decorations, "total_mass_source": "API_explicit_link_mass_kg",
                "distribution": "total_link_mass_distributed_over_retained_physical_solids",
                "separate_label_volume_or_inertia_invented": False}
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(export_dir / "meshes", out_dir / "meshes")
    (out_dir / "physical_meshes").mkdir()
    shutil.copy2(spec_path, out_dir / "kinematics.json")
    _write_json(out_dir / "physics_profile.json", profile)
    manifest = {"schema_version": 1, "urdf_path": str(out_dir / "object.urdf"),
                "profile_path": str(out_dir / "physics_profile.json"), "spec_path": str(out_dir / "kinematics.json"),
                "source": {"base_run": str(base_run), "urdf_sha256": file_hash(source_urdf),
                           "spec_sha256": file_hash(spec_path),
                           "export_manifest_sha256": file_hash(export_dir / "export_manifest.json")},
                "geometry_meshes": [], "collision_meshes": [], "link_properties": link_properties,
                "collision_policy": profile["collision"],
                "object_physics": [{"link": p["link"], "object": p["object"], "decision": p["decision"],
                                    **({"support_check": p["support_check"]} if "support_check" in p else {})}
                                   for p in prepared],
                "joint_parameter_semantics": {"damping": "SI passive viscous damping; rotational USD conversion applied by runtime",
                                               "friction": "zero_only_until_URDF_SI_to_PhysX_coefficient_mapping_is_verified"},
                "proxy_validation": {"status": "requires_runtime_cooked_proxy_validation",
                                     "source_geometry": "manifest_reconstructed_authored_objects",
                                     "coverage_policy": "no_mesh_shrinking_no_whole_link_convex_hull",
                                     "requested_approximation": profile["collision"]["approximation"]}}
    from urdf_core.degraded_asset import degradation_records
    manifest['upstream_degradations'] = degradation_records(base_run)
    for entry in prepared:
        changes = [row for row in entry.get('collision_audit', {}).get('components', [])
                   if row.get('geometry_degraded')]
        if changes:
            manifest['upstream_degradations'].append({
                'policy': 'coacd_native_crash_component_fallback.v1',
                'link': entry['link'], 'object': entry['object'],
                'visual_geometry_changed': False, 'components': changes})
    if adaptive:
        manifest["collision_proxies"] = []
        manifest["collision_reduction"] = [{"link": p["link"], "object": p["object"], **p["collision_audit"]}
                                           for p in prepared if "collision_audit" in p]
        if not lightweight:
            manifest["proxy_validation"]["frozen_tolerance_m"] = settings["penetration_tolerance_m"]
    if lightweight:
        manifest["physicalization_schema_version"] = 3
        manifest["proxy_validation"] = {"status": "not_required_functional_policy", "tested": False,
                                        "requested_approximation": "coacd"}
        manifest["joint_parameter_semantics"] = {
            "damping": "SI passive viscous damping; rotational USD conversion applied by runtime",
            "static_friction_effort": "native axis static friction, N or N*m; not URDF Coulomb friction",
            "dynamic_friction_effort": "native axis dynamic friction, N or N*m; not URDF Coulomb friction"}
    inventory_path = base_run / "geometry_inventory" / "inventory.json"
    if inventory_path.is_file():
        manifest["source"]["inventory_sha256"] = file_hash(inventory_path)
    collision_by_link = defaultdict(list)
    for index, entry in enumerate(prepared):
        clean_path = out_dir / "physical_meshes" / f"object_{index:04d}_original.obj"
        _write_obj(clean_path, entry["mesh"])
        manifest["geometry_meshes"].append({"link": entry["link"], "object": entry["object"],
                                            "path": str(clean_path), "sha256": file_hash(clean_path),
                                            "watertight": bool(entry["mesh"].is_watertight),
                                            "cleanup": entry["cleanup"],
                                            "surface_shell_representation": entry.get("surface_shell_representation", []),
                                            "sources": entry["sources"]})
        if lightweight:
            manifest["geometry_meshes"][-1].update(role=entry["decision"]["role"],
                                                   collision_enabled=entry["decision"]["role"] == "contact",
                                                   collision_basis=entry.get("collision_basis", "none_decoration"))
        for part_index, part in enumerate(entry["collision_parts"]):
            path = out_dir / "physical_meshes" / f"object_{index:04d}_collision_{part_index:04d}.obj"
            _write_obj(path, part)
            if not adaptive:
                collision_by_link[entry["link"]].append({"representation": "convexDecomposition", "path": path.relative_to(out_dir).as_posix()})
            manifest["collision_meshes"].append({"link": entry["link"], "object": entry["object"],
                                                 "path": str(path), "sha256": file_hash(path),
                                                 "watertight": bool(part.is_watertight),
                                                 "approximation": "lightweight_proxy" if lightweight else "reference_solid" if adaptive else "convexDecomposition",
                                                 "mass_basis": entry["mass_basis"],
                                                 **({"collision_basis": entry["collision_basis"]} if lightweight else {})})
        if adaptive:
            for part_index, proxy in enumerate(entry["proxies"]):
                path = out_dir / "physical_meshes" / f"object_{index:04d}_proxy_{part_index:04d}.obj"
                _write_obj(path, proxy["mesh"])
                record = {"link": entry["link"], "object": entry["object"], "representation": proxy["representation"],
                          "path": str(path), "sha256": file_hash(path),
                          "triangles": len(proxy["mesh"].faces), "vertices": len(proxy["mesh"].vertices)}
                if "box" in proxy:
                    record["box"] = proxy["box"]
                if "source_component_index" in proxy:
                    record["source_component_index"] = proxy["source_component_index"]
                manifest["collision_proxies"].append(record)
                collision_by_link[entry["link"]].append({**record, "path": path.relative_to(out_dir).as_posix()})
    for link in root.findall("link"):
        name = link.get("name")
        for tag in ("inertial", "collision"):
            for node in link.findall(tag):
                link.remove(node)
        values = link_properties[name]
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": _fmt(values["center_of_mass_m"]), "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": _fmt([values["mass_kg"]])})
        tensor = np.asarray(values["inertia_kg_m2"])
        ET.SubElement(inertial, "inertia", {field: _fmt([tensor[i, j]]) for field, i, j in
                                             (("ixx", 0, 0), ("iyy", 1, 1), ("izz", 2, 2), ("ixy", 0, 1), ("ixz", 0, 2), ("iyz", 1, 2))})
        for proxy in collision_by_link[name]:
            collision = ET.SubElement(link, "collision")
            geometry = ET.SubElement(collision, "geometry")
            if proxy["representation"] == "box":
                box = proxy["box"]
                ET.SubElement(collision, "origin", {"xyz": _fmt(box["center_m"]), "rpy": _fmt(box["rpy_rad"])})
                ET.SubElement(geometry, "box", {"size": _fmt(box["size_m"])})
            else:
                ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
                ET.SubElement(geometry, "mesh", {"filename": proxy["path"], "scale": "1 1 1"})
    for joint in root.findall("joint"):
        params = profile["joints"].get(joint.get("name"))
        if params is not None:
            for node in joint.findall("dynamics"):
                joint.remove(node)
            dynamics = {"damping": _fmt([params["damping"]]), "friction": "0"} if lightweight else {
                key: _fmt([params[key]]) for key in ("damping", "friction")}
            ET.SubElement(joint, "dynamics", dynamics)
    ET.indent(root)
    ET.ElementTree(root).write(out_dir / "object.urdf", encoding="utf-8", xml_declaration=True)
    manifest["urdf_sha256"] = file_hash(out_dir / "object.urdf")
    manifest["profile_sha256"] = file_hash(out_dir / "physics_profile.json")
    manifest["artifacts"] = {path.relative_to(out_dir).as_posix(): file_hash(path)
                              for path in sorted(out_dir.rglob("*")) if path.is_file()}
    _write_json(out_dir / "asset_manifest.json", manifest)
    return manifest
