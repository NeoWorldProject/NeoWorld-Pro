"""Function-preserving component proxies; no visual edits or silent cavity fill.

Mass always comes from the original physical solids. Collision reduction is a
separate operation, with source references and certificates retained for PhysX.
"""
from __future__ import annotations

import numpy as np
import trimesh
import warnings
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .geometry_checks import _Uncertain, _fidelity_meshes, _signed_distance, validate_proxy_representation


def mesh_record(mesh):
    return {"vertices": np.asarray(mesh.vertices), "faces": np.asarray(mesh.faces)}


def _hull_mesh(points):
    """Keep Qhull's vertices exactly; avoid a second mesh library cleanup pass."""
    points = np.asarray(points, dtype=float)
    local = points - points.mean(0)
    hull = ConvexHull(local)
    faces = hull.simplices.copy()
    tri = local[faces]
    inward = np.einsum("ij,ij->i", np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), hull.equations[:, :3]) < 0
    faces[inward] = faces[inward, ::-1]
    result = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    result.remove_unreferenced_vertices()
    return result


def convex_hull_if_exact(mesh):
    """A hull may reduce coplanar tessellation only after proving convexity."""
    try:
        checked = _fidelity_meshes([mesh_record(mesh)], "collision_plan")[0]
    except _Uncertain as error:
        raise ValueError("physical collision source is not certifiable: " + str(error)) from error
    if checked.convex_equations is None:
        return None
    return _hull_mesh(mesh.vertices)


def box_mesh(box):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", box["rpy_rad"]).as_matrix()
    pose[:3, 3] = box["center_m"]
    return trimesh.creation.box(extents=box["size_m"], transform=pose)


def exact_box(mesh):
    """Recognize oriented boxes, without turning bevels or cavities into boxes."""
    bounds = mesh.bounding_box_oriented
    pose = bounds.primitive.transform
    with warnings.catch_warnings():
        # At Euler singularities SciPy chooses an equivalent orientation.
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz").tolist()
    box = {"size_m": bounds.primitive.extents.tolist(), "center_m": pose[:3, 3].tolist(), "rpy_rad": rpy}
    candidate = box_mesh(box)
    proof = validate_proxy_representation({"part": [mesh_record(mesh)]},
                                         {"part": [mesh_record(candidate)]}, 1e-7,
                                         geometry_precision_m=1e-10, max_cells=64)
    return (box, candidate) if proof["passed"] else (None, None)


def split_convex(mesh, limit=32):
    """Exact half-space partition to avoid native hull vertex/plane reduction.

    Each cut uses actual edge-plane intersections. Disjoint half-spaces plus
    containment and volume conservation certify their union, without replacing
    an enclosing concave mesh by its hull. Runtime still checks each cooked part.
    """
    def volume(part):
        # Integrate near the solid, not the link origin. Small translated parts
        # otherwise suffer cancellation large enough to reject an exact cut.
        tri = part.triangles - part.vertices.mean(0)
        return abs(float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6))

    queue, parts, cuts = [(mesh, 0)], [], []
    while queue:
        part, depth = queue.pop()
        if len(part.vertices) <= limit:
            parts.append(part)
            continue
        if depth >= 20 or len(parts) + len(queue) > 512:
            raise ValueError("exact convex partition exceeded its bounded work budget")
        vertices = np.asarray(part.vertices)
        edges = part.edges_unique
        candidates = []
        for axis in range(3):
            plane = float((part.bounds[0, axis] + part.bounds[1, axis]) / 2)
            a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
            selected = (a[:, axis] - plane) * (b[:, axis] - plane) < 0
            a, b = a[selected], b[selected]
            hits = a + ((plane - a[:, axis]) / (b[:, axis] - a[:, axis]))[:, None] * (b - a)
            children = []
            for keep in (vertices[:, axis] <= plane, vertices[:, axis] >= plane):
                points = np.unique(np.vstack((vertices[keep], hits)), axis=0)
                children.append(_hull_mesh(points))
            # A long rail must not be repeatedly sliced along its length while
            # its rounded end keeps the same complexity. Choose the cut that
            # reduces hull complexity, not merely the longest bbox dimension.
            score = (max(len(c.vertices) for c in children), sum(len(c.faces) for c in children), axis)
            candidates.append((score, plane, children))
        score, plane, children = min(candidates, key=lambda item: item[0])
        axis = score[2]
        error = abs(sum(volume(child) for child in children) - volume(part))
        if error > max(1e-16, volume(part) * 1e-8):
            raise ValueError("convex partition did not conserve source volume")
        eq = ConvexHull(vertices).equations
        if any(np.max(child.vertices @ eq[:, :3].T + eq[:, 3]) > 1e-8 for child in children):
            raise ValueError("convex partition escaped its source solid")
        cuts.append({"axis": axis, "plane_link_local_m": plane, "volume_error_m3": error})
        queue.extend((child, depth + 1) for child in children)
    return parts, cuts


def plan_collisions(parts, decision, tolerance, precision):
    """Return references, realizable proxies, and a reproducible reduction audit."""
    recipe = decision.get("collision")
    if recipe:
        proxies = [{"representation": "box", "box": box, "mesh": box_mesh(box)} for box in recipe["boxes"]]
        proof = validate_proxy_representation({"object": [mesh_record(p) for p in parts]},
                                             {"object": [mesh_record(p["mesh"]) for p in proxies]},
                                             tolerance, geometry_precision_m=precision)
        if not proof["passed"]:
            raise ValueError("API compound collision does not preserve the physical solid: " + str(proof))
        return parts, proxies, {"method": "api_compound_boxes", "proof": proof, "reason": recipe["reason"]}
    references, proxies, decisions = [], [], []
    for source in parts:
        hull = convex_hull_if_exact(source)
        if hull is None:
            # A single hull would fill this component's cavities. Retain the
            # source and use decomposition only here, under native fidelity tests.
            references.append(source)
            proxies.append({"representation": "convexDecomposition", "mesh": source})
            decisions.append({"method": "concave_component_decomposition", "source_triangles": len(source.faces)})
            continue
        box, candidate = exact_box(hull)
        if box:
            references.append(source)
            proxies.append({"representation": "box", "box": box, "mesh": candidate})
            decisions.append({"method": "certified_box", "source_triangles": len(source.faces), "proxy_triangles": 12})
            continue
        split, cuts = split_convex(hull)
        references.extend(split)
        proxies.extend({"representation": "convexHull", "mesh": part} for part in split)
        decisions.append({"method": "exact_convex_partition" if cuts else "exact_coplanar_reduction",
                          "source_triangles": len(source.faces), "proxy_triangles": sum(len(p.faces) for p in split),
                          "cuts": cuts, "parts": len(split)})
    return references, proxies, {"method": "component_adaptive", "components": decisions,
                               "source_triangles": sum(len(p.faces) for p in parts),
                               "proxy_triangles": sum(len(p["mesh"].faces) for p in proxies)}


def check_decoration_support(mesh, support_parts, tolerance, precision):
    """Certify the entire visual surface lies in/near its declared contact host.

    Convex containment covers complete triangles. For concave supports use a
    bounded Lipschitz subdivision, never a vertex-only or bounding-box pass.
    """
    try:
        targets = _fidelity_meshes([mesh_record(p) for p in support_parts], "decoration_host")
    except _Uncertain as error:
        raise ValueError("decoration host is not certifiable: " + str(error)) from error
    remaining = list(np.asarray(mesh.triangles))
    work, max_distance = 0, 0.0
    while remaining:
        tri = remaining.pop()
        work += 1
        if work > 20000:
            raise ValueError("decoration support could not be certified within work budget")
        distances = [_signed_distance(tri, target, precision) for target in targets]
        max_distance = max(max_distance, float(np.min(distances, axis=0).max()))
        if any(target.convex_equations is not None and np.max(distance) <= tolerance - precision
               for target, distance in zip(targets, distances)):
            continue
        center = tri.mean(0)
        distance = min(float(_signed_distance([center], target, precision)[0]) for target in targets)
        if distance > tolerance + precision or np.min(distances, axis=0).max() > tolerance + precision:
            raise ValueError("decoration is not covered by its declared contact host within frozen tolerance")
        radius = np.linalg.norm(tri - center, axis=1).max()
        if distance + radius <= tolerance - precision:
            continue
        if radius <= precision:
            raise ValueError("decoration support is uncertain at geometric precision")
        lengths = np.linalg.norm(tri - np.roll(tri, -1, axis=0), axis=1)
        a = int(np.argmax(lengths)); b, c = (a + 1) % 3, (a + 2) % 3
        mid = (tri[a] + tri[b]) / 2
        remaining.extend((np.array([tri[a], mid, tri[c]]), np.array([mid, tri[b], tri[c]])))
    return {"passed": True, "method": "convex_containment_or_lipschitz_surface_bounds",
            "tolerance_m": tolerance, "surface_cells": work, "max_probed_outside_distance_m": max_distance}
