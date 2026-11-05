"""Static closed-solid bounds used to certify authored collision proxies.

Trajectory intersection checks are retired; current native dynamics retain their
own measurements. Unknown static geometry remains explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError


class _Uncertain(RuntimeError):
    pass


def _cross3(a, b):
    """Exact NumPy three-vector arithmetic without generic axis normalization."""
    # Geometry coordinates always share float dtype and (..., 3) shape.
    # Keep the fallback for external callers and unusual broadcasting/dtypes.
    if (type(a) is not np.ndarray or type(b) is not np.ndarray
            or a.shape != b.shape or not a.shape or a.shape[-1] != 3
            or a.dtype != b.dtype or a.dtype.kind != "f"):
        return np.cross(a, b)
    out = np.empty_like(a)
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


def _corners(lo, hi):
    return np.array([[hi[k] if (i >> k) & 1 else lo[k] for k in range(3)] for i in range(8)])


@dataclass
class _Node:
    lo: np.ndarray
    hi: np.ndarray
    faces: np.ndarray | None
    left: object = None
    right: object = None


def _tree(triangles, indices, leaf_size=12):
    chosen = triangles[indices]
    lo, hi = chosen.min(axis=(0, 1)), chosen.max(axis=(0, 1))
    if len(indices) <= leaf_size:
        return _Node(lo, hi, indices)
    axis = int(np.argmax(hi - lo))
    ordered = indices[np.argsort(chosen.mean(axis=1)[:, axis], kind="stable")]
    middle = len(ordered) // 2
    return _Node(lo, hi, None, _tree(triangles, ordered[:middle], leaf_size),
                 _tree(triangles, ordered[middle:], leaf_size))


def _point_triangle_distance(points, triangles, *, per_triangle=False):
    """Squared exact point/triangle distance (projection or the nearest edge)."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e0, e1 = b - a, c - a
    v = points[:, None] - a
    d00, d01, d11 = (np.einsum("ij,ij->i", x, y) for x, y in ((e0, e0), (e0, e1), (e1, e1)))
    d20, d21 = np.einsum("ijk,jk->ij", v, e0), np.einsum("ijk,jk->ij", v, e1)
    denominator = np.maximum(d00 * d11 - d01 * d01, 1e-30)
    u = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    projected = a + u[..., None] * e0 + w[..., None] * e1
    face_distance = np.sum((points[:, None] - projected)**2, axis=2)
    best = np.where((u >= 0) & (w >= 0) & (u + w <= 1), face_distance, np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        t = np.einsum("ijk,jk->ij", points[:, None] - start, edge) / np.maximum(np.sum(edge**2, axis=1), 1e-30)
        closest = start + np.clip(t, 0, 1)[..., None] * edge
        best = np.minimum(best, np.sum((points[:, None] - closest)**2, axis=2))
    return best if per_triangle else best.min(axis=1)


def _inside(points, triangles, eps):
    """Oriented solid angle; points within eps of a face are contact, not interior."""
    output = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), 32):
        p = points[start:start + 32]
        distance = np.full(len(p), np.inf)
        angle = np.zeros(len(p))
        for offset in range(0, len(triangles), 512):
            tri = triangles[offset:offset + 512]
            distance = np.minimum(distance, _point_triangle_distance(p, tri))
            v = tri[None] - p[:, None, None]
            a, b, c = v[:, :, 0], v[:, :, 1], v[:, :, 2]
            la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
            numerator = np.einsum("ijk,ijk->ij", a, _cross3(b, c))
            denominator = (la * lb * lc + np.einsum("ijk,ijk->ij", a, b) * lc
                           + np.einsum("ijk,ijk->ij", b, c) * la + np.einsum("ijk,ijk->ij", c, a) * lb)
            angle += (2 * np.arctan2(numerator, denominator)).sum(axis=1)
        output[start:start + len(p)] = (np.abs(angle) > 2 * math.pi) & (distance > eps * eps)
    return output


class _Mesh:
    def __init__(self, item, label):
        self.label = label
        self.vertices = np.asarray(item["vertices"], dtype=float)
        self.faces = np.asarray(item["faces"], dtype=int)
        if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or not len(self.vertices)
                or not np.isfinite(self.vertices).all() or self.faces.ndim != 2
                or self.faces.shape[1] != 3 or not len(self.faces)
                or self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise ValueError(f"invalid source mesh: {label}")
        self.triangles = self.vertices[self.faces]
        area = np.linalg.norm(_cross3(self.triangles[:, 1] - self.triangles[:, 0],
                                       self.triangles[:, 2] - self.triangles[:, 0]), axis=1)
        self.faces = self.faces[area > 1e-18]
        self.triangles = self.vertices[self.faces]
        if not len(self.faces):
            raise ValueError(f"source mesh has no nondegenerate faces: {label}")
        directed = np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]])
        edges, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
        self.edges = edges
        self.closed = bool(np.all(counts == 2))
        # Coherent orientation is needed for winding-number containment.
        directed_unique = np.unique(directed, axis=0)
        self.oriented = len(directed_unique) == len(directed)
        self.nonmanifold = bool(np.any(counts > 2) or (self.closed and not self.oriented))
        centred = self.triangles - self.vertices.mean(0)
        signed_volume = np.einsum("ij,ij->i", centred[:, 0], _cross3(centred[:, 1], centred[:, 2])).sum()
        self.orientation_sign = -1.0 if signed_volume < 0 else 1.0
        edge_faces = {}
        for index, face in enumerate(self.faces):
            for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge_faces.setdefault(tuple(sorted((int(first), int(second)))), []).append(index)
        neighbours = []
        for index, face in enumerate(self.faces):
            adjacent = {index}
            for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                adjacent.update(edge_faces[tuple(sorted((int(first), int(second))))])
            neighbours.append(self.triangles[sorted(adjacent)].reshape(-1, 3))
        width = max(map(len, neighbours))
        self.neighbours = np.array([np.concatenate([points, np.repeat(points[:1], width - len(points), axis=0)])
                                    for points in neighbours])
        self.tree = _tree(self.triangles, np.arange(len(self.faces)))
        self.radius = float(np.linalg.norm(self.vertices, axis=1).max())


def _signed_distance(points, mesh, precision):
    points = np.asarray(points, dtype=float)
    distances = np.full(len(points), np.inf)
    for offset in range(0, len(mesh.triangles), 512):
        distances = np.minimum(distances, _point_triangle_distance(points, mesh.triangles[offset:offset + 512]))
    distances = np.sqrt(np.maximum(distances, 0))
    if getattr(mesh, "convex_equations", None) is not None:
        eq = mesh.convex_equations
        inside = np.max(points @ eq[:, :3].T + eq[:, 3], axis=1) <= 0
    else:
        inside = _inside(points, mesh.triangles, precision)
    return np.where(inside, -distances, distances)


def _fidelity_meshes(value, label):
    components = value if isinstance(value, list) else value.get("components", [value])
    result = []
    for index, item in enumerate(components):
        mesh = _Mesh(item, f"{label}/{index}")
        mesh.source_object = item.get("source_object", item.get("name", mesh.label))
        if not mesh.closed or not mesh.oriented or mesh.nonmanifold:
            raise _Uncertain(f"proxy fidelity needs closed, oriented physical reference solids: {mesh.label}")
        mesh.convex_equations = None
        try:
            hull = ConvexHull(mesh.vertices)
            origin = mesh.vertices.mean(0)
            tri = mesh.triangles - origin
            volume = abs(float(np.einsum("ij,ij->i", tri[:, 0], _cross3(tri[:, 1], tri[:, 2])).sum() / 6))
            face_centres = mesh.triangles.mean(1)
            hull_gap = np.max(face_centres @ hull.equations[:, :3].T + hull.equations[:, 3], axis=1)
            surface_on_hull = np.max(np.abs(hull_gap)) <= max(1e-12, np.linalg.norm(mesh.tree.hi - mesh.tree.lo) * 1e-9)
            if surface_on_hull and abs(hull.volume - volume) <= max(1e-15, hull.volume * 1e-7):
                mesh.convex_equations = hull.equations
        except QhullError as error:
            raise _Uncertain(f"physical reference is not volumetric: {mesh.label}") from error
        result.append(mesh)
    if not result:
        raise _Uncertain(f"no physical geometry: {label}")
    return result


def _union_signed_distance(point, targets, precision):
    """Exact union minimum with conservative AABB distance lower bounds.

    A positive bbox distance bounds the surface distance below. Inside a bbox
    the signed distance may be negative, so every such candidate is evaluated.
    This culls distant parts without changing tolerances or sampling density.
    """
    point = np.asarray(point)
    lower = [float(np.linalg.norm(np.maximum(np.maximum(target.tree.lo - precision - point,
                                                        point - target.tree.hi - precision), 0)))
             for target in targets]
    best = float("inf")
    for index in np.argsort(lower):
        if lower[index] > max(0.0, best) + precision:
            break
        best = min(best, float(_signed_distance([point], targets[index], precision)[0]))
    return best


def validate_proxy_representation(reference, cooked, tolerance_m, *, max_cells=20000,
                                  max_depth=24, geometry_precision_m=1e-8):
    """Prove bidirectional tolerance-bounded solid coverage or return uncertain.

    Both inputs contain vertices and triangle indices in link-local frames.
    The reference is the closed physical collision asset, including any explicit
    shell thickness, not an open visual sheet. A convex target's Euclidean
    tolerance dilation is convex: containment of every source vertex proves
    containment of the whole source. Remaining regions use adaptive volume cells
    with the 1-Lipschitz bound of signed distance. Finite probes can disprove
    coverage, but cannot authorize a pass without a bound covering their cells.
    """
    tolerance = float(tolerance_m)
    precision = float(geometry_precision_m)
    report = {"passed": False, "complete": False, "status": "uncertain", "issues": [],
              "tolerance_m": tolerance, "links": {}, "cells_evaluated": 0,
              "surface_vertices_probed": 0,
              "method": "convex_containment_and_adaptive_signed_distance_bounds"}
    try:
        if not math.isfinite(tolerance) or tolerance <= 0 or not 0 < precision <= tolerance:
            raise _Uncertain("proxy fidelity tolerances are invalid")
        if not isinstance(reference, dict) or not reference or set(reference) != set(cooked):
            raise _Uncertain("reference/cooked link coverage differs")
        if max_cells < 1 or max_depth < 1:
            raise _Uncertain("proxy fidelity work limits must be positive")

        def covered_by_one_convex(points, targets):
            lo, hi = points.min(0), points.max(0)
            for target in targets:
                if target.convex_equations is None:
                    continue
                if np.any(lo < target.tree.lo - tolerance) or np.any(hi > target.tree.hi + tolerance):
                    continue
                eq = target.convex_equations
                # The exact interior halfspace test avoids surface-distance work
                # for ordinary boxes or convex pieces with identical boundaries.
                if np.max(points @ eq[:, :3].T + eq[:, 3]) <= 0:
                    return True
                if np.all(_signed_distance(points, target, precision) <= tolerance - precision):
                    return True
            return False

        def union_distance(point, targets):
            return _union_signed_distance(point, targets, precision)

        for link in reference:
            sources = _fidelity_meshes(reference[link], f"{link}/reference")
            proxies = _fidelity_meshes(cooked[link], f"{link}/cooked")
            detail = {"reference_components": len(sources), "cooked_components": len(proxies),
                      "convex_certificates": 0, "cells_evaluated": 0, "passed": False}
            report["links"][link] = detail
            for direction, subjects, targets in (("omission", sources, proxies), ("overfill", proxies, sources)):
                for subject in subjects:
                    if covered_by_one_convex(subject.vertices, targets):
                        detail["convex_certificates"] += 1
                        continue
                    # A referenced surface vertex belongs to the closed solid.
                    # Its distance outside the target union is therefore a
                    # rigorous counterexample, including on a very thin tip
                    # that volume subdivision might not reach within budget.
                    # These bounded probes can only reject; passing still needs
                    # convex containment or the complete adaptive proof below.
                    indices = np.unique(subject.faces)
                    if len(indices) > 256:
                        indices = indices[np.linspace(0, len(indices) - 1, 256, dtype=int)]
                    for vertex_index in indices:
                        point = subject.vertices[vertex_index]
                        distance = union_distance(point, targets)
                        report["surface_vertices_probed"] += 1
                        if distance > tolerance + precision:
                            report.update(status="failed", passed=False, complete=True)
                            report["issues"].append({"kind": direction, "link": link, "object": subject.label,
                                "source_object": subject.source_object,
                                "witness_type": "referenced_surface_vertex", "vertex_index": int(vertex_index),
                                "witness_link_local_m": point.tolist(), "distance_outside_target_m": distance})
                            return report
                    stack = [(subject.tree.lo, subject.tree.hi, 0)]
                    while stack:
                        if report["cells_evaluated"] >= max_cells:
                            raise _Uncertain("proxy fidelity cell budget exhausted before containment was proved")
                        lo, hi, depth = stack.pop()
                        report["cells_evaluated"] += 1
                        detail["cells_evaluated"] += 1
                        corners = _corners(lo, hi)
                        if covered_by_one_convex(corners, targets):
                            continue
                        centre = (lo + hi) / 2
                        radius = float(np.linalg.norm(hi - lo) / 2)
                        subject_distance = float(_signed_distance(np.asarray([centre]), subject, precision)[0])
                        if subject_distance > radius + precision:
                            continue  # whole cell lies outside this subject
                        target_distance = union_distance(centre, targets)
                        if target_distance + radius <= tolerance - precision:
                            continue  # whole cell lies within the target tolerance
                        if subject_distance < -precision and target_distance > tolerance + precision:
                            report.update(status="failed", passed=False, complete=True)
                            report["issues"].append({"kind": direction, "link": link, "object": subject.label,
                                                     "source_object": subject.source_object,
                                                     "witness_link_local_m": centre.tolist(),
                                                     "distance_outside_target_m": target_distance})
                            return report
                        if depth >= max_depth or radius <= precision:
                            raise _Uncertain(f"proxy {direction} boundary cannot be certified within precision for {link}")
                        # Divide the longest axis: the deterministic queue covers
                        # every cell without octree's eight-fold memory spikes.
                        axis = int(np.argmax(hi - lo))
                        mid = (lo[axis] + hi[axis]) / 2
                        left_hi, right_lo = hi.copy(), lo.copy()
                        left_hi[axis], right_lo[axis] = mid, mid
                        stack.extend([(right_lo, hi, depth + 1), (lo, left_hi, depth + 1)])
            detail["passed"] = True
        report.update(status="passed", passed=True, complete=True)
    except (_Uncertain, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
        report["issues"].append(str(error))
    return report


