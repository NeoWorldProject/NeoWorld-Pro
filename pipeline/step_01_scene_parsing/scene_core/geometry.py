"""Deterministic OBB arithmetic; diagnostics never modify scene geometry."""

from itertools import combinations
import numpy as np


# Ordering follows the bottom ring then the top ring, for plotting faces/edges.
CORNER_SIGNS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=float)
BOX_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4))
BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def rotation_matrix(rotation_xyz_deg):
    """Right-handed active rotation: column world vector = Rz @ Ry @ Rx @ local."""
    x, y, z = np.deg2rad(np.asarray(rotation_xyz_deg, dtype=float))
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def obb_corners(obb):
    """Return eight world corners (8 x 3), without changing the supplied OBB."""
    local = CORNER_SIGNS * np.asarray(obb["size_m"], dtype=float) / 2.0
    return local @ rotation_matrix(obb["rotation_xyz_deg"]).T + np.asarray(obb["center_m"], dtype=float)


def obb_to_aabb(obb):
    """Legacy six-value AABB: [cx, cy, cz, sx, sy, sz], in world meters."""
    # |R| * size is equivalent to corner extrema and avoids cancellation at
    # large center coordinates. The OBB center is also the AABB center.
    size = np.abs(rotation_matrix(obb["rotation_xyz_deg"])) @ np.asarray(obb["size_m"], dtype=float)
    return [*map(float, obb["center_m"]), *size.tolist()]


def front_direction(obb):
    """The object's designated forward axis is its local -Y axis."""
    return rotation_matrix(obb["rotation_xyz_deg"]) @ np.array([0.0, -1.0, 0.0])


def _obb_overlap(a, b):
    """Separating-axis test for two boxes; this is not a mesh collision test."""
    ra = rotation_matrix(a["rotation_xyz_deg"])
    rb = rotation_matrix(b["rotation_xyz_deg"])
    axes = [*ra.T, *rb.T]
    axes += [np.cross(x, y) for x in ra.T for y in rb.T]
    delta = np.asarray(b["center_m"]) - np.asarray(a["center_m"])
    sa, sb = np.asarray(a["size_m"]) / 2, np.asarray(b["size_m"]) / 2
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-12:
            continue
        axis = axis / length
        radius_a = float(np.abs(ra.T @ axis) @ sa)
        radius_b = float(np.abs(rb.T @ axis) @ sb)
        if abs(float(delta @ axis)) > radius_a + radius_b + 1e-10:
            return False
    return True


def geometry_diagnostics(layout):
    """Describe geometry and graph edges. Nothing here moves or resizes boxes."""
    objects = layout["objects_data"]
    by_name = {obj["name"]: obj for obj in objects}
    bounds = {}
    dimensions = []
    for obj in objects:
        box = obb_to_aabb(obj["obb"])
        center, size = np.asarray(box[:3]), np.asarray(box[3:])
        bounds[obj["name"]] = (center - size / 2, center + size / 2)
        dimensions.append({"object_id": obj["object_id"], "name": obj["name"],
                           "local_size_m": list(obj["obb"]["size_m"]),
                           "aabb_size_m": box[3:],
                           "front_world": front_direction(obj["obb"]).tolist()})
    support_edges = []
    for parent_name, node in layout["scene_tree"].items():
        parent = by_name[parent_name]
        pmin, pmax = bounds[parent_name]
        for child_name in node["children"]:
            child = by_name[child_name]
            cmin, cmax = bounds[child_name]
            xy_overlap = np.maximum(0.0, np.minimum(pmax[:2], cmax[:2]) - np.maximum(pmin[:2], cmin[:2]))
            support_edges.append({
                "parent_id": parent["object_id"], "child_id": child["object_id"],
                "relation": child["support_relation"],
                "child_bottom_minus_parent_top_m": float(cmin[2] - pmax[2]),
                "aabb_xy_overlap_area_m2": float(np.prod(xy_overlap)),
                "child_aabb_inside_parent_aabb": bool(np.all(cmin >= pmin) and np.all(cmax <= pmax)),
            })
    overlaps = []
    for a, b in combinations(objects, 2):
        amin, amax = bounds[a["name"]]
        bmin, bmax = bounds[b["name"]]
        intersection = np.maximum(0.0, np.minimum(amax, bmax) - np.maximum(amin, bmin))
        volume = float(np.prod(intersection))
        if volume > 0.0:
            overlaps.append({"object_ids": [a["object_id"], b["object_id"]],
                             "aabb_intersection_volume_m3": volume,
                             "obb_intersects_or_touches": _obb_overlap(a["obb"], b["obb"])})
    return {
        "kind": "computed_geometry_only",
        "caveats": [
            "OBBs/AABBs enclose objects; overlap is advisory and is not proof of solid-mesh collision.",
            "Vertical gaps use AABB extrema, not actual contact surfaces; inside/attached relations and tilted supports need semantic interpretation.",
            "Fixed world views are layout diagnostics, not input-camera reprojections.",
        ],
        "objects": dimensions, "support_edges": support_edges, "overlap_advisories": overlaps,
    }
