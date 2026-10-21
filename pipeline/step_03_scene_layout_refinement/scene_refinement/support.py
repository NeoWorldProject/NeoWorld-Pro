"""Mesh-derived surface frames and deterministic support-relative four-DOF poses."""
from __future__ import annotations

import numpy as np

from .assets import transformed


class SupportUnresolved(ValueError):
    pass


class AssetGeometryIncompatible(SupportUnresolved):
    pass


def surface_catalog(meshes):
    """Keep actual planar triangles, including holes; never fill convex hulls."""
    result = []
    for index, mesh in enumerate(meshes):
        vertices = np.asarray(mesh["vertices"], float)
        faces = np.asarray(mesh["faces"], int)
        triangles = vertices[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        groups = {}
        for i in np.flatnonzero(lengths > 1e-12):
            normal = cross[i] / lengths[i]
            key = tuple(np.round(np.r_[normal, normal @ triangles[i, 0]], 6))
            groups.setdefault(key, []).append(int(i))
        for key, indices in sorted(groups.items()):
            normal = np.asarray(key[:3]); normal /= np.linalg.norm(normal)
            area = lengths[indices] / 2
            center = np.average(triangles[indices].mean(axis=1), axis=0, weights=area)
            result.append({"region_id": f"{mesh['link']}:mesh_{index}:patch_{len(result)}",
                           "link": mesh["link"], "normal_local": normal.tolist(),
                           "point_local": center.tolist(), "area_m2": float(area.sum()),
                           "triangles_local": triangles[indices].tolist()})
    return result


def surface_frame(normal, point):
    normal = np.asarray(normal, float); normal /= np.linalg.norm(normal)
    axis = np.array([1., 0., 0.])
    u = axis - normal * (axis @ normal)
    if np.linalg.norm(u) < 1e-6:
        axis = np.array([0., 1., 0.])
        u = axis - normal * (axis @ normal)
    u /= np.linalg.norm(u)
    return np.column_stack((u, np.cross(normal, u), normal)), np.asarray(point, float)


def point_on_patch(point, triangles, tolerance=1e-6):
    """Point projection belongs to actual triangle union, not a bounding polygon."""
    point = np.asarray(point, float)
    for tri in np.asarray(triangles, float):
        a, b, c = tri
        e0, e1, rel = b-a, c-a, point-a
        gram = np.array([[e0@e0, e0@e1], [e0@e1, e1@e1]])
        if abs(np.linalg.det(gram)) < 1e-20:
            continue
        uv = np.linalg.solve(gram, [e0@rel, e1@rel])
        if uv.min() >= -tolerance and uv.sum() <= 1+tolerance:
            return True
    return False


def world_region(region, parent, transform, scale):
    rest = np.array(parent["asset"]["body_rest_transforms"][region["link"]], float)
    rest[:3, 3] *= scale
    link_world = np.asarray(transform) @ rest
    point = transformed([np.asarray(region["point_local"]) * scale], link_world)[0]
    normal = link_world[:3, :3] @ np.asarray(region["normal_local"])
    triangles = transformed(np.asarray(region["triangles_local"]).reshape(-1, 3) * scale, link_world).reshape(-1, 3, 3)
    return point, normal / np.linalg.norm(normal), triangles


def resolve_binding(child, parent, resolve=None):
    relation = child["relation"]
    if relation == "on":
        from .support_layers import resolve as resolve_layer
        return resolve_layer(child, parent, resolve)
    if relation not in {"on", "inside", "attached"}:
        raise SupportUnresolved("A child must have an explicit on/inside/attached relation")
    center = np.asarray(child["layout_object"]["obb"]["center_m"])
    points = np.asarray(child["rest_points"]) * child["initial_scale"]
    initial = np.asarray(child["initial_transform"])
    points = transformed(points, initial)
    parent_points = transformed(np.asarray(parent["rest_points"])*parent["initial_scale"], parent["initial_transform"])
    options = []
    internal_surface_count = 0
    for region in parent["surface_catalog"]:
        point, normal, triangles = world_region(region, parent, parent["initial_transform"], parent["initial_scale"])
        if relation in {"on", "inside"} and normal[2] < .7:
            continue
        if relation == "attached" and (center-point) @ normal < -1e-4:
            continue
        if relation == "inside":
            extent = np.ptp(parent_points @ normal)
            # An internal floor must be below the container/rack opening. A
            # solid box top is not a substitute for a missing cavity or hole.
            if (parent_points @ normal).max()-point@normal < max(1e-4, .05*extent):
                continue
            internal_surface_count += 1
        projection = center-normal*((center-point)@normal)
        if not point_on_patch(projection, triangles):
            continue
        gap = float(np.min((points-point) @ normal))
        options.append({"region_id": region["region_id"], "parent_link": region["link"],
                        "distance_m": abs(gap), "normal_world": normal.tolist(), "area_m2": region["area_m2"]})
    options.sort(key=lambda x: (x["distance_m"], -x["area_m2"], x["region_id"]))
    if not options:
        if relation == "inside" and not internal_surface_count:
            raise AssetGeometryIncompatible(f"asset_geometry_incompatible: no real internal support for {child['object_id']}")
        raise SupportUnresolved(f"No real {relation} support patch for {child['object_id']}")
    chosen = options[0]
    source = "deterministic_mesh_proximity"
    ambiguity_tolerance = max(.001, float(np.linalg.norm(np.ptp(points, axis=0)))*.02)
    ambiguous = len(options) > 1 and options[1]["distance_m"]-options[0]["distance_m"] <= ambiguity_tolerance
    if resolve is not None and ambiguous:
        selection = resolve({"child_object_id": child["object_id"], "parent_object_id": parent["object_id"],
                             "relation": relation, "options": options})
        ident = selection.get("parent_region_id", selection.get("region_id")) if isinstance(selection, dict) else selection
        matches = [item for item in options if item["region_id"] == ident]
        if len(matches) != 1:
            raise SupportUnresolved("Binding API must select one actual geometry catalog ID")
        chosen, source = matches[0], "api_catalog_selection"
    region = next(x for x in parent["surface_catalog"] if x["region_id"] == chosen["region_id"])
    frame, origin = surface_frame(region["normal_local"], region["point_local"])
    result = {"parent_id": parent["object_id"], "child_id": child["object_id"], "relation": relation,
            "parent_link": region["link"], "child_link": child["asset"]["root_link"],
            "parent_region_id": region["region_id"], "region": region,
            "support_frame_local": {"basis": frame.tolist(), "origin": origin.tolist()},
            "resolved_from": source, "requires_native_contact": relation in {"on", "inside"}}
    if relation == "inside":
        result["insertion_policy"] = "actual_internal_floor_contact_with_unrestricted_protrusion"
    return result


def axis_rotation(axis, angle):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew@skew)


def solve_pose(obj, binding, parent, parent_transform, parent_scale, parameters, *, release_gap=0.002,
               placement_policy="component_support_alignment.v1", placement_receipt=None):
    values = np.asarray(parameters, float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Surface parameters must be four finite values [du,dv,theta,log_scale]")
    if abs(values[3]) > 10:
        raise ValueError("Scale parameter exceeds finite physical range")
    if placement_policy != "component_support_alignment.v1":
        raise ValueError("Unknown deterministic support placement policy")
    scale = float(obj["initial_scale"] * np.exp(values[3]))
    initial = np.asarray(obj["initial_transform"])
    if binding is None:
        # Independent roots use the world XY frame; caller determines static roots.
        rotation = axis_rotation([0, 0, 1], values[2]) @ initial[:3, :3]
        center = np.asarray(obj["layout_object"]["obb"]["center_m"]) + [values[0], values[1], 0]
        if obj.get("bound_initial_placement"):
            if not np.any(values):
                return initial.copy(), scale, None
            center = transformed([np.asarray(obj["local_center"])*obj["initial_scale"]], initial)[0]
            center += [values[0], values[1], 0]
        transform = np.eye(4); transform[:3, :3] = rotation
        transform[:3, 3] = center - rotation @ (np.asarray(obj["local_center"]) * scale)
        return transform, scale, None
    point, normal, triangles = world_region(binding["region"], parent, parent_transform, parent_scale)
    # Transform frozen parent-local tangent axes, preserving orientation across candidates.
    rest = np.asarray(parent["asset"]["body_rest_transforms"][binding["parent_link"]])
    basis = np.asarray(parent_transform)[:3, :3] @ rest[:3, :3] @ np.asarray(binding["support_frame_local"]["basis"])
    old_parent = np.asarray(parent["initial_transform"])
    parent_rotation_delta = np.asarray(parent_transform)[:3, :3] @ old_parent[:3, :3].T
    rotation = axis_rotation(normal, values[2]) @ parent_rotation_delta @ initial[:3, :3]
    old_center = np.asarray(obj["layout_object"]["obb"]["center_m"])
    if obj.get("bound_initial_placement"):
        old_center = transformed([np.asarray(obj["local_center"])*obj["initial_scale"]], initial)[0]
    parent_local_center = old_parent[:3, :3].T @ (old_center-old_parent[:3, 3])
    center = transformed([parent_local_center * (parent_scale/parent["initial_scale"])], parent_transform)[0]
    center += basis[:, 0]*values[0] + basis[:, 1]*values[1]
    transform = np.eye(4); transform[:3, :3] = rotation
    transform[:3, 3] = center - rotation @ (np.asarray(obj["local_center"]) * scale)
    points = transformed(np.asarray(obj["rest_points"])*scale, transform)
    lowest = float(np.min((points-point)@normal))
    gap = 0.0 if binding["relation"] == "attached" else release_gap
    unchanged = False
    if obj.get("bound_initial_placement"):
        old_point, old_normal, _ = world_region(binding["region"], parent,
                                               parent["initial_transform"], parent["initial_scale"])
        old_points = transformed(np.asarray(obj["rest_points"])*obj["initial_scale"], initial)
        gap = float(np.min((old_points-old_point)@old_normal)) * (scale/obj["initial_scale"])
        unchanged = (not np.any(values) and np.array_equal(parent_transform, parent["initial_transform"])
                     and parent_scale == parent["initial_scale"])
        if not unchanged:
            # Preserve the bound initial pose; later proposals use the actual support surface.
            gap = max(0., gap)
        correction = gap-lowest
        if unchanged:
            # Reuse the exact bound transform at the initial candidate; this is
            # not a second floor-placement solve or a physical acceptance claim.
            transform = initial.copy()
            points = old_points.copy()
            correction = 0.
    elif (obj["asset"].get("support_rest_pose") and obj["asset"]["support_rest_pose"].get("measured_support_pose", True)
          and binding["relation"] in {"on", "inside"}):
        # Geometry already lives in its independently measured support frame.
        # Place that plane, preserving native contact offset and root height.
        correction = float((point - transform[:3, 3]) @ normal)
        gap = lowest + correction
    else:
        correction = gap-lowest
    transform[:3, 3] += normal*correction
    points += normal*correction
    if binding["relation"] == "on":
        from .support_layers import align
        return align(obj, binding, parent, transform, scale, parent_transform, parent_scale, placement_receipt)
    if placement_receipt is not None and unchanged:
        placement_receipt.update(policy=placement_policy, original_bound_initial_preserved=True)
    near = points[((points-point)@normal) <= gap + max(scale*.001, 1e-6)]
    anchor = near.mean(axis=0)-normal*gap
    # Holes remain holes; contact points may bridge separated table triangles.
    projected_center = points.mean(axis=0); projected_center -= normal*((projected_center-point)@normal)
    if not point_on_patch(anchor, triangles) and not point_on_patch(projected_center, triangles):
        raise SupportUnresolved("Candidate contact lies outside the actual supporting surface")
    return transform, scale, anchor
