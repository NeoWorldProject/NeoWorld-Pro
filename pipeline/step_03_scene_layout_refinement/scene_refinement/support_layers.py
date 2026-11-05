"""Named collision support layers and deterministic vertical placement."""
import hashlib
import json

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .assets import transformed

POLICY = 'component_support_alignment.v1'


def collision_parts(obj, transform, scale):
    asset = obj['asset']
    for index, mesh in enumerate(asset['collision_meshes']):
        if mesh.get('collision_enabled', True) is False:
            continue
        local = transformed(mesh['vertices'], asset['body_rest_transforms'][mesh['link']])
        yield index, mesh, transformed(local * scale, transform)


def footprint(obj, transform, scale):
    polygons = []
    for _, _, vertices in collision_parts(obj, transform, scale):
        points = np.unique(vertices[:, :2], axis=0)
        if len(points) < 3:
            continue
        try:
            polygons.append(points[ConvexHull(points).vertices])
        except QhullError:
            continue
    return polygons


def intersection(subject, clip):
    """Intersect two convex projections; holes remain separate components."""
    output = list(subject)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        if not output:
            break
        edge = b - a
        side = lambda p: edge[0]*(p[1]-a[1]) - edge[1]*(p[0]-a[0])
        incoming, output = output, []
        previous = incoming[-1]; dp = side(previous)
        for current in incoming:
            dc = side(current)
            if (dc >= -1e-10) != (dp >= -1e-10):
                output.append(previous + (current-previous) * (dp/(dp-dc)))
            if dc >= -1e-10:
                output.append(current)
            previous, dp = current, dc
    return np.asarray(output)


def contact_projection(triangles, polygons):
    xy = np.asarray(triangles)[:, :, :2]
    low, high = xy.min(1), xy.max(1)
    for poly in polygons:
        keep = np.all(high >= poly.min(0)-1e-10, axis=1) & np.all(low <= poly.max(0)+1e-10, axis=1)
        for tri in xy[keep]:
            cross = np.cross(np.r_[tri[1]-tri[0], 0.], np.r_[tri[2]-tri[0], 0.])[2]
            if abs(cross) < 1e-14:
                continue
            overlap = intersection(poly, tri if cross > 0 else tri[::-1])
            if len(overlap) >= 3:
                area = abs(np.sum(overlap[:, 0]*np.roll(overlap[:, 1], -1)
                                  - overlap[:, 1]*np.roll(overlap[:, 0], -1))) / 2
                if area > 1e-14:
                    return overlap.mean(0)
    return None


def layer_catalog(parent):
    cached = parent.get('_collision_support_layers')
    if cached is not None:
        return cached
    transform, scale = np.asarray(parent['initial_transform']), parent['initial_scale']
    root = parent['asset']['root_link']
    rest = np.array(parent['asset']['body_rest_transforms'][root], float)
    rest[:3, 3] *= scale
    inverse = np.linalg.inv(transform @ rest)
    faces = []
    for index, mesh, vertices in collision_parts(parent, transform, scale):
        triangles = vertices[np.asarray(mesh['faces'], int)]
        cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        upward = (lengths > 1e-12) & (cross[:, 2] >= .7*lengths)
        for face in np.flatnonzero(upward):
            tri = triangles[face]
            faces.append((float(tri[:, 2].max()), index, int(face), mesh, tri, float(lengths[face]/2)))
    faces.sort(key=lambda x: (x[0], x[1], x[2]))
    groups = []
    # This groups evidence only; the exact triangles still determine placement.
    for row in faces:
        if not groups or row[0]-groups[-1][0][0] > .0001:
            groups.append([])
        groups[-1].append(row)
    regions = []
    for group in groups:
        world = np.asarray([x[4] for x in group])
        local = transformed(world.reshape(-1, 3), inverse).reshape(-1, 3, 3) / scale
        identity = [(x[1], x[2]) for x in group]
        region_id = 'support_layer_' + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
        normal = (transform @ rest)[:3, :3].T @ [0., 0., 1.]
        point_world = world.reshape(-1, 3).mean(0); point_world[2] = world[:, :, 2].max()
        point = transformed([point_world], inverse)[0] / scale
        regions.append(dict(region_id=region_id, link=root, normal_local=normal.tolist(),
            point_local=point.tolist(), triangles_local=local.tolist(),
            area_m2=sum(x[5] for x in group)/(scale*scale),
            component_names=sorted({x[3].get('name', x[3]['link']) for x in group}),
            component_ids=sorted({f"{x[3]['link']}:{x[1]}" for x in group}),
            source='collision_component_upper_faces', grouping_tolerance_world_m=.0001))
    parent['_collision_support_layers'] = regions
    return regions


def resolve(child, parent, callback):
    from .support import SupportUnresolved, surface_frame, world_region
    polygons = footprint(child, child['initial_transform'], child['initial_scale'])
    child_points = np.concatenate([v for _, _, v in collision_parts(child, child['initial_transform'], child['initial_scale'])])
    candidates, regions, diagram_triangles = [], {}, {}
    for region in layer_catalog(parent):
        point, normal, triangles = world_region(region, parent, parent['initial_transform'], parent['initial_scale'])
        if contact_projection(triangles, polygons) is None:
            continue
        top = float(triangles[:, :, 2].max())
        regions[region['region_id']] = region
        diagram_triangles[region['region_id']] = triangles.tolist()
        candidates.append(dict(region_id=region['region_id'], parent_link=region['link'],
            component_names=region['component_names'], component_ids=region['component_ids'],
            height_world_m=top, bounds_world_m=[triangles.min((0,1)).tolist(), triangles.max((0,1)).tolist()],
            distance_m=abs(float(child_points[:, 2].min())-top), area_m2=region['area_m2'],
            normal_world=normal.tolist(), center_contact_required=False))
    candidates.sort(key=lambda x: (-x['height_world_m'], -x['area_m2'], x['region_id']))
    for index, option in enumerate(candidates, 1):
        option['diagram_label'] = f'L{index}'
    if not candidates:
        raise SupportUnresolved(f"No actual collision support overlaps {child['object_id']}")
    chosen, source = candidates[0], 'single_collision_support_layer'
    if len(candidates) > 1:
        if callback is not None:
            selection = callback(dict(child_object_id=child['object_id'], parent_object_id=parent['object_id'],
                relation='on', child_bounds_world_m=[child_points.min(0).tolist(), child_points.max(0).tolist()],
                options=candidates, selection_policy='intended_support_layer_not_nearest_distance',
                diagram_triangles_world=diagram_triangles))
            ident = selection.get('region_id', selection.get('parent_region_id')) if isinstance(selection, dict) else selection
            matches = [x for x in candidates if x['region_id'] == ident]
            if len(matches) != 1:
                raise SupportUnresolved('Binding API must select one supplied support layer ID')
            chosen, source = matches[0], 'api_component_layer_selection'
        else:
            chosen = min(candidates, key=lambda x:(x['distance_m'], -x['height_world_m'], x['region_id']))
            source = 'deterministic_collision_layer_proximity'
    region = regions[chosen['region_id']]
    frame, origin = surface_frame(region['normal_local'], region['point_local'])
    return dict(parent_id=parent['object_id'], child_id=child['object_id'], relation='on',
        parent_link=region['link'], child_link=child['asset']['root_link'],
        parent_region_id=region['region_id'], region=region,
        support_frame_local={'basis':frame.tolist(), 'origin':origin.tolist()},
        resolved_from=source, placement_policy=POLICY, requires_native_contact=True,
        selection_evidence=chosen)


def align(child, binding, parent, transform, scale, parent_transform, parent_scale, receipt):
    from .support import SupportUnresolved, world_region
    _, _, triangles = world_region(binding['region'], parent, parent_transform, parent_scale)
    projection = contact_projection(triangles, footprint(child, transform, scale))
    if projection is None:
        raise SupportUnresolved('Child has no projected contact with the selected support layer')
    vertices = np.concatenate([v for _, _, v in collision_parts(child, transform, scale)])
    old_bottom = float(vertices[:, 2].min()); top = float(triangles[:, :, 2].max())
    dz = top - old_bottom
    result = np.array(transform, float, copy=True); result[2, 3] += dz
    if receipt is not None:
        receipt.update(policy=POLICY, status='aligned', applied_translation_m=dz,
            normal_world=[0.,0.,1.], original_bottom_world_m=old_bottom, support_top_world_m=top,
            component_names=binding['region']['component_names'], region_id=binding['parent_region_id'],
            original_transform=np.asarray(transform).tolist(), aligned_transform=result.tolist(),
            acceptance='placement_only_requires_native_physics_semantics_and_independent_reload')
    return result, scale, np.r_[projection, top]
