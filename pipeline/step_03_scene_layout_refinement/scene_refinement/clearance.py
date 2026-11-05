"""Whole-scene convex collision checks and bounded world-XY placement."""
from functools import lru_cache
from itertools import combinations
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.spatial import ConvexHull

POLICY = 'whole_scene_horizontal_clearance.v1'


def runtime_binding():
    from importlib.metadata import version
    return {'policy': POLICY, 'libraries': {name: version(name) for name in
            ('python-fcl', 'numpy', 'scipy', 'trimesh')}}


@lru_cache(maxsize=1024)
def _convex(vertex_bytes, face_bytes):
    import fcl
    import trimesh
    vertices = np.frombuffer(vertex_bytes, dtype=np.float64).reshape(-1, 3)
    faces = np.frombuffer(face_bytes, dtype=np.int64).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not mesh.is_convex or not mesh.is_watertight:
        raise ValueError('Static clearance requires closed per-part convex collision meshes')
    hull = mesh.convex_hull
    encoded = np.column_stack((np.full(len(hull.faces), 3), hull.faces)).ravel()
    normals = np.asarray(hull.face_normals)
    keep = np.sum(normals[hull.face_adjacency[:, 0]]*normals[hull.face_adjacency[:, 1]], axis=1) < 1.-1e-8
    edges = hull.vertices[hull.face_adjacency_edges[keep, 1]]-hull.vertices[hull.face_adjacency_edges[keep, 0]]
    edges /= np.linalg.norm(edges, axis=1)[:, None]
    return fcl.Convex(hull.vertices, len(hull.faces), encoded), normals, np.unique(np.round(edges, 10), axis=0)


def penetration(a, b, shift_a, shift_b, tolerance):
    """Polyhedral separating-axis depth; do not use backend contact-manifold depth."""
    pa, pb = a['points']+shift_a, b['points']+shift_b
    depth = float('inf')
    axes = np.vstack((a['normals'], b['normals']))
    for phase in range(2):
        for offset in range(0, len(axes), 256):
            selected = axes[offset:offset+256]
            aa, bb = pa@selected.T, pb@selected.T
            depth = min(depth, float(np.minimum(aa.max(0)-bb.min(0), bb.max(0)-aa.min(0)).min()))
            if depth <= tolerance:
                return max(0., depth)
        if phase == 0:
            cross = np.cross(a['edges'][:, None, :], b['edges'][None, :, :]).reshape(-1, 3)
            norm = np.linalg.norm(cross, axis=1)
            axes = np.unique(np.round(cross[norm > 1e-10]/norm[norm > 1e-10, None], 10), axis=0)
    return max(0., depth)


class CollisionWorld:
    def __init__(self, objects, body_poses):
        import fcl
        self.objects, self.parts = objects, {}
        for oid, obj in objects.items():
            self.parts[oid] = []
            for index, mesh in enumerate(obj['collision_meshes']):
                if mesh.get('collision_enabled', True) is False:
                    continue
                vertices = np.ascontiguousarray(mesh['vertices'], dtype=np.float64)
                faces = np.ascontiguousarray(mesh['faces'], dtype=np.int64)
                geometry, normals, edges = _convex(vertices.tobytes(), faces.tobytes())
                pose = np.asarray(body_poses[f"{oid}::{mesh['link']}"])
                points = vertices @ pose[:3, :3].T + pose[:3, 3]
                self.parts[oid].append({'index': index, 'pose': pose,
                    'points': points, 'normals': normals @ pose[:3, :3].T, 'edges': edges @ pose[:3, :3].T,
                    'bounds': np.array([points.min(0), points.max(0)]),
                    'native': fcl.CollisionObject(geometry, fcl.Transform(pose[:3, :3], pose[:3, 3]))})
            if not self.parts[oid]:
                raise ValueError('Missing enabled collision geometry: '+oid)
        self.pairs = [(a, b) for a, b in combinations(sorted(objects), 2)
                      if not (objects[a]['is_environment_support'] and objects[b]['is_environment_support'])]

    def check(self, offsets, tolerance):
        import fcl
        boxes = {}
        for oid, parts in self.parts.items():
            shift = np.r_[offsets.get(oid, [0., 0.]), 0.]
            boxes[oid] = np.array([p['bounds']+shift for p in parts])
            for p in parts:
                p['native'].setTransform(fcl.Transform(p['pose'][:3, :3], p['pose'][:3, 3]+shift))
        hits = []
        for a, b in self.pairs:
            aa, bb = boxes[a], boxes[b]
            overlap = np.minimum(aa[:, None, 1], bb[None, :, 1])-np.maximum(aa[:, None, 0], bb[None, :, 0])
            for i, j in np.argwhere(np.all(overlap > 0., axis=2)):
                result = fcl.CollisionResult()
                fcl.collide(self.parts[a][i]['native'], self.parts[b][j]['native'],
                            fcl.CollisionRequest(num_max_contacts=1, enable_contact=True), result)
                if result.is_collision:
                    if not result.contacts:
                        raise ValueError('Collision query returned no penetration measurement')
                    depth = penetration(self.parts[a][i], self.parts[b][j],
                                        np.r_[offsets.get(a, [0., 0.]), 0.],
                                        np.r_[offsets.get(b, [0., 0.]), 0.], tolerance)
                    if not np.isfinite(depth):
                        raise ValueError('Nonfinite static penetration measurement')
                    if depth > tolerance:
                        hits.append({'object_ids': [a, b], 'part_indices': [int(i), int(j)], 'depth_m': depth})
        return sorted(hits, key=lambda h: (-h['depth_m'], h['object_ids'], h['part_indices']))


def movement_maps(scene, objects, parameters, specs):
    specs = {s['object_id']: s for s in specs}
    bases = {}
    for oid in scene['object_order']:
        obj = objects[oid]
        if oid not in specs or obj['is_environment_support'] or obj['relation'] not in ('root', 'on'):
            continue
        binding = scene['bindings'].get(oid)
        frame = np.eye(3)
        if binding:
            parent = obj['parent_id']
            rest = np.asarray(scene['objects'][parent]['asset']['body_rest_transforms'][binding['parent_link']])
            frame = (np.asarray(objects[parent]['initial_transform'])[:3, :3] @ rest[:3, :3]
                     @ np.asarray(binding['support_frame_local']['basis']))
        if np.max(np.abs(frame[2, :2])) <= 1e-8:
            bases[oid] = frame[:2, :2]
    ids = sorted(bases)
    sigmas = np.array([specs[o]['sigma'][i] for o in ids for i in (0, 1)])
    low = np.array([specs[o]['bounds'][i][0]-parameters[o][i] for o in ids for i in (0, 1)])/sigmas
    high = np.array([specs[o]['bounds'][i][1]-parameters[o][i] for o in ids for i in (0, 1)])/sigmas
    maps = {}
    for oid in scene['object_order']:
        parent = objects[oid]['parent_id']
        maps[oid] = maps[parent].copy() if parent in maps else np.zeros((2, len(sigmas)))
        if objects[oid]['is_environment_support']:
            maps[oid][:] = 0
        elif oid in bases:
            index = 2*ids.index(oid)
            maps[oid][:, index:index+2] += bases[oid]*sigmas[index:index+2]
    return ids, maps, low, high, sigmas


def _separations(world, hits, maps, x, low, high, gap):
    a, b = hits[0]['object_ids']
    pa = np.concatenate([world.parts[a][h['part_indices'][0]]['points'] for h in hits])
    pb = np.concatenate([world.parts[b][h['part_indices'][1]]['points'] for h in hits])
    # A common direction per object pair avoids opposing per-hull corrections.
    axes = np.eye(3)[:2]
    choices = []
    for original in axes:
        for sign in (1., -1.):
            axis = sign*original
            row = axis[:2] @ (maps[b]-maps[a])
            norm = float(row @ row)
            if norm < 1e-16:
                continue
            required = float(np.max(pa @ axis)-np.min(pb @ axis)+gap)
            capacity = float(np.where(row >= 0, row*high, row*low).sum())
            if required <= capacity+1e-10:
                choices.append((max(0., required-row@x)**2/norm, tuple(axis), row, required))
    return [(c[2], c[3]) for c in sorted(choices, key=lambda c: c[:2])]


def project(world, scene, parameters, specs, tolerance, max_iterations=12):
    """Return only placements rechecked globally; unknown/infeasible is not clear."""
    start = time.monotonic()
    ids, maps, low, high, sigmas = movement_maps(scene, world.objects, parameters, specs)
    x = np.zeros(len(low)); rows, rhs, history = [], [], []
    bounds = {oid: np.array([np.min([p['bounds'][0] for p in parts], axis=0),
                            np.max([p['bounds'][1] for p in parts], axis=0)]) for oid, parts in world.parts.items()}
    # Preserve existing object-level separation, including all fixed obstacles.
    for a, b in world.pairs:
        gaps = []
        for axis in range(3):
            for left, right in ((a, b), (b, a)):
                gap = bounds[right][0, axis]-bounds[left][1, axis]
                if gap >= 0:
                    gaps.append((gap, axis, left, right))
        if gaps:
            gap, axis, left, right = max(gaps)
            row = maps[right][axis]-maps[left][axis] if axis < 2 else np.zeros(len(x))
            rows.append(row); rhs.append(-gap)
    support = []
    from .support import world_region
    from .support_layers import contact_projection
    for oid, binding in scene['bindings'].items():
        if binding['relation'] != 'on':
            continue
        parent = binding['parent_id']; p = world.objects[parent]
        _, _, triangles = world_region(binding['region'], scene['objects'][parent], p['initial_transform'], p['scale'])
        polygons = []
        for part in world.parts[oid]:
            xy = part['points'][:, :2]
            polygons.append(xy[ConvexHull(xy).vertices])
        support.append((oid, parent, triangles, polygons))
        points = np.concatenate(polygons)
        for axis in range(2):
            row = maps[oid][axis]-maps[parent][axis]
            rows.extend((row, -row))
            rhs.extend((float(triangles[:, :, axis].min()-points[:, axis].max()+1e-7),
                        float(points[:, axis].min()-triangles[:, :, axis].max()+1e-7)))
    status = 'iteration_limit'
    for iteration in range(max_iterations+1):
        offsets = {o: m@x for o, m in maps.items()}
        hits = world.check(offsets, tolerance)
        support_failures = [oid for oid, parent, tri, polys in support if contact_projection(
            tri+np.r_[offsets[parent], 0.], [p+offsets[oid] for p in polys]) is None]
        history.append({'iteration': iteration, 'collisions': hits, 'unsupported': support_failures})
        if not hits and not support_failures:
            status = 'clear'; break
        if iteration == max_iterations or not len(x):
            break
        if support_failures:
            status = 'support_projection_lost'; break
        grouped = {}
        for hit in hits: grouped.setdefault(tuple(hit['object_ids']), []).append(hit)
        next_x = x.copy(); unresolved = False
        for pair_hits in grouped.values():
            options = _separations(world, pair_hits, maps, next_x, low, high, max(1e-5, tolerance*.1))
            chosen = None
            for row, required in options:
                matrix = np.asarray(rows+[row]); target = np.asarray(rhs+[required])
                solved = minimize(lambda y: .5*float(y@y), next_x, jac=lambda y: y, method='SLSQP',
                                  bounds=Bounds(low, high), constraints=[LinearConstraint(matrix, target, np.inf)],
                                  options={'ftol': 1e-11, 'maxiter': 200})
                if solved.success and not np.any(matrix@solved.x < target-1e-8):
                    chosen = (row, required, solved.x); break
            if chosen is None:
                unresolved = True; break
            rows.append(chosen[0]); rhs.append(chosen[1]); next_x = chosen[2]
        if unresolved:
            status = 'bounded_horizontal_constraints_infeasible'; break
        if np.max(np.abs(next_x-x)) < 1e-9:
            status = 'no_progress'; break
        x = next_x
    offsets = {o: (m@x).tolist() for o, m in maps.items()}
    return {'policy': POLICY, 'status': status, 'passed': status == 'clear',
            'scope': 'all_cross_object_enabled_convex_colliders_including_environment',
            'environment_environment_pairs': 'fixed_not_optimized',
            'geometry': 'authored_convex_collision_meshes', 'tolerance_m': tolerance,
            'world_xy_offsets_m': offsets, 'world_z_changed': False,
            'local_uv_corrections_m': {o: (x[2*i:2*i+2]*sigmas[2*i:2*i+2]).tolist() for i, o in enumerate(ids)},
            'iterations': history, 'seconds': time.monotonic()-start, 'physical_acceptance': False}


def apply(stage, scene, objects, parameters, layout, pose_receipts, attachments):
    from pxr import UsdGeom
    from .assets import matrix
    from .composition import _usd_transform
    cache = UsdGeom.XformCache()
    body_poses = {f'{oid}::{link}': matrix(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
                  for oid, o in objects.items() for link, path in o['body_paths'].items()}
    start = time.monotonic()
    world = CollisionWorld(objects, body_poses)
    build_seconds = time.monotonic()-start
    report = project(world, scene, parameters, scene['search_specs'], scene['physics_settings']['penetration_tolerance_m'])
    report.update(geometry_build_seconds=build_seconds, runtime=runtime_binding())
    if report['passed']:
        layouts = {o['object_id']: o for o in layout['objects_data']}
        for oid, xy in report['world_xy_offsets_m'].items():
            delta = np.r_[xy, 0.]
            obj = objects[oid]
            transform = np.array(obj['initial_transform'], copy=True)
            transform[:3, 3] += delta
            _usd_transform(UsdGeom.Xform(stage.GetPrimAtPath(obj['root_prim_path'])), transform)
            obj['initial_transform'] = transform.tolist()
            if obj.get('support_pose_resolution'):
                obj['support_pose_resolution']['post_alignment_world_xy_translation_m'] = list(xy)
            layouts[oid]['obb']['center_m'] = (np.asarray(layouts[oid]['obb']['center_m'])+delta).tolist()
            layouts[oid]['3d_bbox_meters'][:3] = (np.asarray(layouts[oid]['3d_bbox_meters'][:3])+delta).tolist()
            if oid in pose_receipts:
                receipt = pose_receipts[oid]
                receipt['world_support_transform'] = transform.tolist()
                receipt['horizontal_clearance_translation_m'] = delta.tolist()
                for pose in receipt['authored_world_body_poses'].values():
                    for axis in range(2):
                        pose[axis][3] += float(delta[axis])
        for row in attachments:
            delta = np.r_[report['world_xy_offsets_m'][row['parent_id']], 0.]
            if row['fixed_to_world'] and np.any(delta):
                raise ValueError('Cannot move an installation fixed to the environment')
            row['world_anchor'] = (np.asarray(row['world_anchor'])+delta).tolist()
        cache.Clear()
        for oid, obj in objects.items():
            for link, path in obj['body_paths'].items():
                now = matrix(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
                before = body_poses[f'{oid}::{link}']
                if now[2, 3] != before[2, 3] or not np.array_equal(now[:3, :3], before[:3, :3]):
                    raise ValueError('Horizontal clearance changed world Z or link orientation')
    return report
