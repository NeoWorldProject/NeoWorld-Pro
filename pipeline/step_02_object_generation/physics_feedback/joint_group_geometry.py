"""Source-shape equivalence and role-local ground evidence for representative groups.

These are grouping tolerances, never physics acceptance gates. No source or
collision mesh is modified. The single-object harness has a plane as its only
external obstacle; internal collisions remain disabled. This is not scene
collision planning or a proof that a freely moving root can complete a motion.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree, ConvexHull
import trimesh

from urdf_core.run_state import file_hash

# Frozen engineering defaults for *equivalence*, not collision fidelity.
SHAPE_REL = 0.005  # 0.5% of the smaller link's diagonal.
SHAPE_ABS_M = 1e-5
FRAME_ATOL = 1e-5
GROUND_ANGLE_STEP = math.pi / 90


def coordinates_close(a, b):
    return np.allclose(a, b, rtol=0, atol=FRAME_ATOL)


def rotation_close(a, b):
    return np.allclose(a, b, rtol=0, atol=FRAME_ATOL)


def _tolerance(a, b):
    return max(SHAPE_ABS_M, SHAPE_REL * min(np.linalg.norm(a.extents), np.linalg.norm(b.extents)))


def _surface_points(mesh):
    # Vertices, face centers and edge midpoints retain holes/concavities. No AABB
    # or convex hull is used to establish shape equivalence.
    tri = mesh.triangles
    return np.unique(np.concatenate([mesh.vertices, tri.mean(axis=1),
        (tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2,
        (tri[:, 2] + tri[:, 0]) / 2]), axis=0)


def _near_surface(points, mesh, tolerance):
    # Most points need only nearby triangles. Long thin triangles can have
    # distant centers, so unresolved points get an exact, bounded-memory query.
    tree = cKDTree(mesh.triangles_center)
    for chunk in np.array_split(points, max(1, math.ceil(len(points) / 256))):
        indices = tree.query(chunk, k=min(32, len(mesh.faces)))[1]
        if indices.ndim == 1:
            indices = indices[:, None]
        triangles = mesh.triangles[indices].reshape(-1, 3, 3)
        repeated = np.repeat(chunk, indices.shape[1], axis=0)
        nearest = trimesh.triangles.closest_point(triangles, repeated)
        distances = np.linalg.norm(nearest - repeated, axis=1).reshape(len(chunk), -1).min(axis=1)
        unresolved = chunk[distances > tolerance]
        width = max(1, min(32, 200000 // len(mesh.faces)))
        for start in range(0, len(unresolved), width):
            exact = trimesh.proximity.closest_point_naive(mesh, unresolved[start:start + width])[1]
            if np.any(exact > tolerance):
                return False
    return True


class SourceGeometry:
    def __init__(self, parsed, manifest):
        self.parsed, self.manifest = parsed, manifest
        self.meshes, self.matches, self.ground = {}, {}, {}
        self.joints = {j['name']: j for j in parsed['joints']}

    def mesh(self, link):
        if link in self.meshes:
            return self.meshes[link]
        entries = self.parsed['links'][link]['visuals']
        if not entries:
            # Coordinate-only links may match; missing source shapes may not.
            value = None if self.parsed['links'][link]['collisions'] else []
            self.meshes[link] = value
            return value
        parts = []
        for entry in entries:
            kind = entry['geometry']
            if kind == 'mesh':
                path = Path(entry['mesh']).resolve()
                relative = path.relative_to(Path(self.parsed['path']).parent).as_posix()
                if self.manifest.get('artifacts', {}).get(relative) != file_hash(path):
                    raise ValueError('Grouping source geometry is not hash-bound: ' + relative)
                part = trimesh.load(path, force='mesh', process=False)
                part.vertices = part.vertices * entry['scale']
            elif kind == 'box':
                part = trimesh.creation.box(extents=entry['size'])
            elif kind == 'sphere':
                part = trimesh.creation.icosphere(subdivisions=3, radius=entry['radius'])
            elif kind == 'cylinder':
                part = trimesh.creation.cylinder(radius=entry['radius'], height=entry['length'], sections=128)
            else:
                raise ValueError('Unsupported grouping source geometry')
            part.apply_transform(entry['origin'])
            if (not len(part.faces) or not np.isfinite(part.vertices).all()
                    or not np.isfinite(part.area) or part.area <= 0):
                raise ValueError('Invalid grouping source surface')
            parts.append(part)
        value = trimesh.util.concatenate(parts)
        self.meshes[link] = value
        return value

    def equivalent(self, left, right):
        key = tuple(sorted((left, right)))
        if key in self.matches:
            return self.matches[key]
        a, b = self.mesh(left), self.mesh(right)
        if a is None or b is None:
            result = False
        elif isinstance(a, list) or isinstance(b, list):
            result = isinstance(a, list) and isinstance(b, list)
        else:
            tol = _tolerance(a, b)
            # Bounds are only a cheap rejection; actual surfaces must match too.
            result = (np.all(np.abs(a.bounds - b.bounds) <= tol)
                and abs(a.area - b.area) <= 4 * SHAPE_REL * min(a.area, b.area)
                and _near_surface(_surface_points(a), b, tol)
                and _near_surface(_surface_points(b), a, tol))
        self.matches[key] = bool(result)
        return bool(result)

    def ground_profile(self, scope, frames, root):
        name = scope['joint_name']
        if name in self.ground:
            return self.ground[name]
        joint = self.joints[name]
        world = {n: root @ pose for n, pose in frames.items()}
        axis = world[joint['child']][:3, :3] @ joint['axis']
        pivot = world[joint['child']][:3, 3]
        coefficients, diagonals = [], []
        for link in scope['moving_links']:
            mesh = self.mesh(link)
            if mesh is None:
                self.ground[name] = None
                return None
            if isinstance(mesh, list):
                continue
            points = np.unique(mesh.vertices, axis=0)
            # Height extrema depend only on hull vertices. This hull is used
            # solely for analysis, never as a collision proxy or shape test.
            if len(points) > 4 and np.linalg.matrix_rank(points - points[0]) == 3:
                points = points[ConvexHull(points).vertices]
            points = points @ world[link][:3, :3].T + world[link][:3, 3]
            c = np.zeros_like(points)
            if joint['type'] == 'prismatic':
                c[:, 1], c[:, 2] = axis[2], points[:, 2]
            else:
                relative = points - pivot
                parallel = np.outer(relative @ axis, axis)
                c[:, 0] = (relative - parallel)[:, 2]
                c[:, 1] = np.cross(axis, relative)[:, 2]
                c[:, 2] = pivot[2] + parallel[:, 2]
            coefficients.append(c)
            diagonals.append(float(np.linalg.norm(mesh.extents)))
        if not coefficients:
            self.ground[name] = None
            return None
        c = np.concatenate(coefficients)
        lower, upper = scope['full_lower'], scope['full_upper']
        count = 2 if joint['type'] == 'prismatic' else max(3, math.ceil((upper-lower) / GROUND_ANGLE_STEP) + 1)
        if count > 2049:
            self.ground[name] = None  # Unknown evidence never silently merges.
            return None
        q = np.unique(np.r_[np.linspace(lower, upper, count), scope['initial_position']]) - scope['initial_position']
        z = (c[:, 1, None] * q + c[:, 2, None] if joint['type'] == 'prismatic' else
             c[:, 0, None] * np.cos(q) + c[:, 1, None] * np.sin(q) + c[:, 2, None])
        heights = z.min(axis=0)
        tol = max(SHAPE_ABS_M, SHAPE_REL * min(diagonals))
        initial_height = float(heights[np.argmin(np.abs(q))])
        extra = max(0., min(0., initial_height) - float(heights.min()))
        profile = {'joint_name': name, 'q': q.tolist(), 'intrusion_m': np.maximum(0, -heights).tolist(),
            'initial_min_height_m': initial_height,
            'initial_ground_contact': initial_height <= tol,
            'tolerance_m': tol, 'max_source_intrusion_m': float(max(0, -heights.min())),
            'max_additional_source_intrusion_m': extra,
            'contact_class': 'surface_contact_or_clearance' if extra <= tol else 'source_ground_intrusion',
            'basis': 'source_surfaces_in_original_cooked_ground_placement',
            'scope': 'grouping_only; actual native motion scope and acceptance unchanged'}
        self.ground[name] = profile
        return profile

    def same_ground_obstruction(self, a, b, frames, root):
        pa, pb = self.ground_profile(a, frames, root), self.ground_profile(b, frames, root)
        if (pa is None or pb is None or pa['q'] != pb['q']
                or pa['initial_ground_contact'] != pb['initial_ground_contact']):
            return False
        tolerance = min(pa['tolerance_m'], pb['tolerance_m'])
        # Compare intrusion over the full path, not poorly conditioned endpoint
        # roots on a faceted surface already touching the floor. No root motion
        # is assumed here and no free-motion range is increased.
        return bool(np.max(np.abs(np.array(pa['intrusion_m']) - pb['intrusion_m'])) <= tolerance)

    def evidence(self, members):
        return {'policy': 'source_surface_role_local_ground.v1', 'shape_relative_tolerance': SHAPE_REL,
            'shape_absolute_tolerance_m': SHAPE_ABS_M, 'frame_numeric_tolerance': FRAME_ATOL,
            'ground_sampling_step_rad': GROUND_ANGLE_STEP,
            'external_obstacles': 'single_object_test_ground_plane_only',
            'ground_profiles': {n: {k: copy.deepcopy(v) for k, v in self.ground[n].items()
                if k not in {'q', 'intrusion_m'}} for n in members if self.ground.get(n) is not None}}
