"""Frustum evidence for fixed, readable semantic views (no candidate following)."""
import numpy as np


def in_camera(points, camera):
    points = np.asarray(points, dtype=float)
    eye = np.asarray(camera["position"], dtype=float)
    forward = np.asarray(camera["look_at"], dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = points - eye
    z = delta @ forward
    half_x = camera["horizontal_aperture"] / (2 * camera["focal_length"])
    half_y = half_x * camera["resolution"][1] / camera["resolution"][0]
    near, far = camera["clipping_range"]
    return bool(np.all(z > near) and np.all(z < far)
                and np.all(np.abs(delta @ right) <= z * half_x * .98)
                and np.all(np.abs(delta @ up) <= z * half_y * .98))
