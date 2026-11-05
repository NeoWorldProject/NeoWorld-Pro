"""Frame joint sweeps in the initialized world pose without duplicate fixed geometry."""
import numpy as np


def affected_links(parsed, joint_name):
    joint = next(j for j in parsed['joints'] if j['name'] == joint_name)
    descendants = {joint['child']}
    while True:
        expanded = descendants | {j['child'] for j in parsed['joints'] if j['parent'] in descendants}
        if expanded == descendants:
            return descendants
        descendants = expanded


def sweep_points(parsed, original, initial_positions, joint_name, definition, root_transform, initial_points, *, fixed):
    from urdf_core.urdf_validation import urdf_fk
    root_transform = np.asarray(root_transform, dtype=float)
    moving = affected_links(parsed, joint_name) if fixed else set(original)
    points = [initial_points]
    # Retain stationary coordinates once; descendants cover the complete sweep.
    for index, value in enumerate(np.linspace(definition['lower'], definition['upper'], 17)):
        frames = urdf_fk(parsed, {**initial_positions, joint_name: float(value)})
        for name, meshes in original.items():
            if index and name not in moving:
                continue
            for mesh in meshes:
                vertices = np.asarray(mesh['vertices'], dtype=float)
                transformed = vertices @ frames[name][:3, :3].T + frames[name][:3, 3]
                points.append(transformed @ root_transform[:3, :3].T + root_transform[:3, 3])
    return np.concatenate(points)
