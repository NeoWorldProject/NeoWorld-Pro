"""Reconstruct measured states using explicit current joint topology."""
from .rest_state import measured_body_transforms


def initial_state_transforms(stage, bodies, root_link, pose):
    return measured_body_transforms(stage, bodies, root_link, pose)
