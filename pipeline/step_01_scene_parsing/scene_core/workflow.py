"""Dispatcher for scene layout generation and review."""
from .provider import ModelAPI


def run_scene(args, api_factory=ModelAPI):
    if getattr(args, "quality_profile", "fast-v1") != "fast-v1":
        raise ValueError("Unsupported scene parsing profile; use fast-v1")
    if getattr(args, "perception_backend", "api") != "api":
        raise ValueError("Retired perception backend; only API fast-v1 is supported")
    from .layout_workflow import run_scene as run_layout
    return run_layout(args, api_factory)
