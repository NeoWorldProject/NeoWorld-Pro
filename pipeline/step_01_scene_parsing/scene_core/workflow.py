"""Public dispatcher for the supported pure-API fast-v1 parser."""
from .provider import ModelAPI


def run_scene(args, api_factory=ModelAPI):
    if getattr(args, "quality_profile", "fast-v1") != "fast-v1":
        raise ValueError("Unsupported scene parsing profile; use fast-v1")
    if getattr(args, "perception_backend", "api") != "api":
        raise ValueError("Retired perception backend; only API fast-v1 is supported")
    from .fast_workflow import run_scene as fast_run
    return fast_run(args, api_factory)
