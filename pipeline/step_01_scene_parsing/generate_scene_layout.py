"""Single-RGB scene parsing with the bounded, pure-API fast-v1 workflow."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
from scene_core.workflow import run_scene


def build_parser():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__,
        epilog="Only the current fast-v1 profile is supported.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", "--out-dir", "--out_dir", dest='out_dir', type=Path, required=True)
    parser.add_argument("--prompts-dir", "--prompts_dir", dest='prompts_dir', type=Path, default=project / "prompts/scene_layout")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--scene-id", "--scene_id", dest='scene_id', help="Stable output directory name; default is image stem")
    parser.add_argument("--quality-profile", choices=("fast-v1",), default="fast-v1")
    parser.add_argument("--perception-backend", choices=("api",), default="api")
    parser.add_argument("--max-refine-iters", "--max_refine_iters", dest='max_refine_iters', type=int, choices=range(4), default=3,
                        help="Content refinements after the initial candidate, maximum 3; no total time limit")
    parser.add_argument("--pass-score", "--pass_score", dest='pass_score', type=int, default=5, choices=range(1, 6))
    parser.add_argument("--reasoning-effort", "--reasoning_effort", dest='reasoning_effort', default="high", choices=("low", "medium", "high"))
    parser.add_argument("--thinking-budget", "--thinking_budget", dest='thinking_budget', type=int, default=8192, help="Qwen thinking token budget")
    parser.add_argument("--include-environment-crops", "--include_environment_crops", dest='include_environment_crops', action="store_true")
    parser.add_argument("--force-layout", "--force_layout", dest='force_layout', action="store_true", help="Archive this output and start a fresh run")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error("--image must be an existing RGB image")
    if args.thinking_budget < 1:
        parser.error("--thinking_budget must be positive")
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        summary = run_scene(args)
    except Exception as exc:
        from scene_core.provider import safe_error
        print(safe_error(exc), file=sys.stderr, flush=True)
        return 1
    return 0 if summary["accepted"] else (2 if summary["status"] == "unaccepted" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
