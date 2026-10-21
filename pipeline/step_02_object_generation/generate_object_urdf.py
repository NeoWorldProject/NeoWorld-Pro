from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


def __getattr__(name):
    # Native evidence replay does not import the networking SDK.
    if name == "OpenAI":
        from openai import OpenAI
        return OpenAI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def make_image_input(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "application/octet-stream"

    return {
        "type": "input_image",
        "image_url": f"data:{mime};base64,{encode_image(path)}",
    }


def make_text_input(text: str) -> dict:
    return {
        "type": "input_text",
        "text": text,
    }



def _call_model_openai(client, model, content, max_retries=1):
    """One streaming dispatch; the persistent request ledger owns all retries."""
    import time
    import uuid
    from urdf_core.throughput import record_api_event
    from urdf_core.api_metadata import RecordedText, metadata
    from urdf_core.prompt_cache import prepare
    effort = os.environ.get("URDF_REASONING_EFFORT", "").strip()
    if effort and effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Unsupported URDF_REASONING_EFFORT")
    wire, cache_options, cache_receipt = prepare(model, content)
    request = {"model": model, "input": [{"role": "user", "content": wire}], **cache_options}
    if effort:
        request["reasoning"] = {"effort": effort}
    attempt_id, started = uuid.uuid4().hex, time.monotonic()
    record_api_event("start", attempt_id=attempt_id, model=model, reasoning_effort=effort)
    evidence = None
    try:
        configured = client.with_options(max_retries=0, timeout=300) if hasattr(client, "with_options") else client
        with configured.responses.stream(**request) as stream:
            for event in stream:
                pass
            response = stream.get_final_response()
        evidence = metadata(response)
        evidence["prompt_cache_key"] = request["prompt_cache_key"]
        evidence["prompt_cache"] = cache_receipt
        status = getattr(response, 'status', 'completed')
        text = response.output_text
        if not isinstance(text, str):
            raise TypeError('Provider output_text is not a string')
        if status != 'completed':
            details = getattr(response, 'incomplete_details', None)
            reason = details.get('reason') if isinstance(details, dict) else getattr(details, 'reason', None)
            if status == 'incomplete' and reason in {'max_output_tokens', 'content_filter'}:
                from urdf_core.api_requests import RejectedContent
                text = RejectedContent(text, reason)
            else:
                raise RuntimeError(f'Provider returned incomplete response: {status}; reason={reason}')
        record_api_event("success", attempt_id=attempt_id, elapsed_seconds=time.monotonic()-started,
                         **evidence)
        if hasattr(text, "evidence"):
            text.api_metadata = evidence
            return text
        return RecordedText(text, evidence)
    except Exception as error:
        if evidence is not None:
            error.api_metadata = evidence
        record_api_event("failure", attempt_id=attempt_id, elapsed_seconds=time.monotonic()-started,
                         error_type=type(error).__name__, status_code=getattr(error,"status_code",None),
                         api_metadata=evidence)
        raise


def call_model(client: OpenAI, model: str, content: list[dict], max_retries: int = 1) -> str:
    from urdf_core.api_autonomy import assert_autonomous_context
    assert_autonomous_context(content)
    return _call_model_openai(client, model, content, max_retries=1)


def extract_code(text: str) -> str:
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def save_log(result: subprocess.CompletedProcess, log_path: Path) -> None:
    text = "\n".join([
        "=== STDOUT ===",
        result.stdout,
        "",
        "=== STDERR ===",
        result.stderr,
    ])
    write_text(log_path, text)


def combined_log(result: subprocess.CompletedProcess) -> str:
    return "\n".join([
        "=== STDOUT ===",
        result.stdout,
        "",
        "=== STDERR ===",
        result.stderr,
    ])


def add_runtime_arguments(parser):
    parser.add_argument('--runtime-config', type=Path, help='JSON executable and scheduler configuration')
    parser.add_argument('--blender')
    parser.add_argument('--isaac-python', '--isaac_python')
    parser.add_argument('--ffmpeg')
    parser.add_argument('--startup-timeout', type=float, help='Technical startup guard; default 300 seconds')
    parser.add_argument('--no-progress-timeout', type=float, help='Continuous lack of progress guard; default 600 seconds')
    parser.add_argument('--api-concurrency', type=int, help='Shared API scheduler limit; default 64')
    parser.add_argument('--model', default='gpt-6.1-sol')
    parser.add_argument('--reasoning-effort', choices=('low', 'medium', 'high', 'xhigh'), default=None)
    parser.add_argument('--num-model-candidates', '--num_model_candidates', type=int, choices=(3,), default=3)
    parser.add_argument('--max-generation-repairs', type=int, choices=range(4), default=3,
                        help='Repairs of the selected modeling candidate; maximum 3; no total time limit')
    parser.add_argument('--max-articulation-repairs', type=int, choices=range(4), default=3)
    parser.add_argument('--max-physics-repairs', type=int, choices=range(9), default=8)
    parser.add_argument('--joint-test-scope', choices=('representative', 'all'), default=None,
                        help='Default representative; omitted on resume retains the frozen scope')
    parser.add_argument('--physics-free-base', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable free-base motion; disabled for new experiments')
    parser.add_argument('--physics-trial-workers', type=int, choices=range(1, 5), default=2,
                        help='Isolated trial copies per object; shared CPU, memory and GPU availability control native process admission')


def build_parser():
    from urdf_core.paths import RESOURCE_ROOT
    parser = argparse.ArgumentParser(description='Generate and deliver a URDF/USD asset with an iteration-bounded Isaac loop')
    images = parser.add_mutually_exclusive_group()
    images.add_argument('--image', type=Path)
    images.add_argument('--image-dir', '--image_dir', type=Path)
    parser.add_argument('--out-dir', '--out_dir', '--output', type=Path)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--scene-context', '--scene_context', type=Path)
    parser.add_argument('--physics-from-run', '--physics_from_run', type=Path,
                        help='Import an existing verified generation asset into a new experiment')
    parser.add_argument('--physics-api-cache-from-run', type=Path,
                        help='Reuse recorded top-level physics API replies only when the full request matches')
    parser.add_argument('--articulation-from-run', type=Path,
                        help='Reuse generated geometry and articulation; rerun critics and subsequent physics in a new run')
    parser.add_argument('--physics-source-manifest', '--physics_source_manifest', type=Path,
                        help='Explicit hash-bound asset import; preserves parent experiment identity')
    parser.add_argument('--prompts-dir', '--prompts_dir', type=Path, default=RESOURCE_ROOT / 'prompts/urdf')
    parser.add_argument('--extra-model-instruction', '--extra_model_instruction', default='')
    parser.add_argument('--stop-after', '--stop_after', choices=('geometry', 'kinematics', 'all'), default='all',
                        help='Explicit partial work is recorded as incomplete; default all')
    parser.add_argument('--check-environment', action='store_true')
    add_runtime_arguments(parser)
    return parser


def main(argv=None):
    from urdf_core.environment import load_project_env, configure_runtime, inspect_environment
    from urdf_core.delivery_policy import POLICY, resolve
    load_project_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.check_environment:
        config = json.loads(args.runtime_config.read_text()) if args.runtime_config else {}
        for name in ('blender', 'isaac_python', 'ffmpeg'):
            value = getattr(args, name) or config.get(name)
            if value:
                from urdf_core.environment import TOOLS
                os.environ[TOOLS[name]] = value
        print(json.dumps(inspect_environment(), indent=2))
        return
    if args.out_dir is None:
        parser.error('--out-dir is required')
    if args.num_model_candidates < 1 or args.physics_trial_workers < 1:
        parser.error('Candidate and trial-worker counts must be positive')
    if sum(bool(v) for v in (args.physics_from_run, args.physics_source_manifest, args.articulation_from_run)) > 1:
        parser.error('Select one explicit asset source')
    if args.articulation_from_run and args.stop_after == 'geometry':
        parser.error('Articulation reevaluation starts after geometry generation')
    if (args.physics_from_run or args.physics_source_manifest) and args.stop_after != 'all':
        parser.error('Asset imports require the full pipeline')
    args.out_dir = args.out_dir.resolve()
    args.prompts_dir = args.prompts_dir.resolve()
    args.enable_physics = args.stop_after == 'all'
    args.delivery_policy = 'best-effort'
    args.harness_policy = POLICY
    args.physics_acceptance_policy = 'automatic_object_physics_v11'
    args.physics_config = None
    args.isaac_video_fps = 12
    args.isaac_joint_duration = 3.0
    args.alignment_threshold_articulation = 6
    for name in ('physics_from_run', 'physics_source_manifest', 'articulation_from_run', 'scene_context',
                 'physics_api_cache_from_run'):
        value = getattr(args, name)
        if value:
            setattr(args, name, value.resolve())
    if args.image is None and args.image_dir is None:
        if args.physics_source_manifest:
            from urdf_core.campaign_scene_context import source_manifest_image
            args.image, args.scene_context = source_manifest_image(
                args.physics_source_manifest, args.scene_context)
        elif args.physics_from_run or args.articulation_from_run:
            source = json.loads(((args.physics_from_run or args.articulation_from_run) / 'run_manifest.json').read_text())
            images = list(source['config']['source_images'])
            if len(images) != 1:
                parser.error('Multiple source images require --image-dir')
            args.image = Path(images[0])
    if args.image is None and args.image_dir is None:
        parser.error('Provide --image or --image-dir')
    image_dir = args.image_dir or (args.image if args.image and args.image.is_dir() else None)
    args.image_paths = (sorted(p.resolve() for p in image_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))
                       if image_dir else [args.image.resolve()])
    if not args.image_paths or any(not path.is_file() for path in args.image_paths):
        parser.error('No readable input images')
    resolve(args)
    configure_runtime(args)
    from urdf_core.orchestration import run_pipeline
    run_pipeline(args, sys.modules[__name__])


if __name__ == '__main__':
    main()
