"""Shared scene contracts, crop export and content-bound artifact validation."""
from __future__ import annotations
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from .provider import digest_bytes, write_json
from .geometry import geometry_diagnostics
from .visualization import render_layout_visualizations


def extract_json(text: str) -> dict:
    value = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL)
    if match:
        value = match.group(1)
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    parsed = json.loads(value, parse_constant=invalid_constant)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def get_eval_metrics(data: dict) -> tuple[int, str, list]:
    qa = data.get("quality_assessment", {})
    return qa.get("overall_score", 0), str(qa.get("verdict", "")).lower(), data.get("issues", [])


def validate_critic(data: dict, layout: dict, require_perception: bool = False) -> list[str]:
    qa = data.get("quality_assessment")
    if not isinstance(qa, dict):
        return ["Missing quality_assessment object"]
    errors = []
    if require_perception:
        for field in ("localization_pass", "geometry_pass"):
            if type(data.get(field)) is not bool:
                errors.append(f"{field} must be an explicit boolean")
        if data.get("localization_pass") is False or data.get("geometry_pass") is False:
            if qa.get("verdict") == "PASS" or not data.get("issues"):
                errors.append("A failed perception dimension requires issues and a non-PASS verdict")
    score = qa.get("overall_score")
    if type(score) is not int or not 1 <= score <= 5:
        errors.append("overall_score must be an integer from 1 to 5")
    if qa.get("verdict") not in {"PASS", "NEEDS_MINOR_FIX", "NEEDS_MAJOR_REWRITE"}:
        errors.append("Invalid critic verdict")
    if not isinstance(qa.get("summary"), str):
        errors.append("Critic summary must be text")
    issues = data.get("issues")
    if not isinstance(issues, list):
        return errors + ["issues must be a list"]
    valid_ids = {o["object_id"] for o in layout["objects_data"]} | {"Global", "global"}
    for i, issue in enumerate(issues):
        if not isinstance(issue, dict):
            errors.append(f"issues[{i}] must be an object")
            continue
        if issue.get("object_id") not in valid_ids:
            errors.append(f"issues[{i}].object_id must reference an object or Global")
        for key in ("issue_type", "description", "suggestion"):
            if not isinstance(issue.get(key), str):
                errors.append(f"issues[{i}].{key} must be text")
        if issue.get("severity") not in {"minor", "major"}:
            errors.append(f"issues[{i}].severity must be minor or major")
    return errors


def is_pass(data: dict, pass_score: int = 5) -> bool:
    score, verdict, issues = get_eval_metrics(data)
    dimensions = all(data.get(key, True) is True for key in ("localization_pass", "geometry_pass", "orientation_pass"))
    return verdict == "pass" and score >= pass_score and not issues and dimensions


def sanitize_name(name: str) -> str:
    return "".join(c for c in name.lower().replace(" ", "_") if c.isalnum() or c == "_").strip("_") or "object"


def crop_from_bbox(image: Image.Image, bbox: list[float]) -> tuple[Image.Image, list[int]]:
    if len(bbox) != 4 or not all(type(x) in (int, float) and math.isfinite(x) for x in bbox):
        raise ValueError("Invalid crop box")
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("Crop box must be ordered and normalized")
    w, h = image.size
    pixels = [math.floor(x0*w), math.floor(y0*h), math.ceil(x1*w), math.ceil(y1*h)]
    return image.crop(pixels), pixels


def generate_crops_from_layout(image_path: Path, layout: dict, out_dir: Path,
                               skip_environment: bool = True) -> dict:
    crops = out_dir / "crops"
    crops.mkdir(exist_ok=True)
    records = []
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    for obj in layout["objects_data"]:
        record = {"object_id": obj["object_id"], "name": obj["name"]}
        if skip_environment and obj["is_environment_support"]:
            record["status"] = "skipped_environment"
        elif obj["2d_bbox_normalized"] is None:
            record["status"] = "skipped_no_bbox"
        else:
            from .recovery import identity_stem
            name = identity_stem(obj['object_id'], obj['name']) + '.png'
            crop, pixels = crop_from_bbox(image, obj["2d_bbox_normalized"])
            crop_path = crops / name
            crop.save(crop_path)
            record.update({"description": obj["description"], "is_environment_support": obj["is_environment_support"],
                "is_occluded": obj["is_occluded"], "crop_path": str(crop_path.resolve()),
                "crop_bbox_pixels": pixels, "bbox_2d_normalized": obj["2d_bbox_normalized"],
                "bbox_3d_meters": obj["3d_bbox_meters"], "obb": obj["obb"], "status": "success"})
        records.append(record)
    manifest = {"scene_image": str(image_path.resolve()), "layout_source": str((out_dir / "final_layout.json").resolve()),
                "crops_dir": str(crops.resolve()), "skip_environment": skip_environment, "objects": records}
    write_json(out_dir / "crops_manifest.json", manifest)
    return manifest


def _fingerprint(args, config: dict, prompts: dict) -> tuple[str, dict]:
    stage = Path(__file__).resolve().parents[1]
    files = [stage / "generate_scene_layout.py", *sorted((stage / "scene_core").glob("*.py"))]
    binding = {"schema_version": "scene_layout.v2", "image_sha256": digest_bytes(args.image.read_bytes()),
        "scene_id": args.scene_id or args.image.stem, "provider": config,
        "max_refine_iters": args.max_refine_iters, "pass_score": args.pass_score,
        "include_environment_crops": args.include_environment_crops,
        "prompts": {k: digest_bytes(v.encode()) for k, v in prompts.items()},
        "code": {str(p.relative_to(stage)): digest_bytes(p.read_bytes()) for p in files}}
    cache_source = stage.parent / "step_02_object_generation/urdf_core/prompt_cache.py"
    binding["code"]["../step_02_object_generation/urdf_core/prompt_cache.py"] = digest_bytes(cache_source.read_bytes())
    from .fast_workflow import fast_policy
    binding["quality_profile"] = "fast-v1"
    binding["fast_policy"] = fast_policy(args)
    return digest_bytes(json.dumps(binding, sort_keys=True).encode()), binding


def _stage_one_owned(relative: Path) -> bool:
    """Explicit ownership lets any downstream output share this scene folder."""
    if not relative.parts:
        return False
    if relative.parts[0] in {"api_calls", "external_calls", "crops", "perception", "orientation", "fast"}:
        return True
    if len(relative.parts) != 1:
        return False
    return (relative.name in {"inventory.json", "final_layout.json", "crops_manifest.json",
                              "input_audit.json", "refine_summary.json", "perception_input_manifest.json", "run_deadline.json",
                              "degraded_layout.json", "degradation_receipt.json"}
            or bool(re.fullmatch(r"(?:inventory|identity_events|localization(?:_review)?|critic_raw|layout(?:_[a-z0-9]+)?)_round_\d+\.(?:json|png)", relative.name)))


def _artifacts(out_dir: Path) -> dict:
    files = []
    for entry in out_dir.iterdir():
        if not _stage_one_owned(entry.relative_to(out_dir)):
            continue
        files.extend(entry.rglob("*") if entry.is_dir() else [entry])
    return {str(path.relative_to(out_dir)): digest_bytes(path.read_bytes()) for path in sorted(files)
            if path.is_file() and not path.name.endswith(".tmp") and path.name != "run_manifest.json"}


def _verify_artifacts(out_dir: Path, expected: dict) -> bool:
    # Old manifests may contain additional owned diagnostics: verify every
    # recorded file, while allowing independently written downstream products.
    if not isinstance(expected, dict) or not expected:
        return False
    root = out_dir.resolve()
    for relative, expected_hash in expected.items():
        if not isinstance(relative, str) or not relative:
            return False
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            return False
        if not _stage_one_owned(relative_path):
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest_bytes(path.read_bytes()) != expected_hash:
            return False
    return all(expected.get(name) == value for name, value in _artifacts(out_dir).items())


def _archive(out_dir: Path) -> None:
    existing = [p for p in out_dir.iterdir() if p.name != "history"]
    if not existing:
        return
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = out_dir / "history" / name
    destination.mkdir(parents=True)
    for path in existing:
        shutil.move(str(path), str(destination / path.name))


def _archive_failed_delivery(out_dir: Path, manifest: dict) -> None:
    """Retain failed derived output while replaying the original API receipts.

    Completed responses and attempt reservations stay in place, so recovering
    local setup or finalization cannot grant new model or refinement budgets.
    """
    name = datetime.now(timezone.utc).strftime("technical_%Y%m%dT%H%M%S_%fZ")
    destination = out_dir / "history" / name
    destination.mkdir(parents=True)
    write_json(destination / 'run_manifest.json', manifest)
    # A process killed during archival must remain resumable from cached calls.
    write_json(out_dir / 'run_manifest.json', {**manifest, 'status': 'running'})
    for path in list(out_dir.iterdir()):
        if path.name != 'api_calls' and _stage_one_owned(path.relative_to(out_dir)):
            shutil.move(str(path), str(destination / path.name))
