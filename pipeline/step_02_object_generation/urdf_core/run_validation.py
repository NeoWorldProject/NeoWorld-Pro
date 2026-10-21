"""Read-only verification that a run's artifacts match its passing checkpoints."""
from __future__ import annotations

import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must contain an object: {path.name}")
    return value


def reject_assisted_provenance(root: Path, manifest: dict | None = None) -> None:
    """Check before RunState can replace old config; never erase assisted origin."""
    root = Path(root).resolve()
    manifest_path = root / "run_manifest.json"
    if manifest is None:
        manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    config = manifest.get("config", {})
    stages = manifest.get("stages", {})
    assisted = bool(config.get("review_refinement"))
    for stage in stages.values():
        if not isinstance(stage, dict):
            continue
        assisted |= any(stage.get(key) for key in
                        ("review_refinement", "seed_spec_path", "review_feedback_path", "review_key"))
        assisted |= any("review_refinement" in Path(relative).parts
                        or Path(relative).name == "review_feedback.json"
                        for relative in stage.get("artifacts", {}))
    # A removed manifest field cannot launder the retained archived feedback.
    archive = root / "review_refinement"
    assisted |= archive.exists() and (not archive.is_dir() or any(archive.iterdir()))
    if assisted:
        raise ValueError("Run has external review_refinement provenance; automatic acceptance requires --fresh in a new directory")


def verify_isaac_evidence(report_path: Path, spec_path: Path, urdf_path: Path,
                          *, root: Path, critic_path: Path | None = None,
                          threshold: float | None = None, path_resolver=None) -> dict:
    """Verify one current round's numerical, motion and model evidence bindings."""
    from .run_state import file_hash
    root = root.resolve()
    if path_resolver is None:
        path_resolver = lambda value: Path(value).resolve()
    report_path, spec_path, urdf_path = map(lambda p: Path(p).resolve(), (report_path, spec_path, urdf_path))
    evidence = set()

    def artifact(value, digest=None):
        if not isinstance(value, (str, Path)) or not str(value):
            raise ValueError("Isaac evidence has no artifact path")
        path = Path(value)
        if not path.is_absolute():
            path = report_path.parent / path
        path = path_resolver(path)
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Isaac evidence artifact is missing or outside this run: {path}")
        if digest is not None and file_hash(path) != digest:
            raise ValueError(f"Isaac evidence artifact hash changed: {path}")
        evidence.add(path)
        return path

    artifact(report_path)
    report = _read_json(report_path)
    if (report.get("passed") is not True or report.get("status") != "passed"
            or report.get("numerical_passed") is not True or report.get("worker_returncode") != 0
            or isinstance(report.get("worker_returncode"), bool)
            or report.get("errors") != [] or not report.get("poses")
            or any(pose.get("passed") is not True for pose in report["poses"])):
        raise ValueError("Isaac numerical/environment verification did not pass")
    for key, path in (("urdf", urdf_path), ("spec", spec_path)):
        if report.get(key + "_sha256") != file_hash(path):
            raise ValueError(f"Isaac report does not bind current {key}")
        artifact(report.get(key), report[key + "_sha256"])
    motion = report.get("motion_video", {})
    if report.get("motion_videos_passed") is not True or motion.get("passed") is not True or motion.get("enabled") is not True:
        raise ValueError("Isaac motion videos did not pass")
    manifest_path = artifact(motion.get("manifest_path"), motion.get("manifest_sha256"))
    if not isinstance(motion.get("manifest_sha256"), str):
        raise ValueError("Isaac motion manifest must be hash-bound")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported Isaac motion manifest")
    for key, path in (("urdf", urdf_path), ("spec", spec_path)):
        source = manifest.get(key, {})
        if source.get("sha256") != file_hash(path):
            raise ValueError(f"Isaac motion manifest does not bind current {key}")
        artifact(source.get("path"), source["sha256"])
    spec = _read_json(spec_path)
    expected = {joint["name"] for joint in spec["joints"] if joint["type"] != "fixed"}
    joints = manifest.get("joints", [])
    expected_entries = expected or {"__static__"}
    if (set(manifest.get("joint_names", [])) != expected
            or len(joints) != len(expected_entries)
            or {joint.get("name") for joint in joints} != expected_entries):
        raise ValueError("Isaac motion videos do not cover every joint exactly once")
    if not expected and (manifest.get("static") is not True or joints[0].get("synthetic") is not True):
        raise ValueError("Static Isaac evidence must be explicitly synthetic")
    for joint in joints:
        from kinematics_validation.isaac_motion import validate_capture_visibility
        validate_capture_visibility(manifest, joint, spec)
        views = joint.get("views", [])
        if len(views) != 2 or {view.get("name") for view in views} != {"overview", "closeup"}:
            raise ValueError("Isaac joint motion requires overview and closeup video")
        for view in views:
            if not isinstance(view.get("video_sha256"), str):
                raise ValueError("Isaac video must be hash-bound")
            artifact(view.get("video_path"), view["video_sha256"])
            frames = view.get("frames", [])
            if len(frames) < 2:
                raise ValueError("Isaac video must include a motion sequence")
            for frame in frames:
                if not isinstance(frame.get("sha256"), str):
                    raise ValueError("Isaac frame must be hash-bound")
                artifact(frame.get("path"), frame["sha256"])
    video_hash = manifest.get("overview_video_sha256", motion.get("video_sha256"))
    if not isinstance(video_hash, str):
        raise ValueError("Isaac overview video must be hash-bound")
    video = artifact(manifest.get("overview_video_path"), video_hash)
    if path_resolver(Path(motion.get("video_path", "")).resolve()) != video:
        raise ValueError("Isaac report and motion manifest refer to different overview videos")
    result = {"isaac_report_path": str(report_path), "isaac_video_path": str(video),
              "motion_manifest_path": str(manifest_path)}
    if critic_path is not None:
        from .isaac_critic import validate_review_result
        critic_path = artifact(critic_path)
        review = _read_json(critic_path)
        if path_resolver(Path(review.get("report_path", "")).resolve()) != critic_path:
            raise ValueError("Isaac API critic report path differs from this checkpoint")
        validate_review_result(review, spec=spec, verify_evidence=True, path_resolver=path_resolver)
        from .critic_policy import accepted as critic_accepted
        if not critic_accepted(review):
            raise ValueError("Isaac API motion critic reports an asset defect")
        if threshold is not None and review.get("threshold") != threshold:
            raise ValueError("Isaac API critic threshold differs from this request")
        if (review.get("worker_report_sha256") != file_hash(report_path)
                or review.get("motion_manifest_sha256") != file_hash(manifest_path)
                or review.get("spec_sha256") != file_hash(spec_path)):
            raise ValueError("Isaac API critic does not bind this report, motion manifest and spec")
        result["isaac_motion_review_path"] = str(critic_path)
    result["artifacts"] = sorted(evidence)
    return result


def _verify_stage_artifacts(root, manifest):
    """Verify checkpoint bytes without invoking run or revision validators."""
    from .run_state import file_hash

    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("run manifest has no stages")
    for required in ("geometry", "inventory", "articulation_export", "isaac_motion"):
        if required not in stages:
            raise ValueError(f"run manifest is missing required stage: {required}")
    hashed = set()
    export_hashed = set()
    isaac_hashed = set()
    for name, stage in stages.items():
        artifacts = stage.get("artifacts") if isinstance(stage, dict) else None
        if not isinstance(stage, dict) or stage.get("status") != "passed" or not isinstance(artifacts, dict) or not artifacts:
            raise ValueError(f"stage {name} is not a passing artifact checkpoint")
        for relative, digest in artifacts.items():
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("checkpoint artifact path must be relative and safe")
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != digest:
                raise ValueError(f"checkpoint artifact changed, missing or unsafe: {relative}")
            hashed.add(path)
            if name == "articulation_export":
                export_hashed.add(path)
            if name == "isaac_motion":
                isaac_hashed.add(path)
    return hashed, export_hashed, isaac_hashed


def verify_completed_run(object_run_dir: Path, expected_config: dict | None = None) -> dict:
    """Keep downstream acceptance strict while routing history to its own verifier."""
    from .delivery_policy import POLICY
    root = Path(object_run_dir).resolve()
    manifest = _read_json(root / 'run_manifest.json')
    if manifest.get('config', {}).get('harness_policy') != POLICY:
        raise ValueError('Unsupported historical run; use a new output with this release')
    from .asset_delivery import verify_run
    return verify_run(root, expected_config, require_pass=True)
