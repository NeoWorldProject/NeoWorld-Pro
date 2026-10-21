"""Image -> free geometry -> shared kinematics -> verified deterministic export."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .modeling import generate_geometry, run_blender
from .run_state import RunState, file_hash, fingerprint, write_json
from .run_validation import reject_assisted_provenance, verify_isaac_evidence
from .paths import KINEMATICS_VALIDATION, OBJECT_GENERATION, URDF_CORE
from .generation_research import cached_response
from .generation_research import GenerationResearch
from .isaac_cache import evidence_files
from .critic_policy import POLICY, blender_decision, accepted as critic_accepted
from .environment import runtime_identity
from kinematics_validation.isaac_motion import capture_workload


def _hash_files(paths):
    return {str(p.resolve()): file_hash(p) for p in paths}


class CandidateRejected(ValueError):
    """A generated candidate failed an executable quality/structure check."""


def _runtime(args, api, operation, target, spec=None, render=False):
    helper = URDF_CORE / "blender_runtime.py"
    cmd = [args.blender, "--background", "--factory-startup", "--disable-autoexec",
           "--threads", "4", "--python-exit-code", "1", "--python", str(helper), "--", operation,
           "--blend", str(args.out_dir / "geometry_baseline.blend"), "--out-dir", str(target),
           "--ffmpeg", args.ffmpeg]
    if spec is not None:
        cmd.extend(["--spec", str(spec)])
    if render:
        cmd.append("--render")
    target.mkdir(parents=True, exist_ok=True)
    print(f"Blender {operation}: {target}", flush=True)
    result = run_blender(cmd, cwd=target, log_path=target / "runtime.log", api=api)
    if result.returncode:
        if operation == 'build':
            raise CandidateRejected(f"Blender build rejected generated kinematics; inspect {target / 'runtime.log'}")
        from .execution import TechnicalExecutionError
        raise TechnicalExecutionError(f"Deterministic Blender {operation} failed; inspect {target / 'runtime.log'}")


def _isaac_settings(args):
    return {
        "python": str(args.isaac_python),
        "video_fps": getattr(args, "isaac_video_fps", 12),
        "joint_duration": getattr(args, "isaac_joint_duration", 3),
    }


def _isaac_capture_plan(settings, round_dir):
    spec = json.loads((round_dir / "kinematic_spec.json").read_text())
    return capture_workload(spec, fps=settings["video_fps"], duration=settings["joint_duration"])


def _isaac_phase_key(args, round_dir):
    """Bind the exact native workload independently of the outer stage cache."""
    settings = _isaac_settings(args)
    return {"settings": settings, "capture_plan_sha256": fingerprint(_isaac_capture_plan(settings, round_dir)),
            "spec_sha256": file_hash(round_dir / "kinematic_spec.json"),
            "urdf_sha256": file_hash(round_dir / "urdf_export/object.urdf"),
            "sources": _hash_files([KINEMATICS_VALIDATION / "verify_urdf_isaac.py",
                                   KINEMATICS_VALIDATION / "isaac_motion.py"])}


def _articulation_implementation():
    return fingerprint({p.name: file_hash(p) for p in
        [Path(__file__), URDF_CORE / "generation_research.py",
         URDF_CORE / "articulation_import.py",
         URDF_CORE / "blender_runtime.py", URDF_CORE / "blender_mesh_orientation.py", URDF_CORE / "kinematics.py", URDF_CORE / "isaac_cache.py",
         KINEMATICS_VALIDATION / "verify_urdf_isaac.py", KINEMATICS_VALIDATION / "isaac_motion.py",
         KINEMATICS_VALIDATION / "capture_execution_evidence.py", URDF_CORE / "isaac_critic.py",
         URDF_CORE / "isaac_review_diagnostics.py", URDF_CORE / "api_requests.py",
         URDF_CORE / "prompt_payload.py", URDF_CORE / "api_metadata.py", URDF_CORE / "review_frames.py",
         URDF_CORE / "run_validation.py", URDF_CORE / "critic_policy.py", URDF_CORE / "motion_range_guard.py"]})


def _run_isaac(args, round_dir):
    """Dispatch motion validation through the native supervisor's GPU queue."""
    settings = _isaac_settings(args)
    target = round_dir / "isaac_validation"
    target.mkdir(parents=True, exist_ok=True)
    worker = KINEMATICS_VALIDATION / "verify_urdf_isaac.py"
    command = [sys.executable, str(worker), "--urdf", str(round_dir / "urdf_export/object.urdf"),
               "--spec", str(round_dir / "kinematic_spec.json"), "--out-dir", str(target),
               "--isaac-python", settings["python"],
               "--video-fps", str(settings["video_fps"]), "--joint-duration", str(settings["joint_duration"]),
               "--ffmpeg", str(args.ffmpeg)]
    print(f"Isaac motion verification (GPU queue): {target}", flush=True)
    with (target / "orchestration.log").open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait()
        except BaseException:
            from .execution import stop_process
            stop_process(process)
            raise
    result = subprocess.CompletedProcess(command, returncode)
    if result.returncode:
        from .execution import TechnicalExecutionError
        report_path = target / 'report.json'
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        if report.get('status') in ('failed', 'asset_rejected'):
            raise CandidateRejected('Isaac rejected generated kinematics: ' + json.dumps(report.get('errors', [])))
        raise TechnicalExecutionError(f"Isaac technical verification failed after native recovery; see {report_path}")
    return target / "report.json"


def _review_isaac_motion(**kwargs):
    from .isaac_critic import review_isaac_motion
    return review_isaac_motion(**kwargs)


def _critic_frames(args, video, target, spec):
    """Sample each joint segment, including short or numerous secondary joints."""
    target.mkdir(parents=True, exist_ok=True)
    joints = [j for j in spec["joints"] if j["type"] != "fixed"]
    sampling = video.parent / "preview_sampling.json"
    if sampling.is_file():
        samples = json.loads(sampling.read_text()).get("samples", [])
        if samples:
            paths = []
            for index, joint in enumerate(joints or [{"name": None}]):
                group = [s for s in samples if s["joint"] == joint["name"]]
                expected_frames = {index * 90 + offset for offset in (1, 30, 60)}
                selected = [s for s in group if s["scene_frame"] in expected_frames]
                if len(selected) != 3:
                    raise RuntimeError(f"Preview does not cover rest and both limits: {joint['name']}")
                for sample in selected:
                    source = Path(sample["path"]).resolve()
                    if not source.is_relative_to(video.parent.resolve()) or not source.is_file():
                        raise RuntimeError("Preview sample must belong to the current round")
                    copied = target / source.name
                    shutil.copy2(source, copied)
                    label = f"{joint['name'] or 'static'}; joint coordinates {sample['positions']}"
                    paths.append((label, copied))
            return paths
    count = max(1, len(joints))
    # Three pictures per 3-second joint segment; no temporal stride that can skip a joint.
    paths = []
    for joint_index in range(count):
        for sample_index, offset in enumerate((0.0, 29 / 30, 59 / 30)):
            path = target / f"joint_{joint_index:03d}_{sample_index}.png"
            from .media_execution import run_capture
            command = [args.ffmpeg, "-loglevel", "error", "-y", "-ss",
                            str(joint_index * 3.0 + offset), "-i", str(video),
                            "-frames:v", "1", str(path)]
            code = run_capture(command, ledger_path=path.with_suffix('.ffmpeg.json'),
                               binding={'video_sha256': file_hash(video)}, outputs=[path])
            if code:
                raise CandidateRejected(f'Candidate preview could not be decoded: {video}; exit={code}')
            if not path.is_file():
                raise CandidateRejected(f"Missing actual preview frame: {path}")
            paths.append((joints[joint_index]["name"] if joints else "static", path))
    return paths


def _accepted_critique(value):
    if not isinstance(value, dict) or value.get('review_policy') != POLICY:
        raise ValueError('Current articulation critique must use ' + POLICY)
    return critic_accepted(value)


def run_pipeline(args, api):
    """Run generation and, when selected, the object physics loop."""
    from .runtime_snapshot import dispatch
    if dispatch(args):
        return
    if getattr(args, "enable_physics", False) and args.stop_after != "geometry":
        from physics_feedback.loop import run_physics_pipeline
        return run_physics_pipeline(args, api, run_kinematics_pipeline)
    return run_kinematics_pipeline(args, api)


def run_kinematics_pipeline(args, api):
    package = URDF_CORE
    image_paths = args.image_paths
    source_hashes = _hash_files(image_paths)
    from .target_reference import load_reference, reference_content, verify_source_images
    target_reference = load_reference(getattr(args, 'scene_context', None), args.prompts_dir)
    verify_source_images(target_reference, image_paths)
    articulation_import = None
    if getattr(args, "articulation_from_run", None):
        from .articulation_import import bind
        articulation_import = bind(args.articulation_from_run, source_hashes, target_reference)
    if any(getattr(args, key, None) for key in
           ("seed_spec_path", "review_feedback_path", "review_feedback_source_path", "rebuild_from_seed", "_physics_geometry_source", "_physics_geometry_code",
            "_physics_source_digest", "_physics_reference_spec", "_physics_kinematics_seed", "_physics_evidence_content")):
        raise ValueError("External case feedback is not supported; use the automatic pipeline with --fresh")
    reject_assisted_provenance(args.out_dir)
    isaac_settings = _isaac_settings(args)
    provider_identity = fingerprint({"provider": "openai",
        "endpoint": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")})
    config = {"model": args.model, "reasoning_effort": os.environ.get("URDF_REASONING_EFFORT", ""),
              "provider_identity": provider_identity,
              "num_model_candidates": args.num_model_candidates, "source_images": source_hashes,
              "extra_model_instruction": args.extra_model_instruction,
              "target_reference": target_reference,
              "alignment_threshold_articulation": args.alignment_threshold_articulation,
              "blender": args.blender, "runtime": runtime_identity(args),
              "implementation": fingerprint({"articulation": _articulation_implementation(),
                  "generation": _hash_files([URDF_CORE / "modeling.py", URDF_CORE / "blender_mesh_orientation.py", OBJECT_GENERATION / "generate_object_urdf.py"])}),
              "prompts": _hash_files([args.prompts_dir / name for name in (
                  "1_model.md", "debug.md", "1b_model_selection.md", "1c_model_critic.md", "3_articulation.md",
                  "4_articulation_critic.md", "4b_articulation_refine.md", "6_isaac_motion_critic.md")]),
              "acceptance_policy": "automatic_isaac_motion_v1", "isaac": isaac_settings}
    config.update(harness_policy='object_pipeline.v1', delivery_policy='best-effort',
                  geometry_repairs=args.max_generation_repairs,
                  articulation_repairs=getattr(args, 'max_articulation_repairs', 3),
                  runtime_snapshot=getattr(args, '_runtime_snapshot', None))
    if articulation_import:
        config["articulation_import"] = articulation_import
    if any(getattr(args, name, None) for name in
           ('generation_revision', 'generation_review_receipt', 'isaac_capture_recovery')):
        raise ValueError('Historical continuations require their frozen implementation')
    prior_manifest_path = args.out_dir / "run_manifest.json"
    prior_manifest = json.loads(prior_manifest_path.read_text()) if prior_manifest_path.is_file() else None
    if prior_manifest:
        if prior_manifest.get('config', {}).get('harness_policy') != 'object_pipeline.v1':
            raise ValueError('Historical generation runs require their frozen implementation; use a new output directory')
        if prior_manifest.get('config') != config:
            raise ValueError('Frozen generation configuration changed; use a new output directory')
    geometry_key = fingerprint({
        "images": source_hashes, "model": args.model, "effort": config["reasoning_effort"],
        "provider_identity": provider_identity,
        "candidates": args.num_model_candidates, "extra": args.extra_model_instruction,
        "blender": args.blender,
        "implementation": _hash_files([package / "modeling.py", package / "blender_mesh_orientation.py", OBJECT_GENERATION / "generate_object_urdf.py"]),
        "prompts": _hash_files([args.prompts_dir / n for n in ("1_model.md", "debug.md", "1b_model_selection.md", "1c_model_critic.md")]),
        "target_reference": target_reference,
    })
    if articulation_import:
        geometry_key = fingerprint({"geometry": geometry_key, "articulation_import": articulation_import})
    state = RunState(args.out_dir, args.fresh, config)
    from .delivery_policy import RunTiming
    args._run_timing = RunTiming(args.out_dir)
    import threading
    local = threading.local()
    call_artifacts = []

    def ask(stage, content, validator=None):
        from .prompt_payload import prepare_content
        review_content = prepare_content(content)
        call_stage = stage
        request = {"model": args.model, "reasoning_effort": config["reasoning_effort"],
                   "provider_identity": provider_identity, "source_images": source_hashes,
                   "content": [{"type": item["type"], **({"text": item["text"]} if item["type"] == "input_text"
                               else {"image_sha256": fingerprint(item["image_url"])})} for item in review_content]}
        def prepare_client():
            if not hasattr(local, 'client'):
                kwargs = {"timeout": 300, "max_retries": 0}
                if os.environ.get("OPENAI_BASE_URL"):
                    kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
                local.client = api.OpenAI(**kwargs)
        def dispatch():
            return api.call_model(local.client, args.model, review_content, max_retries=1)
        response, paths = cached_response(args.out_dir, call_stage, request, dispatch, prepare=prepare_client)
        call_artifacts.extend(p for p in paths if p not in call_artifacts)
        if validator is not None:
            validator(response)
        return response

    images = (reference_content(target_reference, api, args.out_dir) if target_reference else
              [api.make_image_input(p) for p in image_paths])
    try:
        if state.current("geometry", geometry_key):
            print("Resume: verified current geometry checkpoint", flush=True)
        else:
            state.begin("geometry", geometry_key)
            if articulation_import:
                from .articulation_import import copy_geometry
                artifacts = copy_geometry(articulation_import, args.out_dir)
                print("Imported generated geometry; new articulation reviews remain required.", flush=True)
            else:
                artifacts = generate_geometry(args, api, ask, images)
            state.passed("geometry", geometry_key, artifacts, semantic_evaluation="independent_selected_geometry_review",
                         selection_policy="imported" if articulation_import else "original")
        if args.stop_after == "geometry":
            state.data["status"] = "geometry_complete"
            state.save()
            print("Geometry checkpoint saved; remaining stages have not run.", flush=True)
            return

        from .kinematics import validate_spec, validate_motion_refinement
        from .urdf_validation import validate_urdf
        baseline = args.out_dir / "geometry_baseline.blend"
        inventory_dir = args.out_dir / "geometry_inventory"
        inventory_key = fingerprint({"baseline": file_hash(baseline),
                                     "implementation": _hash_files(sorted(package.glob("*.py")))})
        if not state.current("inventory", inventory_key):
            state.begin("inventory", inventory_key)
            _runtime(args, api, "inspect", inventory_dir)
            files = [p for p in inventory_dir.iterdir() if p.is_file() and p.name != "runtime.log"]
            state.passed("inventory", inventory_key, files)
        inventory = json.loads((inventory_dir / "inventory.json").read_text())
        articulation_inputs = {
            "geometry": geometry_key, "inventory": inventory_key,
            "model": args.model, "effort": config["reasoning_effort"], "provider_identity": provider_identity,
            "threshold": args.alignment_threshold_articulation, "ffmpeg": args.ffmpeg,
            "max_articulation_repairs": getattr(args, 'max_articulation_repairs', 3),
            "isaac": isaac_settings,
            "isaac_verifier": file_hash(KINEMATICS_VALIDATION / "verify_urdf_isaac.py"),
            "isaac_motion_recorder": file_hash(KINEMATICS_VALIDATION / "isaac_motion.py"),
            "isaac_review_sources": _hash_files([package / "isaac_critic.py", package / "isaac_review_diagnostics.py",
                                                   package / "review_frames.py",
                                                   KINEMATICS_VALIDATION / "capture_execution_evidence.py"]),
            "acceptance_policy": "automatic_isaac_motion_v1",
            "prompts": _hash_files([args.prompts_dir / n for n in
                                   ("3_articulation.md", "4_articulation_critic.md", "4b_articulation_refine.md", "6_isaac_motion_critic.md")]),
        }
        articulation_key = fingerprint(articulation_inputs)
        canonical_urdf = args.out_dir / "urdf_export" / "object.urdf"
        canonical_spec = args.out_dir / "kinematics.json"
        if (state.current("articulation_export", articulation_key)
                and state.current("isaac_motion", articulation_key)):
            spec = validate_spec(json.loads(canonical_spec.read_text()), inventory)
            structural = validate_urdf(canonical_urdf, spec)
            if not structural["passed"]:
                raise RuntimeError("Checkpoint files failed independent URDF validation: " + str(structural))
            print("Resume: verified current articulation and export checkpoint", flush=True)
        else:
            state.begin("articulation_export", articulation_key)
            prompt = (args.prompts_dir / "3_articulation.md").read_text()
            if args.extra_model_instruction:
                prompt += "\nAdditional object instruction:\n" + args.extra_model_instruction
            context = [*images, api.make_image_input(args.out_dir / "Res1_3D.png"),
                       api.make_text_input("Selected modeling code (quality acceptance is recorded separately):\n" + (args.out_dir / "1_model" / "Res1_3D_model_selected.py").read_text()),
                       api.make_text_input("Actual evaluated scene inventory:\n" + json.dumps(inventory)),
                       api.make_text_input(prompt)]
            research = GenerationResearch(args.out_dir, "articulation", baseline={
                "images": sorted(source_hashes.values()), "model": args.model,
                "extra_instruction": args.extra_model_instruction,
                "target_reference": target_reference,
                "geometry_baseline": file_hash(args.out_dir / "geometry_baseline.blend"),
                "inventory": inventory,
                "alignment_threshold_articulation": args.alignment_threshold_articulation},
                implementation=_articulation_implementation(),
                limit=getattr(args, 'max_articulation_repairs', 3))
            accepted = False
            while True:
                history = list(research.data["trials"].values())
                last = history[-1] if history else None
                round_index = (len(history) if last and last["status"] in {"pending", "passed"}
                               and last["implementation"] == research.implementation else len(history) + 1)
                token = f"round_{round_index}"
                next_context = last.get("result") if last else None
                trial = research.start(token, context=next_context)
                previous = trial["context"] or {}
                current_spec = previous.get("spec")
                critique = previous.get("critique")
                frame_records = previous.get("frames", [])
                frame_content = []
                for label, path in frame_records:
                    frame_content.extend([api.make_text_input(label), api.make_image_input(Path(path))])
                round_calls_start = len(call_artifacts)
                round_dir = args.out_dir / "3_articulation" / f"round_{round_index}"
                round_dir.mkdir(parents=True, exist_ok=True)
                spec_path = round_dir / "kinematics.json"
                refinement_source = current_spec
                from .motion_range_guard import requirements as motion_requirements, validate as validate_motion_travel
                required_travel = motion_requirements(research.data["trials"], refinement_source)
                content = context if not previous else [*context, *frame_content,
                    api.make_text_input("Current kinematics:\n" + json.dumps(current_spec)),
                    api.make_text_input("Critique:\n" + json.dumps(critique)),
                    api.make_text_input((args.prompts_dir / "4b_articulation_refine.md").read_text())]
                def proposal():
                    if articulation_import and round_index == 1:
                        from .articulation_import import proposal as import_proposal
                        return import_proposal(articulation_import, round_dir / "proposal_response.txt")
                    proposal_content = [*content, api.make_text_input(
                        "Frozen minimum evaluated joint travel (derive corrections without shortening this coverage):\n"
                        + json.dumps(required_travel, ensure_ascii=False)), api.make_text_input(
                        "Prior experiments and reviewed next strategy (observations and hypotheses, not ground truth):\n"
                        + json.dumps(research.refinement_context(token), ensure_ascii=False))]
                    raw = ask(f"kinematics_round_{round_index}_try_1", proposal_content, validator=api.extract_json)
                    raw_path = round_dir / "proposal_response.txt"
                    raw_path.write_text(raw)
                    return raw, [raw_path, *call_artifacts[round_calls_start:]]
                try:
                    raw = research.phase(token, "proposal", proposal)
                    try:
                        proposed = api.extract_json(raw)
                        current_spec = (validate_spec(proposed, inventory) if refinement_source is None else
                                        validate_motion_refinement(refinement_source, proposed, inventory))
                        validate_motion_travel(current_spec, required_travel)
                    except (ValueError, KeyError, TypeError) as error:
                        failure_path = round_dir / "contract_failure.json"
                        write_json(failure_path, {"error": str(error), "response": raw})
                        research.checked(token, passed=False, reason="kinematics_contract", paths=[failure_path], result={
                            "spec": refinement_source, "frames": frame_records,
                            "critique": {"stage": "kinematics_contract", "error": str(error),
                                         "previous_feedback": critique, "invalid_previous_response": raw}})
                        continue

                    def build():
                        write_json(spec_path, current_spec)
                        _runtime(args, api, "build", round_dir, spec_path, render=True)
                        return True, evidence_files(round_dir)
                    research.phase(token, "build", build)
                    report = json.loads((round_dir / "validation_report.json").read_text())
                    if not report.get("passed"):
                        raise CandidateRejected(f"Geometry/kinematics verification failed: {round_dir / 'validation_report.json'}")
                    runtime_spec = round_dir / "kinematic_spec.json"
                    if report.get("spec_sha256") != file_hash(runtime_spec):
                        raise RuntimeError("Blender validation report does not bind the actual exported protocol")
                    # Runtime validation can change a normalized arbitrary axis by
                    # one floating-point ULP. Evaluate/refine the exact protocol
                    # that produced the rig and will be published; do not normalize
                    # it again or weaken the evidence's exact content binding.
                    current_spec = json.loads(runtime_spec.read_text())
                    from .generation_delivery import record
                    record(args.out_dir, round_dir)
                    def get_frames():
                        frames = _critic_frames(args, round_dir / "Res2_Articulation.mp4", round_dir / "critic_frames", current_spec)
                        return [[label, str(path)] for label, path in frames], [p for _, p in frames]
                    frame_records = research.phase(token, "frames", get_frames)
                    frame_content = []
                    for frame_index, (name, path) in enumerate(frame_records):
                        frame_content.extend([api.make_text_input(f"frame_ref: blender::{frame_index}; Joint segment: {name}"), api.make_image_input(Path(path))])
                    critique_content = [*images, *frame_content,
                        api.make_text_input("Kinematics being demonstrated:\n" + json.dumps(current_spec)),
                        api.make_text_input((args.prompts_dir / "4_articulation_critic.md").read_text())]
                    critique_parent = args.out_dir / "4_articulation_refine"
                    critique_path = critique_parent / f"articulation_eval_round_{round_index}.json"
                    def blender_review():
                        review_content = list(critique_content)
                        if POLICY not in (args.prompts_dir / "4_articulation_critic.md").read_text():
                            raise ValueError("Current Blender critic prompt must declare " + POLICY)
                        raw = ask(f"articulation_critic_round_{round_index}", review_content)
                        value = blender_decision(api.extract_json(raw), {f"blender::{i}" for i in range(len(frame_records))})
                        write_json(critique_path, value)
                        return value, [critique_path, *call_artifacts[round_calls_start:]]
                    critique = research.phase(token, "blender_review", blender_review)
                    from .generation_delivery import record
                    record(args.out_dir, round_dir, blender=critique)
                    blender_critique = critique
                    state.begin("isaac_motion", articulation_key)
                    def simulate():
                        report = _run_isaac(args, round_dir)
                        return str(report), evidence_files(report.parent)
                    isaac_report = Path(research.phase(token, "isaac", simulate,
                                                      key=_isaac_phase_key(args, round_dir)))
                    verify_isaac_evidence(isaac_report, round_dir / "kinematic_spec.json",
                                          round_dir / "urdf_export/object.urdf", root=args.out_dir)
                    def review_motion():
                        review = _review_isaac_motion(
                            report_path=isaac_report, spec=current_spec, source_images=image_paths,
                            prompt_path=args.prompts_dir / "6_isaac_motion_critic.md",
                            threshold=args.alignment_threshold_articulation,
                            ask=lambda stage, content: ask(f"isaac_round_{round_index}_{stage}", content),
                            api=api, out_dir=isaac_report.parent / "api_critic",
                            target_reference=target_reference)
                        return review, [p for p in (isaac_report.parent / "api_critic").rglob("*") if p.is_file()]
                    isaac_review = research.phase(token, "isaac_review", review_motion)
                    from .generation_delivery import record
                    record(args.out_dir, round_dir, isaac=isaac_review)
                    if isaac_review.get("verdict") not in {"pass", "fail", "uncertain"}:
                        raise RuntimeError("Isaac motion critic returned no valid verdict")
                    if (critic_accepted(isaac_review)
                            and not _accepted_critique(blender_critique)):
                        research.checked(token,passed=False,reason='blender_critic',paths=[critique_path],
                            result={'spec':current_spec,'critique':blender_critique,'frames':frame_records})
                        continue
                    if not critic_accepted(isaac_review):
                        critique = {"stage": "isaac_motion", "source": "isaac_motion_critic", "verdict": isaac_review["verdict"],
                                    "per_joint": [joint for joint in isaac_review.get("per_joint", [])
                                                  if not critic_accepted(joint)]}
                        frame_content = []
                        frame_records = []
                        rejected_joints = {joint["joint_name"] for joint in isaac_review.get("per_joint", [])
                                           if not critic_accepted(joint)}
                        pictured_joints = set()
                        for frame in isaac_review.get("selected_frames", []):
                            if frame["joint_name"] not in rejected_joints:
                                continue
                            path = Path(frame["path"]).resolve()
                            if (not path.is_relative_to(isaac_report.parent.resolve()) or not path.is_file()
                                    or file_hash(path) != frame.get("sha256")):
                                raise RuntimeError("Isaac refinement frame is not current-round verified evidence")
                            label = (f"Isaac joint {frame['joint_name']}; view {frame['view_name']}; "
                                     f"frame {frame['frame_index']}; time {frame['time_seconds']}; "
                                     f"joint coordinates {frame['positions']}")
                            frame_content.extend([api.make_text_input(label), api.make_image_input(path)])
                            frame_records.append([label, str(path)])
                            pictured_joints.add(frame["joint_name"])
                        if not rejected_joints or pictured_joints != rejected_joints:
                            raise ValueError("Isaac semantic refinement requires motion frames for every rejected joint")
                        print(f"Isaac motion round {round_index} rejected: {isaac_review['verdict']}", flush=True)
                        research.checked(token, passed=False, reason="isaac_critic", paths=[Path(isaac_review["report_path"])], result={
                            "spec": current_spec, "critique": critique, "frames": frame_records})
                        continue
                    checked = verify_isaac_evidence(
                        isaac_report, round_dir / "kinematic_spec.json", round_dir / "urdf_export/object.urdf",
                        root=args.out_dir, critic_path=Path(isaac_review["report_path"]),
                        threshold=args.alignment_threshold_articulation)
                    research.checked(token, passed=True, reason="kinematics_and_isaac_accepted", paths=checked["artifacts"], result={
                        "spec": current_spec, "critique": critique, "frames": frame_records})
                    isaac_evidence = {key: str(Path(checked[key]).relative_to(args.out_dir.resolve()))
                                      for key in ("isaac_report_path", "isaac_motion_review_path", "isaac_video_path")}
                    isaac_artifacts = [*checked["artifacts"], *call_artifacts[round_calls_start:]]
                    isaac_artifacts.extend(p for p in (isaac_report.parent / "api_critic").rglob("*") if p.is_file())
                    isaac_artifacts.extend(p for p in (round_dir / "urdf_export").rglob("*") if p.is_file())
                    export_dir = args.out_dir / "urdf_export"
                    if export_dir.exists():
                        shutil.rmtree(export_dir)
                    shutil.copytree(round_dir / "urdf_export", export_dir)
                    for name in ("pipeline_scene.blend", "Res2_Articulation.mp4", "validation_report.json"):
                        shutil.copy2(round_dir / name, args.out_dir / name)
                    # Use the runtime's normalized spec, whose digest appears in the report.
                    shutil.copy2(round_dir / "kinematic_spec.json", canonical_spec)
                    published_report = json.loads((args.out_dir / "validation_report.json").read_text())
                    published_report["artifacts"] = {
                        "urdf": str(canonical_urdf), "usd": str(export_dir / "object.usd"),
                        "blend": str(args.out_dir / "pipeline_scene.blend"),
                        "video": str(args.out_dir / "Res2_Articulation.mp4"),
                    }
                    from .motion_range_guard import POLICY as range_policy, requirements as motion_requirements, validate as validate_motion_travel
                    required_travel = motion_requirements(research.data["trials"])
                    validate_motion_travel(current_spec, required_travel)
                    published_report["motion_travel_validation"] = {
                        "policy": range_policy, "required": required_travel, "passed": True}
                    published_report["accepted_round"] = round_index
                    published_report["isaac_motion"] = {"passed": True, **isaac_evidence}
                    write_json(args.out_dir / "validation_report.json", published_report)
                    structural = validate_urdf(canonical_urdf, current_spec)
                    if not structural["passed"]:
                        raise RuntimeError("Published URDF failed independent validation: " + str(structural))
                    artifacts = [canonical_spec, args.out_dir / "pipeline_scene.blend",
                                 args.out_dir / "Res2_Articulation.mp4", args.out_dir / "validation_report.json", critique_path]
                    artifacts.extend(p for p in research.path.parent.rglob("*") if p.is_file())
                    artifacts.extend(p for p in export_dir.rglob("*") if p.is_file())
                    artifacts.extend(isaac_artifacts)
                    state.passed("articulation_export", articulation_key, artifacts, accepted_round=round_index,
                                 critic_score=blender_critique["score"], isaac_evidence=isaac_evidence)
                    state.passed("isaac_motion", articulation_key, isaac_artifacts,
                                 accepted_round=round_index, **isaac_evidence)
                    accepted = True
                    break
                except (CandidateRejected, ValueError, KeyError, TypeError) as error:
                    export_receipt = round_dir/'export_complete.json'
                    if export_receipt.is_file():
                        saved = json.loads(export_receipt.read_text())
                        if saved.get('passed') and all(file_hash(round_dir/p) == h for p, h in saved['files'].items()):
                            from .generation_delivery import record
                            record(args.out_dir, round_dir, blender={'passed': False, 'reason': str(error)})
                    diagnostic = round_dir / 'candidate_failure.json'
                    write_json(diagnostic, {'error_type': type(error).__name__, 'error': str(error)})
                    research.checked(token, passed=False, reason='articulation_candidate_rejected', paths=[diagnostic],
                        result={'spec': current_spec or refinement_source, 'frames': frame_records,
                                'critique': {'stage': 'articulation_candidate', 'error': str(error),
                                             'previous_feedback': critique}})
                    continue
            if not accepted:
                raise RuntimeError("Articulation critic did not pass within the configured refinement budget")
        state.data.update(status="passed", isaac_verification="passed")
        state.data.pop("error", None)
        state.save()
        from .generation_delivery import finalize
        finalize(args.out_dir, state, 'quality_accepted')
        print(f"Verified URDF with Isaac motion acceptance: {canonical_urdf}", flush=True)
    except BaseException as error:
        from .generation_delivery import finalize
        from .execution import terminal_failure
        if terminal_failure(error):
            state.fail(f"{type(error).__name__}: {error}")
            raise
        from .delivery_policy import BudgetExhausted
        if isinstance(error, BudgetExhausted):
            if not finalize(args.out_dir,state,f'{type(error).__name__}: {error}'):
                from .degraded_asset import rigid_generation
                rigid_generation(args, api, state, str(error))
            return
        state.fail(f"{type(error).__name__}: {error}")
        raise
