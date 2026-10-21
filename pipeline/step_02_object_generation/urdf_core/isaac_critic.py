"""Evidence-bound semantic review of actual Isaac motion captures.

This module never creates a model client or judges images itself. ``ask`` owns
the audited provider/model/reasoning configuration. Local checks establish
evidence integrity and fail closed on contradictory model verdicts.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from typing import Callable

from .run_state import file_hash, fingerprint, write_json
from .critic_policy import POLICY, component, finalize_attempts
from .review_frames import POLICY as FRAME_POLICY, GUIDE as FRAME_GUIDE, select_joint_evidence
from kinematics_validation.isaac_motion import validate_capture_visibility


SCOPE = ("API semantic review of sampled frames captured from actual Isaac motion; "
         "joint pivot/axis and parent-child behavior, not a dynamics or collision certificate. "
         "Numerical FK agreement is not evidence of semantic correctness.")
VERDICTS = {"pass", "fail", "uncertain"}
MAX_JSON_REPAIRS = 0

# Responsibility wording validated in the independent execution-evidence A/B.
AXIS_GUIDANCE = 'Research review of the named joint, using the same semantic standard as the supplied core rubric.\nNative measurements establish executed motion, not semantic ground truth. A proposed joint or numerical FK match can be wrong.\nIf diagnostic axes images are supplied, they supplement the unchanged source/overview/closeup; they are not an asset texture or a real surface feature. They ignore occlusion. Colored child axes and dashed parent reference axes are projected from measured link poses using actual captured camera readback. Axis length is only a display scale. A rotating marker alone is NEVER evidence that the correct physical part moves or that the axis/pivot is mechanically correct.\nAssess the physical pivot location, motion axis and plane, and ownership/following of the actual intended structural part using the real images and proposed ownership together. A pin can rotate while the intended arm stays still: that must not pass parent_child_motion merely because its marker moves. Conversely, rotational symmetry may make RGB motion unobservable; measured axes can help disambiguate executed orientation but cannot replace semantic evidence. Retain uncertain if the real part, pivot, axis meaning, or ownership is not defensibly observable.\nDiagnostic images carry the SAME frame_ref as their exact original closeup, and are an annotation of that frame, not an extra captured state. Cite only frame_ref values present in this request. An axis overlay cannot justify referencing an absent time/frame or another joint. Do not infer the expected answer from this research protocol. Preserve the requested JSON schema and never raise a score to obtain a pass.'
EXECUTION_GUIDANCE = 'Additional evidence-responsibility clarification for this research arm:\nThe supplied execution_evidence is derived by code from bound native measurements and imported visual geometry checks. It establishes which actual exported visual geometries were bound to measured rigid links and how those links moved. It does NOT establish that the proposed joint, part names, axis, parent, or functional mechanism is correct.\nKeep the semantic decision separate: inspect the original source and unmodified overview/closeup, then judge whether the actual geometry assigned to the moving child is the intended moving structural part, whether its rigid followers are complete, and whether its pivot/axis/direction is mechanically appropriate for the visible mechanism. Movement of only a connector does not establish movement of the intended structural assembly; a rotation incompatible with the visible mechanism remains an axis error even when its numerical execution is perfect.\nDo not require texture landmarks to re-prove execution that the code has established with actual mesh binding and measured transforms. For a textureless rotationally symmetric part, lack of visible RGB texture movement ALONE is not a semantic defect. You may combine verified execution facts with defensible visible shape/source evidence to judge the semantic components. Neither an authored object name, protocol/FK match, rotating marker, nor this clarification alone justifies pass. If the identity/shape of the actual moving geometry, mechanical axis, or required followers remains ambiguous, retain uncertain and explain that semantic uncertainty. Never turn a missing texture feature by itself into a concrete critical defect.\nUse the unchanged component schema and score standard. Cite only supplied frame_ref values, with actual visible geometric/source reasoning, and identify which execution facts support that reasoning. Do not infer an expected answer from this protocol.'


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _movable(spec):
    joints = [j for j in spec["joints"] if j["type"] != "fixed"]
    names = [j["name"] for j in joints]
    if len(names) != len(set(names)):
        raise ValueError("Motion review requires unique movable joint coverage")
    if any(j["type"] not in {"revolute", "continuous", "prismatic"} for j in joints):
        raise ValueError("Unsupported motion joint type")
    return joints


def _targets(spec):
    return _movable(spec) or [{"name": "__static__", "type": "fixed", "parent": spec["root"],
                               "child": spec["root"], "synthetic": True}]


def _checks(joint):
    return (["static_integrity"] if joint["type"] == "fixed" else
            ["translation_axis", "parent_child_motion"] if joint["type"] == "prismatic"
            else ["pivot_position", "axis_direction", "parent_child_motion"])


def _file(path, expected_hash=None):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty Isaac review evidence: {path}")
    digest = file_hash(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"Evidence SHA256 mismatch: {path}")
    return path, digest


def _load(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    json.dumps(value, allow_nan=False)
    return value


def _response(raw, joint, allowed_refs, threshold):
    """Validate structure, then gate semantics without requesting a new score."""
    if not isinstance(raw, dict) or raw.get("joint_name") != joint["name"]:
        raise ValueError("Response must identify exactly the joint being reviewed")
    score = _number(raw.get("score"), "score")
    if not 0 <= score <= 10 or raw.get("verdict") not in VERDICTS:
        raise ValueError("score must be 0..10 and verdict pass/fail/uncertain")
    errors = raw.get("critical_errors")
    if not isinstance(errors, list) or any(not isinstance(x, str) or not x.strip() for x in errors):
        raise ValueError("critical_errors must be a list of nonempty strings (or [])")
    checks = {}
    for name in _checks(joint):
        check = raw.get(name)
        if not isinstance(check, dict) or check.get("verdict") not in VERDICTS:
            raise ValueError(f"Missing or invalid {name} verdict")
        if not isinstance(check.get("reason"), str) or not check["reason"].strip():
            raise ValueError(f"{name} requires a reason")
        refs = check.get("evidence")
        if (not isinstance(refs, list) or not refs or
                any(not isinstance(ref, str) or ref not in allowed_refs for ref in refs)):
            raise ValueError(f"{name} must cite existing frame_ref values from this joint's input")
        checks[name] = {"verdict": check["verdict"], "evidence": refs, "reason": check["reason"]}
        if raw.get("review_policy") == POLICY:
            checks[name]["issue_type"] = component(check)
    decisions = [check["verdict"] for check in checks.values()]
    if raw.get("review_policy") == POLICY:
        verdict = "fail" if "fail" in decisions else "uncertain" if "uncertain" in decisions else "pass"
        defects = [c["reason"] for c in checks.values() if c["issue_type"] == "asset_error"]
        if raw["verdict"] != verdict or bool(errors) != bool(defects):
            raise ValueError("Overall verdict/errors must agree with classified asset defects")
        return {"review_policy": POLICY, "joint_name": joint["name"], **checks, "score": score,
                "model_verdict": raw["verdict"], "verdict": verdict, "passed": verdict == "pass",
                "critical_errors": defects, "accepted_for_pipeline": not defects,
                "observations": [{"component": name, **check} for name, check in checks.items()
                                 if check["issue_type"] in {"observation", "needs_evidence", "out_of_scope"}],
                "consistency_errors": []}
    raise ValueError("Historical critic results require their frozen verifier")


def _select(view, joint, indices):
    frames = view.get("frames")
    if not isinstance(frames, list) or len(frames) < 7:
        raise ValueError(f"{joint['name']}/{view['name']} needs at least seven captured frames")
    ordered = sorted(frames, key=lambda f: f["frame_index"])
    ids = [f["frame_index"] for f in ordered]
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Captured frame indices must be unique nonnegative integers")
    times = [_number(f.get("time_seconds"), "frame time_seconds") for f in ordered]
    if any(t < 0 for t in times) or times != sorted(times):
        raise ValueError("Captured frame times must be nonnegative and ordered")
    static = joint["type"] == "fixed"
    qs = ([0.0] * len(ordered) if static else
          [_number(f.get("positions", {}).get(joint["name"]), "captured joint position") for f in ordered])
    if not static and (max(qs) - min(qs) <= 1e-6 or min(abs(q) for q in qs) > 1e-4):
        raise ValueError("Motion evidence must include q=0 and visibly distinct commanded positions")
    if joint["type"] not in {"continuous", "fixed"}:
        lo, hi = joint["limit"]["lower"], joint["limit"]["upper"]
        tolerance = max(1e-4, abs(hi-lo)*1e-5)
        if abs(min(qs)-lo) > tolerance or abs(max(qs)-hi) > tolerance:
            raise ValueError("Motion capture does not cover both declared joint limits")
    if indices is not None:
        if (not isinstance(indices, list) or not 7 <= len(indices) <= 9 or
                len(indices) != len(set(indices)) or any(i not in ids for i in indices)):
            raise ValueError("critic_frame_indices must identify seven to nine actual frames per view")
        selected = [frame for frame in ordered if frame["frame_index"] in indices]
    else:
        # Include both temporal ends, q=0 and motion extrema, then fill the
        # largest remaining time gaps. No images are interpolated or invented.
        chosen = {0, len(ordered)-1, min(range(len(qs)), key=lambda i: abs(qs[i])),
                  min(range(len(qs)), key=qs.__getitem__), max(range(len(qs)), key=qs.__getitem__)}
        while len(chosen) < min(9, len(ordered)):
            chosen.add(max((i for i in range(len(ordered)) if i not in chosen),
                           key=lambda i: (min(abs(i-j) for j in chosen), -i)))
        selected = [ordered[i] for i in sorted(chosen)]
    picked_q = [0.0 if static else f["positions"][joint["name"]] for f in selected]
    if (min(abs(q) for q in picked_q) > 1e-4 or
            abs(min(picked_q)-min(qs)) > 1e-5 or abs(max(picked_q)-max(qs)) > 1e-5):
        raise ValueError("Selected critic frames omit zero or a captured motion endpoint")
    return selected


def _execution_evidence(report, manifest, remember):
    """Bind supplementary execution facts without assigning semantic truth."""
    descriptor = manifest.get("execution_evidence")
    reported = report.get("execution_evidence")
    if descriptor is None and reported is None:
        return None
    if (descriptor != reported or not isinstance(descriptor, dict)
            or set(descriptor) != {"path", "sha256"}):
        raise ValueError("Execution evidence report/manifest bindings differ")
    path = remember(descriptor["path"], descriptor["sha256"])
    value = _load(path)
    if (value.get("schema_version") != "isaac_execution_evidence.v1"
            or value.get("status") not in {"verified", "unavailable"}):
        raise ValueError("Invalid execution evidence schema/status")
    from kinematics_validation import capture_execution_evidence as execution_module
    declared_source = manifest.get("sources", {}).get("execution_module_sha256")
    if (not isinstance(declared_source, str) or not declared_source
            or declared_source != report.get("execution_module_sha256")
            or declared_source != value.get("execution_module_sha256")):
        raise ValueError("Execution evidence source differs between report and manifest")
    # Generation and physics may load identical frozen code from different
    # snapshot paths. Bind its bytes, not the verifier's incidental import path.
    _file(execution_module.__file__, declared_source)
    dependencies = value.get("artifacts", [])
    if not isinstance(dependencies, list):
        raise ValueError("Execution evidence artifacts must be explicit file bindings")
    if value.get("input_binding") is not None:
        dependencies = [value["input_binding"], *dependencies]
    for binding in dependencies:
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError("Invalid execution dependency binding")
        remember(binding["path"], binding["sha256"])
    if value["status"] == "unavailable":
        return value
    if value.get("mode") != "kinematic_playback" or value.get("input_binding") is None:
        raise ValueError("Verified execution evidence needs its native input ledger and playback mode")
    if value.get("frame_state_sha256") != execution_module.measurement_digest(manifest):
        raise ValueError("Execution evidence differs from the captured native frame states")
    return value


def _prepare(report_path, spec, source_images, prompt_path, *, path_resolver=None,
             review_context=None, diagnostics_dir=None, write_diagnostics=False):
    """Preflight all joints before the first paid API request."""
    evidence = {}
    resolve = path_resolver or (lambda value: Path(value).resolve())

    def remember(path, digest=None):
        path, value = _file(resolve(path), digest)
        evidence[path] = value
        return path

    report_path = remember(report_path)
    report = _load(report_path)
    if report.get("passed") is not True or report.get("status") != "passed":
        raise ValueError("Isaac worker must finish successfully before semantic review")
    manifest_path = remember(report.get("motion_video", {})["manifest_path"])
    manifest = _load(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported Isaac motion manifest schema")
    for label in ("urdf", "spec"):
        record = manifest[label]
        path = remember(record["path"], record["sha256"])
        # The recorder keeps immutable copies under motion/sources. Bind both
        # the reported original and snapshot by content, not pathname equality.
        if not report.get(label):
            raise ValueError(f"Worker report has no original {label} path")
        remember(report[label], record["sha256"])
        if report.get(label + "_sha256", record["sha256"]) != record["sha256"]:
            raise ValueError(f"Motion manifest {label} hash differs from worker report")
        if label == "spec" and fingerprint(_load(path)) != fingerprint(spec):
            raise ValueError("Motion manifest spec differs from the protocol being reviewed")
    if (report.get("verifier_source_sha256") and manifest.get("sources", {}).get("verifier_sha256")
            != report["verifier_source_sha256"]):
        raise ValueError("Motion manifest and worker verifier source hashes differ")
    sources = [remember(path) for path in source_images]
    if not sources:
        raise ValueError("Isaac semantic review requires the original source image(s)")
    prompt_path = remember(prompt_path)
    joints = _targets(spec)
    expected = {j["name"] for j in joints}
    movable_names = {j["name"] for j in _movable(spec)}
    listed = manifest.get("joint_names", [])
    records = manifest.get("joints", [])
    if (len(listed) != len(set(listed)) or set(listed) != movable_names or
            len(records) != len(expected) or {j.get("name") for j in records} != expected):
        raise ValueError("Motion manifest must cover every movable joint exactly once")
    if not movable_names and (manifest.get("static") is not True or records[0].get("synthetic") is not True):
        raise ValueError("Static capture must explicitly identify its synthetic observation target")
    by_name = {record["name"]: record for record in records}
    selected = []
    for joint in joints:
        record = by_name[joint["name"]]
        if any(record.get(k) != joint[k] for k in ("type", "parent", "child")):
            raise ValueError("Motion manifest joint type/parent/child differs from spec")
        views = record.get("views", [])
        if len(views) != 2 or {v.get("name") for v in views} != {"overview", "closeup"}:
            raise ValueError("Every joint requires overview and closeup motion videos")
        validate_capture_visibility(manifest, record, spec)
        for view in views:
            remember(view["video_path"], view["video_sha256"])
            # Missing frames anywhere in a supplied clip are capture failures,
            # even if that frame would not have been selected for the model.
            for frame in view.get("frames", []):
                _file(resolve(frame["path"]), frame["sha256"])
            for frame in _select(view, joint, record.get("critic_frame_indices")):
                path = remember(frame["path"], frame["sha256"])
                positions = frame["positions"]
                if set(positions) != movable_names:
                    raise ValueError("Selected frame must identify positions for all movable joints")
                for q in positions.values():
                    _number(q, "frame position")
                selected.append({"joint_name": joint["name"], "view_name": view["name"],
                                 "frame_ref": f"{joint['name']}::{view['name']}::{frame['frame_index']}",
                                 "path": str(path), "sha256": evidence[path],
                                 "frame_index": frame["frame_index"], "time_seconds": frame["time_seconds"],
                                 "positions": positions, "measured_positions": frame.get("measured_positions", {}),
                                 "camera": view.get("camera", {}),
                                 **({"visibility": view["visibility"]} if "visibility" in view else {})})
    if manifest.get("overview_video_path"):
        remember(manifest["overview_video_path"])
    execution = _execution_evidence(report, manifest, remember)
    summaries = {}
    if execution is not None:
        from .execution_review_summary import review_summaries
        refs_by_joint = {joint["name"]: [] for joint in joints}
        for frame in selected:
            refs_by_joint[frame["joint_name"]].append(frame["frame_ref"])
        summaries = review_summaries(execution, manifest, refs_by_joint)
    if review_context is not None:
        review_context["execution_evidence"] = manifest.get("execution_evidence")
        review_context["execution_summaries"] = summaries
    if diagnostics_dir is not None:
        from .isaac_review_diagnostics import axis_diagnostics
        diagnostics, diagnostic_evidence = axis_diagnostics(report, manifest, selected, diagnostics_dir,
            write=write_diagnostics, path_resolver=resolve)
        for path, digest in diagnostic_evidence.items():
            remember(path, digest)
        if review_context is not None:
            review_context["axis_diagnostics"] = diagnostics
    return report_path, manifest_path, manifest, sources, prompt_path, joints, selected, evidence


def validate_review_result(result, *, spec, verify_evidence=True, path_resolver=None):
    """Recheck a persisted review and its exact evidence before cache reuse."""
    joints = _targets(spec)
    if (not isinstance(result, dict) or result.get("schema_version") != 1 or
            result.get("spec_fingerprint") != fingerprint(spec)):
        raise ValueError("Isaac semantic review does not match the current spec")
    if verify_evidence and result.get('target_reference'):
        from .target_reference import load_reference
        reference = result['target_reference']
        if load_reference(reference['context']['context_file']['path'], Path(reference['prompt_path']).parent) != reference:
            raise ValueError('Isaac review target reference changed')
    threshold = _number(result.get("threshold"), "threshold")
    if not 0 <= threshold <= 10:
        raise ValueError("Invalid review threshold")
    records = result.get("per_joint", [])
    if (not isinstance(records, list) or any(not isinstance(x, dict) for x in records) or
            len(records) != len(joints) or {x.get("joint_name") for x in records} != {j["name"] for j in joints}):
        raise ValueError("Isaac semantic review has incomplete or duplicate joint coverage")
    frames = result.get("selected_frames")
    if (not isinstance(frames, list) or not frames or
            any(not isinstance(f, dict) or not {"joint_name", "frame_ref", "path", "sha256"} <= set(f)
                for f in frames) or len({f["frame_ref"] for f in frames}) != len(frames)):
        raise ValueError("Isaac semantic review has invalid or duplicate frame records")
    by_name = {r["joint_name"]: r for r in records}
    frame_selection = result.get("api_frame_selection")
    if frame_selection is not None and (not isinstance(frame_selection, dict)
            or frame_selection.get("policy") != FRAME_POLICY
            or set(frame_selection.get("joints", {})) != {j["name"] for j in joints}):
        raise ValueError("Invalid API frame-selection policy or joint coverage")
    checked = []
    for joint in joints:
        record = by_name[joint["name"]]
        refs = {f["frame_ref"] for f in result["selected_frames"] if f["joint_name"] == joint["name"]}
        if frame_selection is not None:
            plan = frame_selection["joints"][joint["name"]]
            sent_refs = plan.get("sent_frame_refs", [])
            if (not sent_refs or len(sent_refs) != len(set(sent_refs)) or not set(sent_refs) <= refs):
                raise ValueError("API frame selection cites missing/duplicate frames")
            refs = set(sent_refs)
        raw = {**record, "verdict": record.get("model_verdict")}
        if result.get("review_policy") == POLICY:
            attempts = record.get("review_attempts")
            if not isinstance(attempts, list) or not attempts:
                raise ValueError("Versioned review requires its actual evidence attempts")
            checked_attempts = [_response({**a, "verdict": a.get("model_verdict")}, joint, refs, threshold)
                                for a in attempts]
            if attempts != checked_attempts:
                raise ValueError("Stored evidence attempt differs from its classified verdict")
            validated = finalize_attempts(checked_attempts)
            if record != validated:
                raise ValueError("Stored evidence exception contradicts its review attempts")
        else:
            validated = _response(raw, joint, refs, threshold)
        if any(record.get(k) != validated[k] for k in ("passed", "verdict", "consistency_errors")):
            raise ValueError("Stored Isaac review contradicts its model evidence")
        checked.append(validated)
    passed = all(r["passed"] for r in checked)
    verdict = "pass" if passed else "fail" if any(r["verdict"] == "fail" for r in checked) else "uncertain"
    critical = [f"{r['joint_name']}: {error}" for r in checked for error in r["critical_errors"]]
    if result.get("review_policy") == POLICY:
        if result.get("accepted_for_pipeline") is not all(r["accepted_for_pipeline"] for r in checked):
            raise ValueError("Pipeline acceptance differs from classified evidence")
        if result.get("observations") != [{"joint_name": r["joint_name"], "items": r["observations"]}
                                          for r in checked if r["observations"]]:
            raise ValueError("Observation summary differs from its review")
    if (result.get("passed") is not passed or result.get("verdict") != verdict or
            result.get("score") != min(r["score"] for r in checked) or
            result.get("critical_errors") != critical):
        raise ValueError("Stored overall verdict contradicts per-joint results")
    if verify_evidence:
        resolve = path_resolver or (lambda value: Path(value).resolve())
        root = Path(result["report_path"]).resolve().parent
        if not result.get("evidence_artifacts"):
            raise ValueError("Review has no bound evidence")
        for name, digest in result["evidence_artifacts"].items():
            _file(resolve((root / name).resolve()), digest)
        for frame in result["selected_frames"]:
            key = os.path.relpath(Path(frame["path"]).resolve(), root)
            if result["evidence_artifacts"].get(key) != frame["sha256"]:
                raise ValueError("Selected frame is not bound by review evidence hashes")
        context = {}
        prepared = _prepare(Path(result["worker_report_path"]), spec,
                            [Path(source["path"]) for source in result["source_images"]],
                            Path(result["prompt_path"]), path_resolver=resolve, review_context=context,
                            diagnostics_dir=root / "axis_diagnostics" if "axis_diagnostics" in result else None)
        if (result.get("execution_evidence") != context["execution_evidence"]
                or result.get("execution_summary_sha256", {}) !=
                    {name: fingerprint(summary) for name, summary in context["execution_summaries"].items()}):
            raise ValueError("Stored execution evidence differs from the bound native capture")
        if "axis_diagnostics" in result and result["axis_diagnostics"] != context.get("axis_diagnostics"):
            raise ValueError("Stored diagnostic axes differ from measured frames/cameras")
        if frame_selection is not None:
            for joint in joints:
                _, _, expected = select_joint_evidence(
                    [f for f in prepared[6] if f["joint_name"] == joint["name"]], joint, spec=spec,
                    execution_summary=context["execution_summaries"].get(joint["name"]),
                    diagnostics=context.get("axis_diagnostics", {}).get("joints", {}).get(joint["name"]))
                if frame_selection["joints"][joint["name"]] != expected:
                    raise ValueError("Stored API frame selection differs from measured evidence")
        expected_frames = [{**frame, "path": str(resolve(frame["path"]))}
                           for frame in result["selected_frames"]]
        if prepared[6] != expected_frames:
            raise ValueError("Stored frame selection differs from the bound capture manifest")
        hashes = {str(path): digest for path, digest in prepared[7].items()}
        expected_hashes = {str(resolve((root / name).resolve())): digest
                           for name, digest in result["evidence_artifacts"].items()}
        if hashes != expected_hashes:
            raise ValueError("Stored evidence set differs from the capture inputs")
    return result


def review_isaac_motion(*, report_path: Path, spec: dict, source_images: list[Path],
                        prompt_path: Path, threshold: float,
                        ask: Callable[[str, list[dict]], str], api, out_dir: Path, path_resolver=None,
                        target_reference=None) -> dict:
    """Review every movable joint using actual overview/closeup capture frames."""
    threshold = _number(threshold, "threshold")
    if not 0 <= threshold <= 10:
        raise ValueError("threshold must be 0..10")
    out_dir = Path(out_dir).resolve()
    context = {}
    (report_path, manifest_path, manifest, sources, prompt_path,
     joints, selected, evidence) = _prepare(report_path, spec, source_images, prompt_path,
        review_context=context, diagnostics_dir=out_dir / "axis_diagnostics", write_diagnostics=True,
        path_resolver=path_resolver)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompt_path.read_text(encoding="utf-8")
    if target_reference:
        from .target_reference import reference_content
        reference = reference_content(target_reference, api, out_dir)
    else:
        reference = []
        for image_index, path in enumerate(sources, 1):
            reference.extend([api.make_text_input(f"Original source image {image_index}"), api.make_image_input(path)])
    per_joint, frame_plans = [], {}
    for index, joint in enumerate(joints, 1):
        frames, overlay_frames, frame_plans[joint["name"]] = select_joint_evidence(
            [f for f in selected if f["joint_name"] == joint["name"]], joint, spec=spec,
            execution_summary=context["execution_summaries"].get(joint["name"]),
            diagnostics=context["axis_diagnostics"]["joints"].get(joint["name"]))
        content = [api.make_text_input(prompt), *reference,
                   api.make_text_input("Complete proposed kinematic protocol:\n" + json.dumps(spec)),
                   api.make_text_input("Review this joint only:\n" + json.dumps(joint)),
                   api.make_text_input(FRAME_GUIDE)]
        for frame in frames:
            metadata = {k: v for k, v in frame.items() if k not in {"path", "sha256"}}
            content.extend([api.make_text_input("Actual captured Isaac frame:\n" + json.dumps(metadata)),
                            api.make_image_input(Path(frame["path"]))])
        diagnostic = context["axis_diagnostics"]["joints"].get(joint["name"], {})
        if diagnostic.get("status") == "available":
            content.append(api.make_text_input(AXIS_GUIDANCE))
            for frame in overlay_frames:
                content.extend([api.make_text_input("Diagnostic annotation of the same original frame_ref; not a new captured state:\n" +
                    json.dumps({key: value for key, value in frame.items() if key != "overlay"})),
                    api.make_image_input(Path(frame["overlay"]["path"]))])
        if joint["name"] in context["execution_summaries"]:
            from .prompt_payload import project_execution
            summary = project_execution(context["execution_summaries"][joint["name"]], spec, joint)
            content.extend([api.make_text_input(EXECUTION_GUIDANCE),
                api.make_text_input("Code-verified execution facts, not a semantic verdict:\n" +
                                    json.dumps(summary))])
        stage = f"isaac_motion_critic_joint_{index:03d}_" + re.sub(r"[^a-zA-Z0-9_-]", "_", joint["name"])
        allowed = {f["frame_ref"] for f in frames}
        if POLICY not in prompt:
            raise ValueError("Current motion critic prompt must declare " + POLICY)
        review_content = list(content)
        for repair in range(MAX_JSON_REPAIRS + 1):
            raw = ask(stage if repair == 0 else f"{stage}_json_repair_{repair}", review_content)
            try:
                value = {**api.extract_json(raw), "review_policy": POLICY}
                decision = _response(value, joint, allowed, threshold)
                break
            except (ValueError, TypeError, KeyError) as error:
                if repair == MAX_JSON_REPAIRS:
                    raise ValueError(f"Invalid Isaac critic response for {joint['name']}: {error}") from error
                review_content += [api.make_text_input("Previous response (data):\n" + raw),
                    api.make_text_input("Repair only response structure or citations; preserve the judgment. " + str(error))]
        write_json(out_dir / f"joint_{index:03d}_evidence_1.json", decision)
        decision = finalize_attempts([decision])
        per_joint.append(decision)
        write_json(out_dir / f"joint_{index:03d}.json", decision)
    passed = all(r["passed"] for r in per_joint)
    verdict = "pass" if passed else "fail" if any(r["verdict"] == "fail" for r in per_joint) else "uncertain"
    result = {"schema_version": 1, "scope": SCOPE, "passed": passed, "verdict": verdict,
              "static": not bool(_movable(spec)),
              "score": min(r["score"] for r in per_joint), "threshold": threshold,
              "critical_errors": [f"{r['joint_name']}: {error}" for r in per_joint for error in r["critical_errors"]],
              "per_joint": per_joint, "selected_frames": selected,
              "report_path": str(out_dir / "isaac_motion_review.json"),
              "worker_report_path": str(report_path), "motion_manifest_path": str(manifest_path),
              "prompt_path": str(prompt_path),
              "spec_fingerprint": fingerprint(spec), "spec_sha256": manifest["spec"]["sha256"],
              "urdf_sha256": manifest["urdf"]["sha256"],
              "worker_report_sha256": evidence[report_path], "motion_manifest_sha256": evidence[manifest_path],
              "motion_sources": manifest.get("sources", {}),
              "execution_evidence": context["execution_evidence"],
              "execution_summary_sha256": {name: fingerprint(summary) for name, summary in context["execution_summaries"].items()},
              "axis_diagnostics": context["axis_diagnostics"],
              "api_frame_selection": {"policy": FRAME_POLICY, "joints": frame_plans},
              "source_images": [{"path": str(p), "sha256": evidence[p]} for p in sources],
              "evidence_artifacts": {os.path.relpath(p, out_dir): digest for p, digest in evidence.items()},
              "feedback": {"source": "isaac_motion_critic", "scope": SCOPE, "verdict": verdict,
                           "per_joint": per_joint}}
    if POLICY in prompt:
        result["review_policy"] = POLICY
        result["accepted_for_pipeline"] = all(r["accepted_for_pipeline"] for r in per_joint)
        result["observations"] = [{"joint_name": r["joint_name"], "items": r["observations"]}
                                  for r in per_joint if r["observations"]]
    if target_reference:
        result['target_reference'] = target_reference
    validate_review_result(result, spec=spec, path_resolver=path_resolver)
    write_json(Path(result["report_path"]), result)
    return result
