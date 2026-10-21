"""Bounded diagnostic evidence and conservative numerical candidate ordering.

This module does not decide acceptance. The native verifier remains authoritative.
Unknown/unmeasured checks are explicit and never assigned a fabricated improvement.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path

from urdf_core.run_state import file_hash, fingerprint
from .trajectory_io import first_trajectory_sample, iter_trajectory_samples


def _diagnostic_speed(contract):
    # Recognition is diagnostic compatibility only: it neither activates a
    # new acceptance policy nor permits old contracts to discard a former gate.
    return (contract.get("acceptance_policy") in {'object_functional_physics.v11'}
            and contract.get("joint_peak_velocity_policy") == "diagnostic_only")


def _scope_policy(contract):
    if "motion_scope_policy" not in contract:
        return None
    value = contract["motion_scope_policy"]
    if value != "fixed_full_ground_operable.v1":
        raise ValueError("Unknown motion scope cannot default to full-range diagnostics")
    return value


def _diagnostic_operation(trial, contract, *, first_sample=None):
    """Bind diagnostic endpoints to the original native initialization receipt.

    The formal verifier owns URDF/cooked-source reconstruction and acceptance.
    Here the hash-bound t0 plan is independently derived by the same numerical
    scope adapter, then compared with setup and measured scope metadata. A
    model-provided tested range, absent proof or unknown policy is not a fallback
    command. Physical lower/upper limits in the returned joint remain unchanged.
    """
    name, kind = trial.get("joint_name"), trial.get("kind")
    target = contract.get("joints", {}).get(name, {})
    lower, upper = target.get("lower", 0.), target.get("upper", 0.)
    policy = _scope_policy(contract)
    if policy is None or kind not in {"fixed_joint", "free_joint"}:
        return target, lower, upper
    if name not in contract["joints"]:
        raise ValueError("Scoped diagnostic trial has no known operation joint")
    if first_sample is None:
        path = Path(trial.get("trajectory_path", ""))
        if not path.is_file() or file_hash(path) != trial.get("trajectory_sha256"):
            raise ValueError("Scoped diagnostic trajectory is missing or its hash changed")
        # This reader validates the entire stream, including a gzip footer.
        first_sample = first_trajectory_sample(path)
    if (not isinstance(first_sample, dict) or type(first_sample.get("time_s")) not in (int, float)
            or first_sample["time_s"] != 0.):
        raise ValueError("Scoped diagnostics require the original time-zero receipt")
    from .metrics import _motion_scope
    from .control import scoped_motion_joint
    operation = _motion_scope(first_sample, contract, kind, name)
    if operation is not None:
        lower, upper = operation["lower"], operation["upper"]
    expected = {"policy": policy, "full_lower": target["lower"], "full_upper": target["upper"],
                "tested_lower": lower, "tested_upper": upper,
                "full_range_claimed": lower == target["lower"] and upper == target["upper"]}
    measured = trial.get("metrics", {})
    measured = measured.get("metrics", measured)
    if measured.get("motion_scope") != expected:
        raise ValueError("Diagnostic tested range differs from the native scope receipt")
    setup, binding = trial.get("setup", {}), first_sample.get("motion_scope_binding", {})
    geometry = binding.get("geometry_artifact", {})
    if (setup.get("motion_scope_plan") != first_sample.get("motion_scope_plan")
            or setup.get("motion_scope_geometry") != geometry
            or setup.get("cooked_geometry_sha256") != binding.get("cooked_geometry_sha256")
            or setup.get("fixture_clearance_m") != binding.get("fixture_clearance_m")):
        raise ValueError("Diagnostic setup differs from the native scope receipt")
    geometry_path = Path(geometry.get("path", ""))
    if not geometry_path.is_file() or file_hash(geometry_path) != geometry.get("sha256"):
        raise ValueError("Diagnostic cooked-scope artifact is missing or its hash changed")
    return scoped_motion_joint(contract, name, operation), lower, upper


def trial_list(report):
    trials = list(report.get("trials", []))
    reload_trial = report.get("reload_check", {}).get("trial")
    if isinstance(reload_trial, dict):
        trials.append(reload_trial)
    return trials


def numeric_summary(report, contract):
    """Ratios <= 1 pass; completed trial/check passes are protected on promotion."""
    cfg, definitions = contract["config"], contract.get("joints", {})
    diagnostic_speed = _diagnostic_speed(contract)
    from .metrics import _passive_velocity_diagnostic_only, _upright_required
    diagnostic_tail_speed = _passive_velocity_diagnostic_only(contract)
    upright_required = _upright_required(contract)
    _scope_policy(contract)
    checks, passed_trials, stages = {}, [], {"ground_settle": [], "fixed_joint": [], "free_joint": [], "reload": []}
    pos = float(cfg.get("position_tolerance_m") or max(.001, .001 * contract.get("diagonal_m", 1.)))
    angle = float(cfg.get("orientation_tolerance_deg", 1.))
    window = float(cfg.get("zero_effort_tail_s", 1.))

    def ratio(key, value, limit, minimum=False):
        if (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                and isinstance(limit, (int, float)) and math.isfinite(limit) and limit > 0):
            checks[key] = max(0., 1. + (limit - value) / limit) if minimum else max(0., value / limit)

    for trial in trial_list(report):
        if trial.get("status") in {"skipped", "skipped_by_prior_failure"}:
            continue
        identity = str(trial.get("id", trial.get("kind", "unknown")))
        kind = trial.get("kind")
        from .motion_acceptance import enabled, hold_tolerance
        relaxed = enabled(contract, kind)
        stage = "reload" if kind in {"reload", "reload_supported", "support_rest_reload"} else kind
        if stage in stages:
            stages[stage].append(trial.get("passed") is True)
        if trial.get("passed") is True:
            passed_trials.append(identity)
        m = trial.get("metrics", {})
        errors = m.get("errors", trial.get("errors", []))
        m = m.get("metrics", m)
        p = identity + "/"
        # Trial outcome protects observation/ordering/coverage checks without a
        # numeric metric; it is not a surrogate physical distance.
        for error in errors:
            checks[p + "unresolved/" + fingerprint(str(error))] = 2.
        if upright_required and kind in {"ground_settle", "free_joint", "reload"}:
            ratio(p + "tilt", m.get("max_tilt_deg"), cfg.get("tilt_limit_deg", 10.))
        if kind == "free_joint":
            ratio(p + "displacement", m.get("max_horizontal_displacement_m"), m.get("free_slip_tolerance_m", cfg.get("free_slip_tolerance_m")))
        ratio(p + "tail_position", m.get("tail_position_span_m"), pos)
        ratio(p + "tail_orientation", m.get("tail_orientation_span_deg"), angle)
        for name, joint in m.get("joints", {}).items():
            spec = definitions.get(name, {})
            tol = hold_tolerance(spec, contract) if relaxed else pos if spec.get("type") == "prismatic" else math.radians(angle)
            jp = p + "joint/" + name + "/"
            if kind in {"fixed_joint", "free_joint"} and not diagnostic_speed:
                ratio(jp + "speed", joint.get("peak_velocity"), spec.get("max_velocity"))
            ratio(jp + "effort", joint.get("peak_active_effort"), spec.get("max_effort"))
            accepted_incidental = {x['joint_name'] for x in trial.get('metrics', {}).get('incidental_motion', {}).get('items', [])
                                   if x['verdict'] == 'acceptable'}
            for field in ("max_initial_q_deviation", "max_hold_deviation", "tail_q_span"):
                if field == 'max_hold_deviation' and name in accepted_incidental:
                    continue
                ratio(jp + field, joint.get(field), tol)
            if not diagnostic_tail_speed:
                ratio(jp + "tail_velocity", joint.get("tail_peak_velocity"), tol / window)
            lo, hi = spec.get("lower"), spec.get("upper")
            if spec.get("type") != "continuous" and isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                slack = max(1e-5, (hi-lo)*.2) if relaxed else max(1e-5, (hi-lo)*.001)
                if isinstance(joint.get("min_q"), (int, float)):
                    ratio(jp + "lower_limit", max(0., lo-joint["min_q"]), slack)
                if isinstance(joint.get("max_q"), (int, float)):
                    ratio(jp + "upper_limit", max(0., joint["max_q"]-hi), slack)
        _, tested_lower, tested_upper = _diagnostic_operation(trial, contract)
        span = tested_upper - tested_lower
        for key, value in m.get("endpoint_errors", {}).items():
            ratio(p + "endpoint/" + key, value, span * (.8 if relaxed else cfg.get("endpoint_tolerance_fraction", .02)))
        ratio(p + "coverage", m.get("coverage"), .8 if relaxed else cfg.get("coverage_min", .98), True)
        for key, value in m.get("endpoint_actual_holds_s", {}).items():
            ratio(p + "hold/" + key, value, cfg.get("endpoint_hold_s", 1.), True)
        pre_motion = m.get("pre_motion", {})
        if "passed" in pre_motion:
            checks[p + "pre_motion/state"] = 0. if pre_motion["passed"] is True else 2.
        for section, tail in (("tail_link", m.get("passive_tail", {})),
                              ("pre_motion_link", pre_motion.get("stability") or {})):
            for name, values in tail.get("links", {}).items():
                lp = p + section + "/" + name + "/"
                for field, limit in (("position_span_m", pos), ("orientation_span_deg", angle),
                                     ("peak_native_linear_velocity_m_s", pos/window),
                                     ("peak_native_angular_velocity_rad_s", math.radians(angle)/window)):
                    if diagnostic_tail_speed and field.startswith("peak_native_"):
                        continue
                    ratio(lp + field, values.get(field), limit)
    ordered = ["ground_settle", "fixed_joint", "free_joint", "reload"]
    applicable = [s for s in ordered if (s != "ground_settle" or contract.get("stable_required", True))
                  and (s not in {"fixed_joint", "free_joint"} or definitions)
                  and (s != "free_joint" or contract.get("stable_required", True))]
    progress = 0
    for stage in applicable:
        values = stages[stage]
        if not values or not all(values):
            break
        progress += 1
    # Check pass retention is across observed numeric checks AND whole trials.
    return {"passed": report.get("passed") is True, "stage_progress": progress,
            "passed_trials": sorted(passed_trials), "checks": checks,
            "protected_checks": sorted(k for k, v in checks.items() if v <= 1. + 1e-6),
            "normalized_violation": max((max(0., v-1.) for v in checks.values()), default=None)}


COMPARISON_POLICY = "observed_trial_progress.v2"


def compare_candidate(candidate, best, *, policy=COMPARISON_POLICY):
    """Newly discovered failures after an early exit are not old regressions.

    Promote only after all
    previously failing trials actually pass and every protected check remains
    observed and passing. It changes search retention, never acceptance.
    """
    if policy != COMPARISON_POLICY:
        raise ValueError("Unknown candidate comparison policy")
    decision = _compare_observed_checks(candidate, best)
    if best and decision["reason"] != "regressed previously passed checks" and not decision["promote"]:
        observed = {key.split("/", 1)[0] for key in best["checks"]} | set(best["passed_trials"])
        failed = {key.split("/", 1)[0] for key, value in best["checks"].items() if value > 1. + 1e-6}
        new_errors_in_old_trials = [key for key, value in candidate["checks"].items()
            if value > 1. + 1e-6 and key not in best["checks"] and key.split("/", 1)[0] in observed]
        if failed and failed <= set(candidate["passed_trials"]) and not new_errors_in_old_trials:
            decision = {"promote": True, "reason": "previous failures resolved; later untested trials now observed",
                        "resolved_trials": sorted(failed)}
    return {**decision, "policy": COMPARISON_POLICY}


def _compare_observed_checks(candidate, best):
    """Conservative tie handling; absent evidence cannot claim an improvement."""
    if best is None:
        return {"promote": True, "reason": "initial measured baseline"}
    missing = set(best["passed_trials"]) - set(candidate["passed_trials"])
    regressions = [k for k in best["protected_checks"]
                   if k not in candidate["checks"] or candidate["checks"][k] > 1. + 1e-6]
    if missing or regressions:
        return {"promote": False, "reason": "regressed previously passed checks",
                "regressed_trials": sorted(missing), "regressed_checks": sorted(regressions)}
    if candidate["passed"]:
        return {"promote": True, "reason": "all numerical acceptance checks passed"}
    if candidate["stage_progress"] != best["stage_progress"]:
        return {"promote": candidate["stage_progress"] > best["stage_progress"], "reason": "stage progress"}
    new_unresolved = [k for k in candidate["checks"] if "/unresolved/" in k and k not in best["checks"]]
    if new_unresolved:
        return {"promote": False, "reason": "new unquantified failure needs investigation",
                "new_failures": sorted(new_unresolved)}
    # Compare the same previously failed numeric checks, not a moving objective.
    failing = [k for k, v in best["checks"].items() if v > 1. + 1e-6 and "/unresolved/" not in k]
    if failing and all(k in candidate["checks"] for k in failing):
        before = max(best["checks"][k]-1. for k in failing)
        after = max(max(0., candidate["checks"][k]-1.) for k in failing)
        new_failures = [k for k, v in candidate["checks"].items()
                        if v > 1. + 1e-6 and k not in best["checks"] and "/unresolved/" not in k]
        if not new_failures and after < before - 1e-6:
            return {"promote": True, "reason": "lower normalized violation of existing failed checks",
                    "before": before, "after": after}
    return {"promote": False, "reason": "no demonstrated numerical improvement"}


def _steps(path):
    # Native artifacts are JSONL. Unit fixtures may deliberately omit native
    # traces; this does not bypass the independent acceptance verifier.
    yield from iter_trajectory_samples(path)


def _trajectory_phase(time_s, settings, segments, duration, scoped, relaxed=False):
    if duration is None:
        return "passive_ground"
    t = float(time_s) - float(settings["settle_duration_s"])
    if t < 0:
        return "pre_motion"
    if t >= duration - 1e-8 if scoped else t > duration:
        return "active_return_end" if relaxed else "zero_effort_tail"
    for ordinal, (start, end, a, b) in enumerate(segments):
        if t <= end:
            return ("hold" if a == b else "move") + f"_segment_{ordinal}"
    return "operation"


def trajectory_windows(trial, contract, *, radius=3):
    """Retain signal extrema and each commanded phase, including active return."""
    raw_path = trial.get("trajectory_path")
    if not raw_path:
        return {"available": False, "reason": "trial has no trajectory artifact"}
    path = Path(raw_path)
    if not path.is_file() or file_hash(path) != trial.get("trajectory_sha256"):
        raise ValueError("Diagnostic trajectory is missing or its hash changed")
    anchors, count = {}, 0
    settings, joints = contract["config"], contract.get("joints", {})
    from .motion_acceptance import enabled, hold_tolerance
    relaxed = enabled(contract, trial.get("kind"))
    diagnostic_speed = _diagnostic_speed(contract)
    scope_policy = _scope_policy(contract)
    segments, duration = [], None
    motion = trial.get("kind") in {"fixed_joint", "free_joint"} and trial.get("joint_name") in joints
    scoped = scope_policy is not None and trial.get("kind") in {"fixed_joint", "free_joint"}
    tested_span = None
    if motion and not scoped:
        from .control import target_schedule
        segments, duration = target_schedule(joints[trial["joint_name"]], settings, contract=contract)
    root_initial = None
    invalid_native_sample = False
    try:
        for i, step in enumerate(_steps(path)):
            if not isinstance(step, dict) or "time_s" not in step:
                # Still consume the stream: a non-native fixture must not hide
                # a corrupt gzip footer behind this diagnostic-only fallback.
                invalid_native_sample = True
                continue
            count += 1
            if scoped and i == 0:
                from .control import target_schedule
                target_joint, lower, upper = _diagnostic_operation(trial, contract, first_sample=step)
                segments, duration = target_schedule(target_joint, settings, contract=contract)
                tested_span = upper - lower
            signals = {}
            for name, definition in joints.items():
                q = step.get("joint_positions", {}).get(name)
                qd = step.get("joint_velocities", {}).get(name)
                effort = step.get("joint_efforts", {}).get(name)
                target = step.get("joint_targets", {}).get(name)
                if isinstance(qd, (int, float)):
                    signals[name + "/speed"] = abs(qd) / definition["max_velocity"]
                if isinstance(effort, (int, float)):
                    signals[name + "/effort"] = abs(effort) / definition["max_effort"]
                if isinstance(q, (int, float)) and isinstance(target, (int, float)):
                    span = tested_span if scoped and name == trial["joint_name"] else definition["upper"]-definition["lower"]
                    tolerance = max(1e-8, span * (.8 if relaxed else settings.get("endpoint_tolerance_fraction", .02)))
                    signals[name + "/tracking_error"] = abs(q-target)/tolerance
                if isinstance(q, (int, float)):
                    tol = hold_tolerance(definition, contract) if relaxed else settings.get("position_tolerance_m", .001) if definition["type"] == "prismatic" else math.radians(settings.get("orientation_tolerance_deg", 1.))
                    signals[name + "/initial_q_deviation"] = abs(q-definition["initial_position"])/tol
            if duration is not None:
                name = trial["joint_name"]
                phase = _trajectory_phase(step["time_s"], settings, segments, duration, scoped, relaxed)
                value = signals.get(name + "/tracking_error", 0.)
                # A global effort maximum can select only upward motion, and
                # global tracking error can select only the passive tail. Keep
                # an actual observation from every commanded phase so neither
                # omission can be mistaken for a missing active return.
                item = anchors.setdefault(name + "/phase/" + phase, {
                    "first": i, "max": i, "max_ratio": value,
                    "acceptance_gate": False, "reference_kind": "commanded_trajectory_phase",
                    "first_means": "first native sample in this commanded phase"})
                if value > item["max_ratio"]:
                    item.update(max=i, max_ratio=value)
            position = step.get("root_position")
            if isinstance(position, list) and len(position) == 3:
                root_initial = root_initial or position
                tol = settings.get("free_slip_tolerance_m", .005)
                signals["root/horizontal_displacement"] = math.hypot(position[0]-root_initial[0], position[1]-root_initial[1])/tol
            geometry = step.get("geometry", {})
            if geometry.get("passed") is False or geometry.get("issues"):
                signals["contact/geometry_diagnostic"] = 2.
            if step.get("contacts", {}).get("observed") is False:
                signals["contact/unobserved"] = 2.
            # Depth has no acceptance threshold in schema 3. Preserve extrema
            # as observations rather than treating a millimetre as failure.
            depths = [max(float(item.get("depth_m", 0.)), -float(item.get("separation_m", 0.)), 0.)
                      for item in geometry.get("diagnostics", [])
                      if item.get("type") in {"ground_penetration", "ground_contact_penetration"}]
            if depths:
                signals["contact/diagnostic_depth_m"] = max(depths)
            for name, value in signals.items():
                if not math.isfinite(value):
                    continue
                item = anchors.setdefault(name, {"first": None, "max": None, "max_ratio": -1.})
                cutoff = 0. if name == "contact/diagnostic_depth_m" else 1.
                if value > cutoff and item["first"] is None:
                    item["first"] = i
                if value > item["max_ratio"]:
                    item.update(max=i, max_ratio=value)
                if diagnostic_speed and name.endswith("/speed"):
                    item.update(acceptance_gate=False, reference_kind="trajectory_planning_speed",
                                first_means="first crossing of diagnostic planning reference")
                if name == "contact/diagnostic_depth_m":
                    item.update(acceptance_gate=False, units="m", first_means="first positive diagnostic depth")
        if scoped and (invalid_native_sample or count == 0):
            raise ValueError("Scoped diagnostic trajectory lacks a native initialization sample")
        if invalid_native_sample:
            return {"available": False, "reason": "not a native per-step JSONL trace", "sha256": file_hash(path)}
        selected = {i for item in anchors.values() for anchor in (item["first"], item["max"]) if anchor is not None
                    for i in range(max(0, anchor-radius), min(count, anchor+radius+1))}
        samples = {}
        fields = ("time_s", "root_position", "root_quaternion_wxyz", "joint_positions", "joint_velocities",
                  "joint_targets", "joint_efforts", "contacts", "geometry", "link_velocities")
        for i, step in enumerate(_steps(path)):
            if i in selected:
                samples[str(i)] = {k: copy.deepcopy(step[k]) for k in fields if k in step}
                samples[str(i)]["derived_trajectory_phase"] = _trajectory_phase(
                    step["time_s"], settings, segments, duration, scoped, relaxed)
        return {"available": True, "path": str(path.resolve()), "sha256": file_hash(path),
                "sample_count": count, "radius_steps": radius, "anchors": anchors, "samples": samples}
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Cannot decode bound diagnostic trajectory") from error


def build_evidence(report_path, contract, failures, *, asset_dir=None):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    failed_ids = {str(f['trial_id']) for f in failures if f.get('trial_id') is not None}
    # Acceptance independently replays every measured trial. Diagnostic windows
    # for passed trials are discarded by the API projection; do not scan their
    # long traces twice merely to discard the result. Preserve full local traces
    # and all numerical pass summaries. Unexpected/inconclusive trials retain
    # their windows, as do any explicitly named failures.
    trials = [t for t in trial_list(report)
              if t.get('status') not in {'skipped', 'skipped_by_prior_failure'}]
    selected = [t for t in trials if t.get('passed') is not True
                or str(t.get('id', t.get('kind'))) in failed_ids]
    result = {"schema_version": 1, "report": {"path": str(report_path.resolve()), "sha256": file_hash(report_path)},
            "full_report": report, "observed_failures": copy.deepcopy(failures),
            "numeric_summary": numeric_summary(report, contract),
            "trajectory_window_policy": "failed_or_inconclusive_trials.v1",
            "trajectory_windows": {str(t.get("id", t.get("kind"))): trajectory_windows(t, contract)
                                   for t in selected}}
    if asset_dir is not None:
        manifest_path = Path(asset_dir) / "asset_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        result["collision_geometry"] = {
            "asset_manifest_sha256": file_hash(manifest_path),
            "interpretation": "Source and generated proxy bounds in each authored link frame; bounds alone do not establish geometric fidelity or physical acceptance.",
            "parts": manifest.get("collision_reduction", [])}
    return result
