"""Measured symptoms and strictly scoped patches for the simplified loop."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re

from urdf_core.paths import RESOURCE_ROOT
SKILL_PATH = RESOURCE_ROOT / "skills/isaac-physics-refine/SKILL.md"
GROUND_FIELDS = {"static_friction_effort", "dynamic_friction_effort", "damping"}
CONTROL_FIELDS = {"kp", "kd", "speed_fraction"}
CONTACT_FIELDS = {"static_friction", "dynamic_friction"}
SPEED_FRACTION_RANGE = (.05, .8)
_TRIAL_ID_MARKER = re.compile(r'''(?<![\w-])(?P<quote>["'`]?)trial_id(?P=quote)\s*[:=]\s*''')
_BARE_TRIAL_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


def diagnostic_trial_ids(value, *, failures, phase, evidence_report=None):
    """Bind explicit references only; preserve explanatory evidence verbatim.

    Structured targets must agree with the current failure/phase. Explanations
    may also cite other trials in the code-provided native report, including
    passes that a repair should preserve. Such citations never authorize a
    patch. Without that report, retain the historical strict citation scope.
    """
    allowed = {f["trial_id"] for f in failures
               if f["phase"] == phase and f["failure_id"] == value.get("failure_id")}
    observed = {f["trial_id"] for f in failures}
    context_ids = set(allowed)
    if evidence_report is not None:
        trials = list(evidence_report.get("trials", []))
        reload_trial = evidence_report.get("reload_check", {}).get("trial")
        if isinstance(reload_trial, dict):
            trials.append(reload_trial)
        for trial in trials:
            identity = trial.get("id")
            if not isinstance(identity, str) or not identity:
                raise ValueError("Observed evidence report has an invalid trial ID")
            context_ids.add(identity)

    def reject(detail):
        example = ({"evidence": [sorted(allowed)[0]]} if allowed else None)
        raise ValueError("Diagnosis must cite a currently observed failure and its trial IDs. "
                         + detail + ". Allowed trial IDs: " + json.dumps(sorted(allowed), ensure_ascii=False)
                         + (". Valid bare-ID JSON example: " + json.dumps(example, ensure_ascii=False)
                            if example else ". No legal citation exists for this failure_id and phase"))

    evidence = value.get("evidence")
    if not allowed:
        reject("No matching observed failure")
    if (not isinstance(evidence, list) or not evidence
            or any(not isinstance(item, str) for item in evidence)):
        reject("evidence must remain a nonempty list of strings")
    refs = []
    if "trial_ids" in value:
        structured = value["trial_ids"]
        if (not isinstance(structured, list) or not structured
                or any(not isinstance(item, str) or not item for item in structured)):
            reject("trial_ids must be a nonempty list of original trial ID strings")
        if set(structured) - allowed:
            reject("Unknown or cross-phase/cross-failure target trial reference(s): "
                   + json.dumps(sorted(set(structured) - allowed), ensure_ascii=False))
        refs.extend(structured)
    for original in evidence:
        text = original.strip()
        # Known IDs can contain characters that require quotes in a marker.
        # A standalone identifier-like token is also a claimed ID, so an
        # unknown bare citation cannot hide beside a valid structured one.
        if text in observed | context_ids or _BARE_TRIAL_TOKEN.fullmatch(text):
            refs.append(text)
            continue
        for marker in _TRIAL_ID_MARKER.finditer(text):
            tail = text[marker.end():]
            if tail[:1] in {"'", '"', "`"}:
                end = tail.find(tail[0], 1)
                if end < 0:
                    reject("Unclosed quoted trial_id annotation")
                identity, suffix = tail[1:end], tail[end + 1:]
            else:
                token = _BARE_TRIAL_TOKEN.match(tail)
                if token is None:
                    reject("trial_id annotation requires an exact ID")
                identity, suffix = token.group(), tail[token.end():]
            if (not identity or suffix.lstrip().startswith("::")
                    or (suffix and not suffix[0].isspace() and suffix[0] not in ":,;)]}.")):
                reject("Malformed trial_id annotation; use the bare-ID JSON example")
            refs.append(identity)
    if not set(refs) & allowed:
        reject("Explanatory text alone supplies no explicit trial binding")
    invalid = set(refs) - context_ids
    if invalid:
        reject("Unknown or cross-phase/cross-failure trial reference(s): "
               + json.dumps(sorted(invalid), ensure_ascii=False))
    return frozenset(set(refs) & allowed)


def _finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _known_geometry_termination(trial):
    if trial.get("terminated_early") != "geometry_failed":
        return False
    for evidence in trial.get("geometry_failures", []):
        if evidence.get("complete") is not True:
            continue
        for issue in evidence.get("issues", []):
            kind = issue.get("type")
            if kind == "ground_penetration":
                value = issue.get("depth_m")
                if _finite_number(value) and value > 0:
                    return True
            elif kind == "ground_contact_penetration":
                value = issue.get("separation_m")
                if _finite_number(value) and value < 0:
                    return True
            elif kind == "structural_separation":
                if any(_finite_number(issue.get(key)) and issue[key] > 0
                       for key in ("error_m", "rotation_error_rad")):
                    return True
    return False


def _solver_mismatch_after_complete_settle(trial, metrics, errors, contract):
    # Solver position corrections need not integrate sampled native velocity.
    # Escalate only a materially failed, complete, passive posture experiment;
    # this is a reason to inspect solver evidence, not proof of its root cause.
    if not isinstance(contract, dict) or contract.get("schema_version") not in (2, 3):
        return False
    if (trial.get("terminated_early") is not None or trial.get("geometry_failures")
            or any(word in " ".join(str(e).lower() for e in errors)
                   for word in ("geometry", "contact", "structure", "penetrat", "missing", "incomplete", "invalid", "nonfinite"))):
        return False
    settings = contract.get("config", {})
    duration = metrics.get("duration_s")
    expected = trial.get("expected_duration_s")
    actual = trial.get("actual_duration_s")
    dt = settings.get("physics_dt_s")
    frozen_duration = settings.get("settle_duration_s")
    values = (duration, expected, actual, dt, frozen_duration)
    if not all(_finite_number(v) and v > 0 for v in values):
        return False
    if (abs(expected - frozen_duration) > 1e-8 or abs(actual - expected) > dt * 1.01
            or abs(duration - actual) > dt * 1.01
            or not isinstance(trial.get("expected_physics_steps"), int)
            or trial["expected_physics_steps"] < 2
            or trial.get("completed_physics_steps") != trial["expected_physics_steps"]):
        return False
    joints = metrics.get("joints", {})
    definitions = contract.get("joints", {})
    if not joints or set(joints) != set(definitions):
        return False
    if any(data.get("peak_active_effort") != 0 for data in joints.values()):
        return False
    for name, data in joints.items():
        tolerance = (settings.get("position_tolerance_m") if definitions[name].get("type") == "prismatic"
                     else math.radians(settings.get("orientation_tolerance_deg", 0)))
        values = [data.get(key) for key in ("min_q", "max_q", "peak_velocity", "max_initial_q_deviation")]
        if not _finite_number(tolerance) or tolerance <= 0 or not all(_finite_number(v) for v in values):
            continue
        lower, upper, velocity, initial_deviation = values
        if (initial_deviation > tolerance + 1e-8 and upper - lower > tolerance + 1e-8
                and upper - lower > max(tolerance, velocity * duration * 1.05 + 1e-5)):
            return True
    return False


def observed_failures(report, contract=None):
    """Route numeric failures; the API chooses causes, not fabricated symptoms."""
    result = []
    trials = list(report.get("trials", []))
    reload_trial = report.get("reload_check", {}).get("trial")
    if isinstance(reload_trial, dict):
        trials.append(reload_trial)
    for trial in trials:
        if trial.get("passed") is True or trial.get("status") == "skipped":
            continue
        kind = trial.get("kind", "unknown")
        phase = "ground_settle" if kind in {"ground_settle", "reload", "fall"} else kind
        measured = trial.get("metrics", {})
        metrics = measured.get("metrics", measured)
        errors = measured.get("errors", trial.get("errors", []))
        kinds = set()
        # Native metric errors append link/joint identifiers after a colon.
        # Those identifiers are evidence, not symptom descriptions: a held
        # joint named "tilt_axis" must not fabricate a base-tipping failure.
        # Preserve the original errors in the returned evidence below.
        clauses = []
        for error in errors:
            clause = str(error).lower()
            if clause.startswith("pre-motion: "):
                clause = clause[len("pre-motion: "):]
            # A phase label is not evidence that the stable tail failed.
            # Preserve the original error below, including its object identity.
            if clause.startswith("ground settle "):
                clause = clause[len("ground settle "):]
            clauses.append(clause.partition(":")[0])
        text = " ".join(clauses)
        if any(word in text for word in ("penetrat", "geometry", "contact", "structure")):
            kinds.add("external_contact")
        if any(word in text for word in ("tilt", "tipp")):
            kinds.add("rest_pose_drift" if phase == "ground_settle" else "base_tipping")
        if any(word in text for word in ("slid", "horizontal")):
            kinds.add("base_sliding")
        if any(word in text for word in ("initial pose", "initial-pose", "initial_pose", "pose drift")):
            kinds.add("rest_pose_drift")
        if any(word in text for word in ("settle", "tail", "stable")):
            kinds.add("not_settled")
        if any(word in text for word in ("held", "hold deviation", "non-target", "non-tested joint")):
            kinds.add("held_joint_drift")
        if any(word in text for word in ("speed", "velocity", "limit", "overshoot")):
            kinds.add("speed_or_limit")
        if any(word in text for word in ("traverse", "range", "endpoint", "return", "coverage")):
            joint = metrics.get("joints", {}).get(trial.get("joint_name"), {})
            saturation = joint.get("saturated_duration_s", 0)
            kinds.add("effort_saturated" if saturation >= .25 else "tracking_lag")
        if phase == "ground_settle" and any("joint" in str(e).lower() for e in errors):
            kinds.add("joint_sag")
        if any(word in text for word in ("missing", "nonfinite", "mapping", "invalid", "incomplete")):
            kinds.add("joint_configuration")
        if _known_geometry_termination(trial):
            # Missing tails and range coverage follow from the deliberate stop.
            # Preserve the proven collision failure as the primary diagnosis.
            kinds = {"external_contact"}
        elif phase == "ground_settle" and _solver_mismatch_after_complete_settle(
                trial, metrics, errors, contract):
            kinds.add("solver_state_inconsistent")
        for failure in sorted(kinds or {"unknown"}):
            result.append({"failure_id": failure, "phase": phase,
                           "trial_id": trial.get("id", kind), "joint_name": trial.get("joint_name"),
                           "errors": errors, "metrics": metrics,
                           "terminated_early": trial.get("terminated_early"),
                           "geometry_failures": copy.deepcopy(trial.get("geometry_failures", []))})
    return result


def failure_phase(failures):
    for phase in ("ground_settle", "fixed_joint", "free_joint"):
        if any(item["phase"] == phase for item in failures):
            return phase
    return "unknown"


def allowed_paths(profile, controller, phase):
    paths = []
    if phase in {"ground_settle", "return_to_ground"}:
        for link, data in profile["links"].items():
            for part, obj in data["objects"].items():
                for key in ("mass_kg", "density_kg_m3"):
                    if key in obj:
                        paths.append(f"profile.links.{link}.objects.{part}.{key}")
        for joint in profile["joints"]:
            paths += [f"profile.joints.{joint}.{key}" for key in sorted(GROUND_FIELDS)]
        # Passive ground trials also exercise contact materials, including on
        # assets without actuated joints. Keep the existing material bounds;
        # changing them here spends ground budget and revalidates all phases.
        for link in profile["links"]:
            paths += [f"profile.links.{link}.material.{key}" for key in sorted(CONTACT_FIELDS)]
    elif phase in {"fixed_joint", "free_joint", "return_to_fixed"}:
        for joint in controller:
            fields = CONTROL_FIELDS if phase in {"fixed_joint", "return_to_fixed"} else {"speed_fraction"}
            paths += [f"controller.{joint}.{key}" for key in sorted(fields)]
        if phase == "free_joint":
            for link in profile["links"]:
                paths += [f"profile.links.{link}.material.{key}" for key in sorted(CONTACT_FIELDS)]
    return paths


def _path_locations(profile, controller):
    # Map exact authored names rather than split on '.', since Blender names
    # commonly contain periods (e.g. Cube.001).
    locations = {}
    for phase in ("ground_settle", "fixed_joint", "free_joint"):
        for path in allowed_paths(profile, controller, phase):
            if path.startswith("controller."):
                body, field = path[len("controller."):].rsplit(".", 1)
                locations[path] = (controller[body], field)
    for link, data in profile["links"].items():
        for part, obj in data["objects"].items():
            for key in ("mass_kg", "density_kg_m3"):
                if key in obj:
                    locations[f"profile.links.{link}.objects.{part}.{key}"] = (obj, key)
        for key in CONTACT_FIELDS:
            locations[f"profile.links.{link}.material.{key}"] = (data["material"], key)
    for joint, data in profile["joints"].items():
        for key in GROUND_FIELDS:
            locations[f"profile.joints.{joint}.{key}"] = (data, key)
    return locations


def refinement_parameter_rules():
    """Code-owned legal ranges, not advice about which parameter to change.

    Permissions still come from allowed_paths. Initial baselines and frozen
    efforts are already supplied verbatim; do not duplicate every part/joint.
    """
    return {
        "policy": "explicit_refinement_parameter_ranges.v4",
        "all_patch_values": "Finite numbers; booleans are not numbers.",
        "scope": "Only paths in the chosen route's allowed_paths may change. These ranges do not grant additional permissions.",
        "mass_kg_or_density_kg_m3": "Finite and > 0; no multiplier limit relative to the initial value.",
        "joint_friction_effort": "0 <= dynamic_friction_effort <= static_friction_effort <= the joint's frozen contract.max_effort; use the proposed values together.",
        "joint_damping": "damping >= 0; SI units, natively applied once.",
        "contact_friction": "0 <= dynamic_friction <= static_friction <= 2; use the proposed values together.",
        "controller": {"kp": "finite and > 0", "kd": "finite and >= 0",
                       "speed_fraction_inclusive": list(SPEED_FRACTION_RANGE)},
        "speed_fraction_meaning": "Fraction of the frozen target scheduling speed; this is not an instantaneous measured-speed acceptance gate.",
    }


def validate_parameter_bounds(profile, initial, controller, contract):
    for link, data in profile["links"].items():
        for part, obj in data["objects"].items():
            for field in ("mass_kg", "density_kg_m3"):
                if field in obj:
                    value = obj[field]
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or value <= 0):
                        raise ValueError(f"Part {part} {field} must be finite and > 0")
    for name, data in profile["joints"].items():
        cap = contract["joints"][name]["max_effort"]
        if not 0 <= data["dynamic_friction_effort"] <= data["static_friction_effort"] <= cap:
            raise ValueError(f"{name}: require 0 <= dynamic <= static <= frozen effort")
    for name, data in controller.items():
        if set(data) != CONTROL_FIELDS | {"initial_position"} or any(isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) for v in data.values()):
            raise ValueError(f"Invalid controller: {name}")
        if data["kp"] <= 0:
            raise ValueError(f"controller.{name}.kp must be > 0; received {data['kp']}")
        if data["kd"] < 0:
            raise ValueError(f"controller.{name}.kd must be >= 0; received {data['kd']}")
        lower, upper = SPEED_FRACTION_RANGE
        if not lower <= data["speed_fraction"] <= upper:
            raise ValueError(f"controller.{name}.speed_fraction must lie in [{lower}, {upper}] inclusive; received {data['speed_fraction']}")


def apply_proposal(value, *, profile, initial_profile, controller, contract, spec, failures, phase, remaining,
                   evidence_report=None, joint_groups=None):
    from .protocol import validate_profile
    if not isinstance(value, dict) or value.get("verdict") not in {"refine", "uncertain", "unrepairable"}:
        raise ValueError("Need refine/uncertain/unrepairable verdict")
    if set(value) - {"verdict", "failure_id", "evidence", "trial_ids", "reason", "hypothesis", "expected_effect", "route", "patches"}:
        raise ValueError("Unknown diagnostic fields")
    diagnostic_trial_ids(value, failures=failures, phase=phase, evidence_report=evidence_report)
    if not str(value.get("reason", "")).strip():
        raise ValueError("Diagnosis needs a reason")
    patches = value.get("patches")
    if not isinstance(patches, list):
        raise ValueError("patches must be a list")
    if joint_groups is not None:
        from .joint_groups import expand_patches
        patches = expand_patches(patches, joint_groups)
    if value["verdict"] != "refine":
        if patches:
            raise ValueError("Uncertain/unrepairable cannot mutate parameters")
        return {**value, "patches": []}
    # Failure IDs identify observed symptoms, never a closed list of causes.
    # Any evidence-grounded hypothesis can propose a legal stage-owned patch;
    # actual numerical retesting and the best-candidate gate decide its effect.
    if "hypothesis" in value and (not isinstance(value["hypothesis"], str) or not value["hypothesis"].strip()):
        raise ValueError("hypothesis must be nonempty explanatory text")
    route = value.get("route")
    if not ((route == phase and phase in {"ground_settle", "fixed_joint", "free_joint"})
            or (phase in {"fixed_joint", "free_joint"} and route == "return_to_ground")
            or (phase == "free_joint" and route == "return_to_fixed")):
        raise ValueError("Repair route violates stage ownership")
    family = "ground" if route in {"ground_settle", "return_to_ground"} else "motion"
    if remaining[family] <= 0:
        raise ValueError(f"{family} repair budget exhausted")
    if not patches or not str(value.get("expected_effect", "")).strip():
        raise ValueError("Refine requires patches and expected measured effect")
    updated, controls = copy.deepcopy(profile), copy.deepcopy(controller)
    locations = _path_locations(updated, controls)
    allowed, seen = set(allowed_paths(updated, controls, route)), set()
    changed = False
    for patch in patches:
        if not isinstance(patch, dict) or set(patch) != {"path", "value"}:
            raise ValueError("Each patch has exactly path/value")
        path, number = patch["path"], patch["value"]
        if path not in allowed or path in seen:
            raise ValueError("Forbidden/duplicate parameter path: " + str(path))
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError("Patch values must be finite numbers")
        seen.add(path)
        owner, key = locations[path]
        previous = owner[key]
        changed |= previous != number
        owner[key] = number
    if not changed:
        raise ValueError("Refinement must change a permitted parameter")
    updated = validate_profile(updated, spec)
    validate_parameter_bounds(updated, initial_profile, controls, contract)
    if joint_groups is not None:
        from .joint_groups import inconsistencies
        if inconsistencies(joint_groups, updated, controls):
            raise ValueError('Refinement must preserve shared group parameters')
    return {**value, "patches": patches, "profile": updated, "controller": controls, "budget_family": family}
