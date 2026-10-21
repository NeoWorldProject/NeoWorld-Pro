"""Validate complete physics evidence; persist reusable independent metric receipts."""
from __future__ import annotations

import json
import math
from pathlib import Path

from urdf_core.run_state import file_hash
from .critic import bound_file, read_json
from .trajectory_io import first_trajectory_sample, iter_trajectory_samples


def recompute_trial_metrics(trajectory_path, *, kind, joint_name, contract,
                            include_initial_sample=False):
    """Re-evaluate the complete JSONL, optionally retaining its parsed first sample.

    The optional pair is returned only after every sample and the gzip footer
    have been validated. It is local to this invocation, never a reader cache.
    """
    from .metrics import evaluate_trial

    trace = list(iter_trajectory_samples(trajectory_path))
    reload_kind = "ground_settle"
    result = evaluate_trial(trace, kind=reload_kind if kind == "reload" else kind,
                            joint_name=joint_name, contract=contract)
    if include_initial_sample:
        if not trace:
            raise ValueError("Physics trajectory contains no samples")
        return result, trace[0]
    return result


def _same_metrics(left, right):
    """JSON equality with small numeric roundoff tolerance and strict bool types."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (math.isfinite(left) and math.isfinite(right)
                and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-10))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_metrics(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_metrics(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def validate_trial_metrics(trial, recomputed, contract):
    """Require report aggregates to match recorded measurements and termination."""
    identity = trial.get("id", "unknown")
    if trial.get("passed") is not recomputed.get("passed"):
        raise ValueError(f"Physics trial verdict contradicts its recorded trajectory: {identity}")
    # Failure messages are an unordered collection: link iteration in a worker
    # can differ from a verifier in another Python process. Keep duplicates and
    # exact text, and keep every numerical/ordered measurement strictly bound.
    def canonical_failures(result):
        if (isinstance(result, dict) and isinstance(result.get("errors"), list)
                and all(isinstance(error, str) for error in result["errors"])):
            return dict(result, errors=sorted(result["errors"]))
        return result

    if not _same_metrics(canonical_failures(trial.get("metrics")), canonical_failures(recomputed)):
        raise ValueError(f"Physics trial metrics differ from independently recomputed evidence: {identity}")
    terminated = trial.get("terminated_early")
    if terminated not in (None, "geometry_failed", "geometry_observation_incomplete"):
        raise ValueError(f"Unknown early physics termination reason: {identity}")
    if terminated is not None and trial.get("passed") is not False:
        raise ValueError(f"An early-terminated physics trial cannot pass: {identity}")
    # New workers publish these fields; require their mutual agreement whenever
    # present, while older completed traces remain independently re-evaluable.
    expected_duration = trial.get("expected_duration_s", trial.get("duration_s"))
    if expected_duration is not None:
        if (not isinstance(expected_duration, (float, int)) or isinstance(expected_duration, bool)
                or not math.isfinite(expected_duration) or expected_duration <= 0):
            raise ValueError(f"Invalid planned physics duration: {identity}")
        if trial.get("duration_s", expected_duration) != expected_duration:
            raise ValueError(f"Physics trial changed its declared planned duration: {identity}")
        dt = float(contract["config"]["physics_dt_s"])
        actual_duration = recomputed.get("metrics", {}).get("duration_s")
        if actual_duration is None or not math.isfinite(actual_duration):
            raise ValueError(f"Physics trial lacks measured duration: {identity}")
        if terminated is None and (actual_duration < expected_duration - dt * 1.05
                                   or actual_duration > expected_duration + dt * 1.05):
            raise ValueError(f"Complete physics trial does not cover its planned duration: {identity}")
        if actual_duration > expected_duration + dt * 1.05:
            raise ValueError(f"Physics trajectory exceeds its planned duration: {identity}")
        if "actual_duration_s" in trial and not math.isclose(trial["actual_duration_s"], actual_duration,
                                                              rel_tol=1e-8, abs_tol=dt * 1.01):
            raise ValueError(f"Physics trial actual duration differs from the trajectory: {identity}")
        expected_steps = math.ceil(expected_duration / dt) + 1
        if "expected_physics_steps" in trial and trial["expected_physics_steps"] != expected_steps:
            raise ValueError(f"Physics trial expected step count differs from its frozen duration: {identity}")
    steps = recomputed.get("metrics", {}).get("physics_steps")
    if "completed_physics_steps" in trial and trial["completed_physics_steps"] != steps:
        raise ValueError(f"Physics trial completed step count differs from the trajectory: {identity}")


def validate_staged_trial_coverage(report, contract, expected_trials):
    """Validate a measured prefix and exact, evidence-backed omissions.

    Return True only for an explicitly declared short circuit. Legacy complete
    reports retain their existing full-set validation; partial schema-1
    contracts are never accepted. Real recorded trial metrics are checked by
    verify_simulation_report after this structural check.
    """
    trials = report.get("trials", [])
    actual = {(t.get("kind"), t.get("joint_name")) for t in trials}
    if len(actual) != len(trials):
        raise ValueError("Physics evidence contains duplicate trial identities")
    if "evaluation_complete" not in report:
        if (actual != expected_trials or report.get("blocked_trials")
                or report.get("stopped_after_phase") is not None):
            raise ValueError("Physics evidence does not cover the frozen, applicable trial set")
        return False
    if report["evaluation_complete"] is not True:
        raise ValueError("Physics evaluation is incomplete")
    from .complete_evaluation import POLICY as COMPLETE_POLICY, validate_coverage
    if report.get('trial_evaluation_policy') == COMPLETE_POLICY:
        return validate_coverage(report, expected_trials)
    phase_order = ["ground_settle", "fixed_joint", "free_joint"]
    policy = report.get("trial_evaluation_policy")
    from .trial_order import LOCAL_POLICY, validate_local_coverage
    if policy == LOCAL_POLICY:
        return validate_local_coverage(report, contract, expected_trials)
    from .trial_order import POLICY as ORDER_POLICY, verify_plan
    orders = {phase: list(contract['joints']) for phase in ('fixed_joint', 'free_joint')}
    if policy == ORDER_POLICY:
        orders = verify_plan(report.get('trial_order', {}), contract)
    if policy not in (None, "first_failed_trial.v1", ORDER_POLICY):
        raise ValueError("Unknown physics trial evaluation policy")
    first_failed_trial = policy in ("first_failed_trial.v1", ORDER_POLICY)
    by_phase = {phase: [t for t in trials if t.get("kind") == phase] for phase in phase_order}
    if any(t.get("kind") not in phase_order for t in trials):
        raise ValueError("Unknown physics phase in staged evaluation")
    recorded_order = [phase_order.index(t["kind"]) for t in trials]
    if recorded_order != sorted(recorded_order):
        raise ValueError("Physics trials violate ground/fixed/free phase order")
    first_failure, failed_ids, expected_blocked = None, [], []
    for phase in phase_order:
        required = {identity for identity in expected_trials if identity[0] == phase}
        present = {(t["kind"], t.get("joint_name")) for t in by_phase[phase]}
        if first_failure is None:
            failed_ids = [t["id"] for t in by_phase[phase] if t.get("passed") is not True]
            if first_failed_trial:
                order = ([None] if phase == "ground_settle" else orders[phase])
                expected_order = [name for name in order if (phase, name) in required]
                measured_order = [t.get("joint_name") for t in by_phase[phase]]
                if (measured_order != expected_order[:len(measured_order)]
                        or not present <= required):
                    raise ValueError("First-failure evaluation must preserve the frozen trial order")
                if failed_ids and (len(failed_ids) != 1 or by_phase[phase][-1]["id"] != failed_ids[0]):
                    raise ValueError("First-failure evaluation ran another trial after a measured failure")
            if present != required and (not first_failed_trial or not failed_ids):
                raise ValueError("Physics evidence does not cover the frozen, applicable trial set for phase " + phase)
            if failed_ids:
                first_failure = phase
                expected_blocked.extend({"kind": kind, "joint_name": name,
                                         "status": "skipped_by_prior_failure",
                                         "blocked_by_phase": first_failure, "failed_trial_ids": failed_ids}
                                        for kind, name in sorted(required - present))
        else:
            if present:
                raise ValueError("Physics evaluation ran a stage blocked by prior failure")
            expected_blocked.extend({"kind": kind, "joint_name": name,
                                     "status": "skipped_by_prior_failure",
                                     "blocked_by_phase": first_failure, "failed_trial_ids": failed_ids}
                                    for kind, name in sorted(required))
    if first_failure is None:
        if (report.get("acceptance_complete") is not True or report.get("blocked_trials") != []
                or report.get("stopped_after_phase") is not None):
            raise ValueError("Complete staged evaluation has inconsistent completion/blocking fields")
        return False
    blocked = report.get("blocked_trials")
    if (not isinstance(blocked, list) or len(blocked) != len(expected_blocked)
            or any(not isinstance(item, dict) for item in blocked)
            or sorted(json.dumps(item, sort_keys=True) for item in blocked)
            != sorted(json.dumps(item, sort_keys=True) for item in expected_blocked)):
        raise ValueError("Blocked trials do not exactly match the measured failed stage")
    expected_reload = {"passed": False, "status": "skipped_by_prior_failure",
                       "blocked_by_phase": first_failure, "failed_trial_ids": failed_ids}
    if (report.get("acceptance_complete") is not False
            or report.get("stopped_after_phase") != first_failure
            or report.get("reload_check") != expected_reload
            or report.get("numerical_passed") is not False
            or report.get("passed") is not False or report.get("status") != "physics_failed"):
        raise ValueError("Stage rejection cannot claim complete acceptance or unmeasured reload evidence")
    return True


def verify_motion_scope_source(plan, geometry_binding, setup, contract, joint_name, *, asset_dir, report_dir, rest_pose=None):
    """Rebuild the operation from the recorded cooked points and original URDF.

    The source is native cooking, not a visual-mesh guess. These measurements
    are saved once per trial and included in the immutable artifact inventory.
    """
    import hashlib
    import numpy as np
    from .motion_scope import plan_joint_motion, validate_motion_plan
    from urdf_core.urdf_validation import read_urdf, urdf_fk
    validate_motion_plan(plan)
    if joint_name not in contract["joints"]:
        raise ValueError("Motion scope references an unknown joint")
    path = bound_file(geometry_binding.get("path", ""), geometry_binding.get("sha256"), report_dir)
    cooked = read_json(path)
    digest = hashlib.sha256()
    for name, meshes in sorted(cooked.items()):
        digest.update(name.encode())
        for mesh in meshes:
            digest.update(np.asarray(mesh["vertices"], dtype=np.float64).tobytes())
            digest.update(np.asarray(mesh["faces"], dtype=np.int64).tobytes())
    if digest.hexdigest() != setup.get("cooked_geometry_sha256"):
        raise ValueError("Motion scope geometry differs from native cooked geometry")
    joint = contract["joints"][joint_name]
    parsed = read_urdf(Path(asset_dir) / "object.urdf")
    initial = {name: item["initial_position"] for name, item in contract["joints"].items()}
    frames = urdf_fk(parsed, initial)
    points = np.concatenate([np.asarray(mesh["vertices"]) @ frames[name][:3, :3].T + frames[name][:3, 3]
                             for name, meshes in cooked.items() for mesh in meshes])
    ground_root = np.eye(4)
    ground_root[2, 3] = -float(points[:, 2].min())
    if rest_pose is not None:
        from .support_rest_pose import validate_support_rest_pose, support_world_matrix
        validate_support_rest_pose(rest_pose, contract)
        ground_root = support_world_matrix(rest_pose)
    if not np.allclose(plan["root_transform"], ground_root, rtol=0, atol=1e-10):
        raise ValueError("Motion scope must start at the prescribed cooked-ground pose or its bound measured rest pose")
    rebuilt = plan_joint_motion(parsed, cooked, initial,
        plan["root_transform"], joint_name, joint["lower"], joint["upper"])
    if rebuilt != plan:
        raise ValueError("Motion scope does not reproduce from the recorded native geometry")
    if setup.get("motion_scope_plan") != plan or setup.get("motion_scope_geometry") != geometry_binding:
        raise ValueError("Trial setup and recorded motion scope differ")
    return path


def verify_simulation_report(report_path, *, root, contract_path, asset_dir, require_pass=True):
    root, report_path, contract_path, asset_dir = map(lambda p: Path(p).resolve(),
                                                   (root, report_path, contract_path, asset_dir))
    report = read_json(report_path)
    if report.get("schema_version") != 1 or report.get("status") not in {"passed", "physics_failed"}:
        raise ValueError("Physics worker did not complete a valid evaluation")
    completed = report.get("evaluation_complete", report.get("acceptance_complete"))
    if (completed is not True or (require_pass and report.get("acceptance_complete") is not True)
            or type(report.get("worker_returncode")) is not int or report["worker_returncode"] != 0
            or report.get("errors") != []):
        raise ValueError("Physics worker completion/exit/error evidence is incomplete")
    from .verify_physics_isaac import source_hashes
    from .measurement_reuse import runtime_sources_match
    expected_sources = source_hashes()
    sources_match = runtime_sources_match(report.get("source_hashes"), expected_sources)
    if (report.get("source_hashes") != report.get("worker_sources") or not sources_match):
        raise ValueError("Physics supervisor/worker sources differ from the current runtime implementation")
    if require_pass and (report.get("passed") is not True or report.get("numerical_passed") is not True
                         or report.get("status") != "passed"):
        raise ValueError("Physics measurements did not pass")
    inputs = report.get("inputs", {})
    expected_inputs = {"urdf": asset_dir / "object.urdf", "spec": asset_dir / "kinematics.json",
                       "profile": asset_dir / "physics_profile.json", "contract": contract_path,
                       "asset_manifest": asset_dir / "asset_manifest.json"}
    from .trial_order import POLICY as ORDER_POLICY, LOCAL_POLICY
    if report.get('trial_evaluation_policy') in (ORDER_POLICY, LOCAL_POLICY):
        expected_inputs['trial_order'] = contract_path.parent / 'trial_order.json'
    sampling_path = contract_path.parent / 'representative_motion_input.json'
    if sampling_path.is_file():
        expected_inputs['representative_motion'] = sampling_path
    elif 'representative_motion' in report.get('inputs', {}) or 'representative_motion' in report:
        raise ValueError('Representative scope lacks the frozen per-candidate annex')
    from .incidental_motion import INPUT_FILE as INCIDENTAL_INPUT, POLICY as INCIDENTAL_POLICY
    incidental_path = contract_path.parent / INCIDENTAL_INPUT
    if incidental_path.exists():
        expected_inputs['incidental_motion'] = incidental_path
        if report.get('incidental_motion_policy') != INCIDENTAL_POLICY:
            raise ValueError('Report omitted its per-trial incidental-motion policy')
    elif report.get('incidental_motion_policy') or 'incidental_motion' in report.get('inputs', {}):
        raise ValueError('Incidental-motion review has no frozen input')
    artifacts = {report_path}
    for name, expected in expected_inputs.items():
        record = inputs.get(name, {})
        path = bound_file(record.get("path", ""), record.get("sha256"), root)
        if path != expected.resolve():
            raise ValueError(f"Physics report references a different {name}")
        artifacts.add(path)
    if 'trial_order' in expected_inputs:
        plan = read_json(expected_inputs['trial_order'])
        if report.get('trial_order') != plan:
            raise ValueError('Trial ordering does not match its bound scheduling input')
        source = plan.get('source_report')
        if source:
            artifacts.add(bound_file(source['path'], source['sha256'], root))
    spec = read_json(expected_inputs["spec"])
    from urdf_core.kinematics import validate_spec
    from .protocol import validate_profile
    spec = validate_spec(spec)
    validate_profile(read_json(expected_inputs["profile"]), spec)
    names = {j["name"] for j in spec["joints"] if j["type"] != "fixed"}
    contract = read_json(contract_path)
    if contract.get("schema_version") != 3:
        raise ValueError("Frozen test contract requires schema version 3")
    if contract.get("schema_version") == 3:
        from .protocol import validate_current_contract
        validate_current_contract(contract)
        if report.get("acceptance_policy") != contract["acceptance_policy"]:
            raise ValueError("Worker report does not bind the current functional acceptance policy")
    if (contract.get("acceptance_policy") in {'object_functional_physics.v11'}
            and contract.get("structural_joints") != spec["joints"]):
        raise ValueError("Frozen structural joints differ from the normalized source kinematics")
    if set(contract["joints"]) != names:
        raise ValueError("Frozen contract does not cover the current joints")
    from .protocol import free_base_required
    needs_free = free_base_required(contract)
    stable_required = contract.get("stable_required", True)
    if contract.get("free_base_policy") != report.get("free_base_policy"):
        raise ValueError("Report does not bind frozen free-base applicability")
    expected_trials = ({('ground_settle', None)} if stable_required else set())
    expected_trials |= {('fixed_joint', name) for name in names}
    if needs_free:
        expected_trials |= {('free_joint', name) for name in names}
    from .representative_motion import verify_report_plan
    representative, representative_files = verify_report_plan(report, contract, asset_dir)
    artifacts.update(representative_files)
    if representative is not None:
        expected_trials = {identity for identity in expected_trials if identity[0] == 'ground_settle'
                           or identity[1] in representative['selected'][identity[0]]}
    trials = report.get("trials", [])
    measured_rest_candidate = None
    if report.get('free_motion_initialization_policy') == 'measured_stable_rest.v1' and report.get('support_rest_pose', {}).get('path'):
        record = report['support_rest_pose']
        measured_rest_candidate = read_json(bound_file(record['path'], record['sha256'], root))
    if contract.get("motion_scope_policy"):
        if report.get("motion_scope_policy") != contract["motion_scope_policy"]:
            raise ValueError("Physics report does not bind its motion scope policy")
        excluded = report.get("inapplicable_ground_operations")
        if not isinstance(excluded, list):
            raise ValueError("Missing explicit ground operation applicability evidence")
        omitted = set()
        for item in excluded:
            name, plan, setup = item.get("joint_name"), item.get("motion_scope_plan"), item.get("setup", {})
            if (item.get("kind") != "free_joint" or name not in names or not needs_free
                    or item.get("status") != "not_applicable_ground_blocked"
                    or item.get("passed", False) is not None or item.get("motion_passed", False) is not None
                    or item.get("ground_coverage_of_full_range") != 0 or name in omitted):
                raise ValueError("Invalid declaration of a ground-blocked operation")
            artifacts.add(verify_motion_scope_source(plan, setup.get("motion_scope_geometry", {}), setup,
                contract, name, asset_dir=asset_dir, report_dir=report_path.parent, rest_pose=measured_rest_candidate))
            if plan["free_status"] != "ground_blocked":
                raise ValueError("An operable ground interval cannot be omitted")
            if setup.get("fixed_root") is not False or setup.get("fixture_clearance_m") is not None:
                raise ValueError("Ground applicability was not evaluated in the free ground setup")
            fixed = [t for t in trials if t.get("kind") == "fixed_joint"]
            fixed_names = set(representative['selected']['fixed_joint']) if representative else names
            required_fixed = (set() if report.get('trial_evaluation_policy') == 'all_independent_trials.v1' else {name} if report.get('trial_evaluation_policy') == LOCAL_POLICY else fixed_names)
            if not required_fixed <= {t.get("joint_name") for t in fixed if t.get("passed") is True}:
                raise ValueError("Ground scope cannot bypass a failed or missing fixed full-range test")
            omitted.add(name)
        expected_trials -= {("free_joint", name) for name in omitted}
    elif report.get("inapplicable_ground_operations") or report.get("motion_scope_policy"):
        raise ValueError("Historical contracts cannot acquire new motion exclusions")
    short_circuited = validate_staged_trial_coverage(report, contract, expected_trials)
    ids = [t.get("id") for t in trials]
    if len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("Physics trials need unique identifiers")
    motion_initial_samples = {}
    metric_reviews = []
    evidence_roots = {}
    def verify_trial(trial, *, must_pass):
        if must_pass and trial.get("passed") is not True:
            raise ValueError(f"Failed physics trial: {trial['id']}")
        from .trial_checkpoint import trial_evidence_root
        evidence_root = trial_evidence_root(trial, report, report_dir=report_path.parent, root=root)
        evidence_roots[trial['id']] = evidence_root
        trajectory_path = None
        from .deferred_capture import POLICY as DEFERRED_POLICY, NUMERIC_DELIVERY_POLICY, verify_media
        deferred = report.get('media_policy') in (DEFERRED_POLICY, NUMERIC_DELIVERY_POLICY)
        numeric_delivery = report.get('media_policy') == NUMERIC_DELIVERY_POLICY
        prefixes = ('trajectory',) if numeric_delivery or (deferred and not report.get('passed')) else ('trajectory', 'video')
        for prefix in prefixes:
            path = bound_file(trial.get(prefix + "_path", ""), trial.get(prefix + "_sha256"), evidence_root)
            artifacts.add(path)
            if prefix == "trajectory":
                trajectory_path = path
        scoped_motion = (contract.get("motion_scope_policy")
                         and trial.get("kind") in {"fixed_joint", "free_joint"})
        options = {"include_initial_sample": True} if scoped_motion else {}
        from .verification_receipt import verified_metrics
        recomputed, metric_review = verified_metrics(trajectory_path, root=root,
            kind=trial.get("kind"), joint_name=trial.get("joint_name"), contract=contract,
            inputs=inputs, evaluator=recompute_trial_metrics, **options)
        metric_reviews.append(dict(metric_review, trial_id=trial['id']))
        if scoped_motion:
            recomputed, first = recomputed
            if trial.get('kind') == 'free_joint':
                motion_initial_samples[trial['id']] = first
        from .incidental_motion import pending_names, verify_receipt, apply_decision
        pending = pending_names(recomputed, trial.get('kind'))
        if trial.get('incidental_motion_review'):
            if 'incidental_motion' not in expected_inputs or not pending:
                raise ValueError('Unauthorized incidental-motion decision')
            decision, review_files = verify_receipt(trial['incidental_motion_review'], raw=recomputed,
                trial=trial, report=report, root=evidence_root)
            artifacts.update(review_files)
            recomputed = apply_decision(recomputed, trial['kind'], decision)
        elif pending and 'incidental_motion' in expected_inputs:
            raise ValueError('Hold deviations require a completed API review')
        validate_trial_metrics(trial, recomputed, contract)
        if scoped_motion:
            binding = first.get("motion_scope_binding", {})
            setup = trial.get("setup", {})
            artifacts.add(verify_motion_scope_source(first.get("motion_scope_plan"),
                binding.get("geometry_artifact", {}), setup, contract, trial.get("joint_name"),
                asset_dir=asset_dir, report_dir=evidence_root,
                rest_pose=measured_rest_candidate if trial.get('kind') == 'free_joint' else None))
            if (binding.get("urdf_sha256") != file_hash(asset_dir / "object.urdf")
                    or binding.get("cooked_geometry_sha256") != setup.get("cooked_geometry_sha256")
                    or binding.get("fixture_clearance_m") != setup.get("fixture_clearance_m")):
                raise ValueError("Motion trace source binding differs from its trial setup")
        frames = trial.get("frames", [])
        if deferred:
            artifacts.update(verify_media(trial, report, contract, evidence_root))
        from .capture_retention import verify_retained_captures
        retention = verify_retained_captures(trial, evidence_root)
        if retention is not None:
            artifacts.add(retention)
        if len(frames) < 2 and not numeric_delivery and not (deferred and trial.get('recording_mode') == 'numeric_only'):
            raise ValueError("Missing actual dynamics frames")
        previous_t = -1.0
        indices = set()
        for frame in frames:
            artifacts.add(bound_file(frame.get("path", ""), frame.get("sha256"), evidence_root))
            if (type(frame.get("frame_index")) is not int or frame["frame_index"] < 0
                    or not isinstance(frame.get("time_seconds"), (int, float))
                    or isinstance(frame["time_seconds"], bool) or not math.isfinite(frame["time_seconds"])
                    or frame["frame_index"] in indices or frame["time_seconds"] <= previous_t):
                raise ValueError("Frame ordering/index is inconsistent")
            indices.add(frame["frame_index"])
            previous_t = frame["time_seconds"]
    for trial in trials:
        verify_trial(trial, must_pass=require_pass)
    delivered = bound_file(report.get("delivered_usd_path", ""), report.get("delivered_usd_sha256"), root)
    artifacts.add(delivered)
    rest_policy = report.get('free_motion_initialization_policy')
    rest_operations = report.get('inapplicable_ground_operations', [])
    if rest_policy is None:
        if any(s.get('initialization', {}).get('mode') == 'support_rest_pose'
               for s in motion_initial_samples.values()):
            raise ValueError('Free motion uses an unbound support-rest initialization')
    elif rest_policy != 'measured_stable_rest.v1' or not stable_required or not needs_free:
        raise ValueError('Invalid measured-rest free motion policy')
    elif motion_initial_samples or rest_operations:
        from .support_rest_pose import validate_support_rest_pose, validate_rest_initialization, support_world_matrix
        import numpy as np
        record = report['support_rest_pose']
        rest_path = bound_file(record.get('path', ''), record.get('sha256'), root)
        artifacts.add(rest_path)
        candidate = read_json(rest_path)
        ground_trials = [t for t in trials if t['kind'] == 'ground_settle' and t.get('passed') is True]
        if len(ground_trials) != 1 or record.get('source_trial_id') != ground_trials[0]['id']:
            raise ValueError('Free motion has no passed source Stable trial')
        ground = ground_trials[0]
        source = bound_file(ground['trajectory_path'], ground['trajectory_sha256'], evidence_roots[ground['id']])
        bindings = {'usd': {'path': str(delivered), 'sha256': file_hash(delivered)},
                    'profile': {'path': str(expected_inputs['profile']), 'sha256': file_hash(expected_inputs['profile'])},
                    'contract': {'path': str(contract_path), 'sha256': file_hash(contract_path)},
                    'source_trajectory': {'path': str(source), 'sha256': file_hash(source)}}
        validate_support_rest_pose(candidate, contract, source_trace=list(iter_trajectory_samples(source)), bindings=bindings)
        for first in motion_initial_samples.values():
            validate_rest_initialization(first, candidate, contract)
        for operation in rest_operations:
            if not np.allclose(operation['motion_scope_plan']['root_transform'], support_world_matrix(candidate), rtol=0, atol=1e-10):
                raise ValueError('Ground-blocked operation did not use its measured Stable pose')
    if short_circuited:
        # Every recorded trial has now been independently recomputed and bound;
        # downstream placeholders cannot stand in for simulation evidence.
        return {"report": report, "artifacts": sorted(artifacts), "contract": contract,
                "metric_verification": metric_reviews}
    reload = report.get("reload_check", {})
    current = contract.get("schema_version") == 3
    if require_pass and (reload.get("passed") is not True or reload.get("free_root") is not True
                         or (not current and reload.get("external_contact_observed") is not True)):
        raise ValueError("Delivered free-base USD did not pass reload verification")
    reload_trial = reload.get("trial")
    allowed_reload = {"reload"} if stable_required else {"reload_supported"}
    if not isinstance(reload_trial, dict) or reload_trial.get("kind") not in allowed_reload:
        raise ValueError("Delivered asset needs a recorded reload trial")
    verify_trial(reload_trial, must_pass=require_pass)
    if not current:
        artifacts.add(bound_file(reload.get("external_contact_path", ""), reload.get("external_contact_sha256"), evidence_roots[reload_trial['id']]))
    if contract.get("schema_version") == 3:
        record = report.get("support_rest_pose", {})
        failed_ground = (report.get('trial_evaluation_policy') == 'all_independent_trials.v1'
                         and any(t['kind']=='ground_settle' and t.get('passed') is False for t in trials))
        if stable_required and failed_ground:
            if record != {'status':'ground_failed','verified':False}:
                raise ValueError('Failed ground cannot claim a validated support rest pose')
        elif stable_required:
            from .support_rest_pose import validate_support_rest_pose, pose_matrix, support_world_matrix
            import numpy as np
            candidate_path = bound_file(record.get("path", ""), record.get("sha256"), root)
            artifacts.add(candidate_path)
            candidate = read_json(candidate_path)
            ground_trials = [t for t in trials if t["kind"] == "ground_settle"]
            if (len(ground_trials) != 1 or ground_trials[0]["passed"] is not True
                    or record.get("source_trial_id") != ground_trials[0]["id"]
                    or record.get("reload_trial_id") != reload_trial["id"]):
                raise ValueError("Support-rest pose does not bind its successful ground and independent reload trials")
            ground = ground_trials[0]
            ground_path = bound_file(ground["trajectory_path"], ground["trajectory_sha256"], evidence_roots[ground['id']])
            ground_trace = list(iter_trajectory_samples(ground_path))
            bindings = {"usd": {"path": str(delivered), "sha256": file_hash(delivered)},
                        "profile": {"path": str(expected_inputs["profile"]), "sha256": file_hash(expected_inputs["profile"])},
                        "contract": {"path": str(contract_path), "sha256": file_hash(contract_path)},
                        "source_trajectory": {"path": str(ground_path), "sha256": file_hash(ground_path)}}
            validate_support_rest_pose(candidate, contract, source_trace=ground_trace, bindings=bindings)
            reload_path = bound_file(reload_trial["trajectory_path"], reload_trial["trajectory_sha256"], evidence_roots[reload_trial['id']])
            first = first_trajectory_sample(reload_path)
            initial = first.get("initialization", {})
            expected_matrix = pose_matrix(initial.get("expected_root_position"), initial.get("expected_root_quaternion_wxyz"))
            if (initial.get("mode") != "support_rest_pose" or initial.get("ground_alignment_required") is not False
                    or initial.get("expected_joint_positions") != candidate["joint_positions"]
                    or not np.allclose(expected_matrix, support_world_matrix(candidate), rtol=0, atol=1e-10)):
                raise ValueError("Reload did not apply the full measured support-rest pose and joint state without regrounding")
            verified = reload.get("passed") is True
            if record.get("verified") is not verified or record.get("status") != ("verified" if verified else "rejected"):
                raise ValueError("Support-rest pose verification status contradicts independent reload")
            if require_pass and not verified:
                raise ValueError("Support-rest pose did not pass independent reloading")
        elif record != {"status": "not_applicable", "verified": False}:
            raise ValueError("A supported-only object cannot claim a free support-rest pose")
    numerical = all(trial.get("passed") is True for trial in trials) and not report.get("blocked_trials")
    passed = numerical and reload.get("passed") is True
    if (report.get("numerical_passed") is not numerical or report.get("passed") is not passed
            or (report["status"] == "passed") != passed):
        raise ValueError("Physics aggregate verdict contradicts trial/reload evidence")
    return {"report": report, "artifacts": sorted(artifacts), "contract": contract,
            "metric_verification": metric_reviews}



def verify_physics_run(root, expected_config=None):
    from urdf_core.delivery_policy import POLICY
    root = Path(root).resolve()
    manifest = read_json(root / 'run_manifest.json')
    if manifest.get('config', {}).get('harness_policy') == POLICY:
        from urdf_core.asset_delivery import verify_run
        return verify_run(root, expected_config, require_pass=True)
    raise ValueError('Unsupported historical physics run')
