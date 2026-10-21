"""Reduce complete, measured scene traces without trusting the simulator's verdict."""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation
from .protocol import DEFAULT_CONFIG


DEFAULT_PHYSICS = {key: DEFAULT_CONFIG[key] for key in (
    "certified_early_exit", "compact_evidence", "shared_shader_cache",
    "compress_trajectories", "async_evidence", "physics_dt_s", "gravity_m_s2",
    "search_duration_s", "final_duration_s", "settle_window_s",
    "penetration_tolerance_m", "geometry_tolerance_m", "position_tolerance_min_m",
    "position_tolerance_fraction", "orientation_tolerance_deg", "drift_min_m",
    "drift_fraction", "tilt_limit_deg", "worker_timeout_s", "video_fps",
    "render_width", "render_height",
)}


def physics_settings(config=None):
    values = dict(DEFAULT_PHYSICS)
    config = config or {}
    values.update({k: v for k, v in config.items() if k in values})
    values.update(config.get("physics", {}))
    for key, value in values.items():
        if key in {"certified_early_exit", "compact_evidence", "shared_shader_cache", "compress_trajectories", "async_evidence"} and type(value) is int and value in (0, 1):
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid physical setting {key}: {value}")
    if values["geometry_tolerance_m"] > values["penetration_tolerance_m"]:
        raise ValueError("geometry precision cannot exceed penetration tolerance")
    return values


def _pose(value):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
            or np.linalg.det(pose[:3, :3]) < .999):
        raise ValueError("Missing/nonfinite/nonrigid body transform")
    return pose


def _contact_owner(path, owners):
    ancestors = [known for known in owners if path == known or path.startswith(known + "/")]
    return owners[max(ancestors, key=len)] if ancestors else None


def _contact_ids(step, path_to_object):
    pairs = set()
    for pair in step["contacts"].get("pairs", []):
        ids = [_contact_owner(pair.get(f"body{index}", ""), path_to_object)
               or _contact_owner(pair.get(f"collider{index}", ""), path_to_object)
               for index in (0, 1)]
        if all(ids) and ids[0] != ids[1]:
            pairs.add(frozenset(ids))
    return pairs


def _anchored_objects(objects, contacts, attachments):
    """Resolve support toward a real environment; touching floating clusters fail."""
    anchored = {oid for oid, obj in objects.items() if obj.get("is_environment_support")}
    outgoing = {}
    for oid, obj in objects.items():
        relation = obj.get("relation", obj.get("support_relation", "root"))
        parent = obj.get("parent_id")
        if relation == "attached":
            evidence = attachments.get(oid, {})
            outgoing[oid] = {parent} if evidence.get("observed") is True and parent else set()
        else:
            neighbours = {other for pair in contacts if oid in pair for other in pair if other != oid}
            outgoing[oid] = neighbours & {parent} if parent else neighbours
    changed = True
    while changed:
        next_set = anchored | {oid for oid, targets in outgoing.items() if targets & anchored}
        changed = next_set != anchored
        anchored = next_set
    return anchored


def reduce_native_trace(trace, manifest, *, duration_s, config=None, proxy_fidelity=None):
    """Every physics step must contain native body poses and contacts.

    Scales never change acceptance thresholds: all diagonals are frozen by scene
    preparation. Attachment constraint residuals do not excuse solid penetration.
    """
    cfg = physics_settings(config)
    report = {"passed": False, "status": "environment_error", "complete": False,
              "violation": 1.0, "physical_residual": 1.0, "errors": [],
              "issues": [], "objects": {}, "thresholds": cfg, "unresolved_geometry": []}
    errors, issues, residuals = report["errors"], report["issues"], []
    try:
        objects = manifest["objects"]
        if not objects or len(trace) < 2:
            raise ValueError("Scene/trace is empty")
        times = np.asarray([step["time_s"] for step in trace], dtype=float)
        dt = float(cfg["physics_dt_s"])
        if (not np.isfinite(times).all() or abs(times[0]) > 1e-8
                or np.any(np.abs(np.diff(times) - dt) > max(1e-8, dt * .001))
                or times[-1] < duration_s - 1e-8):
            raise ValueError("Incomplete physics-step trace or duration")
        tail = times >= times[-1] - cfg["settle_window_s"] - 1e-8
        if times[tail][-1] - times[tail][0] < cfg["settle_window_s"] - dt * 1.01:
            raise ValueError("Insufficient settling observation window")
        paths = {path: oid for oid, obj in objects.items() for path in obj["body_paths"].values()}
        link_paths = {path: (oid, link) for oid, obj in objects.items() for link, path in obj["body_paths"].items()}
        # Object-level self collision belongs to stage two, not scene layout.
        ignored = {frozenset(((oid, a), (oid, b))) for oid, obj in objects.items()
                   for a in obj["body_paths"] for b in obj["body_paths"] if a != b}
        native_depth, worst_native = 0., None
        required_keys = {f"{oid}::{link}" for oid, obj in objects.items() for link in obj["body_paths"]}
        for index, step in enumerate(trace):
            if set(step.get("body_poses", {})) != required_keys:
                raise ValueError(f"Incomplete body observation at step {index}")
            for value in step["body_poses"].values():
                _pose(value)
            if step.get("contacts", {}).get("observed") is not True:
                raise ValueError(f"Missing native contact observation at step {index}")
            for pair in step["contacts"].get("pairs", []) + step["contacts"].get("events", []):
                links = [_contact_owner(pair.get(f"body{i}", ""), link_paths)
                         or _contact_owner(pair.get(f"collider{i}", ""), link_paths) for i in (0, 1)]
                if not all(links):
                    raise ValueError(f"Unbound native contact body/collider at step {index}")
                if all(links) and (links[0] == links[1] or frozenset(links) in ignored):
                    continue
                for point in pair.get("points", []):
                    separation = float(point["separation_m"])
                    if not math.isfinite(separation):
                        raise ValueError(f"Nonfinite native separation at step {index}")
                    if -separation > native_depth:
                        native_depth = -separation
                        worst_native = {"step": index, "body0": pair.get("body0"), "body1": pair.get("body1"),
                                        "object_ids": [link[0] if link else None for link in links]}
        report["max_native_penetration_m"] = native_depth
        residuals.append(native_depth / cfg["penetration_tolerance_m"])
        if native_depth > cfg["penetration_tolerance_m"] + 1e-12:
            issues.append({"kind": "native_contact_penetration", "depth_m": native_depth, **worst_native})
        contacts = [_contact_ids(step, paths) for step in trace]
        anchored = [_anchored_objects(objects, pairs, step.get("attachments", {}))
                    for step, pairs in zip(trace, contacts)]
        for oid, obj in objects.items():
            if obj.get("is_environment_support"):
                continue
            root_link = obj.get("root_link") or next(iter(obj["body_paths"]))
            poses = np.asarray([step["body_poses"][f"{oid}::{root_link}"] for step in trace])
            diagonal = float(obj.get("initial_diagonal_m", 0))
            if not math.isfinite(diagonal) or diagonal <= 0:
                raise ValueError(f"Missing initial diagonal for {oid}")
            position_limit = max(cfg["position_tolerance_min_m"], cfg["position_tolerance_fraction"] * diagonal)
            drift_limit = max(cfg["drift_min_m"], cfg["drift_fraction"] * diagonal)
            linear_velocity_limit = position_limit / cfg["settle_window_s"]
            angular_velocity_limit = math.radians(cfg["orientation_tolerance_deg"]) / cfg["settle_window_s"]
            positions = poses[:, :3, 3]
            rotation = Rotation.from_matrix(poses[:, :3, :3])
            tail_rotation = rotation[np.flatnonzero(tail)]
            settling = float(np.linalg.norm(positions[tail] - positions[tail][0], axis=1).max())
            angular = float(np.degrees((tail_rotation[0].inv() * tail_rotation).magnitude()).max())
            drift = float(np.linalg.norm(positions - positions[0], axis=1).max())
            up = rotation.apply([0, 0, 1])
            reference_up = np.asarray(obj.get("orientation_reference_world", poses[0, :3, :3]))[:, 2]
            tilt = float(np.degrees(np.arccos(np.clip(up @ reference_up, -1, 1))).max())
            reference_rotation = Rotation.from_matrix(np.asarray(obj.get("orientation_reference_world", poses[0, :3, :3])))
            major_angle = float(np.degrees((reference_rotation.inv() * rotation).magnitude()).max())
            ratios = [settling / position_limit, angular / cfg["orientation_tolerance_deg"],
                      tilt / cfg["tilt_limit_deg"], major_angle / cfg["tilt_limit_deg"]]
            if manifest.get("state_kind") == "settled":
                # The already-settled deliverable must remain near its reload state.
                ratios.append(drift / drift_limit)
            link_settling = {}
            for link in obj["body_paths"]:
                observed = np.asarray([trace[index]["body_poses"][f"{oid}::{link}"]
                                       for index in np.flatnonzero(tail)])
                translation = float(np.linalg.norm(observed[:, :3, 3] - observed[0, :3, 3], axis=1).max())
                orientation = Rotation.from_matrix(observed[:, :3, :3])
                angle = float(np.degrees((orientation[0].inv() * orientation).magnitude()).max())
                key = f"{oid}::{link}"
                linear = np.asarray([trace[index]["body_velocities"][key]["linear_m_s"]
                                     for index in np.flatnonzero(tail)], dtype=float)
                angular_velocity = np.asarray([trace[index]["body_velocities"][key]["angular_rad_s"]
                                               for index in np.flatnonzero(tail)], dtype=float)
                if (linear.shape != (int(tail.sum()), 3) or angular_velocity.shape != linear.shape
                        or not np.isfinite(linear).all() or not np.isfinite(angular_velocity).all()):
                    raise ValueError("Missing/nonfinite native rigid-link velocity: " + key)
                peak_linear = float(np.linalg.norm(linear, axis=1).max())
                peak_angular = float(np.linalg.norm(angular_velocity, axis=1).max())
                ratios.extend([translation / position_limit, angle / cfg["orientation_tolerance_deg"]])
                link_settling[link] = {"translation_m": translation, "rotation_deg": angle,
                                      "peak_linear_velocity_m_s": peak_linear,
                                      "peak_angular_velocity_rad_s": peak_angular}
            relation, parent = obj.get("relation", obj.get("support_relation", "root")), obj.get("parent_id")
            supported = []
            if relation == "attached":
                for step in trace:
                    attachment = step.get("attachments", {}).get(oid)
                    if not attachment or attachment.get("observed") is not True:
                        raise ValueError(f"Missing attachment observation for {oid}")
                    ratios.extend([float(attachment["position_error_m"]) / position_limit,
                                   float(attachment["rotation_error_deg"]) / cfg["orientation_tolerance_deg"]])
                supported = [oid in anchored[index] for index in np.flatnonzero(tail)]
            else:
                for index in np.flatnonzero(tail):
                    supported.append(oid in anchored[index])
            if not all(math.isfinite(value) for value in ratios):
                raise ValueError(f"Nonfinite physical residual for {oid}")
            support_fraction = sum(supported) / len(supported)
            ratios.append(0.0 if support_fraction == 1.0 else 1.0 + (1.0 - support_fraction))
            residuals.extend(ratios)
            detail = {"settling_translation_m": settling, "settling_rotation_deg": angular,
                      "stable_required": obj.get("stable_required", True),
                      "independent_ground_test": "upstream_object_stage" if obj.get("stable_required", True) else "not_applicable",
                      "scene_pose_stability_required": True,
                      "max_root_displacement_m": drift, "max_tilt_deg": tilt,
                      "max_major_orientation_deg": major_angle,
                      "tail_support_fraction": support_fraction, "relation": relation,
                      "link_settling": link_settling,
                      "initial_diagonal_m": diagonal, "position_limit_m": position_limit,
                      "linear_velocity_limit_m_s": linear_velocity_limit,
                      "angular_velocity_limit_rad_s": angular_velocity_limit,
                      "native_velocity_policy": "diagnostic_only",
                      "native_velocity_acceptance_gate": False,
                      "drift_limit_m": drift_limit, "passed": max(ratios) <= 1 + 1e-6}
            report["objects"][oid] = detail
            if not detail["passed"]:
                issues.append({"kind": "unstable_or_unsupported", "object_id": oid})
        residual = float(max(residuals, default=0))
        if issues and residual <= 1:
            residual = 2.0
        report.update(duration_s=float(times[-1]), physics_steps=len(trace),
                      complete=True, passed=not issues,
                      status="physics_failed" if issues else "passed", known_infeasible=bool(issues))
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        errors.append(str(exc))
    return report



def evaluate_scene_trace(trace, manifest, *, duration_s, config=None, proxy_fidelity=None, native_articulations=None):
    from .physics import evaluate
    return evaluate(trace, manifest, duration_s=duration_s, config=physics_settings(config),
                    native_articulations=native_articulations)
