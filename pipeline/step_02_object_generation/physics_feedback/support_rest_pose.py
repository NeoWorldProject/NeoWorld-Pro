"""Measured, immutable support-rest candidates; no asset-specific pose fixes."""
from __future__ import annotations

import copy
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .protocol import json_hash, validate_current_contract


def pose_matrix(position, quaternion_wxyz):
    xyz = np.asarray(position, dtype=float)
    q = np.asarray(quaternion_wxyz, dtype=float)
    if xyz.shape != (3,) or q.shape != (4,) or not np.isfinite(xyz).all() or not np.isfinite(q).all():
        raise ValueError("Invalid support-rest root pose")
    if abs(float(np.linalg.norm(q)) - 1) > 1e-5:
        raise ValueError("Support-rest quaternion is not normalized")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = xyz
    return result


def joint_initial_tolerance(joint, contract):
    return (float(contract["config"]["position_tolerance_m"]) if joint["type"] == "prismatic"
            else math.radians(float(contract['ground_initial_rotary_tolerance_deg']
                if contract.get('acceptance_policy') == 'object_functional_physics.v11'
                else contract["config"]["orientation_tolerance_deg"])))


def _rigid_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), rtol=0, atol=1e-8)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, rtol=0, atol=1e-8)):
        raise ValueError("Support-rest root transform must be rigid and finite")
    return matrix


def support_world_matrix(candidate):
    """Compose the measured world pose; v1's support frame is the identity."""
    if not isinstance(candidate, dict) or candidate.get("schema_version") not in {"support_rest_pose.v1", "support_rest_pose.v2"}:
        raise ValueError("Invalid support-rest pose schema")
    return _rigid_matrix(candidate.get("T_world_support")) @ _rigid_matrix(candidate.get("T_support_root"))


def validate_rest_initialization(sample, candidate, contract):
    """Require a free trial to restore the measured state, without regrounding."""
    validate_support_rest_pose(candidate, contract)
    initial = sample.get('initialization', {})
    matrix = pose_matrix(initial.get('expected_root_position'), initial.get('expected_root_quaternion_wxyz'))
    if (initial.get('mode') != 'support_rest_pose' or initial.get('ground_alignment_required') is not False
            or initial.get('expected_joint_positions') != candidate['joint_positions']
            or not np.allclose(matrix, support_world_matrix(candidate), rtol=0, atol=1e-10)):
        raise ValueError('Free motion did not restore its measured Stable pose and joint state')


def make_support_rest_pose(trace, contract, bindings):
    validate_current_contract(contract)
    if contract.get("stable_required") is not True or not trace:
        raise ValueError("Support-rest pose requires a measured stable-required ground trial")
    sample = trace[-1]
    names = set(contract["joints"])
    measured = sample.get("joint_positions", {})
    if set(measured) != names:
        raise ValueError("Support-rest pose lacks the measured joint state")
    q_initial = {name: joint["initial_position"] for name, joint in contract["joints"].items()}
    errors = {name: float(measured[name]) - q_initial[name] for name in names}
    for name, error in errors.items():
        if not math.isfinite(error) or abs(error) > joint_initial_tolerance(contract["joints"][name], contract) + 1e-8:
            raise ValueError("Measured support-rest joint left its frozen initial tolerance: " + name)
    required = {"usd", "profile", "contract", "source_trajectory"}
    if set(bindings) != required or any(not isinstance(v, dict) or set(v) != {"path", "sha256"}
                                       or not isinstance(v["path"], str) or not v["path"]
                                       or not isinstance(v["sha256"], str) or len(v["sha256"]) != 64
                                       for v in bindings.values()):
        raise ValueError("Support-rest pose requires exact USD/profile/contract/trajectory bindings")
    world_support = np.eye(4)
    support_root = pose_matrix(sample["root_position"], sample["root_quaternion_wxyz"])
    result = {"schema_version": "support_rest_pose.v1",
              "candidate_kind": "measured_ground_settle",
              "coordinate_frame": "world_ground_z0",
              "T_world_support": world_support.tolist(), "T_support_root": support_root.tolist(),
              "joint_positions": {name: float(measured[name]) for name in sorted(names)},
              "frozen_q_initial": q_initial, "q_initial_error": errors,
              "source_time_s": float(sample["time_s"]), "bindings": copy.deepcopy(bindings),
              "contract_content_sha256": json_hash(contract),
              "application_policy": "restore_full_pose_and_measured_q_without_regrounding; zero_velocity_and_active_effort"}
    json.dumps(result, allow_nan=False)
    return result


def validate_support_rest_pose(candidate, contract, *, source_trace=None, bindings=None):
    validate_current_contract(contract)
    schema = "support_rest_pose.v1"
    if not isinstance(candidate, dict) or candidate.get("schema_version") != schema:
        raise ValueError("Invalid support-rest pose schema")
    if bindings is not None and candidate.get("bindings") != bindings:
        raise ValueError("Support-rest pose differs from the required evidence bindings")
    if source_trace is not None:
        expected = make_support_rest_pose(source_trace, contract, bindings if bindings is not None else candidate.get("bindings", {}))
        if json_hash(candidate) != json_hash(expected):
            raise ValueError("Support-rest pose differs from its bound measured ground state")
        return copy.deepcopy(candidate)
    matrix = support_world_matrix(candidate)
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    sample = {"root_position": matrix[:3, 3], "root_quaternion_wxyz": q[[3, 0, 1, 2]],
              "joint_positions": candidate.get("joint_positions", {}), "time_s": candidate.get("source_time_s")}
    expected = make_support_rest_pose([sample], contract, candidate.get("bindings", {}))
    if schema == "support_rest_pose.v1":
        # Preserve the historical validator's matrix round-trip semantics.
        expected["T_support_root"] = candidate["T_support_root"]
    else:
        # Only the rotation round-trip can introduce harmless floating point
        # differences. Keep the reconstructed frame/translation checks exact.
        for row in range(3):
            expected["T_support_root"][row][:3] = candidate["T_support_root"][row][:3]
    if json_hash(candidate) != json_hash(expected):
        raise ValueError("Support-rest pose does not bind the full frozen policy/state")
    return copy.deepcopy(candidate)
