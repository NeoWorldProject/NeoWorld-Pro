"""Batched read-side selection for unchanged native capture certificates."""

def review_summaries(evidence, manifest, selections):
    """Validate the complete read once, then select facts for all joints.

    No persistent cache: every invocation rechecks the complete measurement
    digest, coverage and each selected native transform. Capture source files
    stay unchanged so existing source-bound evidence remains verifiable.
    """
    from kinematics_validation.capture_execution_evidence import measurement_digest, _frames, _matrix
    if not evidence or evidence.get("status") == "unavailable":
        return {}
    if evidence.get("schema_version") != "isaac_execution_evidence.v1" or evidence.get("status") != "verified":
        raise ValueError("Invalid execution evidence declaration")
    if evidence.get("frame_state_sha256") != measurement_digest(manifest):
        raise ValueError("Execution evidence measurement digest differs")
    rows = _frames(manifest)
    if evidence.get("frame_count") != len(rows):
        raise ValueError("Execution frame coverage differs")
    captured = {row["frame_ref"]: row for row in rows}
    names = {link["link"] for link in evidence["links"]}
    if len(names) != len(evidence["links"]):
        raise ValueError("Duplicate evidence body")
    return {joint: _select(evidence, captured, names, joint, refs, _matrix)
            for joint, refs in selections.items()}


def _select(evidence, captured, names, joint_name, frame_refs, _matrix):
    selected = []
    for ref in frame_refs:
        if not isinstance(ref, str) or not ref.startswith(joint_name + "::") or ref not in captured:
            raise ValueError("Selected frame does not belong to this joint")
        frame = captured[ref]
        if frame["numeric_passed"] is not True or set(frame["link_poses"]) != names:
            raise ValueError("Invalid selected measured frame")
        for value in frame["link_poses"].values():
            _matrix(value)
        selected.append({"frame_ref": ref, "actual_joint_positions": frame["measured_positions"],
                         "actual_world_from_link": frame["link_poses"]})
    return {"scope": "Source-bound imported visual point geometry and actual native link execution; no semantic verdict.",
            "not_proven": ["Authored labels identify the intended real-world part", "The proposed axis, pivot and parent are mechanically correct", "All required functional structures/followers belong to the moving body", "Point coverage independently certifies surface topology or material identity"],
            "geometry_evidence_method": "Complete per-source visual points associated with actual imported Mesh prims under their rigid body; full bidirectional point coverage without sampling. Native link transforms are measured at captured frames.",
            "links": evidence["links"], "frames": selected}
