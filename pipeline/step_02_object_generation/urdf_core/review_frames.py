"""Select API images from an already verified motion capture."""
from __future__ import annotations

import math

POLICY = "joint_review_keyframes.v1"
GUIDE = ("Images show selected keyframes of the captured motion. Numerical evidence may include "
         "other time samples; cite only frame_ref values attached to the supplied images.")


def _positions(rows, joint):
    values = [float(row["measured_positions"][joint["name"]]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite measured joint position")
    return values


def _unexpected_motion(summary, spec, joint):
    if not summary or not spec:
        return "missing measured execution scope"
    import numpy as np

    moving = {joint["child"]} if joint["type"] != "fixed" else set()
    while True:
        children = {j["child"] for j in spec["joints"] if j["parent"] in moving}
        if children <= moving:
            break
        moving |= children
    links = {row["link"] for row in summary["links"]}
    for name in links:
        poses = np.asarray([f["actual_world_from_link"][name] for f in summary["frames"]], dtype=float)
        if poses.shape != (len(summary["frames"]), 4, 4) or not np.isfinite(poses).all():
            return "incomplete measured link poses"
        if name not in moving and np.max(np.abs(poses - poses[0])) > 1e-5:
            return "unexpected motion outside the moving branch"
    return None


def _indices(rows, joint):
    if joint["type"] == "fixed":
        return {0, len(rows) // 2, len(rows) - 1}
    q = _positions(rows, joint)
    span = max(q) - min(q)
    if span <= 1e-8:
        raise ValueError("insufficient measured travel")
    chosen = {0, len(rows) - 1, min(range(len(q)), key=lambda i: abs(q[i])),
              min(range(len(q)), key=q.__getitem__), max(range(len(q)), key=q.__getitem__)}
    # Preserve every observed reversal, including repeated excursions.
    previous = 0
    for i, (a, b) in enumerate(zip(q, q[1:])):
        sign = 1 if b - a > span * 1e-6 else -1 if a - b > span * 1e-6 else 0
        if sign and previous and sign != previous:
            chosen.add(i)
        if sign:
            previous = sign
    # Add real mid-travel images, instead of filling uniform time gaps/holds.
    while len(chosen) < min(5, len(rows)):
        candidates = []
        ordered = sorted(chosen)
        for a, b in zip(ordered, ordered[1:]):
            if b - a > 1 and abs(q[b] - q[a]) > span * 1e-6:
                middle = min(range(a + 1, b), key=lambda i: abs(q[i] - (q[a] + q[b]) / 2))
                candidates.append((abs(q[b] - q[a]), -a, middle))
        if not candidates:
            break
        chosen.add(max(candidates)[2])
    # Long rotations need intermediate states even when the endpoints look alike.
    if joint["type"] in {"revolute", "continuous"}:
        while True:
            gaps = [(a, b) for a, b in zip(sorted(chosen), sorted(chosen)[1:])
                    if abs(q[b] - q[a]) > math.pi / 2 + 1e-6]
            if not gaps:
                break
            for a, b in gaps:
                if b - a <= 1:
                    raise ValueError("rotation exceeds available angular sampling")
                chosen.add(min(range(a + 1, b), key=lambda i: abs(q[i] - (q[a] + q[b]) / 2)))
    return chosen


def select_joint_evidence(frames, joint, *, spec=None, execution_summary=None, diagnostics=None):
    """Return unchanged source records plus a deterministic API-selection receipt."""
    overlays = list((diagnostics or {}).get("frames", []))
    reason = None
    try:
        grouped = {view: sorted([f for f in frames if f["view_name"] == view],
                               key=lambda f: f["frame_index"])
                   for view in ("overview", "closeup")}
        first, second = grouped["overview"], grouped["closeup"]
        if not first or len(first) != len(second) or len(first) + len(second) != len(frames):
            raise ValueError("incomplete synchronized views")
        if ([f["frame_index"] for f in first] != [f["frame_index"] for f in second]
                or any(abs(float(a["time_seconds"]) - float(b["time_seconds"])) > 1e-8
                       for a, b in zip(first, second))):
            raise ValueError("unsynchronized views")
        if any(f.get("numeric_passed") is False or f.get("geometry_passed") is False for f in frames):
            raise ValueError("recorded native anomaly")
        if joint["type"] != "fixed":
            q1, q2 = _positions(first, joint), _positions(second, joint)
            if any(abs(a - b) > 1e-8 for a, b in zip(q1, q2)):
                raise ValueError("view measurements disagree")
            tolerance = 1e-5 if joint["type"] == "prismatic" else 1e-4
            if any(abs(float(f["positions"][joint["name"]]) - q) > tolerance
                   for f, q in zip(first, q1)):
                raise ValueError("measured motion differs from commanded travel")
        reason = _unexpected_motion(execution_summary, spec, joint)
        if reason:
            raise ValueError(reason)
        selected_indices = {first[i]["frame_index"] for i in _indices(first, joint)}
        sent = [f for f in frames if f["frame_index"] in selected_indices]
        selected_refs = {f["frame_ref"] for f in sent}
        available = [f for f in overlays if f["frame_ref"] in selected_refs]
        if overlays and len(available) < 2:
            raise ValueError("insufficient paired axis annotations")
        if available:
            available = sorted(available, key=lambda f: f["frame_index"])
            initial = available[0]
            q0 = float(initial["measured_positions_for_annotation_only"][joint["name"]])
            extreme = max(available, key=lambda f: abs(
                float(f["measured_positions_for_annotation_only"][joint["name"]]) - q0))
            if extreme["frame_ref"] == initial["frame_ref"]:
                extreme = available[-1]
            sent_overlays = [initial, extreme]
        else:
            sent_overlays = []
    except (KeyError, ValueError, TypeError, IndexError, OverflowError) as error:
        reason = str(error)
        sent, sent_overlays = list(frames), overlays
    receipt = {"policy": POLICY, "fallback_reason": reason,
               "available_frame_refs": [f["frame_ref"] for f in frames],
               "sent_frame_refs": [f["frame_ref"] for f in sent],
               "sent_overlay_refs": [f["frame_ref"] for f in sent_overlays],
               "available_image_count": len(frames) + len(overlays),
               "sent_image_count": len(sent) + len(sent_overlays)}
    return sent, sent_overlays, receipt
