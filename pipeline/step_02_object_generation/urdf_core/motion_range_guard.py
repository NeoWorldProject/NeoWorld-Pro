"""Preserve evaluated joint travel across autonomous generation repairs."""
from __future__ import annotations
import math

POLICY = "preserve_evaluated_motion_travel.v1"


def requirements(trials, reference=None):
    result = {}
    specs = [t.get("result", {}).get("spec") for t in trials.values()
             if t.get("status") in {"passed", "failed"} and isinstance(t.get("result"), dict)]
    specs.append(reference)
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        for joint in spec.get("joints", []):
            kind = joint.get("type")
            if kind not in {"continuous", "revolute", "prismatic"}:
                continue
            name = joint["name"]
            row = {"type": kind}
            if kind != "continuous":
                span = float(joint["limit"]["upper"]) - float(joint["limit"]["lower"])
                if not math.isfinite(span) or span <= 0:
                    raise ValueError("Invalid evaluated joint range: " + name)
                row["minimum_travel"] = span
            old = result.get(name)
            if old is None or (old["type"] != "continuous" and (
                    kind == "continuous" or row.get("minimum_travel", 0) > old.get("minimum_travel", 0))):
                result[name] = row
    return result


def validate(spec, required):
    joints = {j["name"]: j for j in spec["joints"]}
    errors = []
    for name, old in required.items():
        j = joints.get(name)
        if not j or j.get("type") == "fixed":
            errors.append(name + ": required movement missing")
            continue
        if old["type"] == "continuous":
            if j["type"] != "continuous":
                errors.append(name + ": unbounded rotation became bounded")
        elif j["type"] != "continuous":
            span = float(j["limit"]["upper"]) - float(j["limit"]["lower"])
            tolerance = max(1e-10, old["minimum_travel"] * 1e-7)
            if span + tolerance < old["minimum_travel"]:
                errors.append(f"{name}: travel {span:g} is below previously evaluated {old['minimum_travel']:g}")
    if errors:
        raise ValueError("Motion travel preservation violation: " + "; ".join(errors))
    return spec
