"""Explicit input-prefix caching; responses and judgments remain independent."""
from __future__ import annotations

import hashlib
import json
import re

POLICY = "input_prefix_cache.v1"


def cache_key(model, content):
    prefix = json.dumps(content, ensure_ascii=False, allow_nan=False, separators=(",", ":"))[:4096]
    return "nw-" + hashlib.sha256((model + "\n" + prefix).encode()).hexdigest()[:40]


def supported(model):
    match = re.match(r"^gpt-(\d+)\.(\d+)(?:-|$)", model)
    return bool(match and tuple(map(int, match.groups())) >= (5, 6))


def object_boundaries(content):
    """Locate stable evidence before the changing review or repair suffix."""
    dynamic = (
        "Review this joint only:", "Current kinematics:", "Critique:",
        "Correct this initial physical contract error", "Previous complete proposal:",
        "Prior experiments and reviewed next strategy", "Prior refinement outcomes",
        "Actual captured Isaac frame:", "frame_ref:", "Candidate 1;",
    )
    stop = next((i for i, item in enumerate(content)
                 if item.get("text", "").startswith(dynamic)), len(content))
    physics_repair = any("name: isaac-physics-refine" in x.get("text", "") for x in content)
    if physics_repair:
        stop = min(stop, next((i for i, x in enumerate(content)
                              if x.get("text", "").lstrip().startswith(("{", "["))), len(content)))
    ends = []
    for i, item in enumerate(content[:stop]):
        text = item.get("text", "")
        if text.startswith(("Measured geometry inventory:", "Actual evaluated scene inventory:",
                            "Complete proposed kinematic protocol:")):
            ends.append(i)
        elif text == "Unmodified target crop:" and i + 1 < stop:
            if content[i + 1].get("type") == "input_image":
                ends.append(i + 1)
        elif physics_repair and "name: isaac-physics-refine" in text:
            ends.append(i)
    if stop:
        ends.append(stop - 1)
    return sorted(set(ends))[-4:]


def prepare(model, content, *, boundaries=None):
    """Return wire blocks, SDK options and a credential-free cache receipt."""
    wire = [dict(x) for x in content]
    if any("prompt_cache_breakpoint" in x for x in wire):
        raise ValueError("Cache breakpoints are owned by the provider adapter")
    # Keep older model behavior and the existing accounting key.
    key = cache_key(model, content)
    options = {"prompt_cache_key": key}
    receipt = {"policy": POLICY, "mode": "legacy", "prompt_cache_key": key, "boundaries": []}
    if not supported(model):
        return wire, options, receipt
    ends = object_boundaries(content) if boundaries is None else sorted(set(boundaries))
    if len(ends) > 4 or any(type(i) is not int or i < 0 or i >= len(wire) for i in ends):
        raise ValueError("Expected at most four valid cache boundaries")
    for i in ends:
        if wire[i].get("type") not in {"input_text", "input_image"}:
            raise ValueError("Unsupported cache content block")
        wire[i]["prompt_cache_breakpoint"] = {"mode": "explicit"}
    options["extra_body"] = {"prompt_cache_options": {"mode": "explicit", "ttl": "30m"}}
    receipt.update(mode="explicit", ttl="30m", boundaries=ends)
    return wire, options, receipt
