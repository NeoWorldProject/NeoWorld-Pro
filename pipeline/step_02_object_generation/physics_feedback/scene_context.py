"""Image-bound, pre-test applicability of standalone ground stability."""
from __future__ import annotations

import json
import math
from pathlib import Path

from urdf_core.run_state import file_hash, fingerprint, write_json

RULE_VERSION = "scene_support_applicability.v1"
MODES = {"horizontal_plane": True, "wall_mounted": False, "hanging": False,
         "inclined_stand": False, "inserted": False, "clamped": False,
         "leaning": False, "external_support": False, "unclear": None}
PROMPT = """Classify whether THIS target in THIS original scene needs standalone
ground-settle and free-base tests. Use the original RGB, target crop, parent and
support relations together. The actual placement in the current scene controls:
horizontal_plane means independently supported by a horizontal floor, tabletop,
counter or shelf => stable_required=true. Wall mounting, hanging, an inclined
stand, hole/slot insertion, clamping, leaning or other necessary external support
=> false. A tube inserted in a rack is false; the rack on a table is true. A phone
lying flat is true; the same phone resting in an inclined holder is false.
Do not infer the answer from what this object category COULD do elsewhere. Do not
use past simulation results, failures, or acceptance labels to decide applicability.
If the evidence is insufficient or conflicting, use unclear and null; never silently
exempt. Return exactly: {"stable_required":true|false|null,
"support_mode":"horizontal_plane|wall_mounted|hanging|inclined_stand|inserted|clamped|leaning|external_support|unclear",
"reason":"scene-specific rationale", "visual_evidence":"visible contact/support cues",
"confidence":"clear|unclear"}. This classification is frozen before physics tests.
False skips only ground/free tests as not_applicable, keeps applicable fixed-base
joint tests and delivery reload checks, and does not imply scene stability passed.
"""


def bound(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def build_context(layout_path, image_path, target_id, crop_path):
    value = {"schema_version": "object_scene_context.v1", "layout": bound(layout_path),
             "scene_image": bound(image_path), "target_crop": bound(crop_path),
             "target_id": target_id}
    return value


def load_context(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get("schema_version") != "object_scene_context.v1":
        raise ValueError("Unsupported object scene context")
    for key in ("layout", "scene_image", "target_crop"):
        entry = value[key]
        if not Path(entry["path"]).is_absolute() or file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("Scene context evidence changed: " + key)
    layout = json.loads(Path(value["layout"]["path"]).read_text())
    objects = layout["objects_data"]
    matches = [o for o in objects if o["object_id"] == value["target_id"]]
    if len(matches) != 1:
        raise ValueError("Scene context requires one unique target")
    target = matches[0]
    from PIL import Image, ImageChops
    bbox = target.get("2d_bbox_normalized")
    if (not isinstance(bbox, list) or len(bbox) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox)
            or not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1)):
        raise ValueError("Scene context needs a valid target image footprint")
    with Image.open(value["scene_image"]["path"]) as original, Image.open(value["target_crop"]["path"]) as crop:
        width, height = original.size
        pixels = [math.floor(bbox[0]*width), math.floor(bbox[1]*height),
                  math.ceil(bbox[2]*width), math.ceil(bbox[3]*height)]
        expected = original.convert("RGB").crop(pixels)
        if crop.size != expected.size or ImageChops.difference(crop.convert("RGB"), expected).getbbox() is not None:
            raise ValueError("Target crop does not match its original scene and footprint")
    parents = {}
    for name, node in layout["scene_tree"].items():
        for child in node["children"]:
            if child in parents:
                raise ValueError("Scene context target has multiple parents")
            parents[child] = name
    by_name = {o["name"]: o for o in objects}
    ancestry, seen, name = [], {target["name"]}, target["name"]
    while name in parents:
        name = parents[name]
        if name in seen or name not in by_name:
            raise ValueError("Invalid scene support ancestry")
        seen.add(name)
        ancestry.append(by_name[name])
    value.update(target=target, parent=ancestry[0] if ancestry else None,
                 support_ancestry=ancestry, context_file=bound(path), rule_version=RULE_VERSION)
    value["fingerprint"] = fingerprint(value)
    return value


def context_content(context, api):
    if load_context(context["context_file"]["path"]) != context:
        raise ValueError("Scene context changed after applicability was frozen")
    return [api.make_text_input("Original scene RGB followed by target crop; use their joint context."),
            api.make_image_input(Path(context["scene_image"]["path"])),
            api.make_image_input(Path(context["target_crop"]["path"])),
            api.make_text_input(json.dumps({k: context[k] for k in
                ("rule_version", "target", "parent", "support_ancestry")}, ensure_ascii=False))]


def validate_decision(value):
    keys = {"stable_required", "support_mode", "reason", "visual_evidence", "confidence"}
    if not isinstance(value, dict) or set(value) != keys or value["support_mode"] not in MODES:
        raise ValueError("Malformed support applicability response")
    expected = MODES[value["support_mode"]]
    if value["stable_required"] is not expected:
        raise ValueError("Support mode and stability applicability disagree")
    if value["confidence"] not in {"clear", "unclear"} or any(
            not isinstance(value[k], str) or not value[k].strip() for k in ("reason", "visual_evidence")):
        raise ValueError("Support classification needs explicit evidence")
    return value


def classify(context, api, ask, *, repair_budget=None, max_reviews=2, modern=False):
    content = context_content(context, api) + [api.make_text_input(PROMPT)]
    attempts = []
    modern = modern or repair_budget is not None
    if repair_budget is not None:
        max_reviews = repair_budget.limit + 1
    for index in range(max_reviews):
        stage = ('scene_support' if index == 0 else f'scene_support_refine_{index}' if modern else 'scene_support_focused_review')
        if index and repair_budget is not None:
            repair_budget.reserve(stage, 'support_classification', {'context': context['fingerprint'], 'index': index})
        reply = ask(stage, content)
        if index and repair_budget is not None:
            repair_budget.complete(stage, status='responded')
        try:
            value = validate_decision(api.extract_json(reply))
            attempts.append({"decision": value})
            if value["confidence"] == "clear" and value["stable_required"] is not None:
                return {**({'recovery_policy': 'physics_budget.v2'} if modern else {}), "status": "classified", "rule_version": RULE_VERSION,
                        "context_fingerprint": context["fingerprint"], "decision": value, "attempts": attempts}
        except (ValueError, TypeError, KeyError) as exc:
            attempts.append({"error": str(exc)})
        content += [api.make_text_input(("Use the next reserved refinement to reinspect actual contact and parent support. " if modern else "One focused review is permitted. Reinspect actual contact and parent support. ") +
                    "If still uncertain, return unclear/null. Previous response: " + reply)]
    return {**({'recovery_policy': 'physics_budget.v2'} if modern else {}), "status": "pending", "rule_version": RULE_VERSION,
            "context_fingerprint": context["fingerprint"], "decision": None, "attempts": attempts}


def verify_classification(path, context):
    """Bind a separate historical classification to actual recorded API replies."""
    import generate_object_urdf as api
    from .loop import _audit_content
    from urdf_core.prompt_payload import prepare_content
    path = Path(path)
    saved = json.loads(path.read_text())
    directory = path.parent / "model_calls"
    def replay(stage, content):
        requests = sorted(directory.glob("*_" + stage + ".request.json"))
        completed = [(p, p.with_name(p.name.removesuffix(".request.json") + ".response.txt")) for p in requests]
        completed = [(p,r) for p,r in completed if r.is_file()]
        if len(completed) != 1:
            raise ValueError("Classification needs exactly one completed reply per decision stage")
        request, response = completed[0]
        record = json.loads(request.read_text())
        if record["content"] != _audit_content(prepare_content(content)) or record["logical_stage"] != stage:
            raise ValueError("Classification request differs from current scene evidence/rule")
        return response.read_text()
    modern = saved.get('recovery_policy') == 'physics_budget.v2'
    reviews = len(saved.get('attempts', [])) if modern else 2
    if modern and not 1 <= reviews <= 9:
        raise ValueError('Invalid bounded support classification history')
    replayed = classify(context, api, replay, max_reviews=reviews, modern=modern)
    if any(saved.get(key) != value for key, value in replayed.items()):
        raise ValueError("Frozen classification differs from its recorded API decisions")
    return saved
