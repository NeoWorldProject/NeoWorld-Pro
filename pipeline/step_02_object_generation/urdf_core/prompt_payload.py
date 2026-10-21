"""Compact API evidence; native files, measurements and acceptance stay unchanged."""
from __future__ import annotations

import copy
import json
from collections import Counter

POLICY = "object_prompt_payload.v1"
GUIDE = (
    "Evidence encoding: object_prompt_payload.v1 is lossless JSON. "
    "In its data/shared entries, @ref indexes shared, @str indexes strings, "
    "@columns with @rows describes records, and @object contains literal key/value pairs. "
    "Expand these references when interpreting evidence. Names are identifiers, not proof "
    "of semantic correctness. Return the requested ordinary JSON, not this encoding."
)


def dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def encode(value):
    counts, strings = Counter(), Counter()

    def count(v):
        if isinstance(v, (dict, list)):
            serial = dump(v)
            if len(serial) >= 160:
                counts[serial] += 1
            for x in (v.values() if isinstance(v, dict) else v):
                count(x)
        elif isinstance(v, str) and len(v) >= 48:
            strings[v] += 1

    count(value)
    names = sorted(s for s, n in strings.items() if n >= 3)
    name_ids = {s: i for i, s in enumerate(names)}
    shared, ids = [], {}
    reserved = {"@ref", "@str", "@columns", "@rows", "@object"}

    def pack(v, allow_ref=True):
        if isinstance(v, str) and v in name_ids:
            return {"@str": name_ids[v]}
        if not isinstance(v, (dict, list)):
            return v
        serial = dump(v)
        if allow_ref and counts[serial] >= 2:
            if serial not in ids:
                item = pack(v, False)
                ids[serial] = len(shared)
                shared.append(item)
            return {"@ref": ids[serial]}
        if isinstance(v, dict):
            if reserved.intersection(v):
                return {"@object": [[k, pack(x)] for k, x in v.items()]}
            return {k: pack(x) for k, x in v.items()}
        if (len(v) >= 3 and all(isinstance(x, dict) and x for x in v)
                and all(list(x) == list(v[0]) for x in v)):
            keys = list(v[0])
            return {"@columns": keys, "@rows": [[pack(x[k]) for k in keys] for x in v]}
        return [pack(x) for x in v]

    data = pack(value)
    result = {"encoding": POLICY, "strings": names, "shared": shared, "data": data}
    return result if len(dump(result)) < len(dump(value)) else value


def decode(value):
    if not isinstance(value, dict) or value.get("encoding") != POLICY:
        return value
    def expand(v):
        if isinstance(v, list):
            return [expand(x) for x in v]
        if not isinstance(v, dict):
            return v
        if set(v) == {"@ref"}:
            return expand(value["shared"][v["@ref"]])
        if set(v) == {"@str"}:
            return value["strings"][v["@str"]]
        if set(v) == {"@object"}:
            return {k: expand(x) for k, x in v["@object"]}
        if set(v) == {"@columns", "@rows"}:
            return [dict(zip(v["@columns"], map(expand, row))) for row in v["@rows"]]
        return {k: expand(x) for k, x in v.items()}
    return expand(value["data"])


def split_json(text):
    """Only whole JSON or a labelled JSON block, never embedded code/examples."""
    starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
    for start in starts:
        part = text[start:].lstrip()
        if not part.startswith(("{", "[")):
            continue
        try:
            value = json.loads(part)
        except ValueError:
            continue
        if isinstance(value, (dict, list)):
            return text[:start], value
    return None


def prepare_content(content):
    from .api_autonomy import assert_autonomous_context
    assert_autonomous_context(content)
    result, encoded = [], False
    for item in content:
        row = dict(item)
        if row.get("type") == "input_text":
            split = split_json(row["text"])
            if split:
                prefix, value = split
                if isinstance(value, dict) and value.get("encoding") == POLICY:
                    packed = value
                else:
                    packed = encode(value)
                encoded |= isinstance(packed, dict) and packed.get("encoding") == POLICY
                row["text"] = prefix + dump(packed)
        result.append(row)
    if encoded and not any(x.get("text") == GUIDE for x in result):
        result.insert(0, {"type": "input_text", "text": GUIDE})
    return result


def project_execution(summary, spec, joint):
    """Keep global identity/shape context, detailed observed movers and neighbours."""
    import numpy as np

    if not summary or not summary.get("frames") or not summary.get("links"):
        return summary
    try:
        links = {x["link"]: x for x in summary["links"]}
        if len(links) != len(summary["links"]):
            raise ValueError("duplicate link")
        frames = summary["frames"]
        matrices = {}
        bounds = {}
        moving = set()
        for name, link in links.items():
            poses = np.asarray([f["actual_world_from_link"][name] for f in frames], dtype=float)
            box = np.asarray(link["bound_link_local_aabb_m"], dtype=float)
            if (poses.shape != (len(frames), 4, 4) or box.shape != (2, 3)
                    or not np.isfinite(poses).all() or not np.isfinite(box).all()
                    or np.any(box[0] > box[1])):
                raise ValueError("incomplete spatial evidence")
            corners = np.array([[box[i, 0], box[j, 1], box[k, 2], 1.]
                                for i in (0, 1) for j in (0, 1) for k in (0, 1)])
            world = np.einsum("fij,kj->fki", poses, corners)[..., :3]
            bounds[name] = (world.min(axis=(0, 1)), world.max(axis=(0, 1)))
            matrices[name] = poses
            # This threshold selects extra detail; it never decides physical pass.
            if np.max(np.abs(poses - poses[0])) > 1e-9:
                moving.add(name)
        detailed = {joint["parent"], joint["child"]} | moving
        if not detailed <= links.keys():
            raise ValueError("missing related body")
        descendants = {joint["child"]}
        while True:
            new = {j["child"] for j in spec["joints"] if j["parent"] in descendants}
            if new <= descendants:
                break
            descendants |= new
        detailed |= descendants
        if not detailed <= links.keys():
            raise ValueError("missing descendant")
        seeds = list(descendants | moving)
        for name, (lo, hi) in bounds.items():
            if any(np.all(hi + .01 >= bounds[s][0]) and np.all(bounds[s][1] + .01 >= lo)
                   for s in seeds):
                detailed.add(name)
        projected = []
        for name, link in links.items():
            row = {k: copy.deepcopy(v) for k, v in link.items()
                   if k not in {"visual_meshes", "body_path"}}
            meshes = link["visual_meshes"]
            row["mesh_count"] = len(meshes)
            row["mesh_binding_max_distance_m"] = max((m["max_distance_m"] for m in meshes), default=0.)
            if name in detailed:
                row["visual_meshes"] = [{k: copy.deepcopy(v) for k, v in m.items()
                    if k not in {"source_mesh", "imported_prim_paths", "visual_id"}} for m in meshes]
            row["detail_scope"] = "detailed" if name in detailed else "global_summary"
            projected.append(row)
        return {**summary, "links": projected,
                "api_projection": {"policy": POLICY, "detail_links": sorted(detailed),
                    "observed_moving_links": sorted(moving), "all_frame_transforms_retained": True,
                    "scope_note": "Global labels/bounds and all captured transforms remain. "
                    "Detailed mesh rows cover the target chain, observed movers and spatial neighbours. "
                    "Source paths and binding identities are checked locally; these facts are not a semantic verdict."}}
    except (ValueError, KeyError, TypeError, IndexError):
        # Ambiguous scope keeps the complete evidence instead of omitting bodies.
        return summary
