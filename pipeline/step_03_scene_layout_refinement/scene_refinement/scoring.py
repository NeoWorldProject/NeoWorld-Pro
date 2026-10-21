"""Shared candidate reward and independent semantic vote reduction."""
import hashlib
import json

def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def _semantic(records):
    values = [r["response"].get("score") for r in records
              if r.get("status") == "done" and isinstance(r.get("response"), dict)
              and r["response"].get("uncertain") is False
              and type(r["response"].get("score")) is int and r["response"]["score"] in (0, 1)]
    # A disagreeing repeat is uncertainty; no majority manufactured by more calls.
    resolved = bool(values) and len(values) == len(records) and len(set(values)) == 1
    value = values[0] if resolved else None
    return {"valid": resolved, "sample_count": len(values), "attempt_count": len(records),
            "uncertain": not resolved, "reliable": resolved and len(values) >= 2,
            "score": value, "mean": value, "scores": None,
            "mean_range": max(values) - min(values) if values else None}


def candidate_reward(physics, semantic, distance, *, semantic_weight=.05, deviation_weight=.01):
    """A physical plateau dominates every bounded semantic contribution."""
    if not physics["passed"]:
        return -max(0., float(physics["violation"]))
    semantic_term = semantic_weight * semantic if semantic is not None else 0.
    return 1. + semantic_term - deviation_weight * min(1., max(0., distance))
