"""Frozen engineering defaults and evidence primitives for scene refinement."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from .hashing import file_hash

SCHEMA_VERSION = "scene_refinement.v5_support_layers"
DEFAULT_CONFIG = {
    "certified_early_exit": 1, "compact_evidence": 1,
    "shared_shader_cache": 1, "compress_trajectories": 1,
    "async_evidence": 1,
    "population_size": 16, "elite_count": 4, "max_iterations": 10,
    "block_size": 3, "min_std_fraction": .2,
    "translation_std_fraction": .05, "yaw_std_deg": 15., "log_scale_std": .1,
    "translation_bound_std": 4., "yaw_bound_deg": 60.,
    "scale_min": 2 / 3, "scale_max": 1.5, "api_budget": 0,
    "semantic_weight": .05, "deviation_weight": .01,
    "physical_signal_epsilon": 1e-8,
    "physics_dt_s": 1 / 240, "gravity_m_s2": 9.81,
    "search_duration_s": 3., "final_duration_s": 8., "settle_window_s": 1.,
    "penetration_tolerance_m": .0005, "geometry_tolerance_m": 1e-6,
    "position_tolerance_min_m": .001, "position_tolerance_fraction": .001,
    "orientation_tolerance_deg": 1., "drift_min_m": .005, "drift_fraction": .02,
    "tilt_limit_deg": 10., "worker_timeout_s": 7200., "video_fps": 12,
    "render_width": 1024, "render_height": 768,
}


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    path = Path(path)
    data = gzip.open(path, "rt", encoding="utf-8").read() if path.suffix == ".gz" else path.read_text(encoding="utf-8")
    return json.loads(data, parse_constant=reject)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".gz":
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    else:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_config(value=None):
    supplied = read_json(value) if isinstance(value, (str, Path)) else dict(value or {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_CONFIG):
        raise ValueError("Unknown scene refinement configuration fields")
    config = {**DEFAULT_CONFIG, **supplied}
    switches = {"certified_early_exit", "compact_evidence", "shared_shader_cache",
                "compress_trajectories", "async_evidence"}
    for key, val in config.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"{key} must be finite numeric data")
        if isinstance(DEFAULT_CONFIG[key], int) and type(val) is not int:
            raise ValueError(f"{key} must be an integer")
        if key in switches:
            if val not in (0, 1):
                raise ValueError(f"{key} must be 0 or 1")
        elif key == "api_budget":
            if val < 0:
                raise ValueError(f"{key} must be nonnegative (0 selects automatic/unlimited)")
        elif val <= 0:
            raise ValueError(f"{key} must be positive")
    if not 2 <= config["elite_count"] <= config["population_size"]:
        raise ValueError("Require 2 <= elite_count <= population_size")
    if config['population_size'] != 16 or config['elite_count'] != 4:
        raise ValueError('Scene refinement requires 16 candidates and 4 elites per generation')
    if config['max_iterations'] > 10:
        raise ValueError('Scene refinement permits at most 10 generations')
    if config['api_budget'] != 0:
        raise ValueError('Scene quality search ends at its generation budget; no separate total API budget')
    if config['final_duration_s'] < 8. or config['settle_window_s'] < 1.:
        raise ValueError('Scene acceptance requires at least an 8-second reload and a 1-second settling window')
    if not 0 < config["min_std_fraction"] <= 1:
        raise ValueError("min_std_fraction must lie in (0, 1]")
    if not config["scale_min"] < 1 < config["scale_max"]:
        raise ValueError("Scale bounds must straddle one")
    if config['deviation_weight'] >= 1:
        raise ValueError('deviation_weight must be below one to preserve physical-pass priority')
    if config["settle_window_s"] > min(config["search_duration_s"], config["final_duration_s"]):
        raise ValueError("Both rollouts must contain the complete settle window")
    if config["geometry_tolerance_m"] > config["penetration_tolerance_m"]:
        raise ValueError("Geometry resolution exceeds the penetration tolerance")
    if config["final_duration_s"] < config["search_duration_s"]:
        raise ValueError("Final validation cannot be shorter than search")
    frequency = 1 / config["physics_dt_s"]
    if round(frequency) < 1 or abs(frequency - round(frequency)) > 1e-6:
        raise ValueError("physics_dt_s must be the reciprocal of an integer USD simulation frequency")
    return config


def validate_score(value, object_ids=None):
    """Binary relation check; uncertainty remains missing, never a fabricated score."""
    if not isinstance(value, dict) or type(value.get("uncertain")) is not bool:
        raise ValueError("Semantic review requires an explicit uncertain boolean")
    score = value.get("score")
    if value["uncertain"]:
        if score is not None:
            raise ValueError("Uncertain semantic score must be null")
    elif type(score) is not int or score not in (0, 1):
        raise ValueError("Semantic score must be 1 (reasonable), 0 (clear contradiction), or null")
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Semantic review needs evidence")
    for item in evidence:
        if not isinstance(item, dict) or not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("Evidence needs a reason")
        ids = item.get("object_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
            raise ValueError("Evidence needs object IDs")
        if object_ids is not None and set(ids) - set(object_ids):
            raise ValueError("Semantic evidence refers to unknown objects")
    return value
