"""Frozen engineering defaults and evidence primitives for scene refinement."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from .hashing import file_hash

SCHEMA_VERSION = "scene_refinement.v6_adaptive_rollout"
DEFAULT_CONFIG = {
    "certified_early_exit": 1, "compact_evidence": 1,
    "shared_shader_cache": 1, "compress_trajectories": 1,
    "async_evidence": 1, "trajectory_retention": "compact",
    "population_size": 16, "elite_count": 4, "max_iterations": 6,
    "candidate_workers": 0, "max_error_diagnoses": 5,
    "block_size": 3, "min_std_fraction": .2,
    "translation_std_fraction": .05, "yaw_std_deg": 15., "log_scale_std": .1,
    "translation_bound_std": 4., "yaw_bound_deg": 60.,
    "scale_min": 2 / 3, "scale_max": 1.5, "api_budget": 0,
    "semantic_weight": .05, "deviation_weight": .01,
    "physical_signal_epsilon": 1e-8,
    "physics_dt_s": 1 / 240, "gravity_m_s2": 9.81,
    "rollout_max_duration_s": 5., "settle_window_s": .5,
    "early_stop_translation_m": .0001, "early_stop_rotation_deg": .1,
    "reuse_static_geometry": 0, "geometry_cache_mib": 256,
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
    value = json.loads(data, parse_constant=reject)
    from .candidate_storage import SCHEMA, unpack as unpack_candidate
    if isinstance(value, dict) and value.get('storage_schema') == SCHEMA:
        return unpack_candidate(value, path)
    from .geometry_storage import unpack
    return unpack(value)


def write_json(path, value):
    import os
    import uuid
    from urdf_core.storage import require_space
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict) and value.get('schema_version') == 'scene_candidate.v1':
        from .candidate_storage import pack
        value = pack(value, path)
    if isinstance(value, dict) and value.get('schema_version') == 'scene_prepared.v1':
        from .geometry_storage import pack_scene
        value = pack_scene(value, path)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        if path.suffix == '.gz':
            with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=1) as handle:
                json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        else:
            text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
            if isinstance(value, dict) and value.get('storage_schema') in {'scene_asset_references.v1', 'scene_prepared_references.v1'}:
                require_space(path.parent, len(text.encode('utf-8')))
            temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_rollout_settings(config):
    if config['settle_window_s'] != .5 or not .5 < config['rollout_max_duration_s'] <= 5.:
        raise ValueError('Adaptive scene rollouts require a 0.5-second window and at most 5 seconds')
    if (config['early_stop_translation_m'] > config['position_tolerance_min_m']
            or config['early_stop_rotation_deg'] > config['orientation_tolerance_deg']):
        raise ValueError('Early stop tolerances cannot exceed acceptance tolerances')
    frequency = 1/config['physics_dt_s']
    if round(frequency) < 1 or abs(frequency-round(frequency)) > 1e-6:
        raise ValueError('physics_dt_s must be the reciprocal of an integer USD simulation frequency')
    for key in ('rollout_max_duration_s', 'settle_window_s'):
        if abs(config[key]*frequency-round(config[key]*frequency)) > 1e-6:
            raise ValueError(key+' must contain an integer number of physics steps')


def load_config(value=None):
    supplied = read_json(value) if isinstance(value, (str, Path)) else dict(value or {})
    if 'reuse_local_mesh_arrays' in supplied:
        raise ValueError('reuse_local_mesh_arrays was removed; delete it from the scene configuration')
    if isinstance(supplied, dict) and set(supplied) & {"search_duration_s", "final_duration_s"}:
        raise ValueError("retired 3/8-second configuration; use rollout_max_duration_s in a new run, or the old frozen source")
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_CONFIG):
        raise ValueError("Unknown scene refinement configuration fields")
    config = {**DEFAULT_CONFIG, **supplied}
    switches = {"certified_early_exit", "compact_evidence", "shared_shader_cache",
                "compress_trajectories", "async_evidence", "reuse_static_geometry"}
    for key, val in config.items():
        if key == "trajectory_retention":
            if val not in ("compact", "full"):
                raise ValueError("trajectory_retention must be compact or full")
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"{key} must be finite numeric data")
        if isinstance(DEFAULT_CONFIG[key], int) and type(val) is not int:
            raise ValueError(f"{key} must be an integer")
        if key in switches:
            if val not in (0, 1):
                raise ValueError(f"{key} must be 0 or 1")
        elif key in {"api_budget", "candidate_workers", "max_error_diagnoses"}:
            if val < 0:
                raise ValueError(f"{key} must be nonnegative (0 selects automatic/unlimited)")
        elif val <= 0:
            raise ValueError(f"{key} must be positive")
    if not 2 <= config["elite_count"] <= config["population_size"]:
        raise ValueError("Require 2 <= elite_count <= population_size")
    if config['population_size'] != 16 or config['elite_count'] != 4:
        raise ValueError('Scene refinement requires 16 candidates and 4 elites per generation')
    if config['max_error_diagnoses'] > 5:
        raise ValueError('max_error_diagnoses cannot exceed 5')
    if config['candidate_workers'] > config['population_size']:
        raise ValueError('candidate_workers cannot exceed the generation population')
    if config['api_budget'] != 0:
        raise ValueError('Scene quality search ends at its generation budget; no separate total API budget')
    if not 0 < config["min_std_fraction"] <= 1:
        raise ValueError("min_std_fraction must lie in (0, 1]")
    if not config["scale_min"] < 1 < config["scale_max"]:
        raise ValueError("Scale bounds must straddle one")
    if config['deviation_weight'] >= 1:
        raise ValueError('deviation_weight must be below one to preserve physical-pass priority')
    if config['geometry_tolerance_m'] > config['penetration_tolerance_m']:
        raise ValueError('Geometry resolution exceeds the penetration tolerance')
    validate_rollout_settings(config)
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
