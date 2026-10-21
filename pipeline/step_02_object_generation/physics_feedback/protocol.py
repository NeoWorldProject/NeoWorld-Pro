"""Versioned physical assumptions and immutable object-level test contracts.

Only the model supplies object-specific masses and effort budgets.  This module
normalizes units and freezes the acceptance criteria before dynamics are run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from urdf_core.kinematics import validate_spec


DEFAULT_CONFIG = {
    "physics_dt_s": 1 / 240, "video_fps": 12, "gravity_m_s2": 9.81,
    "settle_duration_s": 8.0, "settle_window_s": 1.0,
    "position_tolerance_m": None, "orientation_tolerance_deg": 1.0,
    "tilt_limit_deg": 10.0, "coverage_min": 0.98, "release_height_m": None,
    "angular_velocity_rad_s": 0.25, "linear_velocity_m_s": 0.02,
    "penetration_tolerance_m": 0.0005, "geometry_tolerance_m": 1e-6,
    "sweep_resolution_m": 0.0005, "max_sweep_substeps": 128,
    "endpoint_tolerance_fraction": 0.02, "endpoint_hold_s": 1.0,
    "endpoint_tracking_allowance_s": .25,
    "trial_timeout_s": 600.0, "zero_effort_tail_s": 1.0,
    "free_slip_tolerance_m": None,
    "collision_threshold": .05,
    "collision_max_vertices": 32, "collision_seed": 0,
}

CURRENT_CONTRACT_SCHEMA = 3
GROUND_HOLD_ACCEPTANCE_POLICY = "object_functional_physics.v11"
CURRENT_GROUND_POLICY = "passive_root_settle_q_initial"
MOTION_SCOPE_POLICY = "fixed_full_ground_operable.v1"


def validate_orientation_policy(value):
    """Freeze the API's orientation applicability, separately from stability."""
    if (not isinstance(value, dict) or set(value) != {"upright_required", "reason"}
            or type(value.get("upright_required")) is not bool
            or not isinstance(value.get("reason"), str) or not value["reason"].strip()):
        raise ValueError("orientation_policy requires exactly upright_required boolean and nonempty reason")
    return copy.deepcopy(value)


def validate_free_base_policy(value):
    """Independent functional-use classification; never a dynamics repair."""
    if (not isinstance(value, dict) or set(value) != {"required", "reason"}
            or type(value.get("required")) is not bool
            or not isinstance(value.get("reason"), str) or not value["reason"].strip()):
        raise ValueError("free_base_policy requires exactly required boolean and nonempty reason")
    return copy.deepcopy(value)


def free_base_required(contract):
    if contract.get('acceptance_policy') != GROUND_HOLD_ACCEPTANCE_POLICY:
        raise ValueError('Unsupported object acceptance policy; expected v11')
    policy = validate_free_base_policy(contract.get('free_base_policy'))
    return bool(contract.get('stable_required', True) and policy['required'])


def validate_current_contract(contract):
    expected = {'schema_version': CURRENT_CONTRACT_SCHEMA,
                'acceptance_policy': GROUND_HOLD_ACCEPTANCE_POLICY,
                'ground_policy': CURRENT_GROUND_POLICY,
                'penetration_policy': 'diagnostic_only',
                'joint_peak_velocity_policy': 'diagnostic_only',
                'external_probe_policy': 'not_required',
                'motion_reference_policy': 'pre_motion_stable_root',
                'collision_policy': 'external_and_ground_only',
                'motion_scope_policy': MOTION_SCOPE_POLICY,
                'passive_tail_velocity_policy': 'diagnostic_only',
                'ground_initial_rotary_tolerance_deg': 10.0}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError('Unsupported or changed frozen v11 contract')
    validate_orientation_policy(contract.get('orientation_policy'))
    validate_free_base_policy(contract.get('free_base_policy'))
    from .motion_acceptance import enabled
    enabled(contract)
    joints = contract.get('structural_joints')
    if (not isinstance(joints, list)
            or {j['name'] for j in joints if j['type'] != 'fixed'} != set(contract['joints'])
            or any(j['parent'] not in contract['required_links'] or j['child'] not in contract['required_links'] for j in joints)):
        raise ValueError('Contract requires complete structural joint definitions')
    from .test_modes import FIXED_BASE_POLICY
    if contract.get('fixed_base_policy') != FIXED_BASE_POLICY:
        raise ValueError('Contract requires elevated full-range fixed-base tests')
    return contract


# Isaac Sim 5.1 / omni.usd.schema.physx 107.3.26 schema.usda and
# omni.kit.property.physx database.py: native names and conservative UI bounds.
# minThickness explicitly allows zero in the schema; use the UI upper bound.
# The local omni.convexdecomposition.plugin rejects errorPercentage < 0.01
# (native diagnostic: "must be between 0.01 and 25"). Intersect that backend
# range with the conservative UI maximum of 20 instead of accepting UI's zero.
# These are cooking parameters, never acceptance-tolerance overrides.
DECOMPOSITION_PARAMETERS = {
    "hull_vertex_limit": {"usd_name": "hullVertexLimit", "kind": "int", "min": 8, "max": 64},
    "max_convex_hulls": {"usd_name": "maxConvexHulls", "kind": "int", "min": 1, "max": 2048},
    "voxel_resolution": {"usd_name": "voxelResolution", "kind": "int", "min": 50000, "max": 5000000},
    "error_percentage": {"usd_name": "errorPercentage", "kind": "number", "min": 0.01, "max": 20},
    "min_thickness_m": {"usd_name": "minThickness", "kind": "number", "min": 0, "max": 10},
    "shrink_wrap": {"usd_name": "shrinkWrap", "kind": "bool"},
}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _number(value, label, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if value < 0 or (not zero and value == 0):
        raise ValueError(f"{label} must be {'nonnegative' if zero else 'positive'}")
    return float(value)


def load_config(path=None):
    """Return independent defaults, with strict optional JSON/dict overrides."""
    value = {} if path is None else copy.deepcopy(path) if isinstance(path, dict) else json.loads(Path(path).read_text())
    if isinstance(value, dict) and "collision_face_limit" in value:
        raise ValueError("collision_face_limit has been removed; use a new configuration without input simplification, or the historical frozen runtime.")
    if not isinstance(value, dict) or set(value) - set(DEFAULT_CONFIG):
        raise ValueError("physics_config must contain only supported configuration keys")
    result = {**DEFAULT_CONFIG, **value}
    for key, value in result.items():
        if value is None and key in {"position_tolerance_m", "release_height_m", "free_slip_tolerance_m"}:
            continue
        _number(value, key, zero=key in {"release_height_m", "collision_seed"})
    for key in ("video_fps", "max_sweep_substeps", "collision_max_vertices", "collision_seed"):
        if type(result[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if not 4 <= result["collision_max_vertices"] <= 32 or result["collision_threshold"] > 1:
        raise ValueError("invalid frozen collision engineering settings")
    if not 0 < result["coverage_min"] <= 1 or not 0 < result["endpoint_tolerance_fraction"] < 0.5:
        raise ValueError("invalid coverage or endpoint tolerance fraction")
    if result["settle_window_s"] > result["settle_duration_s"]:
        raise ValueError("settle window exceeds settle duration")
    if result["physics_dt_s"] > 1 / result["video_fps"]:
        raise ValueError("physics timestep must not exceed video frame period")
    if result["geometry_tolerance_m"] > result["penetration_tolerance_m"]:
        raise ValueError("geometry precision must not exceed penetration tolerance")
    return result


def _validate_profile_v3(profile, spec):
    """Current per-part masses and native articulation friction, all in SI."""
    allowed = {"schema_version", "stable_required", "stability_reason", "links", "joints", "collision"}
    if set(profile) - allowed or type(profile.get("stable_required")) is not bool:
        raise ValueError("schema 3 requires stable_required boolean and supported fields")
    if not isinstance(profile.get("stability_reason"), str) or not profile["stability_reason"].strip():
        raise ValueError("schema 3 requires a nonempty stability_reason")
    result = copy.deepcopy(profile)
    expected = {link["name"]: set(link["objects"]) for link in spec["links"]}
    if not isinstance(result.get("links"), dict) or set(result["links"]) != set(expected):
        raise ValueError("physics profile must cover exactly every link")
    contacts = 0
    for name, params in result["links"].items():
        if not isinstance(params, dict) or set(params) - {"material", "objects"}:
            raise ValueError("schema 3 uses per-part mass, not link mass: " + name)
        material = params.setdefault("material", {})
        if not isinstance(material, dict) or set(material) - {"static_friction", "dynamic_friction", "restitution"}:
            raise ValueError("invalid contact material: " + name)
        for key, default in (("static_friction", .5), ("dynamic_friction", .5), ("restitution", 0)):
            material[key] = _number(material.get(key, default), name + "." + key, zero=True)
        if material["dynamic_friction"] > material["static_friction"] or material["static_friction"] > 2 or material["restitution"] > 1:
            raise ValueError("invalid contact material bounds: " + name)
        objects = params.get("objects")
        if not isinstance(objects, dict) or set(objects) != expected[name]:
            raise ValueError("schema 3 must cover every authored part: " + name)
        for obj, part in objects.items():
            keys = {"role", "reason", "mass_kg", "density_kg_m3", "mass_model", "thickness_m", "approximation_reason"}
            if not isinstance(part, dict) or set(part) - keys:
                raise ValueError("invalid per-part physics fields: " + obj)
            if part.get("role") not in {"contact", "decoration"} or not isinstance(part.get("reason"), str) or not part["reason"].strip():
                raise ValueError("part needs contact/decoration and a reason: " + obj)
            contacts += part["role"] == "contact"
            if ("mass_kg" in part) == ("density_kg_m3" in part):
                raise ValueError("part requires exactly one mass or density: " + obj)
            key = "mass_kg" if "mass_kg" in part else "density_kg_m3"
            part[key] = _number(part[key], obj + "." + key)
            part.setdefault("mass_model", "auto")
            if part["mass_model"] not in {"auto", "surface_shell"}:
                raise ValueError("unsupported part mass_model: " + obj)
            if "thickness_m" in part:
                part["thickness_m"] = _number(part["thickness_m"], obj + ".thickness_m")
            if part["mass_model"] == "surface_shell" and ("thickness_m" not in part or not str(part.get("approximation_reason", "")).strip()):
                raise ValueError("surface_shell requires thickness and approximation_reason: " + obj)
    if not contacts:
        raise ValueError("an externally collidable asset requires at least one contact part")
    active = {joint["name"] for joint in spec["joints"] if joint["type"] != "fixed"}
    joints = result.setdefault("joints", {})
    if not isinstance(joints, dict) or set(joints) - active:
        raise ValueError("physics profile contains unknown/fixed joints")
    for name in sorted(active):
        params = joints.setdefault(name, {})
        keys = {"static_friction_effort", "dynamic_friction_effort", "damping"}
        if not isinstance(params, dict) or set(params) - keys:
            raise ValueError("unsupported native joint parameters: " + name)
        for key in keys:
            params[key] = _number(params.get(key, 0), name + "." + key, zero=True)
        if params["dynamic_friction_effort"] > params["static_friction_effort"]:
            raise ValueError("dynamic friction effort exceeds static friction effort: " + name)
    policy = {"approximation": "coacd", "enabled": True, "self_collision": False}
    if result.get("collision") != policy:
        raise ValueError("schema 3 requires fixed CoACD policy, enabled external collision and disabled self collision")
    return result


def validate_profile(profile, spec):
    """Normalize API physical parameters without supplying object-specific mass."""
    spec = validate_spec(spec)
    if isinstance(profile, dict) and type(profile.get("schema_version")) is int and profile["schema_version"] == 3:
        return _validate_profile_v3(profile, spec)
    if not isinstance(profile, dict) or type(profile.get("schema_version")) is not int or profile["schema_version"] not in (1, 2):
        raise ValueError("physics profile schema_version must be 1 or 2")
    if set(profile) - {"schema_version", "links", "joints", "collision"}:
        raise ValueError("unknown top-level physics profile fields")
    result = copy.deepcopy(profile)
    links = result.get("links")
    if not isinstance(links, dict) or set(links) != {link["name"] for link in spec["links"]}:
        raise ValueError("physics profile must cover exactly every link")
    for name, params in links.items():
        if not isinstance(params, dict):
            raise ValueError(f"invalid link profile: {name}")
        if set(params) - {"mass_kg", "density_kg_m3", "reason", "material", "thickness_m", "approximation_reason", "mass_model", "objects"}:
            raise ValueError(f"unknown physics link parameters: {name}")
        if ("mass_kg" in params) == ("density_kg_m3" in params):
            raise ValueError(f"{name} requires exactly one of mass_kg and density_kg_m3")
        key = "mass_kg" if "mass_kg" in params else "density_kg_m3"
        params[key] = _number(params[key], name + "." + key)
        if "thickness_m" in params:
            params["thickness_m"] = _number(params["thickness_m"], name + ".thickness_m")
        params.setdefault("mass_model", "auto")
        if params["mass_model"] not in {"auto", "surface_shell"}:
            raise ValueError(f"unsupported mass_model for {name}")
        if params["mass_model"] == "surface_shell" and ("thickness_m" not in params or not str(params.get("approximation_reason", "")).strip()):
            raise ValueError(f"surface_shell on {name} requires thickness_m and approximation_reason")
        material = params.setdefault("material", {})
        if not isinstance(material, dict) or set(material) - {"static_friction", "dynamic_friction", "restitution"}:
            raise ValueError(f"invalid contact material on {name}")
        for field, default in (("static_friction", .5), ("dynamic_friction", .5), ("restitution", 0.0)):
            material[field] = _number(material.get(field, default), name + "." + field, zero=True)
        if material["restitution"] > 1 or material["dynamic_friction"] > material["static_friction"]:
            raise ValueError(f"invalid restitution or static/dynamic friction on {name}")
        object_names = next(set(link["objects"]) for link in spec["links"] if link["name"] == name)
        objects = params.get("objects", {})
        if not isinstance(objects, dict) or set(objects) - object_names:
            raise ValueError(f"unknown object physics classifications on {name}")
        if result["schema_version"] == 2 and set(objects) != object_names:
            raise ValueError(f"schema 2 must classify every authored object on {name}")
        if result["schema_version"] == 1 and objects:
            raise ValueError("object classifications require physics profile schema 2")
        for obj, decision in objects.items():
            if not isinstance(decision, dict) or set(decision) - {"role", "reason", "represented_by", "mass_treatment", "collision"}:
                raise ValueError(f"invalid object physics classification: {obj}")
            if (decision.get("role") not in {"contact", "decoration"}
                    or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()):
                raise ValueError(f"{obj} needs contact/decoration role and API reason")
            if decision["role"] == "decoration":
                if (decision.get("represented_by") not in object_names - {obj}
                        or decision.get("mass_treatment") != "included_in_link_mass" or "mass_kg" not in params
                        or "collision" in decision):
                    raise ValueError(f"{obj}: decoration needs a same-link contact host and explicit total link mass")
            elif set(decision) & {"represented_by", "mass_treatment"}:
                raise ValueError(f"contact object {obj} cannot delegate its mass or collision")
            if "collision" in decision:
                recipe = decision["collision"]
                if (not isinstance(recipe, dict) or set(recipe) != {"representation", "boxes", "reason"}
                        or recipe["representation"] != "compound_boxes" or not isinstance(recipe["reason"], str) or not recipe["reason"].strip()
                        or not isinstance(recipe["boxes"], list) or not 1 <= len(recipe["boxes"]) <= 256):
                    raise ValueError(f"{obj}: invalid compound_boxes collision recipe")
                for box in recipe["boxes"]:
                    if not isinstance(box, dict) or set(box) != {"size_m", "center_m", "rpy_rad"}:
                        raise ValueError(f"{obj}: a box needs size_m, center_m, rpy_rad in link frame")
                    for key, values in box.items():
                        if (not isinstance(values, list) or len(values) != 3
                                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                                or (key == "size_m" and min(values) <= 0)):
                            raise ValueError(f"{obj}: invalid box {key}")
        for obj, decision in objects.items():
            if decision["role"] == "decoration" and objects.get(decision["represented_by"], {}).get("role") != "contact":
                raise ValueError(f"{obj}: decoration host must be a contact object, not another decoration")
    active = {joint["name"] for joint in spec["joints"] if joint["type"] != "fixed"}
    joints = result.setdefault("joints", {})
    if not isinstance(joints, dict) or set(joints) - active:
        raise ValueError("physics profile contains unknown/fixed joints")
    for name in sorted(active):
        params = joints.setdefault(name, {})
        if not isinstance(params, dict) or set(params) - {"damping", "friction"}:
            raise ValueError(f"unsupported passive joint parameters on {name}")
        for key in ("damping", "friction"):
            params[key] = _number(params.get(key, 0), name + "." + key, zero=True)
        if params["friction"] != 0:
            raise ValueError("nonzero URDF joint Coulomb friction is unsupported until its SI mapping to PhysX is verified; use contact material friction and joint damping")
    policy = {"approximation": "convexDecomposition", "enabled": True, "self_collision": True}
    collision = result.get("collision", policy)
    if not isinstance(collision, dict) or set(collision) - {*policy, "decomposition"}:
        raise ValueError("unsupported collision policy fields")
    allowed_approximation = {"convexDecomposition"} if result["schema_version"] == 1 else {"adaptive"}
    if (collision.get("approximation") not in allowed_approximation
            or collision.get("enabled") is not True or collision.get("self_collision") is not True):
        raise ValueError("collision and self_collision must remain enabled; schema 1 uses convexDecomposition, schema 2 uses adaptive")
    if "decomposition" in collision:
        requested = collision["decomposition"]
        if not isinstance(requested, dict) or set(requested) - set(DECOMPOSITION_PARAMETERS):
            raise ValueError("unsupported collision.decomposition parameters")
        for key, value in requested.items():
            rule, label = DECOMPOSITION_PARAMETERS[key], "collision.decomposition." + key
            if rule["kind"] == "bool":
                if type(value) is not bool:
                    raise ValueError(f"{label} must be a boolean")
                continue
            if rule["kind"] == "int" and type(value) is not int:
                raise ValueError(f"{label} must be an integer")
            number = _number(value, label, zero=True)
            if not rule["min"] <= number <= rule["max"]:
                raise ValueError(f"{label} must be in [{rule['min']}, {rule['max']}]")
    # Do not insert native defaults into a previously approved profile. Isaac
    # authors requested values and records all actual schema values separately.
    result["collision"] = copy.deepcopy(collision)
    return result


def make_test_contract(spec, effort_budgets, config=None, diagonal_m=1.0, *, previous=None, range_review=None, profile=None, controller=None, acceptance_policy=None, orientation_policy=None, free_base_policy=None):
    """Freeze all test criteria; amendments retain existing effort ceilings.

    A changed range requires a separately stored API review, supplied as
    {approved: true, reason: ...}. The orchestrator binds that review's evidence.
    """
    spec = validate_spec(spec)
    if profile is None or profile.get('schema_version') != 3:
        raise ValueError('Current contracts require profile schema 3')
    modern = True
    selected_policy = acceptance_policy or (previous or {}).get('acceptance_policy', GROUND_HOLD_ACCEPTANCE_POLICY)
    if selected_policy not in {GROUND_HOLD_ACCEPTANCE_POLICY}:
        raise ValueError('Unsupported new contract acceptance policy')
    if previous and selected_policy != previous.get('acceptance_policy'):
        raise ValueError('Contract policy changes require explicit acceptance migration')
    if selected_policy in {GROUND_HOLD_ACCEPTANCE_POLICY}:
        orientation_policy = validate_orientation_policy((previous or {}).get('orientation_policy') if orientation_policy is None else orientation_policy)
        if previous and orientation_policy != previous.get('orientation_policy'):
            raise ValueError('orientation_policy is frozen during refinement')
    elif orientation_policy is not None:
        raise ValueError('Historical contracts cannot introduce an orientation policy')
    if selected_policy in {GROUND_HOLD_ACCEPTANCE_POLICY}:
        free_base_policy = validate_free_base_policy((previous or {}).get('free_base_policy') if free_base_policy is None else free_base_policy)
        if previous and free_base_policy != previous.get('free_base_policy'):
            raise ValueError('free_base_policy is frozen during refinement')
    elif free_base_policy is not None:
        raise ValueError('Historical contracts cannot introduce free-base exclusions')
    if profile is not None:
        profile = validate_profile(profile, spec)
    diagonal_m = _number(diagonal_m, 'diagonal_m')
    settings = load_config(config)
    settings['position_tolerance_m'] = settings['position_tolerance_m'] or max(0.001, 0.001 * diagonal_m)
    settings['release_height_m'] = settings['release_height_m'] or min(0.02, max(0.002, 0.01 * diagonal_m))
    frozen = {'settle_duration_s': 8.0, 'settle_window_s': 1.0, 'release_height_m': 0.0, 'zero_effort_tail_s': 1.0, 'endpoint_hold_s': 1.0, 'endpoint_tracking_allowance_s': 0.25, 'position_tolerance_m': max(0.001, 0.001 * diagonal_m), 'orientation_tolerance_deg': 1.0, 'tilt_limit_deg': 10.0, 'free_slip_tolerance_m': max(0.005, 0.02 * diagonal_m)}
    supplied = config if isinstance(config, dict) else load_config(config) if config else {}
    for key, value in frozen.items():
        if supplied.get(key) is not None and supplied[key] != value:
            raise ValueError('schema 2 fixed acceptance threshold cannot be overridden: ' + key)
    settings.update(frozen)
    if not isinstance(effort_budgets, dict):
        raise ValueError('effort_budgets must be an object')
    active = {j['name'] for j in spec['joints'] if j['type'] != 'fixed'}
    if set(effort_budgets) - active:
        raise ValueError('effort budgets contain unknown or fixed joints')
    controller = copy.deepcopy(controller or {})
    if not isinstance(controller, dict) or set(controller) - active:
        raise ValueError('controller contains unknown/fixed joints')
    old_joints = previous.get('joints', {}) if previous else {}
    if previous:
        validate_current_contract(previous)
        if previous.get('schema_version') != CURRENT_CONTRACT_SCHEMA or set(old_joints) - active:
            raise ValueError('refinement cannot remove accepted active joints')
        supplied = config if isinstance(config, dict) else load_config(config) if config else {}
        if any((value is not None and previous['config'].get(key) != value for key, value in supplied.items())):
            raise ValueError('test thresholds cannot change during refinement')
        settings = copy.deepcopy(previous['config'])
        if previous.get('stable_required') != profile['stable_required']:
            raise ValueError('stable_required is frozen during refinement')
    joints = {}
    ranges_changed = False
    for joint in spec['joints']:
        if joint['type'] == 'fixed':
            continue
        name, kind = (joint['name'], joint['type'])
        old = old_joints.get(name)
        budget = effort_budgets.get(name, old)
        if not isinstance(budget, dict) or not isinstance(budget.get('reason'), str) or (not budget['reason'].strip()):
            raise ValueError(f'{name} needs API max_effort and a nonempty reason')
        cap = _number(budget.get('max_effort'), name + '.max_effort')
        if old and cap != old['max_effort']:
            raise ValueError(f'{name} effort budget is frozen')
        if old and (kind == 'prismatic') != (old['type'] == 'prismatic'):
            raise ValueError(f'{name} may not change its physical unit')
        supplied_controller = controller.get(name, {})
        if not isinstance(supplied_controller, dict) or set(supplied_controller) - {'kp', 'kd', 'speed_fraction', 'initial_position'}:
            raise ValueError('unsupported controller fields: ' + name)
        q0 = supplied_controller.get('initial_position', old.get('initial_position', 0.0) if old else 0.0)
        if type(q0) not in (int, float) or not math.isfinite(q0):
            raise ValueError('initial_position must be finite: ' + name)
        if old and q0 != old.get('initial_position', 0.0):
            raise ValueError('initial_position is frozen: ' + name)
        lower, upper = (q0 - 2 * math.pi, q0 + 2 * math.pi) if kind == 'continuous' else (joint['limit']['lower'], joint['limit']['upper'])
        ranges_changed |= bool(old and (old['type'], old['lower'], old['upper']) != (kind, lower, upper))
        speed = settings['linear_velocity_m_s'] if kind == 'prismatic' else settings['angular_velocity_rad_s']
        speed = min(speed, joint.get('limit', {}).get('velocity', speed))
        if not lower <= q0 <= upper:
            raise ValueError('initial_position must lie inside joint limits: ' + name)
        span = upper - lower
        kp = cap / max(span * settings['endpoint_tolerance_fraction'], 1e-06)
        joints[name] = {'type': kind, 'lower': lower, 'upper': upper, 'max_effort': cap, 'effort_unit': 'N' if kind == 'prismatic' else 'N*m', 'reason': budget['reason'], 'max_velocity': speed, 'kp': kp, 'kd': cap / max(speed, 1e-06) * 0.1, 'waypoints': [q0, lower, upper, q0]}
        defaults = {'kp': kp, 'kd': cap / max(speed, 1e-06) * 0.1, 'speed_fraction': 0.8}
        for key, default in defaults.items():
            value = supplied_controller.get(key, old.get(key, default) if old else default)
            joints[name][key] = _number(value, name + '.' + key, zero=key in {'kp', 'kd'})
        if joints[name]['speed_fraction'] > 1:
            raise ValueError('speed_fraction must lie in (0, 1]')
        joints[name]['initial_position'] = float(q0)
    if ranges_changed and (not isinstance(range_review, dict) or range_review.get('approved') is not True or (not isinstance(range_review.get('reason'), str)) or (not range_review['reason'].strip())):
        raise ValueError('joint range changes require an independent approved API range review')
    result = {'schema_version': CURRENT_CONTRACT_SCHEMA, 'revision': previous.get('revision', 0) + 1 if previous else 1, 'spec_sha256': json_hash(spec), 'diagonal_m': diagonal_m, 'config': settings, 'joints': joints}
    result.update(stable_required=profile['stable_required'], stability_reason=profile['stability_reason'], required_links=[link['name'] for link in spec['links']], collision_policy='external_and_ground_only', ground_policy=CURRENT_GROUND_POLICY, acceptance_policy=selected_policy, penetration_policy='diagnostic_only', joint_peak_velocity_policy='diagnostic_only', external_probe_policy='not_required', motion_reference_policy='pre_motion_stable_root')
    if selected_policy in {GROUND_HOLD_ACCEPTANCE_POLICY}:
        result['motion_scope_policy'] = MOTION_SCOPE_POLICY
    if selected_policy in {GROUND_HOLD_ACCEPTANCE_POLICY}:
        result['passive_tail_velocity_policy'] = 'diagnostic_only'
    if selected_policy in {GROUND_HOLD_ACCEPTANCE_POLICY}:
        result['orientation_policy'] = copy.deepcopy(orientation_policy)
    if selected_policy in {GROUND_HOLD_ACCEPTANCE_POLICY}:
        result['free_base_policy'] = copy.deepcopy(free_base_policy)
    if selected_policy in {GROUND_HOLD_ACCEPTANCE_POLICY}:
        from .motion_acceptance import freeze
        freeze(result, spec)
    if selected_policy in {GROUND_HOLD_ACCEPTANCE_POLICY}:
        from .test_modes import FIXED_BASE_POLICY
        result['fixed_base_policy'] = FIXED_BASE_POLICY
    if selected_policy == GROUND_HOLD_ACCEPTANCE_POLICY:
        result['ground_initial_rotary_tolerance_deg'] = 10.0
    if previous:
        result['previous_contract_sha256'] = json_hash(previous)
        if 'motion_scope_policy' in previous:
            result['motion_scope_policy'] = previous['motion_scope_policy']
    if ranges_changed:
        result['range_review'] = copy.deepcopy(range_review)
    return result


def validate_contract_refinement(previous, proposed):
    """Check a loaded candidate contract before accepting external amendments."""
    if proposed.get("schema_version") != previous.get("schema_version") or previous.get("config") != proposed.get("config"):
        raise ValueError("frozen physics test configuration changed")
    if any(previous.get(k) != proposed.get(k) for k in ('motion_acceptance', 'structural_joints', 'fixed_base_policy', 'ground_initial_rotary_tolerance_deg')):
        raise ValueError('Frozen motion acceptance or structural definition changed')
    if previous.get("motion_scope_policy") != proposed.get("motion_scope_policy"):
        raise ValueError("Model repairs cannot change the frozen motion scope policy")
    if previous.get("schema_version") in (2, 3) and any(previous.get(k) != proposed.get(k) for k in ("stable_required", "required_links", "ground_policy", "collision_policy", "acceptance_policy", "penetration_policy", "motion_reference_policy", "joint_peak_velocity_policy", "external_probe_policy", "passive_tail_velocity_policy", "orientation_policy", "free_base_policy")):
        raise ValueError("frozen schema 2 acceptance policy changed")
    for name, old in previous["joints"].items():
        new = proposed.get("joints", {}).get(name)
        if not new or any(new.get(k) != old[k] for k in ("max_effort", "effort_unit")):
            raise ValueError("accepted motion or frozen force budget changed: " + name)
        if previous.get("schema_version") in (2, 3) and new.get("initial_position") != old.get("initial_position"):
            raise ValueError("frozen initial joint position changed: " + name)
        if any(new.get(k) != old[k] for k in ("type", "lower", "upper")):
            review = proposed.get("range_review", {})
            if review.get("approved") is not True or not str(review.get("reason", "")).strip():
                raise ValueError("unreviewed joint range amendment: " + name)
    return copy.deepcopy(proposed)
