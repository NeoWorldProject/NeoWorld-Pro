"""Schema-3 component collisions, independent of source mass geometry.

Exact boxes remain native boxes. Other physical components enter CoACD without
input mesh simplification; its convex outputs are limited to
32 vertices without forcing a hull-count limit or merging separate parts.
This is a collision-generation policy, not a proxy-fidelity certificate.
"""
from __future__ import annotations

from importlib.metadata import version
import warnings

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .coacd_worker import CoacdSystemError, repeated_native_crash, resource_failure, run_coacd


POLICY = {"threshold": 0.05, "max_convex_hull": -1,
          "merge": False, "decimate": True, "max_ch_vertex": 32, "seed": 0}


class CollisionHullError(ValueError):
    """Invalid library output is a generation error, not a physical trial."""
    def __init__(self, message, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(message)


def exact_box(mesh):
    """Recognize six box planes; never substitute an OBB for a curved solid.

    This finite primitive recognition checks original triangles, closedness and
    volume. It does not call the historical symmetric coverage/proof engine.
    """
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        return None
    bounds = mesh.bounding_box_oriented
    pose = np.asarray(bounds.primitive.transform)
    size = np.asarray(bounds.primitive.extents)
    tolerance = max(float(np.linalg.norm(size)) * 1e-8, 1e-10)
    if np.any(size <= tolerance):
        return None
    local = (np.asarray(mesh.vertices) - pose[:3, 3]) @ pose[:3, :3]
    triangles = local[np.asarray(mesh.faces)]
    # Every positive-area triangle must lie on one of the six outer planes.
    planes = np.any(np.all(np.abs(triangles - size / 2) <= tolerance, axis=1)
                    | np.all(np.abs(triangles + size / 2) <= tolerance, axis=1), axis=1)
    if not np.all(planes) or np.any(np.abs(local) > size / 2 + tolerance):
        return None
    expected = float(np.prod(size))
    if not np.isclose(abs(float(mesh.volume)), expected, rtol=1e-7, atol=expected * 1e-10):
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz").tolist()
    box = {"size_m": size.tolist(), "center_m": pose[:3, 3].tolist(), "rpy_rad": rpy}
    return {"representation": "box", "box": box,
            "mesh": trimesh.creation.box(extents=size, transform=pose)}


def _validate_source_geometry(mesh):
    """Reject invalid input before any optional primitive-recognition fallback."""
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
            or faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces)
            or not np.isfinite(vertices).all() or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("collision source component must contain finite nonempty triangle geometry with valid indices")
    if not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError("collision source component must have finite positive surface area")


def _coacd_hull(vertices, faces, vertex_limit=None):
    """Check library output before treating it as a convex collision solid."""
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces)
    finite = bool(np.isfinite(vertices).all())
    valid_arrays = (vertices.ndim == 2 and vertices.shape[1:] == (3,) and len(vertices)
                    and faces.ndim == 2 and faces.shape[1:] == (3,) and len(faces)
                    and np.issubdtype(faces.dtype, np.integer) and faces.min() >= 0
                    and faces.max() < len(vertices))
    diagnostic = {"vertices": len(vertices) if vertices.ndim else 0,
                  "faces": len(faces) if faces.ndim else 0, "finite_vertices": finite,
                  "vertex_limit": vertex_limit}
    if not valid_arrays or not finite:
        raise CollisionHullError("CoACD output contains invalid vertex/face arrays",
                                 {"code": "coacd_invalid_arrays", **diagnostic})
    hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    hull.remove_unreferenced_vertices()
    # A collapsed sheet can make trimesh's mass integration warn while deriving
    # a zero volume. Record that invalid value without hiding the failure.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        volume = float(hull.volume)
        closed, convex = bool(hull.is_watertight), bool(hull.is_convex)
    diagnostic.update(vertices=len(hull.vertices), watertight=closed, convex=convex,
                      volume_m3=volume if np.isfinite(volume) else None,
                      extents_m=hull.extents.tolist())
    if (len(hull.vertices) < 4 or (vertex_limit is not None and len(hull.vertices) > vertex_limit)
            or not closed or not convex or not np.isfinite(volume) or abs(volume) <= 0):
        raise CollisionHullError("CoACD output is not a finite closed convex hull within the vertex budget",
                                 {"code": "coacd_invalid_hull", **diagnostic})
    if volume < 0:
        hull.invert()
    return hull


def _bounded_convex_piece(raw, vertex_limit):
    """Reduce one already validated CoACD convex piece, never a concave source."""
    if len(raw.vertices) <= vertex_limit:
        return raw.copy()
    import fast_simplification
    vertices, faces = fast_simplification.simplify(
        np.asarray(raw.vertices, dtype=np.float64), np.asarray(raw.faces, dtype=np.int32),
        target_count=2 * vertex_limit - 4)
    reduced = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    reduced.remove_unreferenced_vertices()
    _validate_source_geometry(reduced)
    if len(reduced.vertices) > vertex_limit:
        raise ValueError("Alternative hull simplification did not reach the unchanged vertex budget")
    # Quadric reduction can make locally concave faces. Reconstruct only this
    # one convex piece from its bounded reduced points; no pieces are merged.
    bounded = reduced.convex_hull
    return _coacd_hull(bounded.vertices, bounded.faces, vertex_limit)


def _decompose(source, policy):
    def run(decimate):
        return run_coacd(source.vertices, source.faces, **{**policy, "decimate": decimate})
    result = run(policy["decimate"])
    if not result:
        raise ValueError("CoACD produced no convex collision components")
    try:
        return [_coacd_hull(vertices, faces, policy["max_ch_vertex"]) for vertices, faces in result], None
    except CollisionHullError as original:
        # CoACD's optional vertex decimator can collapse a valid convex solid
        # to a zero-volume sheet. One bounded representation fallback preserves
        # decomposition/seed/threshold and replaces only that decimation step.
        # Raw decomposition must itself be valid; no global source convex hull.
        fallback = {"trigger": original.diagnostics, "original_hull_count": len(result),
                    "method": "coacd_without_decimation_then_per_piece_quadric_convex",
                    "retry_count": 1, "seed": policy["seed"], "threshold": policy["threshold"],
                    "vertex_limit": policy["max_ch_vertex"], "raw_hull_count": None}
        try:
            raw_result = run(False)
            fallback["raw_hull_count"] = len(raw_result)
            if not raw_result:
                raise ValueError("CoACD without decimation produced no convex components")
            pieces = [_coacd_hull(vertices, faces) for vertices, faces in raw_result]
            hulls = [_bounded_convex_piece(piece, policy["max_ch_vertex"]) for piece in pieces]
            fallback.update(status="recovered", raw_vertices=[len(p.vertices) for p in pieces],
                            final_vertices=[len(h.vertices) for h in hulls],
                            final_faces=[len(h.faces) for h in hulls])
            return hulls, fallback
        except (ValueError, RuntimeError) as error:
            fallback.update(status="failed", fallback_error=getattr(error, "diagnostics", {"reason": str(error)}))
            raise CollisionHullError("CoACD generated invalid collision hulls and the bounded representation fallback failed",
                                     {"code": "coacd_representation_fallback_failed", **fallback}) from error


def _part_convex_proxy(source, vertex_limit):
    hull = source.convex_hull
    hull = _coacd_hull(hull.vertices, hull.faces, vertex_limit)
    return {"representation": "convexHull", "mesh": hull}


def _native_crash_proxy(source, policy, error):
    audit = {"policy": "coacd_native_crash_component_fallback.v1", "geometry_degraded": True,
             "visual_geometry_changed": False, "trigger": error.diagnostics,
             "may_fill_cavities": True, "content_refinements_used": 0}
    try:
        proxy = _part_convex_proxy(source, policy["max_ch_vertex"])
        audit["method"] = "native_crash_part_convex_hull"
    except (ValueError, RuntimeError) as hull_error:
        if resource_failure(hull_error):
            raise OSError(str(hull_error)) from hull_error
        points = np.asarray(source.vertices)
        size = np.maximum(np.ptp(points, axis=0), 1e-4)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(center)
        _coacd_hull(mesh.vertices, mesh.faces, policy["max_ch_vertex"])
        proxy = {"representation": "box", "mesh": mesh,
                 "box": {"size_m": size.tolist(), "center_m": center.tolist(), "rpy_rad": [0., 0., 0.]}}
        audit.update(method="native_crash_part_box", convex_error=str(hull_error), minimum_box_extent_m=1e-4)
    return proxy, audit


def effective_policy(config=None):
    if config is not None and "collision_face_limit" in config:
        raise ValueError("collision_face_limit has been removed; CoACD receives unsimplified source meshes. Use a new configuration or the historical frozen runtime.")
    policy = dict(POLICY)
    for config_key, key in (("collision_threshold", "threshold"),
                            ("collision_max_vertices", "max_ch_vertex"), ("collision_seed", "seed")):
        if config is not None and config_key in config:
            policy[key] = config[config_key]
    return policy


def plan_lightweight_collisions(parts, *, config=None):
    """Produce per-component convex proxies with recorded algorithm settings.

    ``parts`` are source-connected solids (or one parity-oriented nested-shell
    solid). Never split nested cavity shells into independently filled hulls.
    """
    policy = effective_policy(config)
    proxies, decisions = [], []
    for index, source in enumerate(parts):
        _validate_source_geometry(source)
        recognition_diagnostic = None
        try:
            box = exact_box(source)
        except (ValueError, RuntimeError) as error:
            # A valid authored solid can trigger a degenerate temporary hull in
            # trimesh's OBB search. Exact-box recognition is only an optimization;
            # failing it must not block the normal, unchanged decomposition path.
            # Source validation above and all downstream output checks still fail
            # normally; this handler covers only the optional recognizer.
            box = None
            recognition_diagnostic = {"status": "failed", "error_type": type(error).__name__,
                                      "message": str(error), "fallback": "coacd_on_source_mesh"}
        if box is not None:
            proxies.append({**box, "source_component_index": index})
            decisions.append({"component_index": index, "method": "exact_native_box",
                              "source_faces": len(source.faces), "proxy_count": 1, "proxy_faces": 12})
            continue
        try:
            hulls, fallback = _decompose(source, policy)
        except CoacdSystemError as error:
            if not repeated_native_crash(error.diagnostics):
                raise
            proxy, crash_audit = _native_crash_proxy(source, policy, error)
            proxies.append({**proxy, "source_component_index": index})
            decisions.append({**crash_audit, "component_index": index,
                "source_faces": len(source.faces), "coacd_input_faces": len(source.faces),
                "proxy_count": 1, "proxy_faces": len(proxy["mesh"].faces),
                "proxy_vertices": len(proxy["mesh"].vertices)})
            continue
        component_proxies = []
        for hull in hulls:
            component_proxies.append({"representation": "convexHull", "mesh": hull,
                                      "source_component_index": index})
        proxies.extend(component_proxies)
        decision = {"component_index": index, "method": "coacd",
                    "source_faces": len(source.faces), "coacd_input_faces": len(source.faces),
                    "source_bounds_m": source.bounds.tolist(),
                    "proxy_count": len(component_proxies),
                    "proxy_faces": sum(len(p["mesh"].faces) for p in component_proxies),
                    "proxy_vertices": sum(len(p["mesh"].vertices) for p in component_proxies)}
        if recognition_diagnostic is not None:
            decision["optional_box_recognition"] = recognition_diagnostic
        if fallback is not None:
            decision["invalid_output_fallback"] = fallback
        decisions.append(decision)
    if not proxies:
        raise ValueError("contact part has no collision components")
    audit = {"method": "component_lightweight_coacd", "policy": policy, "components": decisions,
             "input_mesh_policy": "source_components_without_simplification.v1",
             "proxy_fidelity_tested": False,
             "versions": {name: version(name) for name in ("trimesh", "fast-simplification", "coacd")}}
    source_bounds = np.array([p.bounds for p in parts])
    proxy_bounds = np.array([p["mesh"].bounds for p in proxies])
    source_bounds = np.array([source_bounds[:, 0].min(axis=0), source_bounds[:, 1].max(axis=0)])
    proxy_bounds = np.array([proxy_bounds[:, 0].min(axis=0), proxy_bounds[:, 1].max(axis=0)])
    audit["bounds_diagnostic"] = {
        "frame": "authored_link_local_m", "source_bounds_m": source_bounds.tolist(),
        "proxy_bounds_m": proxy_bounds.tolist(),
        "outward_excess_lower_m": np.maximum(0, source_bounds[0] - proxy_bounds[0]).tolist(),
        "outward_excess_upper_m": np.maximum(0, proxy_bounds[1] - source_bounds[1]).tolist()}
    if any(row.get("geometry_degraded") for row in decisions):
        audit["geometry_degraded"] = True
    return [p["mesh"] for p in proxies], proxies, audit
