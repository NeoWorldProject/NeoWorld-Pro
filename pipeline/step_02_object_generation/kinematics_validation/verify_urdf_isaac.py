"""Bounded Isaac Sim 5.1 import/kinematics smoke check (no fall or wind).

Run with ordinary Python or Isaac's python.sh. The parent process supervises an
isolated Isaac worker so a Kit startup/shutdown hang cannot leave a passing
report. Angular positions passed to Articulation are radians; USD angular
attributes are never used as radian joint state substitutes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

# Direct CLI execution and importlib loading both need the sibling recorder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaac_motion import (MOTION_SOURCE_SHA256, sha256_file, screenshot_sample_indices,
                          capture_workload)
from capture_execution_evidence import (EXECUTION_SOURCE_SHA256, create_input_binding,
    verify_input_binding, match_imported_visuals, build_execution_evidence, artifact as execution_artifact)

# Isaac launches this script independently of the object-generation entrypoint.
# Only this bootstrap needs the enclosing source directory; runtime resources
# are resolved by urdf_core.paths, and outputs remain caller-supplied.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from urdf_core.isaac_cache import gpu_worker_slot, prepare_runtime_cache

DEFAULT_ISAAC_PYTHON = None
VERIFIER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def write_report(path: Path, report: dict) -> None:
    """Atomically publish strict JSON, including while the worker is running."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    if report.get('run_id') and report.get('stage') not in (None, 'starting', 'input_validation', 'simulation_app_launch'):
        from urdf_core.native_progress import write_execution_progress
        write_execution_progress(path.parent, report['run_id'], {'stage': report.get('stage'),
            'capture_events': report.get('capture_progress', {}).get('journal', {}).get('committed_events')})


class CaptureProgress:
    """Keep every measurement once, with bounded full-report checkpoint writes.

    The append-only journal is diagnostic evidence, never an acceptance report
    or a simulation-resume point. Flush each complete event so worker termination
    preserves observations after the last aggregate checkpoint. Original report
    and manifest contents remain authoritative for successful acceptance.
    """
    def __init__(self, root, report, *, interval_s=60.0, clock=time.monotonic):
        self.root, self.report, self.clock = Path(root), report, clock
        self.interval_s = float(interval_s)
        if not math.isfinite(self.interval_s) or self.interval_s <= 0:
            raise ValueError("Checkpoint interval must be finite and positive")
        self.path = self.root / f"capture_journal.{report['run_id']}.jsonl"
        self.stream = self.path.open("xb")
        self.digest, self.bytes, self.events = hashlib.sha256(), 0, 0
        self.last_checkpoint, self.checkpoints = self.clock(), 0
        self.manifests = {}
        self.append("header", {key: report.get(key) for key in (
            "urdf_sha256", "spec_sha256", "verifier_source_sha256", "motion_source_sha256",
            "execution_module_sha256", "capture_input_binding", "capture_workload")})
        self.checkpoint(force=True)

    def append(self, kind, data):
        event = {"sequence": self.events, "run_id": self.report["run_id"],
                 "kind": kind, "data": data}
        encoded = (json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        self.stream.write(encoded)
        self.stream.flush()
        self.digest.update(encoded)
        self.bytes += len(encoded)
        self.events += 1
        if kind != 'header':
            from urdf_core.native_progress import write_execution_progress
            write_execution_progress(self.root, self.report['run_id'], {'capture_events': self.events, 'kind': kind})
        return event["sequence"]

    def track_manifest(self, path, manifest):
        self.manifests[Path(path)] = manifest

    def checkpoint(self, *, force=False):
        if not force and self.clock() - self.last_checkpoint < self.interval_s:
            return False
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.checkpoints += 1
        self.report["capture_progress"] = {
            "policy": "append_native_evidence_periodic_checkpoint.v1",
            "checkpoint_interval_s": self.interval_s, "full_checkpoint_count": self.checkpoints,
            "journal": {"path": str(self.path), "committed_bytes": self.bytes,
                        "committed_events": self.events, "prefix_sha256": self.digest.hexdigest()},
            "scope": "Diagnostic observations only; incomplete captures never become accepted or resumable simulations."}
        for path, manifest in self.manifests.items():
            write_report(path, manifest)
        write_report(self.root / "report.json", self.report)
        self.last_checkpoint = self.clock()
        return True

    def close(self):
        if not self.stream.closed:
            try:
                self.checkpoint(force=True)
            finally:
                self.stream.close()


def inspect_capture_journal(report, root):
    """Read complete diagnostic events after interruption; never promote a pass.

    Verify the saved checkpoint prefix and source header before advertising any
    later complete lines. An interrupted final line is not a committed event.
    """
    binding = report.get("capture_progress", {}).get("journal")
    if not binding:
        return None
    path = Path(binding["path"])
    if path != Path(root) / f"capture_journal.{report['run_id']}.jsonl" or path.is_symlink():
        raise ValueError("Capture journal path does not match this execution")
    digest, size, count, prefix_verified = hashlib.sha256(), 0, 0, False
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            event = json.loads(line)
            if event.get("sequence") != count or event.get("run_id") != report["run_id"]:
                raise ValueError("Capture journal event identity or order changed")
            if count == 0:
                expected = {key: report.get(key) for key in (
                    "urdf_sha256", "spec_sha256", "verifier_source_sha256", "motion_source_sha256",
                    "execution_module_sha256", "capture_input_binding", "capture_workload")}
                if event.get("kind") != "header" or event.get("data") != expected:
                    raise ValueError("Capture journal source binding changed")
            digest.update(line)
            size += len(line)
            count += 1
            if size == binding["committed_bytes"]:
                if count != binding["committed_events"] or digest.hexdigest() != binding["prefix_sha256"]:
                    raise ValueError("Capture journal checkpoint prefix changed")
                prefix_verified = True
    if not prefix_verified:
        raise ValueError("Capture journal lost its committed checkpoint prefix")
    return {"path": str(path), "complete_events": count, "complete_bytes": size,
            "complete_prefix_sha256": digest.hexdigest(), "ignored_tail_bytes": path.stat().st_size-size,
            "checkpoint_prefix_verified": True, "acceptance_evidence": False}


def pose_error(expected, actual) -> dict:
    import numpy as np

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.shape != (4, 4) or actual.shape != (4, 4):
        raise ValueError("Poses must be 4 by 4 matrices")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Non-finite pose matrix")
    relative_rotation = expected[:3, :3].T @ actual[:3, :3]
    # atan2 remains accurate near zero; acos(trace) loses precision there.
    sine = np.linalg.norm([
        relative_rotation[2, 1] - relative_rotation[1, 2],
        relative_rotation[0, 2] - relative_rotation[2, 0],
        relative_rotation[1, 0] - relative_rotation[0, 1],
    ]) / 2.0
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    return {
        "translation_m": float(np.linalg.norm(expected[:3, 3] - actual[:3, 3])),
        "rotation_rad": float(math.atan2(sine, cosine)),
    }


def tensor_transform(value):
    """Physics tensors use [x,y,z,qx,qy,qz,qw], not scalar-first quaternions."""
    import numpy as np

    value = np.asarray(value, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("Invalid Isaac link transform")
    x, y, z, w = value[3:]
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-12:
        raise ValueError("Zero quaternion from Isaac")
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    result = np.eye(4)
    result[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    result[:3, 3] = value[:3]
    return result


def transform_points(matrix, points):
    import numpy as np

    points = np.asarray(points, dtype=float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def read_capture_camera(stage, product_path, pixels_shape, renderer_parameters):
    """Bind an RGB capture to actual USD and renderer camera readback.

    USD matrices and camera_params matrices use row vectors. Saved projection
    matrices are transposed to the same column-vector convention as link_poses.
    The renderer projection, rather than a focal-length reconstruction, includes
    its actual render-product aspect handling. No simulator step occurs here.
    """
    import numpy as np
    from pxr import Usd, UsdGeom, UsdRender

    def matrix(value, name):
        result = np.asarray(value, dtype=float)
        if result.size != 16 or not np.isfinite(result).all():
            raise ValueError("Invalid captured camera matrix: " + name)
        return result.reshape(4, 4)

    product = UsdRender.Product(stage.GetPrimAtPath(str(product_path)))
    if not product:
        raise ValueError("Capture render product is absent from the actual stage")
    targets = product.GetCameraRel().GetTargets()
    if len(targets) != 1:
        raise ValueError("Capture render product must bind exactly one camera")
    camera = UsdGeom.Camera(stage.GetPrimAtPath(targets[0]))
    if not camera or camera.GetProjectionAttr().Get() != UsdGeom.Tokens.perspective:
        raise ValueError("Diagnostic capture requires the actual perspective camera")
    time_code = Usd.TimeCode.Default()  # These capture cameras are static.
    world_from_camera = matrix(camera.ComputeLocalToWorldTransform(time_code), "USD world transform").T
    camera_from_world = np.linalg.inv(world_from_camera)
    native = renderer_parameters
    if not isinstance(native, dict) or native.get("cameraModel") != "pinhole":
        raise ValueError("Missing actual pinhole camera_params readback")
    view_raw = matrix(native.get("cameraViewTransform"), "renderer view")
    projection_raw = matrix(native.get("cameraProjection"), "renderer projection")
    # Some camera_params descriptions call this camera-to-world despite the
    # shipped translated-camera example containing the world-to-camera inverse.
    # Keep raw output and establish its meaning against actual USD transforms.
    matches = [name for name, expected in (("world_to_camera", camera_from_world),
                                           ("camera_to_world", world_from_camera))
               if np.allclose(view_raw.T, expected, rtol=1e-5, atol=1e-5)]
    if not matches:
        raise ValueError("Renderer camera transform does not match the actual USD camera")
    resolution = np.asarray(native.get("renderProductResolution"))
    expected_resolution = [int(pixels_shape[1]), int(pixels_shape[0])]
    if resolution.shape != (2,) or resolution.tolist() != expected_resolution:
        raise ValueError("Camera readback resolution differs from the captured RGB")
    usd_resolution = product.GetResolutionAttr().Get()
    if usd_resolution is not None and list(usd_resolution) != expected_resolution:
        raise ValueError("USD render product resolution differs from the captured RGB")
    focal = float(camera.GetFocalLengthAttr().Get())
    aperture = [float(camera.GetHorizontalApertureAttr().Get()), float(camera.GetVerticalApertureAttr().Get())]
    offsets = [float(camera.GetHorizontalApertureOffsetAttr().Get()), float(camera.GetVerticalApertureOffsetAttr().Get())]
    clipping = list(camera.GetClippingRangeAttr().Get())
    actual_parameters = [focal, *aperture, *offsets, *clipping]
    if (not np.isfinite(actual_parameters).all() or min([focal, *aperture, clipping[0]]) <= 0
            or clipping[1] <= clipping[0] or not np.isfinite(camera_from_world).all()):
        raise ValueError("Non-finite or invalid actual camera properties")
    renderer_values = {}
    for key in ("cameraFocalLength", "cameraAperture", "cameraApertureOffset", "cameraNearFar", "metersPerSceneUnit"):
        value = np.asarray(native.get(key), dtype=float)
        if not np.isfinite(value).all():
            raise ValueError("Missing or invalid renderer camera parameter: " + key)
        renderer_values[key] = value.tolist()
    return {"schema_version": 1, "passed": True, "status": "passed",
            "source": "actual_usd_and_render_product.v1",
            "timing": "after_accepted_rgb_render_before_next_command", "camera_prim_path": str(targets[0]),
            "render_product_path": str(product_path), "camera_is_static": True,
            "matrix_convention": "column_vectors; camera_forward_minus_Z; pixel_origin_top_left",
            "world_from_camera": world_from_camera.tolist(), "camera_from_world": camera_from_world.tolist(),
            "projection_matrix": projection_raw.T.tolist(), "resolution": expected_resolution,
            "usd": {"focal_length": focal, "aperture": aperture, "aperture_offset": offsets,
                    "clipping_range": clipping, "projection": str(camera.GetProjectionAttr().Get()),
                    "raw_frustum_projection_matrix": matrix(camera.GetCamera(time_code).frustum.ComputeProjectionMatrix(),
                                                               "USD projection").T.tolist(),
                    "aspect_ratio_conform_policy": str(product.GetAspectRatioConformPolicyAttr().Get()),
                    "pixel_aspect_ratio": product.GetPixelAspectRatioAttr().Get()},
            "renderer": {"cameraModel": native["cameraModel"], **renderer_values,
                         "cameraViewTransform_row_matrix": view_raw.tolist(),
                         "cameraProjection_row_matrix": projection_raw.tolist(),
                         "view_transform_matches": matches},
            "pixel_mapping": "u=(ndc_x+1)*width/2; v=(1-ndc_y)*height/2; no camera fitting"}


def capture_camera_evidence(stage, product_path, pixels_shape, annotator, setup_error=None):
    """Optional diagnostic evidence; unavailable camera data never fails motion."""
    try:
        if annotator is None:
            raise RuntimeError(setup_error or "camera_params annotator unavailable")
        return read_capture_camera(stage, product_path, pixels_shape, annotator.get_data())
    except Exception as error:
        return {"schema_version": 1, "passed": False, "status": "unavailable",
                "source": "actual_usd_and_render_product.v1", "render_product_path": str(product_path),
                "error": f"{type(error).__name__}: {error}", "marker_eligible": False,
                "scope": "Optional camera evidence only; original RGB/native motion acceptance is unchanged"}


def attach_camera_readback(rep, product, capture_resources):
    """Keep optional graph nodes alive until all native motion reads finish.

    Detaching an annotator can delete graph prims, invalidating Isaac 5.1's
    articulation handle even though its rigid bodies were not removed. This
    lifetime rule therefore applies to auxiliary annotators as well as RGB.
    """
    annotator = None
    try:
        annotator = rep.AnnotatorRegistry.get_annotator("camera_params")
        # Retain before attach: a partially attached graph also needs deferred
        # cleanup. Its failure remains auxiliary evidence, not motion failure.
        capture_resources.keep(annotator, product, auxiliary=True)
        annotator.attach([product])
        return annotator, None
    except Exception as error:
        return annotator, f"{type(error).__name__}: {error}"


def point_cloud_error(expected, actual) -> dict:
    """Symmetric vertex distances detect flipped meshes even with equal bboxes."""
    import numpy as np

    expected, actual = np.asarray(expected), np.asarray(actual)
    if not len(expected) or not len(actual):
        raise ValueError("Empty visual point cloud")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Non-finite visual point cloud")
    try:
        from scipy.spatial import cKDTree
        forward = cKDTree(actual).query(expected)[0]
        reverse = cKDTree(expected).query(actual)[0]
        sampled = False
    except ImportError:
        # Bound memory and work on runtimes without scipy. Record the sampling.
        def distances(source, target):
            source = source[np.linspace(0, len(source)-1, min(256, len(source)), dtype=int)]
            nearest = np.full(len(source), np.inf)
            for start in range(0, len(target), 1024):
                diff = source[:, None, :] - target[None, start:start+1024, :]
                nearest = np.minimum(nearest, np.sqrt(np.min(np.sum(diff*diff, axis=2), axis=1)))
            return nearest
        forward, reverse = distances(expected, actual), distances(actual, expected)
        sampled = True
    return {
        "max_distance_m": float(max(np.max(forward), np.max(reverse))),
        "rms_distance_m": float(np.sqrt(np.mean(np.concatenate([forward, reverse])**2))),
        "expected_vertices": len(expected), "actual_vertices": len(actual),
        "sampled": sampled,
    }


def capture_rgb(orchestrator, annotator, max_attempts: int = 6, *, settings=None,
                setting_evidence=None, rt_subframes=4):
    """Initialize/step the Replicator capture graph and wait for usable RGB.

    Rendering SimulationContext alone does not necessarily initialize a new
    offscreen render product. Zero delta time preserves the measured joint pose.
    The worker watchdog stops a blocked native render call.
    """
    import numpy as np
    if type(rt_subframes) is not int or rt_subframes < 1:
        raise ValueError('Render subframes must be a positive integer')

    last_error = "no RGB frame"
    for attempt in range(max_attempts):
        evidence = {"attempt": attempt + 1, "accepted": False}
        if setting_evidence is not None:
            setting_evidence.append(evidence)
        try:
            if settings is not None:
                # Stage/render setup can restore a preset after app startup.
                # Enforce the required setting at the actual capture boundary.
                evidence["before_render"] = configure_fractional_opacity(settings)
            orchestrator.step(rt_subframes=rt_subframes, pause_timeline=False,
                              delta_time=0.0, wait_for_render=True)
            if settings is not None:
                effective = {key: settings.get(key) for key in evidence["before_render"]}
                evidence["after_render"] = effective
                if any(value != True for value in effective.values()):
                    last_error = "Fractional opacity was reset during rendering"
                    evidence["error"] = last_error
                    continue
            data = annotator.get_data()
            if isinstance(data, dict):
                data = data.get("data")
            pixels = np.asarray(data)
            if pixels.size == 0 or pixels.ndim != 3 or pixels.shape[2] < 3:
                last_error = "RGB annotator is not ready"
                continue
            pixels = pixels[:, :, :3]
            if not np.isfinite(pixels).all() or np.ptp(pixels) == 0:
                last_error = "RGB annotator returned a blank or non-finite image"
                continue
            evidence["accepted"] = True
            return pixels.astype(np.uint8)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            evidence["error"] = last_error
    raise RuntimeError(f"Offscreen RGB capture failed after {max_attempts} attempts: "
                       f"{last_error}; see renderer/GPU diagnostics in kit.log")


def detach_render_product(annotator, product) -> None:
    """Pass paths: Isaac 5.1 only normalizes a bare HydraTexture, not a list."""
    path = product if isinstance(product, str) else product.path
    annotator.detach([path])


class CaptureResources:
    """Keep render prims alive until all PhysX measurements are complete.

    Isaac 5.1's Articulation prim-deletion callback clears its physics view even
    when an unrelated render-product prim is removed. Suspend completed captures
    using the documented Hydra update switch; defer any USD deletion to cleanup.
    """
    def __init__(self):
        self.products = []
        self.auxiliary = set()
        self.auxiliary_cleanup_errors = []

    def keep(self, annotator, product, *, auxiliary=False):
        self.products.append((annotator, product))
        if auxiliary:
            self.auxiliary.add((id(annotator), str(product.path)))

    def suspend(self, product):
        product.hydra_texture.set_updates_enabled(False)

    def close(self):
        errors = []
        # HydraTexture.destroy resets matching SyntheticData graphs (its path
        # can prefix later products). Disconnect every annotator before any
        # product deletion, otherwise later detach calls can see invalid nodes.
        for annotator, product in self.products:
            try:
                detach_render_product(annotator, product)
            except Exception as exc:
                error = f"detach: {type(exc).__name__}: {exc}"
                if (id(annotator), str(product.path)) in self.auxiliary:
                    self.auxiliary_cleanup_errors.append({"render_product": str(product.path), "error": error})
                else:
                    errors.append(error)
        destroyed = set()
        for _, product in self.products:
            # RGB and camera_params share the same actual render product.
            identity = str(product.path)
            if identity in destroyed:
                continue
            destroyed.add(identity)
            try:
                product.destroy()
            except Exception as exc:
                error = f"destroy: {type(exc).__name__}: {exc}"
                if all((id(a), str(p.path)) in self.auxiliary for a, p in self.products
                       if str(p.path) == identity):
                    self.auxiliary_cleanup_errors.append({"render_product": identity, "error": error})
                else:
                    errors.append(error)
        self.products.clear()
        self.auxiliary.clear()
        if errors:
            raise RuntimeError("Capture resource cleanup failed: " + "; ".join(errors))


def configure_fractional_opacity(settings) -> dict:
    """Enable native fractional MTL opacity; leave imported materials intact."""
    result = {}
    for renderer in ("raytracing", "pathtracing"):
        key = f"/rtx/{renderer}/fractionalCutoutOpacity"
        previous = settings.get(key)
        settings.set_bool(key, True)
        effective = settings.get(key)
        result[key] = {"previous": previous, "requested": True, "effective": effective}
        if effective != True:
            raise RuntimeError(f"Renderer did not enable fractional opacity: {key}")
    return result


def finalize_report(report: dict, returncode: int | None, timed_out: bool) -> dict:
    """A stale pass or a crashed/timed-out worker is never a successful check."""
    if timed_out:
        report["errors"].append("Isaac worker startup or progress stalled; its process group was terminated.")
        report.update(passed=False, status="timeout")
    elif returncode != 0 and not (returncode == 2 and (
            report.get('status') == 'failed' and report.get('stage') == 'complete'
            or report.get('status') == 'asset_rejected' and report.get('asset_error'))):
        report["errors"].append(f"Isaac worker exited with code {returncode}; see isaac_stdout.log.")
        report.update(passed=False, status="runtime_error")
    elif report.get("status") != "passed" or not report.get("poses") or report.get("errors"):
        report["passed"] = False
        if report.get("status") in {"starting", "running", "passed"}:
            report["status"] = "incomplete"
            report["errors"].append("Worker exited without completing all pose checks.")
    elif report.get("motion_video", {}).get("enabled") and not (
            report["motion_video"].get("passed") and report.get("motion_videos_passed")
            and report["motion_video"].get("manifest_sha256") and report["motion_video"].get("video_sha256")):
        report.update(passed=False, status="incomplete")
        report["errors"].append("Motion recording was enabled but no complete verified video evidence was published.")
    if not report.get("passed") and "motion_video" in report:
        report["motion_video"]["passed"] = False
        report["motion_videos_passed"] = False
        report["acceptance_complete"] = False
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--isaac-python", type=Path, default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--no-motion-video", dest="motion_video", action="store_false", default=True,
                        help="Diagnostic only: skip continuous motion video, so motion acceptance remains incomplete")
    parser.add_argument("--diagnostic-target-subtree", action="store_true",
                        help="Research only: additionally capture isolated target subtrees at core critic frames; does not change motion acceptance")
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--joint-duration", type=float, default=3.0)
    parser.add_argument("--ffmpeg", default=os.environ.get("NEOWORLD_FFMPEG", "ffmpeg"))
    parser.add_argument("--position-tolerance", type=float, default=2e-4)
    parser.add_argument("--rotation-tolerance", type=float, default=2e-3,
                        help="Radian tolerance for measured link rotations")
    parser.add_argument("--geometry-tolerance", type=float, default=2e-4)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--expected-verifier-sha256", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--expected-motion-sha256", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--expected-execution-sha256", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--input-binding-sha256", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-lock-fd", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.isaac_python is None:
        from urdf_core.environment import resolve_tool
        try:
            args.isaac_python = Path(resolve_tool("isaac_python"))
        except FileNotFoundError as error:
            parser.error(str(error))
    for name in ("position_tolerance", "rotation_tolerance", "geometry_tolerance"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("urdf", "spec", "out_dir", "isaac_python"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.video_fps <= 0 or not math.isfinite(args.joint_duration) or args.joint_duration <= 0:
        parser.error("Video fps and joint duration must be positive and finite")
    if round(args.video_fps * args.joint_duration) < 9:
        parser.error("Each joint video must contain at least nine rendered frames")
    if args.diagnostic_target_subtree and not args.motion_video:
        parser.error("--diagnostic-target-subtree requires the original motion videos")
    return args


def supervised_run(args) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "report.json"
    run_id = uuid.uuid4().hex
    report = {"passed": False, "status": "starting", "errors": [], "poses": [],
              "run_id": run_id, "urdf": str(args.urdf), "spec": str(args.spec),
              "verifier_source_sha256": VERIFIER_SOURCE_SHA256,
              "motion_source_sha256": MOTION_SOURCE_SHA256,
              "execution_module_sha256": EXECUTION_SOURCE_SHA256,
              "urdf_sha256": sha256_file(args.urdf) if args.urdf.is_file() else None,
              "spec_sha256": sha256_file(args.spec) if args.spec.is_file() else None,
              "motion_video": {"passed": False, "enabled": args.motion_video, "artifacts": []},
              "motion_videos_passed": False, "acceptance_complete": False}
    write_report(report_path, report)
    if not args.urdf.is_file() or not args.spec.is_file() or not args.isaac_python.is_file():
        report.update(status="environment_error")
        report["errors"].append("URDF, spec, or Isaac Python launcher does not exist.")
        write_report(report_path, report)
        return 2
    try:
        report["capture_workload"] = capture_workload(
            json.loads(args.spec.read_text(encoding="utf-8")), fps=args.video_fps,
            duration=args.joint_duration,
            motion_video=args.motion_video, diagnostic_target_subtree=args.diagnostic_target_subtree)
    except (ValueError, KeyError, TypeError) as exc:
        report.update(status="input_error")
        report["errors"].append(f"Cannot plan complete capture workload: {exc}")
        write_report(report_path, report)
        return 2
    binding_ref = create_input_binding(args.urdf, args.spec, args.out_dir / "capture_input_binding.json", run_id)
    report["capture_input_binding"] = binding_ref
    write_report(report_path, report)
    runtime = args.out_dir / "runtime"
    env = os.environ.copy()
    for key, folder in {"XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config",
                        "XDG_DATA_HOME": "data", "CUDA_CACHE_PATH": "cuda",
                        "__GL_SHADER_DISK_CACHE_PATH": "shader", "TMPDIR": "tmp"}.items():
        path = runtime / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    from urdf_core.execution import native_owner_environment
    env.update(native_owner_environment())
    command = [str(args.isaac_python), str(Path(__file__).resolve()), "--worker", "--run-id", run_id,
               "--isaac-python", str(args.isaac_python),
               "--expected-verifier-sha256", VERIFIER_SOURCE_SHA256,
               "--expected-motion-sha256", MOTION_SOURCE_SHA256,
               "--expected-execution-sha256", EXECUTION_SOURCE_SHA256,
               "--input-binding-sha256", binding_ref["sha256"],
               "--urdf", str(args.urdf), "--spec", str(args.spec), "--out-dir", str(args.out_dir),
               "--position-tolerance", str(args.position_tolerance),
               "--rotation-tolerance", str(args.rotation_tolerance),
               "--geometry-tolerance", str(args.geometry_tolerance),
               "--video-fps", str(args.video_fps), "--joint-duration", str(args.joint_duration),
               "--ffmpeg", args.ffmpeg]
    if not args.motion_video:
        command.append("--no-motion-video")
    if getattr(args, "diagnostic_target_subtree", False):
        command.append("--diagnostic-target-subtree")
    if not args.headless:
        command.append("--no-headless")
    started = time.monotonic()
    timed_out = False
    returncode = None
    supervisor_errors = []
    cache_policy = None
    try:
        from contextlib import nullcontext
        queued = getattr(args, '_gpu_queue_owned', False)
        with (nullcontext(None) if queued else gpu_worker_slot(getattr(args, "gpu_lock_fd", None))) as lock_fd:
            if queued:
                from urdf_core.gpu_queue import private_runtime_cache
                cache_policy = private_runtime_cache(args.out_dir, args.isaac_python, environ=env)
            else:
                cache_policy = prepare_runtime_cache(args.out_dir, args.isaac_python,
                                                     gpu_lock_fd=lock_fd, environ=env)
            report["runtime_cache_policy"] = cache_policy
            write_report(report_path, report)
            with (args.out_dir / "isaac_stdout.log").open("w", encoding="utf-8") as log:
                from urdf_core.isaac_admission import IsaacProcess, resource_key
                process = IsaacProcess(command, resource_key=resource_key(args.urdf, args.isaac_python), resource_output=args.out_dir, cwd=args.out_dir, env=env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                if getattr(args, '_gpu_task', None) is not None:
                    args._gpu_task.bind_process(process)
                from urdf_core.execution import wait_process, TechnicalExecutionError
                from urdf_core.native_progress import read_execution_progress
                try:
                    returncode = wait_process(process,
                        progress=lambda: read_execution_progress(args.out_dir, run_id))
                except TechnicalExecutionError as exc:
                    timed_out = True
                    returncode = process.poll()
                    supervisor_errors.append(str(exc))
    except (OSError, ValueError) as exc:
        report["errors"].append(f"Isaac worker setup or launch failed: {exc}")
        returncode = -1
        write_report(report_path, report)
    try:
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        if saved.get("run_id") == run_id:
            report = saved
        else:
            report["errors"].append("Worker report has a different run id.")
    except (OSError, ValueError) as exc:
        report["errors"].append(f"Cannot read worker report: {exc}")
    report["errors"].extend(supervisor_errors)
    if cache_policy is not None:
        report["runtime_cache_policy"] = cache_policy
    report = finalize_report(report, returncode, timed_out)
    if not report["passed"]:
        try:
            journal = inspect_capture_journal(report, args.out_dir)
            if journal is not None:
                report["interrupted_capture_journal"] = journal
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["capture_journal_diagnostic_error"] = str(exc)
    log_path = args.out_dir / "isaac_stdout.log"
    if not report["passed"] and log_path.exists():
        needles = ("No device could be created", "GPU Foundation is not initialized",
                   "NVIDIA driver is not loaded", "no CUDA-capable device", "no suitable CUDA GPU")
        diagnostics = [line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                       if any(needle in line for needle in needles)]
        if diagnostics:
            report["renderer_environment_errors"] = diagnostics[:12]
    report["elapsed_seconds"] = round(time.monotonic()-started, 3)
    report["worker_returncode"] = returncode
    write_report(report_path, report)
    print(f"Isaac check: {report['status']}; report: {report_path}")
    return 0 if report["passed"] else 2


def record_target_subtree_sidecar(args, spec, manifest, command, measure, rep,
                                  render_settings, capture_resources, visual_prims):
    """Opt-in native diagnostic captures; never enters the acceptance manifest.

    Run only after the complete original captures. Render products are retained
    until the enclosing worker finishes all native measurements. Failures here
    are reported separately, without changing existing motion/physics gates.
    """
    import omni.usd
    from PIL import Image
    from pxr import UsdGeom
    from isaac_motion import (TARGET_SUBTREE_POLICY, target_subtree_visibility,
                              target_subtree_source_frames)

    folder = args.out_dir / "diagnostic_target_subtree"
    path = folder / "manifest.json"
    if folder.exists() and any(folder.iterdir()):
        return {"status": "failed", "capture_complete": False, "marker_eligible": False,
                "research_only": True, "errors": ["Diagnostic sidecar requires an unused output directory"]}
    result = {"schema_version": 1, "research_only": True, "acceptance_decision": None,
              "status": "running", "capture_complete": False, "marker_eligible": False,
              "visibility_policy": TARGET_SUBTREE_POLICY,
              "scope": "new isolated-target diagnostic RGB, not the original closeup, collision evidence or semantic acceptance",
              "selection_policy": "exact original core critic_frame_indices and commanded positions; no new q samples",
              "camera_policy": "reuse original closeup planning parameters; independently read actual camera for each new render",
              "source_motion_manifest": {"path": str(args.out_dir / "motion_manifest.json"),
                                         "sha256": sha256_file(args.out_dir / "motion_manifest.json")},
              "urdf": manifest["urdf"], "spec": manifest["spec"], "sources": manifest["sources"],
              "joints": [], "errors": [], "rendered_frame_count": 0}
    try:
        folder.mkdir(parents=True, exist_ok=True)
        write_report(path, result)
        for joint_index, joint in enumerate(manifest["joints"]):
            if joint.get("synthetic"):
                continue
            visibility = target_subtree_visibility(spec, joint)
            if not set(visual_prims) <= set(visibility["measured_links"]):
                raise ValueError("Diagnostic visual link coverage differs from the native articulation")
            selected = target_subtree_source_frames(joint)
            closeup = next(view for view in joint["views"] if view["name"] == "closeup")["camera"]
            target_folder = folder / f"joint_{joint_index:03d}"
            target_folder.mkdir()
            camera = rep.create.camera(position=tuple(closeup["position"]),
                                       look_at=tuple(closeup["look_at"]),
                                       focal_length=closeup["focal_length"],
                                       horizontal_aperture=closeup["horizontal_aperture"],
                                       clipping_range=tuple(closeup["clipping_range"]))
            product = rep.create.render_product(camera, tuple(closeup["resolution"]), force_new=True)
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            # Keep before attach, including a partially created optional graph.
            capture_resources.keep(annotator, product, auxiliary=True)
            hidden_attributes = []
            entry = {key: joint[key] for key in ("name", "type", "parent", "child")}
            entry.update(view_name="target_subtree", visibility=visibility,
                         planned_camera=closeup, source_critic_frame_indices=joint["critic_frame_indices"],
                         link_objects={link["name"]: link["objects"] for link in spec["links"]},
                         frames=[], visibility_restored=False)
            result["joints"].append(entry)
            try:
                annotator.attach([product])
                camera_annotator, camera_setup_error = attach_camera_readback(rep, product, capture_resources)
                for link in visibility["hidden_links"]:
                    for prim in visual_prims.get(link, []):
                        attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
                        hidden_attributes.append((attr, attr.HasAuthoredValueOpinion(), attr.Get()))
                        attr.Set(UsdGeom.Tokens.invisible)
                for wide, original in selected:
                    for source in (wide, original):
                        if sha256_file(Path(source["path"])) != source["sha256"]:
                            raise ValueError("Original diagnostic source RGB changed")
                    positions = dict(original["positions"])
                    command(positions)
                    capture_checks = []
                    pixels = capture_rgb(rep.orchestrator, annotator, settings=render_settings,
                                         setting_evidence=capture_checks)
                    index = original["frame_index"]
                    ref = f"{joint['name']}::target_subtree::{index}"
                    label = f"diagnostic:{joint['name']}:target_subtree:{index:06d}"
                    actual, numeric = measure(positions, label)
                    if (set(actual) != set(visibility["measured_links"])
                            or set(numeric["links"]) != set(visibility["measured_links"])
                            or numeric.get("passed") is not True):
                        raise ValueError("Diagnostic frame lacks complete valid native link measurements")
                    actual_camera = capture_camera_evidence(omni.usd.get_context().get_stage(), product.path,
                        pixels.shape, None if camera_setup_error else camera_annotator, camera_setup_error)
                    pose_path = target_folder / f"native_pose_{index:06d}.json"
                    write_report(pose_path, {"schema_version": 1, "research_only": True,
                        "frame_ref": ref, "native_measurement": numeric,
                        "actual_camera": actual_camera, "visibility": visibility})
                    image_path = target_folder / f"frame_{index:06d}.png"
                    Image.fromarray(pixels).save(image_path)
                    entry["frames"].append({"frame_ref": ref, "frame_index": index,
                        "time_seconds": original["time_seconds"],
                        "time_scope": "source motion clip index, not continuous sidecar playback",
                        "path": str(image_path), "sha256": sha256_file(image_path),
                        "positions": positions, "measured_positions": numeric["measured_positions"],
                        "source_measured_positions": original["measured_positions"],
                        "link_poses": {name: value.tolist() for name, value in actual.items()},
                        "actual_camera": actual_camera, "numeric_passed": numeric["passed"],
                        "numeric_pose_name": label,
                        "native_pose": {"path": str(pose_path), "sha256": sha256_file(pose_path)},
                        "source_frames": [{"frame_ref": f"{joint['name']}::{view}::{index}",
                            "path": source["path"], "sha256": source["sha256"]}
                            for view, source in (("overview", wide), ("closeup", original))],
                        "capture_settings": capture_checks, "rendered": True})
                    result["rendered_frame_count"] += 1
                    write_report(path, result)
            finally:
                restore_errors = []
                for attr, authored, value in reversed(hidden_attributes):
                    try:
                        attr.Set(value) if authored else attr.Clear()
                    except Exception as error:
                        restore_errors.append(f"{type(error).__name__}: {error}")
                entry["visibility_restored"] = not restore_errors
                # No detach/destroy until every native read in this worker ends.
                capture_resources.suspend(product)
                if restore_errors:
                    raise RuntimeError("Diagnostic visibility restoration failed: " + "; ".join(restore_errors))
        frames = [frame for joint in result["joints"] for frame in joint["frames"]]
        result["capture_complete"] = True
        result["marker_eligible"] = bool(frames) and all(frame["actual_camera"].get("passed") is True for frame in frames)
        result["status"] = ("not_applicable_static" if not frames else
                            "complete" if result["marker_eligible"] else "camera_unavailable")
    except Exception as error:
        result.update(status="failed", capture_complete=False, marker_eligible=False)
        result["errors"].append(f"{type(error).__name__}: {error}")
    folder.mkdir(parents=True, exist_ok=True)
    write_report(path, result)
    return {"status": result["status"], "capture_complete": result["capture_complete"],
            "marker_eligible": result["marker_eligible"], "research_only": True,
            "manifest_path": str(path), "manifest_sha256": sha256_file(path),
            "rendered_frame_count": result["rendered_frame_count"]}


def record_motion_videos(args, report, spec, parsed, articulation, sim, rep,
                         render_settings, imported_local, dof_names, body_names,
                         capture_resources, visual_prims, progress=None, imported_triangles=None):
    """Record each joint in Isaac; every video frame has its own real capture."""
    import numpy as np
    import omni.usd
    from PIL import Image
    from isaac_motion import (bbox_corners, camera_for_points, closeup_region,
                              encode_video, motion_schedule, subtree_links,
                              capture_visibility, VISIBILITY_POLICY, choose_unoccluded_camera)
    from pxr import UsdGeom
    from urdf_core.urdf_validation import urdf_fk

    root = args.out_dir
    motion_dir = root / "motion"
    sources = motion_dir / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    urdf_snapshot, spec_snapshot = sources / "object.urdf", sources / "spec.json"
    urdf_snapshot.write_bytes(args.urdf.read_bytes())
    spec_snapshot.write_bytes(args.spec.read_bytes())
    joints = [j for j in spec["joints"] if j["type"] != "fixed"]
    sequences = joints or [{"name": "__static__", "type": "fixed", "parent": spec["root"],
                            "child": spec["root"], "synthetic": True}]
    manifest_path = root / "motion_manifest.json"
    manifest = {"schema_version": 1, "passed": False, "static": not joints,
                "urdf": {"path": str(urdf_snapshot), "sha256": sha256_file(urdf_snapshot), "snapshot_only": True},
                "spec": {"path": str(spec_snapshot), "sha256": sha256_file(spec_snapshot)},
                "sources": {"verifier_sha256": VERIFIER_SOURCE_SHA256, "motion_module_sha256": MOTION_SOURCE_SHA256,
                            "execution_module_sha256": EXECUTION_SOURCE_SHA256},
                "joint_names": [j["name"] for j in joints], "fps": args.video_fps,
                "duration_seconds": round(args.video_fps * args.joint_duration) / args.video_fps,
                "capture_method": "Each frame is newly rendered after commanding and measuring real PhysX joint/link state; kinematic playback, not a dynamics test.",
                "camera_policy": "Each clip uses fixed cameras fitted to actual presampled PhysX geometry; closeup includes child subtree and parent connection region.",
                "camera_evidence_policy": "actual_usd_and_render_product.v1",
                "visibility_policy": VISIBILITY_POLICY,
                "joints": [], "overview_video_path": None, "overview_video_sha256": None,
                "overview_frames": []}
    result = report["motion_video"]
    result.update(status="running", manifest_path=str(manifest_path), video_path=None)
    report["motion_pose_count"] = 0
    if progress is not None:
        progress.track_manifest(manifest_path, manifest)
    write_report(manifest_path, manifest)
    zeros = np.zeros((1, len(dof_names)), dtype=np.float32)

    def command(positions):
        if dof_names:
            targets = np.asarray([[positions.get(name, 0.0) for name in dof_names]], dtype=np.float32)
            articulation.set_joint_positions(targets)
            articulation.set_joint_velocities(zeros)
        for _ in range(2):
            sim.step(render=False, update_fabric=True)

    def measure(positions, name):
        measured = np.asarray(articulation.get_joint_positions()).reshape(-1) if dof_names else np.array([])
        tensors = np.asarray(articulation._physics_view.get_link_transforms(), dtype=float)[0]
        actual = {link: tensor_transform(value) for link, value in zip(body_names, tensors)}
        expected = urdf_fk(parsed, positions)
        q_error = max((abs(float(q) - positions.get(joint, 0.0)) for joint, q in zip(dof_names, measured)), default=0.0)
        item = {"name": name, "kind": "motion_frame", "positions": positions,
                "measured_positions": dict(zip(dof_names, measured.tolist())),
                "joint_position_max_error": q_error, "links": {}, "geometry": {},
                "geometry_scope": "Static imported local mesh equivalence was checked above; every frame independently checks the transforms moving those meshes.",
                "passed": bool(np.isfinite(measured).all() and q_error <= args.position_tolerance)}
        for link in body_names:
            error = pose_error(expected[link], actual[link])
            passed = error["translation_m"] <= args.position_tolerance and error["rotation_rad"] <= args.rotation_tolerance
            item["links"][link] = {"expected": expected[link].tolist(), "actual": actual[link].tolist(),
                                    "error": error, "passed": passed}
            item["passed"] &= passed
        if not item["passed"]:
            raise RuntimeError(f"Actual PhysX state differs from the commanded independent URDF pose: {name}")
        return actual, item

    # Measure before choosing cameras: the closeup cannot be based only on an
    # unverified FK estimate or a category-specific camera/part name.
    plans, all_bounds = [], []
    for joint in sequences:
        schedule = motion_schedule(joint, args.video_fps, args.joint_duration)
        descendants = subtree_links(parsed["joints"], joint["child"])
        moving_bounds, parent_points, pivot, observations = [], None, None, []
        visible = capture_visibility(spec, joint, "closeup")["visible_links"]
        triangle_links = imported_triangles or {}
        can_plan_occlusion = (not joint.get("synthetic") and all(
            link in triangle_links and triangle_links[link] is not None for link in visible))
        for frame in schedule["frames"]:
            positions = {name: frame["q"] if name == joint["name"] else 0.0 for name in dof_names}
            command(positions)
            actual, _ = measure(positions, f"camera_plan:{joint['name']}:{frame['frame_index']}")
            for link, points in imported_local.items():
                bounds = bbox_corners(transform_points(actual[link], points))
                all_bounds.append(bounds)
                if link in descendants:
                    moving_bounds.append(bounds)
            if parent_points is None:
                parent_points = transform_points(actual[joint["parent"]], imported_local[joint["parent"]])
                pivot = actual[joint["child"]][:3, 3].copy()
            if can_plan_occlusion and frame["frame_index"] in schedule["anchor_indices"]:
                world_triangles = {name: transform_points(actual[name], triangle_links[name].reshape(-1, 3)).reshape(-1, 3, 3)
                                   for name in visible}
                target = np.concatenate([world_triangles[name] for name in descendants]).mean(axis=1)
                others = [world_triangles[name] for name in visible if name not in descendants]
                observations.append((target, np.concatenate(others) if others else np.empty((0, 3, 3))))
        region, direction = closeup_region(np.concatenate(moving_bounds), parent_points, pivot)
        closeup = camera_for_points(region, direction, resolution=(512, 384))
        closeup, camera_plan = choose_unoccluded_camera(region, closeup, observations)
        closeup = dict(closeup, occlusion_planning=camera_plan)
        plans.append((joint, schedule, closeup, descendants))
        if progress is not None:
            progress.append("camera_plan", {"joint": joint["name"], "measured_frames": len(schedule["frames"]),
                                            "closeup": closeup, "descendant_links": descendants})
            progress.checkpoint()
    overview = camera_for_points(np.concatenate(all_bounds), resolution=(720, 540))
    overview_frame_paths = []
    for joint_index, (joint, schedule, closeup, descendants) in enumerate(plans):
        joint_entry = {key: joint[key] for key in ("name", "type", "parent", "child")}
        joint_entry.update(synthetic=bool(joint.get("synthetic")),
                           positions_range=schedule["positions_range"],
                           anchor_indices=schedule["anchor_indices"],
                           critic_frame_indices=schedule["critic_frame_indices"],
                           descendant_links=descendants, views=[])
        manifest["joints"].append(joint_entry)
        for view_name, camera_spec in (("overview", overview), ("closeup", closeup)):
            visibility = capture_visibility(spec, joint_entry, view_name)
            hidden_attributes = []
            folder = motion_dir / f"joint_{joint_index:03d}_{joint['name']}" / view_name
            folder.mkdir(parents=True, exist_ok=True)
            camera = rep.create.camera(position=tuple(camera_spec["position"]),
                                       look_at=tuple(camera_spec["look_at"]),
                                       focal_length=camera_spec["focal_length"],
                                       horizontal_aperture=camera_spec["horizontal_aperture"],
                                       clipping_range=tuple(camera_spec["clipping_range"]))
            product = rep.create.render_product(camera, tuple(camera_spec["resolution"]), force_new=True)
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([product])
            capture_resources.keep(annotator, product)
            camera_annotator, camera_setup_error = attach_camera_readback(rep, product, capture_resources)
            view = {"name": view_name, "camera": camera_spec, "video_path": str(folder / "motion.mp4"),
                    "video_sha256": None, "frames": [], "visibility": visibility,
                    "camera_evidence_lifetime": "retained_until_all_native_measurements_complete"}
            joint_entry["views"].append(view)
            try:
                for link in visibility["hidden_links"]:
                    for prim in visual_prims.get(link, []):
                        attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
                        hidden_attributes.append((attr, attr.HasAuthoredValueOpinion(), attr.Get()))
                        attr.Set(UsdGeom.Tokens.invisible)
                for frame in schedule["frames"]:
                    positions = {name: frame["q"] if name == joint["name"] else 0.0 for name in dof_names}
                    command(positions)
                    evidence = []
                    pixels = capture_rgb(rep.orchestrator, annotator, settings=render_settings,
                                         setting_evidence=evidence)
                    # Read after the actual render: the manifest describes the
                    # accepted RGB frame, not merely a pre-render command.
                    label = f"motion:{joint['name']}:{view_name}:{frame['frame_index']:06d}"
                    actual, numeric = measure(positions, label)
                    actual_camera = capture_camera_evidence(omni.usd.get_context().get_stage(), product.path,
                        pixels.shape, None if camera_setup_error else camera_annotator, camera_setup_error)
                    report["poses"].append(numeric)
                    report["motion_pose_count"] += 1
                    numeric_event = progress.append("native_pose", numeric) if progress is not None else None
                    path = folder / f"frame_{frame['frame_index']:06d}.png"
                    Image.fromarray(pixels).save(path)
                    captured = {"path": str(path), "sha256": sha256_file(path),
                                "frame_index": frame["frame_index"], "time_seconds": frame["time_seconds"],
                                "positions": positions, "measured_positions": numeric["measured_positions"],
                                "link_poses": {name: value.tolist() for name, value in actual.items()},
                                "actual_camera": actual_camera,
                                "numeric_passed": numeric["passed"], "numeric_pose_name": label,
                                "capture_settings": evidence, "rendered": True}
                    view["frames"].append(captured)
                    if progress is not None:
                        # Full transforms/q live in native_pose once. The final
                        # manifest still retains its original complete records.
                        progress.append("motion_frame", {"joint": joint["name"], "view": view_name,
                            "native_pose_event": numeric_event, **{key: value for key, value in captured.items()
                                if key not in {"link_poses", "measured_positions"}}})
                    if view_name == "overview":
                        overview_frame_paths.append(path)
                        manifest["overview_frames"].append({"joint": joint["name"], "path": str(path),
                            "sha256": captured["sha256"], "joint_frame_index": frame["frame_index"],
                            "frame_index": len(overview_frame_paths) - 1,
                            "time_seconds": (len(overview_frame_paths) - 1) / args.video_fps})
                    if progress is not None:
                        progress.checkpoint()
            finally:
                for attr, authored, value in hidden_attributes:
                    if authored:
                        attr.Set(value)
                    else:
                        attr.Clear()
                capture_resources.suspend(product)
            artifact = encode_video([f["path"] for f in view["frames"]], view["video_path"],
                                    args.video_fps, args.ffmpeg, root=root)
            artifact.update(joint=joint["name"], view=view_name)
            result["artifacts"].append(artifact)
            view["video_sha256"] = artifact["sha256"]
            view["video_probe"] = artifact["probe"]
            if progress is not None:
                progress.append("motion_view_complete", artifact)
                progress.checkpoint()
            else:
                write_report(manifest_path, manifest)
                write_report(root / "report.json", report)
    aggregate = encode_video(overview_frame_paths, motion_dir / "overview_all.mp4",
                             args.video_fps, args.ffmpeg, root=root)
    aggregate.update(joint=None, view="all_overview")
    result["artifacts"].append(aggregate)
    manifest.update(passed=True, overview_video_path=aggregate["video_path"],
                    overview_video_sha256=aggregate["sha256"],
                    rendered_frame_count=report["motion_pose_count"])
    write_report(manifest_path, manifest)
    result.update(passed=True, status="passed", manifest_sha256=sha256_file(manifest_path),
                  video_path=aggregate["video_path"], video_sha256=aggregate["sha256"],
                  rendered_frame_count=report["motion_pose_count"])
    report["motion_videos_passed"] = True
    report["acceptance_complete"] = True
    if progress is not None:
        progress.append("motion_complete", {"manifest_sha256": result["manifest_sha256"],
                                           "video_sha256": result["video_sha256"],
                                           "rendered_frame_count": report["motion_pose_count"]})
        progress.checkpoint(force=True)
    if getattr(args, "diagnostic_target_subtree", False):
        report["diagnostic_target_subtree"] = record_target_subtree_sidecar(
            args, spec, manifest, command, measure, rep, render_settings,
            capture_resources, visual_prims)



def _target_subtree_directory_identity(path, capture_root):
    if (path.is_symlink() or path.parent.is_symlink()
            or not path.resolve().is_relative_to(capture_root.resolve())):
        raise ValueError("Diagnostic sidecar manifest is outside its capture output")
    parent = path.parent.stat()
    return parent.st_dev, parent.st_ino


def _bound_target_subtree_manifest(report, manifest_path, manifest_sha256):
    """Read a sidecar only when its existing publication chain is intact."""
    reference = report.get("diagnostic_target_subtree")
    if reference is None:
        return None
    if not reference.get("manifest_path"):
        if reference.get("capture_complete") or reference.get("marker_eligible"):
            raise ValueError("Completed diagnostic sidecar has no manifest reference")
        return None  # An optional capture may fail before producing a manifest.
    path = Path(reference["manifest_path"])
    expected = manifest_path.parent / "diagnostic_target_subtree" / "manifest.json"
    if path.resolve() != expected.resolve():
        raise ValueError("Diagnostic sidecar manifest is outside its capture output")
    parent_identity = _target_subtree_directory_identity(path, manifest_path.parent)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != reference.get("manifest_sha256"):
        raise ValueError("Diagnostic sidecar manifest hash changed before finalization")
    sidecar = json.loads(raw)
    source = sidecar.get("source_motion_manifest", {})
    if (not source.get("path") or Path(source["path"]).resolve() != manifest_path.resolve()
            or source.get("sha256") != manifest_sha256):
        raise ValueError("Diagnostic sidecar does not bind the exact prepublication motion manifest")
    return path, sidecar, digest, parent_identity


def _write_bound_target_subtree_manifest(sidecar, capture_root, final_sha):
    """Publish through the checked directory, never a replacement parent path."""
    path, value, original_sha, parent_identity = sidecar

    def check_directory():
        if _target_subtree_directory_identity(path, capture_root) != parent_identity:
            raise ValueError("Diagnostic sidecar directory changed during publication")

    check_directory()
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = f".{path.name}.{uuid.uuid4().hex}.tmp"
    created = False
    try:
        parent = os.fstat(directory)
        if (parent.st_dev, parent.st_ino) != parent_identity:
            raise ValueError("Diagnostic sidecar directory changed during publication")

        def check_content(expected_sha=original_sha):
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            with os.fdopen(descriptor, "rb") as stream:
                if hashlib.sha256(stream.read()).hexdigest() != expected_sha:
                    raise ValueError("Diagnostic sidecar changed during execution evidence publication")

        check_content()
        value["source_motion_manifest"]["sha256"] = final_sha
        raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o666, dir_fd=directory)
        created = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
        check_directory()
        check_content()
        # Both names are relative to the same open directory. A parent symlink
        # replacement after the checks cannot redirect this write outside it.
        os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        created = False
        check_directory()
        published_sha = hashlib.sha256(raw).hexdigest()
        check_content(published_sha)
        return published_sha
    finally:
        try:
            if created:
                os.unlink(temporary, dir_fd=directory)
        finally:
            os.close(directory)


def _sidecar_binding_error(report, error):
    """Keep failed evidence untouched and make only the auxiliary view unusable."""
    reference = report["diagnostic_target_subtree"]
    reference["binding_status"] = "invalid" if isinstance(error, ValueError) else "unavailable"
    reference["binding_error"] = type(error).__name__ + ":" + str(error)
    reference["marker_eligible"] = False


def publish_execution_evidence(out_dir, report, binding_ref, links, reasons):
    """Publish optional facts without changing existing numerical/visual gates."""
    try:
        manifest_path = Path(out_dir) / "motion_manifest.json"
        raw_manifest = manifest_path.read_bytes() if manifest_path.is_file() else None
        manifest = json.loads(raw_manifest) if raw_manifest is not None else None
        previous_sha = hashlib.sha256(raw_manifest).hexdigest() if raw_manifest is not None else None
        sidecar = None
        if manifest is not None:
            motion = report.get("motion_video", {})
            diagnostic = report.get("diagnostic_target_subtree")
            requires_reference = (diagnostic is not None and
                                  (diagnostic.get("manifest_path") or diagnostic.get("capture_complete")))
            if ((requires_reference or "manifest_sha256" in motion)
                    and motion.get("manifest_sha256") != previous_sha):
                error = ValueError("Report does not bind the exact prepublication motion manifest")
                if diagnostic is not None:
                    _sidecar_binding_error(report, error)
                # Do not replace an already incorrect declaration with a fresh hash.
                raise error
            try:
                sidecar = _bound_target_subtree_manifest(report, manifest_path, previous_sha)
            except Exception as exc:
                _sidecar_binding_error(report, exc)
        evidence = build_execution_evidence(binding_ref, report, manifest, links, reasons)
        path = Path(out_dir) / "execution_evidence.json"
        write_report(path, evidence)
        reference = execution_artifact(path)
        report["execution_evidence"] = reference
        if manifest is not None:
            if sha256_file(manifest_path) != previous_sha:
                error = ValueError("Motion manifest changed during execution evidence publication")
                if sidecar is not None:
                    _sidecar_binding_error(report, error)
                raise error
            manifest["execution_evidence"] = reference
            manifest.setdefault("sources", {})["execution_module_sha256"] = EXECUTION_SOURCE_SHA256
            write_report(manifest_path, manifest)
            final_sha = sha256_file(manifest_path)
            report["motion_video"]["manifest_sha256"] = final_sha
            if sidecar is not None:
                try:
                    # Only metadata is rebound; every captured RGB/state stays intact.
                    sidecar_sha = _write_bound_target_subtree_manifest(sidecar, manifest_path.parent, final_sha)
                    report["diagnostic_target_subtree"]["manifest_sha256"] = sidecar_sha
                    report["diagnostic_target_subtree"]["binding_status"] = "verified"
                except Exception as exc:
                    _sidecar_binding_error(report, exc)
    except Exception as exc:
        # Even an evidence writer/storage problem is not a new asset failure.
        report["execution_evidence_unavailable"] = type(exc).__name__ + ":" + str(exc)


def is_link_visual_mesh(prim, body):
    """Select visible mesh ownership by USD schemas, within one rigid body."""
    from pxr import UsdGeom, UsdPhysics
    if not prim.IsA(UsdGeom.Mesh):
        return False
    ancestor = prim
    while ancestor != body:
        if not ancestor or ancestor.HasAPI(UsdPhysics.CollisionAPI) or ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            return False
        ancestor = ancestor.GetParent()
    return True


def run_worker(args) -> int:
    import numpy as np
    from urdf_core.kinematics import validate_spec, pose_samples, fk
    from urdf_core.urdf_validation import read_urdf, urdf_fk, read_obj

    capture_resources = CaptureResources()
    report_path = args.out_dir / "report.json"
    report = {"passed": False, "status": "running", "errors": [], "poses": [],
              "run_id": args.run_id, "urdf": str(args.urdf), "spec": str(args.spec),
              "verifier_source_sha256": VERIFIER_SOURCE_SHA256,
              "motion_source_sha256": MOTION_SOURCE_SHA256,
              "execution_module_sha256": EXECUTION_SOURCE_SHA256,
              "urdf_sha256": sha256_file(args.urdf), "spec_sha256": sha256_file(args.spec),
              "motion_video": {"passed": False, "enabled": args.motion_video, "artifacts": []},
              "motion_videos_passed": False, "acceptance_complete": False,
              "supervisor_verifier_source_sha256": args.expected_verifier_sha256,
              "screenshots": [], "stage": "input_validation",
              "tolerances": {"position_m": args.position_tolerance,
                             "rotation_rad": args.rotation_tolerance,
                             "geometry_m": args.geometry_tolerance}}
    app, progress = None, None
    execution_reasons, execution_links = [], []
    binding_ref, input_binding = None, None
    try:
        if getattr(args, "expected_execution_sha256", None) != EXECUTION_SOURCE_SHA256:
            raise ValueError("Auxiliary execution helper lacks matching supervisor source binding")
        binding_path = args.out_dir / "capture_input_binding.json"
        binding_ref = {"path": str(binding_path), "sha256": getattr(args, "input_binding_sha256", None)}
        input_binding = verify_input_binding(binding_ref, args.run_id)
        report["capture_input_binding"] = binding_ref
    except Exception as exc:
        execution_reasons.append("pre_import_binding_unavailable:" + str(exc))
        # No nonexistent or unbound file is advertised as an artifact.
        if binding_ref and (not Path(binding_ref["path"]).is_file() or not binding_ref["sha256"]):
            binding_ref = None
    try:
        if args.expected_verifier_sha256 and args.expected_verifier_sha256 != VERIFIER_SOURCE_SHA256:
            raise RuntimeError("Verifier source changed between supervisor and worker startup; rerun the check")
        if args.expected_motion_sha256 and args.expected_motion_sha256 != MOTION_SOURCE_SHA256:
            raise RuntimeError("Motion helper source changed between supervisor and worker startup")
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        report["capture_workload"] = capture_workload(spec, fps=args.video_fps,
            duration=args.joint_duration,
            motion_video=args.motion_video, diagnostic_target_subtree=args.diagnostic_target_subtree)
        progress = CaptureProgress(args.out_dir, report)
        spec = validate_spec(spec)
        parsed = read_urdf(args.urdf)
        if set(parsed["links"]) != {link["name"] for link in spec["links"]}:
            raise ValueError("Spec and URDF link names differ")
        samples = pose_samples(spec)
        if not samples:
            raise ValueError("No pose samples provided")
        screenshot_indices = screenshot_sample_indices(samples)
        report["screenshot_pose_names"] = [samples[index]["name"] for index in screenshot_indices]
        expected_local, expected_visuals = {}, {}
        for name, link in parsed["links"].items():
            clouds = []
            for visual_index, visual in enumerate(link["visuals"]):
                if "mesh" not in visual:
                    raise ValueError("Isaac geometry check requires exported OBJ mesh visuals")
                points = read_obj(visual["mesh"])["vertices"] * visual["scale"]
                cloud = transform_points(visual["origin"], points)
                clouds.append(cloud)
                expected_visuals[f"{name}::{visual_index}"] = cloud
            if clouds:
                expected_local[name] = np.concatenate(clouds)
        if not expected_local:
            raise ValueError("No mesh visuals found; geometry verification would be empty")
        report["stage"] = "simulation_app_launch"
        progress.checkpoint(force=True)
        # Import Kit only after arguments, spec and expected geometry are validated.
        from isaacsim import SimulationApp
        portable = args.out_dir / "runtime" / "kit"
        portable.mkdir(parents=True, exist_ok=True)
        extra = ["--portable-root", str(portable),
                 f"--/app/tokens/cache={args.out_dir / 'runtime' / 'cache'}",
                 f"--/app/tokens/data={args.out_dir / 'runtime' / 'data'}",
                 f"--/app/tokens/logs={args.out_dir}",
                 "--/app/extensions/fsWatcherEnabled=0", "--/app/extensions/registryEnabled=0",
                 "--/app/extensions/syncRegistryOnStartup=0", "--/app/extensions/preferLocalVersions=1",
                 f"--/log/file={args.out_dir / 'kit.log'}",
                 "--/physics/updateToUsd=true", "--/physics/updateVelocitiesToUsd=true"]
        # SimulationApp also parses sys.argv; keep script options out of Kit.
        from urdf_core.isaac_admission import thread_args
        extra.extend(thread_args())
        sys.argv = [sys.argv[0], "--portable-root", str(portable)]
        app = SimulationApp({"headless": args.headless, "create_new_stage": False,
                             "width": 1024, "height": 768, "multi_gpu": False,
                             "extra_args": extra})
        from urdf_core.native_progress import write_execution_progress
        write_execution_progress(args.out_dir, args.run_id, {'stage': 'runtime_ready'})
        import carb.settings
        render_settings = carb.settings.get_settings()
        report["render_configuration"] = {
            "imported_material_colors_or_opacity_modified": False,
            "capture_checks": [],
        }
        import omni.kit.commands
        import omni.usd
        import omni.replicator.core as rep
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import create_new_stage
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics

        enable_extension("isaacsim.asset.importer.urdf")
        app.update()
        create_new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityMagnitudeAttr().Set(0.0)
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx_scene.CreateEnableGPUDynamicsAttr(False)
        physx_scene.CreateBroadphaseTypeAttr("MBP")
        report["stage"] = "urdf_import"
        progress.checkpoint(force=True)
        ok, config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not ok:
            raise RuntimeError("URDFCreateImportConfig failed")
        config.merge_fixed_joints = False
        config.fix_base = True
        config.import_inertia_tensor = True
        config.distance_scale = 1.0
        config.collision_from_visuals = False
        config.self_collision = False
        ok, root_path = omni.kit.commands.execute(
            "URDFParseAndImportFile", urdf_path=str(args.urdf), import_config=config,
            get_articulation_root=True)
        if not ok or not root_path:
            raise RuntimeError("URDF import failed")
        # Imported visual/collision meshes can be instance proxies. Make this
        # isolated smoke stage editable so collision disabling reaches every mesh.
        while True:
            instances = [prim for prim in stage.Traverse() if prim.IsInstance()]
            if not instances:
                break
            for prim in instances:
                prim.SetInstanceable(False)
        bodies = {}
        disabled_collisions = 0
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                disabled_collisions += 1
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
                if prim.GetName() in parsed["links"]:
                    if prim.GetName() in bodies:
                        raise RuntimeError(f"Ambiguous imported link name {prim.GetName()}")
                    bodies[prim.GetName()] = prim
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(False)
        report["disabled_collision_prims"] = disabled_collisions
        missing = set(parsed["links"]) - set(bodies)
        if missing:
            from urdf_core.content_errors import AssetContentError
            raise AssetContentError(f"Importer omitted rigid link bodies: {sorted(missing)}", stage='URDF_import')
        imported_local, visual_prims, imported_visuals, imported_triangles = {}, {}, {}, {}
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for name, body in bodies.items():
            body_matrix = np.asarray(cache.GetLocalToWorldTransform(body), dtype=float).T
            clouds, triangles, triangle_meshes_only = [], [], True
            for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if not is_link_visual_mesh(prim, body):
                    continue
                visual_prims.setdefault(name, []).append(prim)
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                mesh_world = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
                cloud = transform_points(np.linalg.inv(body_matrix) @ mesh_world, points)
                clouds.append(cloud)
                mesh = UsdGeom.Mesh(prim)
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
                if len(counts) and np.all(counts == 3) and len(indices) == len(counts) * 3:
                    triangles.append(cloud[indices.reshape(-1, 3)])
                else:
                    # Do not invent triangulation of an arbitrary polygon. This
                    # rendering aid can fall back while native checks stay intact.
                    triangle_meshes_only = False
                imported_visuals.setdefault(name, []).append({"path": str(prim.GetPath()), "points": cloud})
            if clouds:
                imported_local[name] = np.concatenate(clouds)
                imported_triangles[name] = np.concatenate(triangles) if triangles and triangle_meshes_only else None
        if set(imported_local) != set(expected_local):
            stage.Export(str(args.out_dir / "import_debug.usda"))
            from urdf_core.content_errors import AssetContentError
            raise AssetContentError(f"Imported visual link coverage {sorted(imported_local)} differs "
                                    f"from URDF visuals {sorted(expected_local)}; see import_debug.usda", stage='URDF_import')
        try:
            if not input_binding or execution_reasons:
                raise ValueError("Pre-import source binding is unavailable")
            # A queued source may have changed before import; never retroactively
            # associate newly read bytes with an older source ledger.
            verify_input_binding(binding_ref, args.run_id)
            execution_links = match_imported_visuals(input_binding, expected_visuals,
                imported_visuals, {name: str(body.GetPath()) for name, body in bodies.items()},
                args.geometry_tolerance)
        except Exception as exc:
            execution_reasons.append("per_mesh_binding_unavailable:" + type(exc).__name__ + ":" + str(exc))
        sim = SimulationContext(physics_dt=1/240, rendering_dt=1/240,
                                stage_units_in_meters=1.0, physics_prim_path="/physicsScene",
                                set_defaults=False)
        sim.initialize_physics()
        sim.play()
        sim.step(render=False, update_fabric=True)
        articulation = Articulation(str(root_path), name="urdf_smoke")
        articulation.initialize()
        if not articulation.is_physics_handle_valid():
            raise RuntimeError("Imported articulation has no valid physics handle")
        dof_names, body_names = list(articulation.dof_names), list(articulation.body_names)
        expected_dofs = {j["name"] for j in spec["joints"] if j["type"] != "fixed"}
        if set(dof_names) != expected_dofs:
            from urdf_core.content_errors import AssetContentError
            raise AssetContentError(f"Imported DOFs differ: {dof_names}, expected {sorted(expected_dofs)}", stage='URDF_import')
        if set(body_names) != set(parsed["links"]):
            from urdf_core.content_errors import AssetContentError
            raise AssetContentError("Imported physics body names differ from URDF links", stage='URDF_import')
        report["dof_names"] = dof_names
        report["body_names"] = body_names
        zeros = np.zeros((1, len(dof_names)), dtype=np.float32)
        if dof_names:
            articulation.set_gains(kps=zeros, kds=zeros)
        # A fixed root is checked against identity, never silently refitted per pose.
        report["stage"] = "pose_verification"
        from urdf_core.isaac_admission import initialized
        initialized()
        progress.checkpoint(force=True)
        for sample_index, sample in enumerate(samples):
            positions = sample["positions"]
            targets = np.asarray([[positions.get(name, 0.0) for name in dof_names]], dtype=np.float32)
            if dof_names:
                articulation.set_joint_positions(targets)
                articulation.set_joint_velocities(zeros)
            # Physics backend forward kinematics follows the commanded state.
            for _ in range(2):
                sim.step(render=False, update_fabric=True)
            measured_q = np.asarray(articulation.get_joint_positions()).reshape(-1) if dof_names else np.array([])
            transforms = np.asarray(articulation._physics_view.get_link_transforms(), dtype=float)[0]
            actual = {name: tensor_transform(value) for name, value in zip(body_names, transforms)}
            expected = urdf_fk(parsed, positions)
            spec_expected = fk(spec, positions)
            result = {"name": sample["name"], "positions": positions,
                      "measured_positions": dict(zip(dof_names, measured_q.tolist())),
                      "links": {}, "geometry": {}, "passed": True}
            if dof_names and np.max(np.abs(measured_q-targets[0])) > args.position_tolerance:
                result["passed"] = False
                report["errors"].append(f"{sample['name']}: measured joint states differ from commanded radians/meters")
            for name in parsed["links"]:
                error = pose_error(expected[name], actual[name])
                source_error = pose_error(spec_expected[name], expected[name])
                accepted = (error["translation_m"] <= args.position_tolerance
                            and error["rotation_rad"] <= args.rotation_tolerance
                            and source_error["translation_m"] <= args.position_tolerance
                            and source_error["rotation_rad"] <= args.rotation_tolerance)
                result["links"][name] = {"expected": expected[name].tolist(), "actual": actual[name].tolist(),
                                         "error": error, "spec_to_urdf_error": source_error, "passed": accepted}
                result["passed"] &= accepted
            for name, points in expected_local.items():
                error = point_cloud_error(transform_points(expected[name], points),
                                          transform_points(actual[name], imported_local[name]))
                error["passed"] = error["max_distance_m"] <= args.geometry_tolerance
                result["geometry"][name] = error
                result["passed"] &= error["passed"]
            if not result["passed"]:
                report["errors"].append(f"{sample['name']}: imported link poses or visual geometry differ from independent URDF FK")
            report["poses"].append(result)
            progress.append("native_pose", result)
            progress.checkpoint()
        report["numerical_passed"] = all(pose["passed"] for pose in report["poses"])
        stage.Export(str(args.out_dir / "imported_pose_scene.usda"))
        # Complete numerical checks before rendering: a GPU failure must preserve
        # the independent CPU physics evidence while keeping overall passed false.
        report["stage"] = "screenshot_capture"
        progress.checkpoint(force=True)
        light = UsdLux.DomeLight.Define(stage, "/SmokeLight")
        light.CreateIntensityAttr(1000.0)
        pose_expected_clouds = []
        for sample in samples:
            sample_frames = urdf_fk(parsed, sample["positions"])
            pose_expected_clouds.extend(transform_points(sample_frames[name], points)
                                        for name, points in expected_local.items())
        all_points = np.concatenate(pose_expected_clouds)
        center = (all_points.min(axis=0)+all_points.max(axis=0))/2
        radius = max(float(np.linalg.norm(all_points.max(axis=0)-all_points.min(axis=0))), 0.1)
        camera = rep.create.camera(position=tuple(center+radius*np.array([1.4,-1.8,1.2])),
                                   look_at=tuple(center), clipping_range=(0.001, radius*30+10))
        product = rep.create.render_product(camera, (1024, 768))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([product])
        capture_resources.keep(rgb, product)
        report["render_configuration"]["fractional_cutout_opacity"] = configure_fractional_opacity(render_settings)
        for sample_index, sample in enumerate(samples):
            if sample_index in screenshot_indices:
                targets = np.asarray([[sample["positions"].get(name, 0.0) for name in dof_names]], dtype=np.float32)
                if dof_names:
                    articulation.set_joint_positions(targets)
                    articulation.set_joint_velocities(zeros)
                for _ in range(2):
                    sim.step(render=False, update_fabric=True)
                capture_settings = []
                report["render_configuration"]["capture_checks"].append(
                    {"pose": sample["name"], "attempts": capture_settings})
                pixels = capture_rgb(rep.orchestrator, rgb, settings=render_settings,
                                     setting_evidence=capture_settings)
                from PIL import Image
                image_path = args.out_dir / f"pose_{sample_index:03d}.png"
                Image.fromarray(pixels).save(image_path)
                report["screenshots"].append({"pose": sample["name"], "path": str(image_path)})
                progress.append("screenshot", {"pose": sample["name"], "sample_index": sample_index,
                    "path": str(image_path), "sha256": sha256_file(image_path),
                    "capture_settings": capture_settings})
                progress.checkpoint()
        capture_resources.suspend(product)
        if args.motion_video:
            report["stage"] = "motion_video_capture"
            progress.checkpoint(force=True)
            record_motion_videos(args, report, spec, parsed, articulation, sim, rep,
                                 render_settings, imported_local, dof_names, body_names,
                                 capture_resources, visual_prims, progress=progress,
                                 imported_triangles=imported_triangles)
        else:
            report["motion_video"].update(status="disabled_diagnostic_only", passed=False)
        report["numerical_passed"] = all(pose["passed"] for pose in report["poses"])
        report["passed"] = not report["errors"] and bool(report["screenshots"])
        if args.motion_video:
            report["passed"] &= report["motion_video"]["passed"]
        report["status"] = "passed" if report["passed"] else "failed"
        report["stage"] = "complete"
    except Exception as exc:
        status = "render_error" if report.get("stage") in {"screenshot_capture", "motion_video_capture"} else "runtime_error"
        from urdf_core.content_errors import AssetContentError
        if isinstance(exc, AssetContentError):
            status = 'asset_rejected'
            report['asset_error'] = exc.evidence
        report.update(passed=False, status=status)
        report["motion_video"]["passed"] = False
        report["motion_videos_passed"] = False
        report["acceptance_complete"] = False
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()
    finally:
        try:
            capture_resources.close()
        except Exception as exc:
            report.update(passed=False, status="render_error", acceptance_complete=False,
                          motion_videos_passed=False)
            report["motion_video"]["passed"] = False
            report["errors"].append(str(exc))
        if capture_resources.auxiliary_cleanup_errors:
            report["camera_evidence_cleanup_errors"] = capture_resources.auxiliary_cleanup_errors
        if progress is not None:
            try:
                progress.append("worker_finished", {"status": report["status"], "stage": report["stage"],
                                                   "errors": report["errors"]})
                progress.close()
            except Exception as exc:
                report.update(passed=False, status="runtime_error", acceptance_complete=False,
                              motion_videos_passed=False)
                report["motion_video"]["passed"] = False
                report["errors"].append("Capture evidence checkpoint failed: " + str(exc))
        publish_execution_evidence(args.out_dir, report, binding_ref, execution_links, execution_reasons)
        write_report(report_path, report)
        if app is not None:
            app.close()
    return 0 if report["passed"] else 2


def queued_run(args) -> int:
    if getattr(args, 'gpu_lock_fd', None) is not None or not all(
            p.is_file() for p in (args.urdf, args.spec, args.isaac_python)):
        return supervised_run(args)
    from urdf_core.gpu_queue import (GpuTask, GpuOutOfMemory, GpuResourceError,
                                    oom_evidence, archive_attempt, record_resource_error)
    task = GpuTask(args.out_dir / 'gpu_admission.json',
        identity={'urdf': sha256_file(args.urdf), 'spec': sha256_file(args.spec),
                  'verifier': VERIFIER_SOURCE_SHA256, 'motion': MOTION_SOURCE_SHA256})
    def execute(_fd):
        args._gpu_queue_owned = True
        args._gpu_task = task
        code = supervised_run(args)
        if code and oom_evidence(args.out_dir / 'isaac_stdout.log', args.out_dir / 'report.json'):
            archive_attempt(args.out_dir, task.state['oom_count'])
            raise GpuOutOfMemory('Isaac kinematics GPU OOM; retry the same candidate')
        if code:
            report = json.loads((args.out_dir / 'report.json').read_text())
            if report.get('status') in ('timeout', 'runtime_error', 'render_error', 'environment_error', 'incomplete'):
                from urdf_core.execution import TechnicalExecutionError
                archive_attempt(args.out_dir, task.state.get('technical_failures', 0), prefix='technical_attempts')
                raise TechnicalExecutionError('Isaac kinematics worker incomplete; retry the same candidate')
        return code
    try:
        return task.run(execute)
    except GpuResourceError as error:
        record_resource_error(args.out_dir, error)
        return 2


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.worker:
        from urdf_core.execution import start_owner_monitor
        start_owner_monitor()
        return run_worker(args)
    return queued_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
