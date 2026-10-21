"""Pure-Python planning, camera framing and encoding for real Isaac captures.

This module never synthesizes motion images. Callers supply frames rendered from
the simulator and measured link geometry; all output paths stay in one run.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

MOTION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
VISIBILITY_POLICY = "full_overview_and_branch_isolated_closeup.v1"
TARGET_SUBTREE_POLICY = "research_target_subtree_only.v1"
CAPTURE_BUDGET_POLICY = "native_capture_workload.v1"


def screenshot_sample_indices(samples) -> list[int]:
    """Rest, both sampled extremes of every joint, and the final coupled pose."""
    if not samples:
        return []
    selected, isolated, combined = {0}, {}, []
    for index, sample in enumerate(samples):
        positions = sample["positions"]
        if len(positions) == 1:
            joint, value = next(iter(positions.items()))
            isolated.setdefault(joint, []).append((float(value), index))
        elif len(positions) > 1:
            combined.append(index)
    for values in isolated.values():
        values.sort()
        selected.update((values[0][1], values[-1][1]))
    if combined:
        selected.add(combined[-1])
    return sorted(selected)


def _motion_frame_count(fps, duration):
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("video fps must be a positive integer")
    try:
        valid_duration = (not isinstance(duration, bool) and isinstance(duration, (int, float))
                          and math.isfinite(duration) and duration > 0)
    except OverflowError:
        valid_duration = False
    if not valid_duration:
        raise ValueError("joint duration must be finite and positive")
    try:
        count = round(fps * duration)
    except (OverflowError, ValueError) as error:
        raise ValueError("Motion frame count must be finite") from error
    if count < 9:
        raise ValueError("fps times joint duration must provide at least nine frames")
    return count


def capture_workload(spec, *, fps=12, duration=3.0, motion_video=True,
                     diagnostic_target_subtree=False):
    """Count the full native capture plan without imposing a runtime limit."""
    from urdf_core.kinematics import pose_samples
    if type(motion_video) is not bool or type(diagnostic_target_subtree) is not bool:
        raise ValueError('Capture feature flags must be booleans')
    if diagnostic_target_subtree and not motion_video:
        raise ValueError('Target-subtree diagnostics require motion video capture')
    frame_count = _motion_frame_count(fps, duration)
    # This is the worker's actual validated sampling function, including the
    # continuous-joint exception and coupled poses. Never budget invalid specs.
    samples = pose_samples(spec)
    source_digest = hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False,
                                              allow_nan=False).encode()).hexdigest()
    joints = [joint for joint in spec["joints"] if joint["type"] != "fixed"]
    sequence_count = max(1, len(joints)) if motion_video else 0
    camera_presamples = sequence_count * frame_count
    motion_frames = 2 * camera_presamples
    # The sidecar reuses nine paired core source indices, rendering one new
    # target-only view per joint. Both source RGBs are verified, not re-rendered.
    diagnostic_frames = 9 * len(joints) if diagnostic_target_subtree else 0
    screenshots = len(screenshot_sample_indices(samples))
    native_poses = len(samples) + camera_presamples + motion_frames + diagnostic_frames
    counts = {
        "movable_joints": len(joints), "links": len(spec["links"]),
        "numeric_poses": len(samples), "screenshots": screenshots,
        "motion_sequences": sequence_count, "frames_per_motion_view": frame_count if motion_video else 0,
        "camera_presamples": camera_presamples, "motion_frames": motion_frames,
        "diagnostic_frames": diagnostic_frames, "native_poses": native_poses,
        "state_link_checks": native_poses * len(spec["links"]),
        "views": 1 + 2 * sequence_count + (len(joints) if diagnostic_target_subtree else 0),
        "render_frames": screenshots + motion_frames + diagnostic_frames,
    }
    from urdf_core.execution import watchdog_settings
    return {'policy_version': 'capture_workload.v2', 'spec_sha256': source_digest,
            'fps': fps, 'duration_s': float(duration), 'motion_video': motion_video,
            'diagnostic_target_subtree': diagnostic_target_subtree, 'counts': counts,
            **watchdog_settings(),
            'total_runtime_limit': None, 'coverage_reduced': False}


def capture_visibility(spec, joint, view_name):
    """Reveal hidden joints without changing any pose or removing their parent.

    Only separate movable branches may be hidden. Fixed structure, the target
    subtree, and its ancestor chain remain visible; the overview is complete.
    This is a rendering aid, never evidence of collision-free operation.
    """
    if view_name not in {"overview", "closeup"}:
        raise ValueError("Unknown motion view")
    links = {link["name"] for link in spec["links"]}
    hidden = set()
    if view_name == "closeup" and not joint.get("synthetic"):
        descendants = set(subtree_links(spec["joints"], joint["child"]))
        parents = {j["child"]: j["parent"] for j in spec["joints"]}
        ancestors, current = set(), joint["parent"]
        while current is not None:
            if current in ancestors:
                raise ValueError("Joint graph contains a cycle")
            ancestors.add(current)
            current = parents.get(current)
        for other in spec["joints"]:
            if (other["type"] != "fixed" and other["child"] not in descendants
                    and other["child"] not in ancestors):
                hidden.update(subtree_links(spec["joints"], other["child"]))
        if hidden & (descendants | ancestors) or not hidden <= links:
            raise ValueError("Diagnostic visibility would hide the target or parent")
    return {"policy": VISIBILITY_POLICY,
            "mode": "complete_object" if view_name == "overview" else "branch_isolation",
            "visible_links": sorted(links - hidden), "hidden_links": sorted(hidden),
            "scope": "render_visibility_only; all joint states and measured links unchanged"}


def validate_capture_visibility(manifest, joint, spec):
    """Legacy captures stay readable; new capture declarations are exact."""
    policy = manifest.get("visibility_policy")
    if policy is None:
        if any("visibility" in view for view in joint.get("views", [])):
            raise ValueError("Undeclared motion visibility policy")
        return
    if policy != VISIBILITY_POLICY:
        raise ValueError("Unsupported motion visibility policy")
    for view in joint.get("views", []):
        if view.get("visibility") != capture_visibility(spec, joint, view["name"]):
            raise ValueError("Motion visibility differs from the joint tree")


def target_subtree_visibility(spec, joint):
    """Optional diagnostic isolation, separate from both acceptance views.

    This intentionally hides the parent geometry. It can reveal which real
    meshes move, but cannot establish attachment, clearance or semantic truth.
    All links still require native measurement, including the hidden parent.
    """
    links = {link["name"] for link in spec["links"]}
    if joint.get("synthetic") or joint.get("type") == "fixed":
        raise ValueError("Target-subtree diagnostics require a movable joint")
    declared = [item for item in spec["joints"] if item["name"] == joint.get("name")]
    if (len(declared) != 1 or
            any(declared[0].get(key) != joint.get(key) for key in ("type", "parent", "child"))):
        raise ValueError("Diagnostic target does not match the declared joint")
    visible = set(subtree_links(spec["joints"], joint["child"]))
    if not visible <= links or joint["parent"] not in links or joint["parent"] in visible:
        raise ValueError("Invalid target subtree or parent")
    return {"policy": TARGET_SUBTREE_POLICY, "mode": "target_subtree_only",
            "visible_links": sorted(visible), "hidden_links": sorted(links - visible),
            "measured_links": sorted(links), "parent_geometry_hidden": True,
            "scope": "diagnostic render only; hidden geometry remains in the measured native articulation; not attachment or semantic acceptance"}


def target_subtree_source_frames(joint):
    """Use only the unchanged core critic indices, with original RGB bindings."""
    indices = joint.get("critic_frame_indices")
    if (not isinstance(indices, list) or not 7 <= len(indices) <= 9
            or any(type(i) is not int or i < 0 for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("Diagnostic selection must use the core critic indices")
    views = {view["name"]: view for view in joint["views"]}
    if set(views) != {"overview", "closeup"} or len(joint["views"]) != 2:
        raise ValueError("Diagnostic sources need the two unchanged core views")
    tables = {name: {f["frame_index"]: f for f in view["frames"]} for name, view in views.items()}
    if any(len(tables[name]) != len(view["frames"]) for name, view in views.items()):
        raise ValueError("Duplicate source frame indices")
    selected = []
    for index in sorted(indices):
        if any(index not in table for table in tables.values()):
            raise ValueError("Core-selected diagnostic source frame is missing")
        wide, close = tables["overview"][index], tables["closeup"][index]
        if (wide["positions"] != close["positions"] or wide["time_seconds"] != close["time_seconds"]
                or not wide.get("numeric_passed") or not close.get("numeric_passed")):
            raise ValueError("Core diagnostic source views disagree or were not measured")
        selected.append((wide, close))
    return selected


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside_run(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact is outside the current run: {path}")
    return path


def critic_frame_indices(frame_count, required=(), count=9):
    """Uniformly cover a clip while retaining zero and limit keyframes."""
    if frame_count < count:
        raise ValueError("Motion clips need at least nine genuinely rendered frames")
    selected = {0, frame_count - 1, *map(int, required)}
    if min(selected) < 0 or max(selected) >= frame_count or len(selected) > count:
        raise ValueError("Invalid required critic frames")
    while len(selected) < count:
        candidates = set(range(frame_count)) - selected
        selected.add(max(candidates, key=lambda i: (min(abs(i - j) for j in selected), -i)))
    return sorted(selected)


def motion_schedule(joint, fps=12, duration=3.0):
    count = _motion_frame_count(fps, duration)
    kind = joint["type"]
    if kind == "continuous":
        lower, upper = -math.pi, math.pi
    elif kind == "fixed":
        lower = upper = 0.0
    else:
        lower, upper = float(joint["limit"]["lower"]), float(joint["limit"]["upper"])
    if not all(math.isfinite(x) for x in (lower, upper)) or not lower <= 0 <= upper:
        raise ValueError("Motion range must be finite and include the input zero pose")
    anchors = [0.0]
    for value in (lower, upper, 0.0):
        if value != anchors[-1]:
            anchors.append(value)
    if len(anchors) == 1:
        anchors.append(anchors[0])
    indices = [round(i * (count - 1) / (len(anchors) - 1)) for i in range(len(anchors))]
    values = np.zeros(count)
    for left, right, lo, hi in zip(indices[:-1], indices[1:], anchors[:-1], anchors[1:]):
        t = np.linspace(0, 1, right - left + 1)
        values[left:right + 1] = lo + (hi - lo) * (1 - np.cos(math.pi * t)) / 2
    return {"positions_range": [lower, upper], "anchor_indices": indices,
            "critic_frame_indices": critic_frame_indices(count, indices),
            "frames": [{"frame_index": i, "time_seconds": i / fps, "q": float(q)}
                       for i, q in enumerate(values)],
            "fps": fps, "duration_seconds": count / fps}


def subtree_links(joints, child):
    result, pending = [], [child]
    while pending:
        link = pending.pop(0)
        if link in result:
            raise ValueError("Joint graph contains a cycle")
        result.append(link)
        pending.extend(j["child"] for j in joints if j["parent"] == link)
    return result


def bbox_corners(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Camera bounds require finite 3D points")
    lo, hi = points.min(0), points.max(0)
    return np.array([[x, y, z] for x in (lo[0], hi[0])
                     for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def closeup_region(child_points, parent_points, pivot):
    """Moving subtree plus the nearby portion of the actual parent's bounds."""
    child = bbox_corners(child_points)
    parent = bbox_corners(parent_points)
    pivot = np.asarray(pivot, dtype=float)
    if pivot.shape != (3,) or not np.isfinite(pivot).all():
        raise ValueError("Invalid measured joint connection")
    radius = max(float(np.linalg.norm(child.max(0) - child.min(0))) * .6, 1e-4)
    lo = np.maximum(parent.min(0), pivot - radius)
    hi = np.minimum(parent.max(0), pivot + radius)
    if np.any(lo > hi):
        connection = np.clip(pivot, parent.min(0), parent.max(0))[None, :]
    else:
        connection = bbox_corners(np.array([lo, hi]))
    points = np.concatenate([child, connection, pivot[None, :]])
    outward = (child.min(0) + child.max(0) - parent.min(0) - parent.max(0)) / 2
    if np.linalg.norm(outward) < 1e-8:
        outward = np.array([1.4, -1.8, .5])
    outward = outward / np.linalg.norm(outward) + np.array([.15, -.12, .5])
    # A low internal child must not pull the Z-up diagnostic camera below its
    # parent. Keep the existing upward bias as a floor, without changing bounds.
    outward[2] = max(outward[2], .5)
    return points, outward


def camera_for_points(points, direction=(1.4, -1.8, 1.2), resolution=(1024, 768)):
    bounds = bbox_corners(points)
    center = (bounds.min(0) + bounds.max(0)) / 2
    radius = max(float(np.linalg.norm(bounds.max(0) - bounds.min(0))) / 2, 1e-4)
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
        raise ValueError("Invalid camera direction")
    direction /= np.linalg.norm(direction)
    focal_length, horizontal_aperture = 24.0, 20.955
    vertical_aperture = horizontal_aperture * resolution[1] / resolution[0]
    half_fov = math.atan(min(horizontal_aperture, vertical_aperture) / (2 * focal_length))
    distance = radius * 1.15 / math.sin(half_fov)
    return {"position": (center + distance * direction).tolist(), "look_at": center.tolist(),
            "focal_length": focal_length, "horizontal_aperture": horizontal_aperture,
            "clipping_range": [max(radius * .002, 1e-5), max(distance + radius * 30, 1.0)],
            "bounds_min": bounds.min(0).tolist(), "bounds_max": bounds.max(0).tolist(),
            "resolution": list(resolution), "fixed_for_entire_clip": True}


def camera_visible_fraction(position, targets, triangles):
    """Segment/triangle visibility estimate; never a semantic acceptance check.

    Double-sided surfaces occlude, including thin panels and open meshes. Only
    intersections strictly between camera and target count; triangle winding,
    watertightness and surface normals are not assumed. No ray library required.
    """
    position = np.asarray(position, dtype=float)
    targets = np.asarray(targets, dtype=float)
    triangles = np.asarray(triangles, dtype=float)
    if (position.shape != (3,) or targets.ndim != 2 or targets.shape[1] != 3
            or not len(targets) or triangles.ndim != 3 or triangles.shape[1:] != (3, 3)
            or not all(np.isfinite(v).all() for v in (position, targets, triangles))):
        raise ValueError("Visibility requires finite camera, target points and triangles")
    if not len(triangles):
        return 1.0
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    area_scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
    relative = position - triangles[:, 0]
    crossed = np.cross(relative, e1)
    blocked = 0
    for target in targets:
        displacement = target - position
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-12:
            raise ValueError("A target cannot coincide with the camera")
        direction = displacement / distance
        h = np.cross(direction, e2)
        determinant = np.einsum("ij,ij->i", e1, h)
        valid = np.abs(determinant) > area_scale * 1e-12
        inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=valid)
        u = inverse * np.einsum("ij,ij->i", relative, h)
        v = inverse * np.einsum("ij,j->i", crossed, direction)
        t = inverse * np.einsum("ij,ij->i", e2, crossed)
        epsilon = max(distance * 1e-8, 1e-10)
        blocked += bool(np.any(valid & (u >= -1e-9) & (v >= -1e-9)
                              & (u + v <= 1 + 1e-9) & (t > epsilon)
                              & (t < distance - epsilon)))
    return 1.0 - blocked / len(targets)


def choose_unoccluded_camera(points, default_camera, observations, *, max_ray_triangle_work=100_000_000):
    """Choose one fixed closeup through real openings, preserving every surface.

    Observations are measured target surface points and all visible non-target
    triangles at initial/end poses. This only changes camera placement. The
    complete overview, independent native checks and semantic judge still decide
    whether the mechanism is correct; a high ray score does not imply a pass.
    """
    policy = "measured_surface_occlusion_camera.v1"
    if not observations:
        return default_camera, {"policy": policy, "status": "no_triangle_observations"}
    prepared = []
    for target, obstacles in observations:
        target = np.asarray(target, dtype=float)
        obstacles = np.asarray(obstacles, dtype=float)
        # Validate even a work-budget fallback; malformed evidence is not a miss.
        if (target.ndim != 2 or target.shape[1] != 3 or not len(target)
                or obstacles.ndim != 3 or obstacles.shape[1:] != (3, 3)
                or not np.isfinite(target).all() or not np.isfinite(obstacles).all()):
            raise ValueError("Malformed camera-planning surface observation")
        selected = np.linspace(0, len(target) - 1, min(len(target), 64), dtype=int)
        prepared.append((target[selected], obstacles))
    directions = [np.asarray(default_camera["position"]) - default_camera["look_at"]]
    for elevation in (0.0, .35, .8):
        for theta in np.arange(16) * (2 * math.pi / 16):
            directions.append(np.array([math.cos(theta), math.sin(theta), elevation]))
    work = len(directions) * sum(len(target) * len(triangles) for target, triangles in prepared)
    evidence = {"policy": policy, "estimated_ray_triangle_work": work,
                "work_limit": max_ray_triangle_work, "observation_count": len(prepared),
                "source": "actual imported visual triangles and measured native transforms",
                "acceptance_gate": False, "geometry_changed": False}
    if work > max_ray_triangle_work:
        return default_camera, dict(evidence, status="bounded_work_fallback")
    candidates = []
    for index, direction in enumerate(directions):
        camera = (default_camera if index == 0 else camera_for_points(
            points, direction, resolution=default_camera["resolution"]))
        scores = [camera_visible_fraction(camera["position"], target, triangles)
                  for target, triangles in prepared]
        candidates.append({"index": index, "camera": camera, "visible_fractions": scores,
                           "min_visible_fraction": min(scores),
                           "mean_visible_fraction": float(np.mean(scores))})
    best = max(candidates, key=lambda x: (x["min_visible_fraction"], x["mean_visible_fraction"], -x["index"]))
    original = candidates[0]
    evidence.update(status="selected", original=original, selected=best,
                    candidate_count=len(candidates))
    return best["camera"], evidence


def encode_video(frame_paths, output, fps, ffmpeg="ffmpeg", *, root):
    """Encode actual capture files and count decoded video frames with ffprobe."""
    output = inside_run(output, root)
    frames = [inside_run(p, root) for p in frame_paths]
    if not frames or any(not p.is_file() for p in frames):
        raise ValueError("Cannot encode missing capture frames")
    if not output.parent.is_dir():
        output.parent.mkdir(parents=True)
    executable = shutil.which(str(ffmpeg))
    if executable is None:
        raise RuntimeError(f"ffmpeg executable is unavailable: {ffmpeg}")
    sibling = Path(executable).with_name("ffprobe")
    probe = str(sibling) if sibling.is_file() else shutil.which("ffprobe")
    if not probe:
        raise RuntimeError("ffprobe is required to verify encoded frame counts")
    listing = output.with_suffix(".frames.json")
    listing.write_text(json.dumps({"fps": fps, "frames": [str(p) for p in frames]}, indent=2) + "\n")
    # image2 generates exactly one uniform timestamp per input image. Concat
    # PNG durations use a separate timebase and can silently repeat/drop frames.
    with tempfile.TemporaryDirectory(prefix=".encode_frames_", dir=output.parent) as temporary:
        sequence = Path(temporary)
        for index, frame in enumerate(frames):
            (sequence / f"capture_{index:09d}.png").symlink_to(frame)
        command = [executable, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                   "-framerate", str(fps), "-start_number", "0", "-i", str(sequence / "capture_%09d.png"),
                   "-frames:v", str(len(frames)), "-c:v", "libx264", "-threads", "2",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
        from urdf_core.media_execution import run_ffmpeg
        result = run_ffmpeg(command, progress_path=output.with_suffix('.encode.progress'), root=root, text=True)
    output.with_suffix(".encode.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode or not output.is_file() or not output.stat().st_size:
        raise RuntimeError(f"Video encoding failed: {output}")
    result = subprocess.run([probe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                             "-show_entries", "stream=nb_read_frames,width,height,codec_name,r_frame_rate",
                             "-of", "json", str(output)], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"Video decoding failed: {output}: {result.stderr}")
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1 or int(streams[0].get("nb_read_frames", -1)) != len(frames):
        raise RuntimeError(f"Encoded video frame count differs from the real captures: {output}")
    numerator, denominator = map(int, streams[0]["r_frame_rate"].split("/"))
    if denominator == 0 or abs(numerator / denominator - fps) > 1e-6:
        raise RuntimeError("Encoded video frame rate differs from the capture schedule")
    return {"video_path": str(output), "sha256": sha256_file(output), "frame_count": len(frames),
            "fps": fps, "duration_seconds": len(frames) / fps, "probe": streams[0],
            "frame_list_path": str(listing), "timing": "image2 input -framerate; one input per encoded frame"}
