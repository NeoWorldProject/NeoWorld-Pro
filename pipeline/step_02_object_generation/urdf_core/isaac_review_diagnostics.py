"""Optional measured-frame annotations for native motion review.

These are diagnostic overlays, never replacement RGB or acceptance decisions.
Projection/drawing is carried from the native-frame A/B experiment; actual
renderer camera and native body transforms are required. Old captures simply
have no available annotation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NOTICE = "DIAGNOSTIC OVERLAY: measured link frames; axes ignore occlusion and are not surface features."
LIMIT = "Motion execution evidence only. Axis meaning, part ownership and functional motion need the real overview/source."


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def rigid(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
        raise ValueError("Missing or invalid measured rigid transform")
    return matrix


def checked_camera(camera):
    if (not isinstance(camera, dict) or camera.get("passed") is not True
            or camera.get("status") != "passed" or camera.get("source") != "actual_usd_and_render_product.v1"
            or camera.get("matrix_convention") != "column_vectors; camera_forward_minus_Z; pixel_origin_top_left"):
        raise ValueError("No eligible actual camera readback; guessed projection is forbidden")
    view = rigid(camera.get("camera_from_world"))
    world = rigid(camera.get("world_from_camera"))
    projection = np.asarray(camera.get("projection_matrix"), dtype=float)
    raw = np.asarray(camera.get("renderer", {}).get("cameraProjection_row_matrix"), dtype=float)
    resolution = camera.get("resolution")
    if (projection.shape != (4, 4) or not np.isfinite(projection).all()
            or np.linalg.matrix_rank(projection) != 4 or projection[0, 0] <= 0 or projection[1, 1] <= 0
            or raw.shape != (4, 4) or not np.array_equal(raw.T, projection)
            or not np.allclose(view @ world, np.eye(4), atol=1e-5)
            or not isinstance(resolution, list) or len(resolution) != 2
            or any(type(x) is not int or x <= 0 for x in resolution)):
        raise ValueError("Invalid actual renderer projection or camera binding")
    return view, projection, resolution


def project(world_points, camera):
    view, projection, (width, height) = checked_camera(camera)
    points = np.asarray(world_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Invalid measured world points")
    local = np.column_stack([points, np.ones(len(points))]) @ view.T
    clip = local @ projection.T
    near, far = camera["usd"]["clipping_range"]
    result = []
    for camera_point, clip_point in zip(local, clip):
        depth = -float(camera_point[2])
        if not near <= depth <= far or clip_point[3] <= 1e-12:
            result.append({"pixel": None, "depth": depth, "reason": "outside_depth_or_behind_camera"})
            continue
        ndc = clip_point[:2] / clip_point[3]
        pixel = [float((ndc[0] + 1) * width / 2), float((1 - ndc[1]) * height / 2)]
        result.append({"pixel": pixel, "depth": depth,
                       "inside_image": 0 <= pixel[0] < width and 0 <= pixel[1] < height})
    return result


def axis_points(measured_pose, length, local_origin=(0, 0, 0)):
    pose = rigid(measured_pose)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Positive finite diagnostic axis length required")
    local = np.asarray(local_origin, dtype=float) + np.vstack([np.zeros(3), np.eye(3) * length])
    return local @ pose[:3, :3].T + pose[:3, 3]


def axes_record(frame, joint, length, parent_anchor):
    camera = frame.get("actual_camera")
    checked_camera(camera)
    measured = frame["link_poses"]
    records = []
    for role, link, origin in (("child", joint["child"], (0, 0, 0)),
                               ("parent_reference", joint["parent"], parent_anchor)):
        world = axis_points(measured[link], length, origin)
        records.append({"role": role, "link": link, "measured_transform": measured[link],
                        "local_origin": list(origin), "axis_length_m": length,
                        "world_points_origin_xyz": world.tolist(), "projection": project(world, camera)})
    return records


def font(size=16):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_axes(rgb, axes):
    output = rgb.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for item in sorted(axes, key=lambda x: x["role"] == "child"):
        origin = item["projection"][0]["pixel"]
        if origin is None:
            continue
        parent = item["role"] == "parent_reference"
        for i, (letter, color) in enumerate(zip("XYZ", ((235, 52, 62), (21, 172, 74), (45, 101, 240))), start=1):
            tip = item["projection"][i]["pixel"]
            if tip is None:
                continue
            a, b = np.asarray(origin), np.asarray(tip)
            delta = b - a
            size = np.linalg.norm(delta)
            if size < 1:
                continue
            if parent:
                for t in np.arange(0, 1, 8 / size):
                    draw.line([tuple(a + t * delta), tuple(a + min(t + 4 / size, 1) * delta)], fill=(25, 25, 25), width=2)
                draw.text(tuple(b + [3, -14]), "p" + letter, font=font(12), fill=(30, 30, 30), stroke_width=2, stroke_fill="white")
            else:
                draw.line([tuple(a), tuple(b)], fill="white", width=6)
                draw.line([tuple(a), tuple(b)], fill=color, width=3)
                direction = delta / size
                normal = np.array([-direction[1], direction[0]])
                draw.polygon([tuple(b), tuple(b - direction * 10 + normal * 4), tuple(b - direction * 10 - normal * 4)], fill=color)
                draw.text(tuple(b + [4, -12]), letter, font=font(15), fill=color, stroke_width=2, stroke_fill="white")
    return output


def _annotation_plan(report, manifest, selected):
    """Derive annotations exclusively from the same measured capture."""
    # Historical reports can predate per-frame native pose records. Their
    # existing visual review remains valid; absent auxiliary records are not a
    # new required observation. Duplicate identities are equally ineligible.
    pose_rows = [item for item in report.get("poses", [])
                 if isinstance(item, dict) and isinstance(item.get("name"), str)]
    measured = {item["name"]: item for item in pose_rows}
    if len(measured) != len(pose_rows):
        measured = {}
    allowed = {item["frame_ref"] for item in selected}
    result = {}
    for joint in manifest.get("joints", []):
        name = joint["name"]
        if joint.get("type") == "fixed":
            result[name] = {"status": "unavailable", "reason": "static observation has no moving frame"}
            continue
        try:
            view = next(v for v in joint["views"] if v["name"] == "closeup")
            frames = {frame["frame_index"]: frame for frame in view["frames"]}
            initial = frames[min(frames)]
            camera_view, projection, (width, _) = checked_camera(initial.get("actual_camera"))
            child_origin = rigid(initial["link_poses"][joint["child"]])[:3, 3]
            parent_inverse = np.linalg.inv(rigid(initial["link_poses"][joint["parent"]]))
            parent_anchor = (parent_inverse @ np.append(child_origin, 1))[:3]
            depth = -(camera_view @ np.append(child_origin, 1))[2]
            length = float(60 * depth / (width / 2 * projection[0, 0]))
            anchors = joint["anchor_indices"]
            indices = sorted(set(anchors + [(a+b)//2 for a, b in zip(anchors, anchors[1:])]))
            indices = [i for i in indices if f"{name}::closeup::{i}" in allowed]
            if len(indices) < 3:
                raise ValueError("insufficient existing selected closeup frames for annotation")
            entries = []
            for index in indices:
                frame = frames[index]
                actual = measured[frame["numeric_pose_name"]]
                if frame.get("numeric_passed") is not True or actual.get("passed") is not True:
                    raise ValueError("native transform observation was not verified")
                for link in (joint["parent"], joint["child"]):
                    if not np.array_equal(frame["link_poses"][link], actual["links"][link]["actual"]):
                        raise ValueError("frame transforms differ from native report")
                entries.append({"frame_ref": f"{name}::closeup::{index}", "frame_index": index,
                    "time_seconds": frame["time_seconds"], "source_rgb_path": frame["path"],
                    "source_rgb_sha256": frame["sha256"], "native_pose": frame["numeric_pose_name"],
                    "camera": frame["actual_camera"], "axes": axes_record(frame, joint, length, parent_anchor),
                    "measured_positions_for_annotation_only": frame["measured_positions"],
                    "overlay_source_pixel_offset": [0, 0], "caption_is_outside_source_image": True})
            result[name] = {"status": "available", "frames": entries}
        except (KeyError, ValueError, TypeError, IndexError, StopIteration, np.linalg.LinAlgError) as error:
            # Missing old capture observations cannot become a new acceptance
            # condition; the unannotated captured views remain reviewable.
            result[name] = {"status": "unavailable", "reason": str(error) or type(error).__name__}
    return result


def _labelled_overlay(raw, axes):
    painted = draw_axes(raw, axes)
    caption = textwrap.wrap(NOTICE + " " + LIMIT, width=76)
    labelled = Image.new("RGB", (painted.width, painted.height + 16 * len(caption) + 8), "white")
    labelled.paste(painted, (0, 0))
    draw = ImageDraw.Draw(labelled)
    for index, line in enumerate(caption):
        draw.text((4, painted.height + 4 + index * 16), line, fill="black", font=font(12))
    return labelled


def axis_diagnostics(report, manifest, selected_frames, out_dir, *, write=False, path_resolver=None):
    """Create or independently recheck optional annotations and their bindings.

    Return ``(metadata, evidence)`` with Path->SHA256 evidence. ``write=False``
    never creates files. Cached PNG pixels must equal projections reconstructed
    from the bound native matrices and actual renderer camera, not just match
    their stored image hash. No additional frame citation is invented.
    """
    resolve = path_resolver or (lambda path: Path(path).resolve())
    out_dir = Path(out_dir).resolve()
    joints = _annotation_plan(report, manifest, selected_frames)
    evidence = {}
    for joint_index, (name, item) in enumerate(joints.items()):
        if item["status"] != "available":
            continue
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", name)
        for entry in item["frames"]:
            raw_path = Path(resolve(entry["source_rgb_path"]))
            if not raw_path.is_file() or digest(raw_path) != entry["source_rgb_sha256"]:
                raise ValueError("Native RGB changed before annotation: " + str(raw_path))
            with Image.open(raw_path) as raw:
                if list(raw.size) != entry["camera"]["resolution"]:
                    raise ValueError("Captured RGB dimensions differ from its actual camera")
                expected = _labelled_overlay(raw, entry["axes"])
            logical_path = out_dir / f"{joint_index:03d}_{safe_name}" / f"frame_{entry['frame_index']:06d}_axes.png"
            path = Path(resolve(logical_path))
            if write and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                expected.save(path)
            if not path.is_file():
                raise ValueError("Missing bound diagnostic annotation: " + str(path))
            with Image.open(path) as stored:
                if stored.size != expected.size or stored.convert("RGB").tobytes() != expected.tobytes():
                    raise ValueError("Diagnostic image differs from measured-frame projection: " + str(path))
            entry["overlay"] = {"path": str(logical_path), "sha256": digest(path)}
            evidence[path] = entry["overlay"]["sha256"]
            evidence[raw_path] = entry["source_rgb_sha256"]
    return {"schema_version": 1, "notice": NOTICE, "scope": LIMIT,
            "parent_reference": "Dashed axes are attached to the initial measured child origin in the actual parent frame.",
            "axis_scale": "Fixed per clip, nominally 60 horizontal pixels at the initial child origin; not geometry dimensions.",
            "joints": joints}, evidence
