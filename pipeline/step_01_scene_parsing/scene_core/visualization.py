"""Fixed diagnostic views of API-produced layouts; no camera fitting or optimization."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image

from .geometry import BOX_EDGES, BOX_FACES, front_direction, obb_corners, rotation_matrix


def _colors(objects):
    palette = plt.get_cmap("tab20")
    return {obj["object_id"]: palette(i % 20) for i, obj in enumerate(sorted(objects, key=lambda o: o["object_id"]))}


def _label(obj):
    text = obj["object_id"] + (" [env]" if obj["is_environment_support"] else "")
    orientation = obj.get('orientation')
    status = orientation.get('front_status') if isinstance(orientation, dict) else None
    return text + (' [front?]' if status in {'symmetric','unobservable','inferred'} else '')


def semantic_front(obj):
    if 'orientation' not in obj:
        return True  # Historical profile images keep their existing convention.
    orientation = obj.get('orientation')
    return not obj['is_environment_support'] and isinstance(orientation, dict) and orientation.get('front_status') in {'observed','inferred'}


def _render_2d(image_path, layout, output, colors):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(image)
    for obj in layout["objects_data"]:
        bbox = obj["2d_bbox_normalized"]
        if bbox is None:
            continue
        left, top, right, bottom = np.asarray(bbox) * [width, height, width, height]
        color = colors[obj["object_id"]]
        ax.add_patch(Rectangle((left, top), right - left, bottom - top, fill=False,
                               edgecolor=color, linewidth=1.8,
                               linestyle="--" if obj["is_environment_support"] else "-"))
        ax.text(left, top, _label(obj), fontsize=8, color="black", va="bottom",
                bbox={"facecolor": color, "alpha": .8, "edgecolor": "none", "pad": 2})
    ax.set_title("Current layout: visible 2D boxes (environment boxes may be absent)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _scene_extent(objects):
    vertices = np.concatenate([obb_corners(obj["obb"]) for obj in objects])
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = (low + high) / 2
    radius = float(np.max(high - low)) * .6
    return center, radius


def _arrow_points(obj):
    center = np.asarray(obj["obb"]["center_m"], dtype=float)
    length = float(max(obj["obb"]["size_m"])) * .55
    return center, center + front_direction(obj["obb"]) * length


def _legend(fig, objects, colors):
    handles = [plt.Line2D([0], [0], color=colors[o["object_id"]], lw=2,
                         linestyle="--" if o["is_environment_support"] else "-",
                         label=f"{_label(o)}: {o['name']}") for o in objects]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(.77, .5), fontsize=8,
               frameon=False)


def _render_isometric(layout, output, colors, center, radius, focus=False):
    objects = layout["objects_data"]
    fig = plt.figure(figsize=(12, max(8, min(16, len(objects) * .3))))
    ax = fig.add_subplot(111, projection="3d")
    for obj in objects:
        vertices, color = obb_corners(obj["obb"]), colors[obj["object_id"]]
        env = obj["is_environment_support"]
        faces = [[vertices[i] for i in face] for face in BOX_FACES]
        facecolors = [color]*len(faces)
        if 'orientation' in obj and semantic_front(obj):
            facecolors[2] = 'darkorange'  # Corner ordering: local -Y face.
        ax.add_collection3d(Poly3DCollection(faces, facecolor=facecolors, edgecolor=color,
                                            linewidth=.8, alpha=.06 if env else .16,
                                            linestyle="--" if env else "-"))
        start, end = _arrow_points(obj)
        if semantic_front(obj):
            inferred = obj.get('orientation',{}).get('front_status') == 'inferred'
            ax.quiver(*start, *(end - start), color='darkorange' if 'orientation' in obj else color,
                      arrow_length_ratio=.2, linewidth=2, linestyle='--' if inferred else '-')
        if 'orientation' in obj and not env:
            r = rotation_matrix(obj['obb']['rotation_xyz_deg'])
            c = np.asarray(obj['obb']['center_m'])
            length = max(obj['obb']['size_m'])*.24
            for axis, tint, name in zip(r.T,('red','green','blue'),('x','y','z')):
                ax.quiver(*c,*(axis*length),color=tint,linewidth=.8)
                ax.text(*(c+axis*length),name,color=tint,fontsize=6)
        label_point = vertices[np.argmax(vertices[:, 2])]
        ax.text(*label_point, _label(obj), fontsize=8, color="black")
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=25, azim=-60)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(("Foreground-only isometric; environment omitted for visibility\n" if focus else
                  "Fixed isometric OBB view; arrows = local -Y front\n") +
                 "Layout diagnostic, not input-camera reprojection" +
                 ('\nOrange = semantic front; dashed = inferred; XYZ = local axes' if any('orientation' in o for o in objects) else ''))
    fig.subplots_adjust(left=.02, right=.75, bottom=.07, top=.92)
    _legend(fig, objects, colors)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _render_projection(layout, output, colors, center, radius, axes, title):
    objects = layout["objects_data"]
    fig, ax = plt.subplots(figsize=(12, max(8, min(16, len(objects) * .3))))
    horizontal, vertical = axes
    for obj in objects:
        vertices, color = obb_corners(obj["obb"]), colors[obj["object_id"]]
        projected = vertices[:, list(axes)]
        env = obj["is_environment_support"]
        for face in BOX_FACES:
            ax.add_patch(Polygon(projected[list(face)], closed=True, facecolor=color,
                                 edgecolor="none", alpha=.015 if env else .025))
        for a, b in BOX_EDGES:
            ax.plot(projected[[a, b], 0], projected[[a, b], 1], color=color,
                    linewidth=1, linestyle="--" if env else "-")
        start, end = _arrow_points(obj)
        start, end = start[list(axes)], end[list(axes)]
        if semantic_front(obj) and np.linalg.norm(end - start) > 1e-12:
            ax.annotate("", xy=end, xytext=start,
                        arrowprops={"arrowstyle": "-|>", "color": 'darkorange' if 'orientation' in obj else color, "lw": 2,
                                    "linestyle": '--' if obj.get('orientation',{}).get('front_status')=='inferred' else '-'})
        elif semantic_front(obj):
            # A forward axis perpendicular to this plane cannot be an in-plane arrow.
            ax.plot(*start, marker="o", markerfacecolor="none", markeredgecolor=color, markersize=5)
        if 'orientation' in obj and semantic_front(obj):
            ax.add_patch(Polygon(projected[list(BOX_FACES[2])],closed=True,facecolor='darkorange',alpha=.14))
        highest = projected[np.argmax(projected[:, 1])]
        ax.annotate(_label(obj), xy=highest, xytext=(2, 3), textcoords="offset points", fontsize=8)
    ax.set_xlim(center[horizontal] - radius, center[horizontal] + radius)
    ax.set_ylim(center[vertical] - radius, center[vertical] + radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{'XYZ'[horizontal]} (m)")
    ax.set_ylabel(f"{'XYZ'[vertical]} (m)")
    ax.grid(alpha=.2)
    ax.set_title(f"{title}; arrows = projected local -Y front\nFixed world view, not input-camera reprojection")
    fig.subplots_adjust(left=.08, right=.75, bottom=.08, top=.9)
    _legend(fig, objects, colors)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_layout_visualizations(image_path, layout, out_dir, round_idx):
    """Write a 2D overlay and four equally scaled world views, returning Paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"layout_2d": out_dir / f"layout_2d_round_{round_idx}.png",
             "layout_3d": out_dir / f"layout_3d_round_{round_idx}.png",
             "top": out_dir / f"layout_top_round_{round_idx}.png",
             "front": out_dir / f"layout_front_round_{round_idx}.png",
             "side": out_dir / f"layout_side_round_{round_idx}.png"}
    paths["isometric"] = paths["layout_3d"]
    objects = layout["objects_data"]
    colors = _colors(objects)
    center, radius = _scene_extent(objects)
    _render_2d(Path(image_path), layout, paths["layout_2d"], colors)
    _render_isometric(layout, paths["isometric"], colors, center, radius)
    for name, axes, title in (("top", (0, 1), "Top: X/Y (look down -Z)"),
                              ("front", (0, 2), "Front: X/Z (look into +Y)"),
                              ("side", (1, 2), "Side: Y/Z (look from +X)")):
        _render_projection(layout, paths[name], colors, center, radius, axes, title)
    return paths
