"""Solid Boolean helpers and bound receipt import for physical preparation."""

from pathlib import Path
import json
import subprocess
import numpy as np
import trimesh
from urdf_core.run_state import file_hash, write_json
from urdf_core.content_errors import AssetContentError
from urdf_core.execution import TechnicalExecutionError, wait_process

POLICY = "physical_boolean_repair.v1"
FOLDER = "physical_geometry"


def source_binding(export, spec):
    return {
        "ownership": {r["name"]: r["objects"] for r in spec["links"]},
        "files": {
            str(p.relative_to(export)): file_hash(p) for p in sorted(Path(export).rglob("*")) if p.is_file()
        },
    }


def partial_pairs(parts):
    from .physicalize import _inside_closed

    result = []
    for i, a in enumerate(parts):
        if not a.is_watertight:
            continue
        for j, b in enumerate(parts):
            if i == j or not b.is_watertight:
                continue
            if not (np.all(a.bounds[0] >= b.bounds[0]) and np.all(a.bounds[1] <= b.bounds[1])):
                continue
            points = a.vertices[np.linspace(0, len(a.vertices) - 1, min(32, len(a.vertices)), dtype=int)]
            try:
                inside = [_inside_closed(b, p) for p in points]
            except ValueError:
                continue
            if any(inside) and not all(inside):
                result.append((i, j))
    return sorted(set(tuple(sorted(pair)) for pair in result))


def discover(objects):
    issues = []
    for entry in objects:
        parts = list(entry["mesh"].split(only_watertight=False, repair=False))
        pairs = partial_pairs(parts)
        pending = set(v for pair in pairs for v in pair)
        while pending:
            group = {min(pending)}
            while True:
                expanded = group | {v for pair in pairs if group.intersection(pair) for v in pair}
                if expanded == group:
                    break
                group = expanded
            pending -= group
            indices = sorted(group)
            operands = [parts[i].copy() for i in indices]
            for mesh in operands:
                if mesh.volume < 0:
                    mesh.invert()
            issues.append(
                {
                    "id": "issue_" + str(len(issues)),
                    "link": entry["link"],
                    "object": entry["object"],
                    "component_indices": indices,
                    "operands": operands,
                    "parts": parts,
                }
            )
    return issues


def render_evidence(group, path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    meshes = group["_issue"]["operands"]
    colors = ["#f2ac40", "#3099d5", "#8dbf54", "#bd70c5"]
    count = len(meshes) + 2
    columns = 2
    rows = (count + 1) // 2
    fig = plt.figure(figsize=(10, 4 * rows))
    for panel in range(count):
        ax = fig.add_subplot(rows, columns, panel + 1, projection="3d")
        selected = list(enumerate(meshes)) if panel < 2 else [(panel - 2, meshes[panel - 2])]
        bounds = np.vstack([m.bounds for _, m in selected])
        lo = bounds.min(0)
        hi = bounds.max(0)
        center = (lo + hi) / 2
        radius = max(hi - lo) * 0.55
        for i, m in selected:
            light = np.array([0.3, -0.4, 0.85])
            light /= np.linalg.norm(light)
            shade = 0.45 + 0.55 * np.maximum(0, m.face_normals @ light)
            facecolors = np.asarray(to_rgb(colors[i % len(colors)]))[None, :] * shade[:, None]
            ax.add_collection3d(
                Poly3DCollection(
                    m.triangles, facecolor=facecolors, edgecolor="none", alpha=0.5 if panel < 2 else 1.0
                )
            )
        ax.set(
            xlim=(center[0] - radius, center[0] + radius),
            ylim=(center[1] - radius, center[1] + radius),
            zlim=(center[2] - radius, center[2] + radius),
        )
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=(25 if panel == 0 else -25 if panel == 1 else 70), azim=-60 if panel != 1 else 120)
        ax.set_axis_off()
        ax.set_title(
            ("Overlay " + str(panel + 1))
            if panel < 2
            else f"Operand {panel-2} | bounds (mm): " + ", ".join(f"{v*1000:.2f}" for v in hi - lo)
        )
    fig.suptitle(
        group["group_id"] + " | " + group["_issue"]["object"] + " | individual panels fitted separately"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_worker(blender, jobs, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    request = folder / "request.json"
    result = folder / "result.json"
    log = folder / "blender.log"
    write_json(request, {"jobs": jobs})
    command = [
        str(blender),
        "--background",
        "--factory-startup",
        "--threads",
        "2",
        "--python",
        str(Path(__file__).with_name("boolean_worker.py")),
        "--",
        str(request.resolve()),
        str(result.resolve()),
    ]
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        code = wait_process(
            process, progress=lambda: (result.stat().st_mtime_ns if result.exists() else None)
        )
    if code or not result.is_file():
        raise TechnicalExecutionError(f"Boolean Blender worker failed ({code}); see {log}")
    rows = json.loads(result.read_text())["results"]
    if len(rows) != len(jobs) or [r["id"] for r in rows] != [j["id"] for j in jobs]:
        raise TechnicalExecutionError("Boolean worker response identity mismatch")
    return rows


def validate_result(row, issue, decision):
    from .physicalize import validate_inertia

    if row.get("status") != "completed":
        raise ValueError(row.get("reason", "Boolean execution rejected"))
    mesh = trimesh.Trimesh(vertices=row["vertices"], faces=row["faces"], process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent or not len(mesh.faces):
        raise ValueError("Boolean output must be nonempty, watertight and coherently oriented")
    props = mesh.mass_properties
    validate_inertia(props["volume"], props["center_mass"], props["inertia"])
    parts = issue["operands"]
    bounds = np.vstack([p.bounds for p in parts])
    lo = bounds.min(0)
    hi = bounds.max(0)
    tolerance = max(float(np.max(hi - lo)) * 1e-5, 1e-8)
    if np.any(mesh.bounds[0] < lo - tolerance) or np.any(mesh.bounds[1] > hi + tolerance):
        raise ValueError("Boolean output escaped operand bounds")
    volume = float(mesh.volume)
    epsilon = max(sum(p.volume for p in parts) * 1e-5, 1e-15)
    if decision["operation"] == "boolean_union":
        if not max(p.volume for p in parts) - epsilon <= volume <= sum(p.volume for p in parts) + epsilon:
            raise ValueError("Union volume is inconsistent with operands")
        if not np.allclose(mesh.bounds, np.array([lo, hi]), rtol=0, atol=tolerance):
            raise ValueError("Union lost operand extent")
    elif volume > parts[decision["base_operand"]].volume + epsilon:
        raise ValueError("Difference increased base volume")
    # Probe actual source surfaces to check operation semantics, including cavities.
    from .physicalize import _inside_closed

    rng = np.random.default_rng(0)
    points = rng.uniform(lo, hi, size=(96, 3))
    for point in points:
        distances = [
            trimesh.proximity.closest_point_naive(p, np.array([point]))[1][0] for p in [*parts, mesh]
        ]
        if min(distances) <= tolerance:
            continue
        occupancy = [_inside_closed(p, point) for p in parts]
        expected = (
            any(occupancy)
            if decision["operation"] == "boolean_union"
            else occupancy[decision["base_operand"]]
            and not any(v for i, v in enumerate(occupancy) if i != decision["base_operand"])
        )
        if _inside_closed(mesh, point) != expected:
            raise ValueError("Boolean result disagrees with selected solid operation")
    return mesh


def apply_saved(objects, export, spec):
    folder = Path(export).parent / FOLDER
    receipt = folder / "receipt.json"
    if not receipt.is_file():
        return objects
    data = json.loads(receipt.read_text())
    if data.get("policy") != POLICY or data.get("source") != source_binding(Path(export), spec):
        raise RuntimeError("Physical boolean source binding changed")
    if data["status"] != "passed":
        raise AssetContentError("Physical boolean preparation failed", stage="source_mass_geometry")
    for relative, digest in data["files"].items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or file_hash(path) != digest:
            raise RuntimeError("Physical boolean evidence changed: " + relative)
    by_key = {(e["link"], e["object"]): e for e in objects}
    for replacement in data["replacements"]:
        entry = by_key[(replacement["link"], replacement["object"])]
        from urdf_core.urdf_validation import read_obj

        mesh = read_obj(folder / replacement["path"])
        entry["mesh"] = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"], process=False)
        entry["cleanup"] = {**entry["cleanup"], "physical_boolean_repair": replacement["decisions"]}
    return objects
