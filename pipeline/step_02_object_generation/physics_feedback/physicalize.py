"""Build physical URDFs from prepared link-local meshes and part parameters."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from urdf_core.storage import copy_asset, copy_asset_tree, freeze_file, asset_pool
from urdf_core.kinematics import validate_spec
from urdf_core.urdf_validation import read_obj
from .protocol import validate_profile, load_config


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _fmt(values):
    return " ".join(format(float(value), ".17g") for value in values)


def _write_obj(path, mesh):
    with Path(path).open("w") as stream:
        stream.write("# Physical geometry, original link-local metre frame.\n")
        for vertex in mesh.vertices:
            stream.write("v " + _fmt(vertex) + "\n")
        for face in mesh.faces:
            stream.write("f " + " ".join(str(int(index) + 1) for index in face) + "\n")


def reconstruct_objects(export_dir, spec, *, clean=True):
    """Load prepared meshes, or recover authored triangles by export provenance."""
    export_dir = Path(export_dir).resolve()
    if clean:
        if not (export_dir.parent / "prepared_geometry/receipt.json").is_file():
            raise ValueError("Physical geometry must be prepared before parameter construction")
        from .mesh_preparation import load

        return load(export_dir.parent, spec)
    from urdf_core.export_storage import read_export_manifest
    manifest = read_export_manifest(export_dir / "export_manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("mesh_frame") != "link_local"
        or manifest.get("pose") != "q_zero"
    ):
        raise ValueError("physicalization requires a link-local q_zero export manifest")
    expected = {(link["name"], name) for link in spec["links"] for name in link["objects"]}
    from .visual_text import inventory_materials, material_is_text, source_text_vertices

    materials_by_object = inventory_materials(export_dir)
    records = {}
    for entry in manifest.get("meshes", []):
        key = (entry.get("link"), entry.get("object"))
        if key not in expected:
            raise ValueError("export manifest has unknown object ownership")
        original_path = Path(entry["path"])
        # Canonical packages retain original absolute paths after publication.
        # Resolve by package-relative mesh name, never use historical outside data.
        path = export_dir / "meshes" / original_path.name
        if not path.resolve().is_relative_to(export_dir) or not path.is_file():
            raise ValueError("missing or escaping source mesh: " + str(path))
        parsed = read_obj(path, physical_copy=not clean)
        vertices, faces = parsed["vertices"], parsed["faces"]
        indices = entry.get("source_vertex_indices")
        triangles = entry.get("source_triangle_vertices")
        reverse = entry.get("backface_source_triangle_indices", [])
        if (
            not isinstance(indices, list)
            or len(indices) != len(vertices)
            or len(set(indices)) != len(indices)
            or not isinstance(triangles, list)
            or not triangles
            or not isinstance(reverse, list)
            or len(faces) != len(triangles) + len(reverse)
        ):
            raise ValueError("source mesh reconstruction provenance is incomplete: " + path.name)
        if any(type(i) is not int or not 0 <= i < len(triangles) for i in reverse):
            raise ValueError("invalid reverse-face provenance")
        for face, triangle in zip(faces, triangles):
            if len(face) != 3 or [indices[i] for i in face] != triangle:
                raise ValueError("authored triangle differs from export provenance: " + path.name)
        for face, index in zip(faces[len(triangles) :], reverse):
            if [indices[i] for i in face] != list(reversed(triangles[index])):
                raise ValueError("added backface differs from export provenance")
        record = records.setdefault(key, {"vertices": {}, "faces": [], "sources": []})
        for index, vertex in zip(indices, vertices):
            if index in record["vertices"] and not np.allclose(
                record["vertices"][index], vertex, rtol=0, atol=1e-10, equal_nan=not clean
            ):
                raise ValueError("material splits disagree on shared authored vertex")
            record["vertices"][index] = vertex
        record["faces"].extend(triangles)
        materials = materials_by_object.get(entry["object"], [])
        index = entry.get("material_index", 0)
        material = entry.get("material", materials[index] if 0 <= index < len(materials) else {})
        is_text = material_is_text(material)
        record.setdefault("text_faces" if is_text else "contact_faces", []).extend(triangles)
        if is_text:
            record.setdefault("text_materials", []).append({"material_index": index, **material})
        record["sources"].append(
            {"path": str(path), "sha256": file_hash(path), "removed_render_backfaces": len(reverse)}
        )
    if set(records) != expected:
        raise ValueError("export manifest does not cover every source object")
    result = []
    for (link, name), record in records.items():
        ordered = sorted(record["vertices"])
        remap = {index: target for target, index in enumerate(ordered)}
        vertices = np.asarray([record["vertices"][i] for i in ordered], dtype=float)
        faces = np.asarray([[remap[i] for i in face] for face in record["faces"]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        cleanup = {"policy": "authored_geometry_without_cleanup"}
        result.append(
            {
                "link": link,
                "object": name,
                "mesh": mesh,
                "sources": record["sources"],
                "cleanup": cleanup,
                "visual_text_vertices": source_text_vertices(record),
                "visual_text_materials": record.get("text_materials", []),
            }
        )
    return result


def validate_profile_geometry(profile, spec, objects):
    """Validate parameter proposals against the prepared physical representation."""
    result = validate_profile(profile, spec)
    if result["schema_version"] != 3:
        raise ValueError("Current physicalization requires a schema-3 part-level profile")
    if not objects or not all(entry.get("prepared_physical") for entry in objects):
        raise ValueError("Physical geometry must be prepared before parameter construction")
    for entry in objects:
        params = result["links"][entry["link"]]["objects"][entry["object"]]
        if params.get("mass_model", "auto") != "auto":
            raise ValueError(
                "Prepared geometry requires mass_model=auto; thickness does not modify its mass volume"
            )
        if entry.get("visual_only_surface") and params["role"] != "decoration":
            raise ValueError("Visual-only markings require decoration role")
    return result


def _surface_mass_info(mesh, params):
    """Integrate a visual-only lamina without inventing normals or a collider.

    An explicitly supplied mass suffices for an ideal thin sheet. Density needs
    an explicit thickness; thickness contributes normal variance to inertia.
    This also supports disconnected/folded print geometry with ambiguous vertex
    normals, because integration uses actual individual triangles.
    """
    triangles = np.asarray(mesh.triangles)
    area = np.asarray(mesh.area_faces)
    total_area = float(area.sum())
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("decorative surface has no finite positive area")
    center = np.average(triangles.mean(axis=1), axis=0, weights=area)
    points = triangles - center
    sums = points.sum(axis=1)
    second = (np.einsum("nki,nkj->nij", points, points) + np.einsum("ni,nj->nij", sums, sums)) / 12
    thickness = params.get("thickness_m")
    if thickness is not None:
        normals = np.asarray(mesh.face_normals)
        second += np.einsum("ni,nj->nij", normals, normals) * float(thickness) ** 2 / 12
    covariance = np.einsum("n,nij->ij", area / total_area, second)
    unit_mass_inertia = np.eye(3) * np.trace(covariance) - covariance
    volume = total_area * float(thickness) if thickness is not None else None
    if "mass_kg" in params:
        mass = float(params["mass_kg"])
    elif volume is not None:
        mass = float(params["density_kg_m3"]) * volume
    else:
        raise ValueError("decorative surface density requires explicit thickness")
    inertia = mass * unit_mass_inertia
    validate_inertia(mass, center, inertia)
    return {
        "volume_m3": volume,
        "center": center,
        "part_mass_kg": mass,
        "part_density_kg_m3": mass / volume if volume is not None else None,
        "part_inertia_kg_m2": inertia,
        "mass_basis": "authored_triangle_surface_lamina",
        "surface_area_m2": total_area,
    }


def _inside_closed(mesh, point):
    """Solid-angle containment without optional spatial-index dependencies."""
    vectors = np.asarray(mesh.triangles) - np.asarray(point)
    lengths = np.linalg.norm(vectors, axis=2)
    if np.any(lengths < 1e-12):
        raise ValueError("touching disconnected shells have ambiguous mass topology")
    a, b, c = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = (
        lengths.prod(axis=1)
        + np.einsum("ij,ij->i", a, b) * lengths[:, 2]
        + np.einsum("ij,ij->i", b, c) * lengths[:, 0]
        + np.einsum("ij,ij->i", c, a) * lengths[:, 1]
    )
    return abs(float(np.arctan2(numerator, denominator).sum() * 2)) > 2 * np.pi


class PhysicalizationGeometryError(ValueError):
    """A physical representation failure with bounded, source-grounded evidence."""

    def __init__(self, message, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(
            message
            + "; geometry_diagnostics="
            + json.dumps(diagnostics, sort_keys=True, allow_nan=False, separators=(",", ":"))
        )


def validate_inertia(mass, center, inertia):
    matrix = np.asarray(inertia, dtype=float)
    if (
        not np.isfinite(mass)
        or mass <= 0
        or np.asarray(center).shape != (3,)
        or not np.isfinite(center).all()
        or matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-14)
    ):
        raise ValueError("invalid mass, center of mass, or inertia tensor")
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = max(float(eigenvalues[-1]) * 1e-9, 1e-18)
    if eigenvalues[0] <= 0 or eigenvalues[-1] > eigenvalues[:2].sum() + tolerance:
        raise ValueError("inertia must be positive definite and obey principal triangle inequalities")
    return eigenvalues.tolist()


def physicalize_asset(
    base_run: Path, profile: dict, out_dir: Path, *, config=None, collision_cache_root=None
) -> dict:
    """Write a new self-contained package and its content-bound lineage manifest."""
    base_run, out_dir = (Path(base_run).resolve(), Path(out_dir).resolve())
    export_dir = base_run / "urdf_export"
    spec_path = base_run / "kinematics.json"
    source_urdf = export_dir / "object.urdf"
    if out_dir == base_run or out_dir.is_relative_to(export_dir) or base_run.is_relative_to(out_dir):
        raise ValueError("derived asset directory must not replace or nest within the base export")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("physical asset output must be empty; prior revisions are immutable")
    spec = validate_spec(json.loads(spec_path.read_text()))
    objects = reconstruct_objects(export_dir, spec)
    profile = validate_profile_geometry(profile, spec, objects)
    settings = load_config(config)
    root = ET.parse(source_urdf).getroot()
    if {node.get("name") for node in root.findall("link")} != {link["name"] for link in spec["links"]}:
        raise ValueError("URDF link identities differ from the shared kinematic protocol")
    link_parts, prepared = (defaultdict(list), [])
    from .collision_batch import CollisionBatch
    from .mesh_preparation import valid_solid

    with CollisionBatch(
        settings, collision_cache_root, out_dir.with_name(out_dir.name + ".collision_progress.json")
    ) as batch:
        for entry in objects:
            source_mesh = entry["mesh"]
            params = profile["links"][entry["link"]]["objects"][entry["object"]]
            if entry.get("visual_only_surface"):
                info = {
                    **entry,
                    **_surface_mass_info(source_mesh, params),
                    "physical_parts": [],
                    "collision_parts": [],
                    "proxies": [],
                    "decision": params,
                    "nested_cavity_shells": False,
                }
            else:
                valid_solid(source_mesh)
                props = source_mesh.mass_properties
                volume = float(props["volume"])
                density = float(params.get("density_kg_m3", params.get("mass_kg", 0) / volume))
                info = {
                    **entry,
                    "physical_parts": [source_mesh],
                    "volume_m3": volume,
                    "center": np.asarray(props["center_mass"]),
                    "unit_density_inertia": np.asarray(props["inertia"]),
                    "part_mass_kg": density * volume,
                    "part_density_kg_m3": density,
                    "part_inertia_kg_m2": np.asarray(props["inertia"]) * density,
                    "mass_basis": entry["audit"]["mass_basis"],
                    "collision_basis": "same_prepared_mass_mesh",
                    "decision": params,
                    "nested_cavity_shells": False,
                }
                if params["role"] == "decoration":
                    info.update(collision_parts=[], proxies=[])
                else:
                    source_mesh.metadata["prepared_physical"] = True
                    batch.submit(info, [source_mesh])
            link_parts[entry["link"]].append(info)
            prepared.append(info)
    if set(link_parts) != {link["name"] for link in spec["links"]}:
        raise ValueError("every rigid link must retain physical mass")
    if not any((p["collision_parts"] for p in prepared)):
        raise ValueError("physical asset needs at least one functional contact part")
    link_properties = {}
    for name, parts in link_parts.items():
        mass = sum((p["part_mass_kg"] for p in parts))
        center = sum((p["center"] * p["part_mass_kg"] for p in parts)) / mass
        inertia = np.zeros((3, 3))
        for part in parts:
            delta = part["center"] - center
            inertia += part["part_inertia_kg_m2"] + part["part_mass_kg"] * (
                np.eye(3) * np.dot(delta, delta) - np.outer(delta, delta)
            )
        eigenvalues = validate_inertia(mass, center, inertia)
        volumes_known = all((p["volume_m3"] is not None for p in parts))
        volume = sum((p["volume_m3"] for p in parts)) if volumes_known else None
        link_properties[name] = {
            "mass_kg": mass,
            "density_kg_m3": mass / volume if volume is not None else None,
            "volume_m3": volume,
            "center_of_mass_m": center.tolist(),
            "inertia_kg_m2": inertia.tolist(),
            "principal_inertias_kg_m2": eigenvalues,
            "aggregation": "per_authored_part_mass_weighted_parallel_axis",
            "objects": [
                {
                    "name": p["object"],
                    "mass_kg": p["part_mass_kg"],
                    "density_kg_m3": p["part_density_kg_m3"],
                    "volume_m3": p["volume_m3"],
                    "center_of_mass_m": p["center"].tolist(),
                    "inertia_kg_m2": p["part_inertia_kg_m2"].tolist(),
                    "mass_basis": p["mass_basis"],
                    "role": p["decision"]["role"],
                }
                for p in parts
            ],
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_asset_tree(export_dir / "meshes", out_dir / "meshes", pool=asset_pool(out_dir))
    (out_dir / "physical_meshes").mkdir()
    copy_asset(spec_path, out_dir / "kinematics.json", pool=asset_pool(out_dir))
    _write_json(out_dir / "physics_profile.json", profile)
    manifest = {
        "schema_version": 1,
        "urdf_path": str(out_dir / "object.urdf"),
        "profile_path": str(out_dir / "physics_profile.json"),
        "spec_path": str(out_dir / "kinematics.json"),
        "source": {
            "base_run": str(base_run),
            "urdf_sha256": file_hash(source_urdf),
            "spec_sha256": file_hash(spec_path),
            "export_manifest_sha256": file_hash(export_dir / "export_manifest.json"),
        },
        "geometry_meshes": [],
        "collision_meshes": [],
        "link_properties": link_properties,
        "collision_policy": profile["collision"],
        "collision_execution": {
            "workers": batch.workers,
            "threads": batch.threads,
            "unique_jobs": len(batch.groups),
            "reused_objects": len(batch.bindings) - len(batch.groups),
        },
        "object_physics": [
            {"link": p["link"], "object": p["object"], "decision": p["decision"]} for p in prepared
        ],
    }
    from urdf_core.degraded_asset import degradation_records

    from .scene_scale import load as load_scene_scale, FILE as SCALE_FILE
    scale_initialization = load_scene_scale(base_run)
    if scale_initialization is not None:
        copy_asset(base_run / SCALE_FILE, out_dir / SCALE_FILE, pool=asset_pool(out_dir))
        manifest['scene_scale'] = scale_initialization
        manifest['source']['scene_scale_sha256'] = file_hash(base_run / SCALE_FILE)

    manifest["upstream_degradations"] = degradation_records(base_run)
    prepared_receipt = base_run / "prepared_geometry" / "receipt.json"
    if prepared_receipt.is_file():
        copy_asset(prepared_receipt, out_dir / "prepared_geometry.json", pool=asset_pool(out_dir))
        manifest["source"]["prepared_geometry_sha256"] = file_hash(prepared_receipt)
        manifest["mass_collision_source"] = "same_prepared_link_local_mesh"
        for row in json.loads(prepared_receipt.read_text())["parts"]:
            if row["audit"].get("geometry_degraded"):
                manifest["upstream_degradations"].append(
                    {
                        "policy": "prepared_physical_mesh.v1",
                        "link": row["link"],
                        "object": row["object"],
                        "visual_geometry_changed": False,
                        "details": row["audit"],
                    }
                )
    preparation = base_run / "physical_geometry" / "receipt.json"
    if preparation.is_file():
        copy_asset(preparation, out_dir / "physical_geometry_preparation.json", pool=asset_pool(out_dir))
        manifest["source"]["physical_geometry_preparation_sha256"] = file_hash(preparation)
    for entry in prepared:
        changes = [
            row
            for row in entry.get("collision_audit", {}).get("components", [])
            if row.get("geometry_degraded")
        ]
        if changes:
            manifest["upstream_degradations"].append(
                {
                    "policy": ("discard_invalid_coacd_hulls.v1"
                               if any(row.get("hull_filtering") for row in changes)
                               else "coacd_native_crash_component_fallback.v1"),
                    "link": entry["link"],
                    "object": entry["object"],
                    "visual_geometry_changed": False,
                    "components": changes,
                }
            )
    manifest["collision_proxies"] = []
    manifest["collision_reduction"] = [
        {"link": p["link"], "object": p["object"], **p["collision_audit"]}
        for p in prepared
        if "collision_audit" in p
    ]
    manifest["physicalization_schema_version"] = 3
    manifest["proxy_validation"] = {
        "status": "not_required_functional_policy",
        "tested": False,
        "requested_approximation": "coacd",
    }
    manifest["joint_parameter_semantics"] = {
        "damping": "SI passive viscous damping; rotational USD conversion applied by runtime",
        "static_friction_effort": "native axis static friction, N or N*m; not URDF Coulomb friction",
        "dynamic_friction_effort": "native axis dynamic friction, N or N*m; not URDF Coulomb friction",
    }
    inventory_path = base_run / "geometry_inventory" / "inventory.json"
    if inventory_path.is_file():
        manifest["source"]["inventory_sha256"] = file_hash(inventory_path)
    collision_by_link = defaultdict(list)
    for index, entry in enumerate(prepared):
        clean_path = out_dir / "physical_meshes" / f"object_{index:04d}_original.obj"
        _write_obj(clean_path, entry["mesh"])
        freeze_file(clean_path, pool=asset_pool(out_dir))
        manifest["geometry_meshes"].append(
            {
                "link": entry["link"],
                "object": entry["object"],
                "path": str(clean_path),
                "sha256": file_hash(clean_path),
                "watertight": bool(entry["mesh"].is_watertight),
                "cleanup": entry["cleanup"],
                "sources": entry["sources"],
            }
        )
        manifest["geometry_meshes"][-1].update(
            role=entry["decision"]["role"],
            collision_enabled=bool(entry["collision_parts"]),
            collision_basis=entry.get("collision_basis", "none_decoration"),
        )
        for part_index, part in enumerate(entry["collision_parts"]):
            path = out_dir / "physical_meshes" / f"object_{index:04d}_collision_{part_index:04d}.obj"
            _write_obj(path, part)
            freeze_file(path, pool=asset_pool(out_dir))
            manifest["collision_meshes"].append(
                {
                    "link": entry["link"],
                    "object": entry["object"],
                    "path": str(path),
                    "sha256": file_hash(path),
                    "watertight": bool(part.is_watertight),
                    "approximation": "lightweight_proxy",
                    "mass_basis": entry["mass_basis"],
                    "collision_basis": entry["collision_basis"],
                }
            )
        for part_index, proxy in enumerate(entry["proxies"]):
            path = out_dir / "physical_meshes" / f"object_{index:04d}_proxy_{part_index:04d}.obj"
            _write_obj(path, proxy["mesh"])
            freeze_file(path, pool=asset_pool(out_dir))
            record = {
                "link": entry["link"],
                "object": entry["object"],
                "representation": proxy["representation"],
                "path": str(path),
                "sha256": file_hash(path),
                "triangles": len(proxy["mesh"].faces),
                "vertices": len(proxy["mesh"].vertices),
            }
            if "box" in proxy:
                record["box"] = proxy["box"]
            if "source_component_index" in proxy:
                record["source_component_index"] = proxy["source_component_index"]
            manifest["collision_proxies"].append(record)
            collision_by_link[entry["link"]].append({**record, "path": path.relative_to(out_dir).as_posix()})
    for link in root.findall("link"):
        name = link.get("name")
        for tag in ("inertial", "collision"):
            for node in link.findall(tag):
                link.remove(node)
        values = link_properties[name]
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": _fmt(values["center_of_mass_m"]), "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": _fmt([values["mass_kg"]])})
        tensor = np.asarray(values["inertia_kg_m2"])
        ET.SubElement(
            inertial,
            "inertia",
            {
                field: _fmt([tensor[i, j]])
                for field, i, j in (
                    ("ixx", 0, 0),
                    ("iyy", 1, 1),
                    ("izz", 2, 2),
                    ("ixy", 0, 1),
                    ("ixz", 0, 2),
                    ("iyz", 1, 2),
                )
            },
        )
        for proxy in collision_by_link[name]:
            collision = ET.SubElement(link, "collision")
            geometry = ET.SubElement(collision, "geometry")
            if proxy["representation"] == "box":
                box = proxy["box"]
                ET.SubElement(
                    collision, "origin", {"xyz": _fmt(box["center_m"]), "rpy": _fmt(box["rpy_rad"])}
                )
                ET.SubElement(geometry, "box", {"size": _fmt(box["size_m"])})
            else:
                ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
                ET.SubElement(geometry, "mesh", {"filename": proxy["path"], "scale": "1 1 1"})
    for joint in root.findall("joint"):
        params = profile["joints"].get(joint.get("name"))
        if params is not None:
            for node in joint.findall("dynamics"):
                joint.remove(node)
            dynamics = {"damping": _fmt([params["damping"]]), "friction": "0"}
            ET.SubElement(joint, "dynamics", dynamics)
    ET.indent(root)
    ET.ElementTree(root).write(out_dir / "object.urdf", encoding="utf-8", xml_declaration=True)
    manifest["urdf_sha256"] = file_hash(out_dir / "object.urdf")
    manifest["profile_sha256"] = file_hash(out_dir / "physics_profile.json")
    manifest["artifacts"] = {
        path.relative_to(out_dir).as_posix(): file_hash(path)
        for path in sorted(out_dir.rglob("*"))
        if path.is_file()
    }
    _write_json(out_dir / "asset_manifest.json", manifest)
    return manifest
