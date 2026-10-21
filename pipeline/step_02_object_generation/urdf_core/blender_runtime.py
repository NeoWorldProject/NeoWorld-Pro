"""Deterministic Blender articulation and export, driven by the kinematic spec.

Run with Blender's --python-exit-code 1. The accepted baseline is always reopened;
evaluated geometry is frozen without changing its world-space shape. Controllers
are rigid, link-local meshes are baked once, and q=0 is the accepted input pose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
import xml.etree.ElementTree as ET

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urdf_core.kinematics import fk, pose_samples, validate_spec
from urdf_core.urdf_validation import (read_obj, read_urdf, urdf_fk, validate_urdf,
                                      validate_export_geometry_contract)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_array(matrix):
    return np.asarray([list(row) for row in matrix], dtype=np.float64)


def transform_points(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def material_info(material):
    rgba = list(material.diffuse_color) if material else [0.65, 0.65, 0.65, 1.0]
    if material and material.use_nodes and material.node_tree:
        node = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
        if node and node.inputs.get("Base Color"):
            rgba = list(node.inputs["Base Color"].default_value)
    return {"name": material.name if material else "default", "rgba": [float(x) for x in rgba]}


def face_components(mesh, world_vertices):
    """Exact connected polygon groups; no assumptions about semantic part names."""
    parents = list(range(len(mesh.polygons)))
    vertex_owner = {}

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for face in mesh.polygons:
        for vertex in face.vertices:
            if vertex in vertex_owner:
                a, b = root(face.index), root(vertex_owner[vertex])
                parents[a] = b
            else:
                vertex_owner[vertex] = face.index
    groups = {}
    for face in mesh.polygons:
        groups.setdefault(root(face.index), []).append(face.index)
    result = []
    for indices in groups.values():
        vertices = sorted({index for face in indices for index in mesh.polygons[face].vertices})
        points = world_vertices[vertices]
        result.append({"polygon_indices": indices, "polygon_count": len(indices),
                       "vertex_count": len(vertices), "bbox_world": {"min": points.min(0).tolist(), "max": points.max(0).tolist()}})
    return sorted(result, key=lambda item: item["polygon_indices"][0])


def capture_geometry():
    from urdf_core.blender_mesh_orientation import prepare_scene
    prepare_scene(validate_only=True)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    records = {}
    for obj in sorted(bpy.context.scene.objects, key=lambda item: item.name):
        # Viewport-hidden parts may still be visible in the accepted render.
        if obj.type not in {"MESH", "CURVE", "FONT"} or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        temporary = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            if temporary is None or not temporary.vertices or not temporary.polygons:
                raise ValueError(f"Visible object {obj.name!r} has no evaluated polygon geometry")
            mesh = temporary.copy()
        finally:
            evaluated.to_mesh_clear()
        local = np.asarray([list(vertex.co) for vertex in mesh.vertices], dtype=np.float64)
        world_matrix = matrix_array(evaluated.matrix_world)
        world = transform_points(world_matrix, local)
        if not np.isfinite(world).all() or not np.isfinite(world_matrix).all():
            raise ValueError(f"Non-finite evaluated geometry: {obj.name}")
        if abs(np.linalg.det(world_matrix[:3, :3])) < 1e-12:
            raise ValueError(f"Singular object transform: {obj.name}")
        mesh.calc_loop_triangles()
        records[obj.name] = {"source": obj, "mesh": mesh, "world": world, "matrix": world_matrix,
                             "triangles": np.asarray([list(tri.vertices) for tri in mesh.loop_triangles], dtype=np.int64)}
    if not records:
        raise ValueError("Baseline contains no visible renderable geometry")
    return records


def inventory_for(records, baseline):
    entries = []
    for name, record in records.items():
        mesh, world = record["mesh"], record["world"]
        entries.append({"name": name, "type": record["source"].type,
                        "vertex_count": len(world), "polygon_count": len(mesh.polygons),
                        "triangle_count": len(record["triangles"]), "matrix_world": record["matrix"].tolist(),
                        "bbox_world": {"min": world.min(0).tolist(), "max": world.max(0).tolist()},
                        "materials": [material_info(material) for material in mesh.materials],
                        "components": face_components(mesh, world)})
    points = np.concatenate([record["world"] for record in records.values()])
    return {"schema_version": 1, "coordinate_system": {"units": "metres", "up": "+Z", "forward": "-Y"},
            "pose": "q_zero", "baseline": str(baseline), "baseline_sha256": file_hash(baseline),
            "scene_diagonal": float(np.linalg.norm(points.max(0) - points.min(0))), "objects": entries}


def save_inspection(records, inventory, out_dir):
    arrays, objects = {}, []
    for index, (name, record) in enumerate(records.items()):
        prefix = f"object_{index:04d}"
        mesh = record["mesh"]
        arrays[prefix + "_world_vertices"] = record["world"]
        arrays[prefix + "_triangles"] = record["triangles"]
        arrays[prefix + "_triangle_materials"] = np.asarray([tri.material_index for tri in mesh.loop_triangles], dtype=np.int32)
        objects.append({"name": name, "array_prefix": prefix})
    np.savez_compressed(out_dir / "geometry_snapshot.npz", **arrays)
    write_json(out_dir / "geometry_snapshot.json", {"schema_version": 1, "baseline_sha256": inventory["baseline_sha256"],
               "arrays": "geometry_snapshot.npz", "objects": objects})
    write_json(out_dir / "inventory.json", inventory)


def split_record(record, name, polygon_indices):
    """Extract evaluated polygons, preserving UVs, material slots and positions."""
    original = record["mesh"]
    indices = sorted(polygon_indices)
    used = sorted({index for face in indices for index in original.polygons[face].vertices})
    remap = {old: new for new, old in enumerate(used)}
    mesh = bpy.data.meshes.new(name + "__evaluated_split")
    mesh.from_pydata([original.vertices[index].co[:] for index in used], [],
                     [[remap[index] for index in original.polygons[face].vertices] for face in indices])
    for material in original.materials:
        mesh.materials.append(material)
    for dest, source_index in zip(mesh.polygons, indices):
        source = original.polygons[source_index]
        dest.material_index, dest.use_smooth = source.material_index, source.use_smooth
    for layer in original.uv_layers:
        output = mesh.uv_layers.new(name=layer.name)
        for dest, source_index in zip(mesh.polygons, indices):
            for dest_loop, source_loop in zip(dest.loop_indices, original.polygons[source_index].loop_indices):
                output.data[dest_loop].uv = layer.data[source_loop].uv
    mesh.update()
    mesh.calc_loop_triangles()
    return {"source": record["source"], "mesh": mesh, "matrix": record["matrix"],
            "world": record["world"][used], "source_vertex_indices": used,
            "triangles": np.asarray([list(tri.vertices) for tri in mesh.loop_triangles], dtype=np.int64)}


def apply_splits(records, spec):
    result = dict(records)
    for split in spec.get("splits", []):
        source = result.pop(split["object"])
        for part in split["parts"]:
            result[part["name"]] = split_record(source, part["name"], part["polygon_indices"])
    return result


def origin_matrix(origin):
    matrix = Euler(origin["rpy"], "XYZ").to_matrix().to_4x4()
    matrix.translation = Vector(origin["xyz"])
    return matrix


def make_empty(name, parent=None):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = Matrix.Identity(4)
    return obj


def build_rig(records, spec):
    frames = fk(spec)
    controllers = {spec["root"]: make_empty("URDF_LINK__" + spec["root"])}
    joints, pending = {}, list(spec["joints"])
    rig_objects = list(controllers.values())
    while pending:
        for joint in pending[:]:
            if joint["parent"] not in controllers:
                continue
            pivot = make_empty("URDF_JOINT__" + joint["name"], controllers[joint["parent"]])
            pivot.matrix_basis = origin_matrix(joint["origin"])
            child = make_empty("URDF_LINK__" + joint["child"], pivot)
            child.rotation_mode = "AXIS_ANGLE"
            if joint["type"] != "fixed":
                child.rotation_axis_angle = [0.0, *joint["axis"]]
            controllers[joint["child"]], joints[joint["name"]] = child, (joint, child)
            rig_objects.extend([pivot, child])
            pending.remove(joint)
    owned, geometry = {}, {}
    for link in spec["links"]:
        for name in link["objects"]:
            record = records[name]
            mesh = record["mesh"].copy()
            bake = np.linalg.inv(frames[link["name"]]) @ record["matrix"]
            mesh.transform(Matrix(bake.tolist()))
            if np.linalg.det(bake[:3, :3]) < 0:
                mesh.flip_normals()
            mesh.update()
            obj = bpy.data.objects.new(name + "__rigged", mesh)
            bpy.context.scene.collection.objects.link(obj)
            obj.parent = controllers[link["name"]]
            obj.matrix_parent_inverse = Matrix.Identity(4)
            obj.matrix_basis = Matrix.Identity(4)
            obj["source_object"] = name
            obj["urdf_link"] = link["name"]
            geometry[name], owned[name] = obj, link["name"]
            rig_objects.append(obj)
    for source in {record["source"] for record in records.values()}:
        source.hide_render = True
        source.hide_set(True)
    bpy.context.view_layer.update()
    return {"controllers": controllers, "joints": joints, "geometry": geometry,
            "owned": owned, "objects": rig_objects}


def set_positions(rig, positions=None):
    positions = positions or {}
    for name, (joint, obj) in rig["joints"].items():
        q = float(positions.get(name, 0.0))
        obj.location = Vector(joint["axis"]) * q if joint["type"] == "prismatic" else Vector((0, 0, 0))
        if joint["type"] in {"revolute", "continuous"}:
            obj.rotation_axis_angle = [q, *joint["axis"]]
    bpy.context.view_layer.update()


def actual_world_vertices(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        vertices = np.asarray([list(vertex.co) for vertex in mesh.vertices], dtype=float)
        return transform_points(matrix_array(evaluated.matrix_world), vertices)
    finally:
        evaluated.to_mesh_clear()


def fmt(values):
    return " ".join(format(float(value), ".12g") for value in values)


def backface_source_indices(triangles, material):
    """OBJ lacks sidedness flags: preserve Blender's non-culling materials.

    Closed authored meshes can also have inward winding; do not infer intended
    visibility from topology or replace the source faces by guessed normals.
    """
    if material is not None and material.use_backface_culling:
        return []
    return list(range(len(triangles)))


def write_obj(path, mesh, triangles, rgba, backface_indices=()):
    used = sorted({index for triangle in triangles for index in triangle.vertices})
    remap = {old: index + 1 for index, old in enumerate(used)}
    uv_layer = mesh.uv_layers.active
    # Original triangles remain first and unchanged. Added back faces share
    # vertices, reverse UV corner order and negate normals without shell offsets.
    faces = [(triangle, False) for triangle in triangles]
    faces.extend((triangles[index], True) for index in backface_indices)
    # Mesh importers can ignore URDF inline colors. Carry the same base color
    # in a companion MTL so a native OBJ import preserves the material group.
    # Export filenames contain only validated link names and numeric indices.
    material_name = "material_" + path.stem
    mtl_path = path.with_suffix(".mtl")
    mtl_path.write_text(f"newmtl {material_name}\nKd {fmt(rgba[:3])}\nd {fmt([rgba[3]])}\n",
                        encoding="utf-8")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Link-local evaluated geometry; metre units.\n")
        handle.write(f"mtllib {mtl_path.name}\nusemtl {material_name}\n")
        for index in used:
            handle.write("v " + fmt(mesh.vertices[index].co) + "\n")
        normals = []
        for triangle, reverse in faces:
            loops = list(triangle.loops)
            for loop in reversed(loops) if reverse else loops:
                if uv_layer:
                    handle.write("vt " + fmt(uv_layer.data[loop].uv) + "\n")
                try:
                    normal = mesh.corner_normals[loop].vector
                except (AttributeError, IndexError):
                    normal = triangle.normal
                normals.append(tuple(-value if reverse else value for value in normal))
        for normal in normals:
            handle.write("vn " + fmt(normal) + "\n")
        corner = 1
        for triangle, reverse in faces:
            indices = []
            vertices = list(triangle.vertices)
            for vertex in reversed(vertices) if reverse else vertices:
                indices.append(f"{remap[vertex]}/{corner}/{corner}" if uv_layer else f"{remap[vertex]}//{corner}")
                corner += 1
            handle.write("f " + " ".join(indices) + "\n")
    return used


def export_urdf(rig, spec, out_dir):
    export_dir = out_dir / "urdf_export"
    # Rebuild this runtime's complete package; renamed links must not leave
    # unreferenced meshes from an earlier attempt in the published directory.
    if export_dir.is_symlink():
        raise ValueError("Refusing to replace a symlinked export directory")
    if export_dir.exists() and any(export_dir.iterdir()):
        marker = export_dir / ".urdf_core_owned"
        manifest = export_dir / "export_manifest.json"
        owned = marker.is_file() and marker.read_text() == "urdf_core_v1\n"
        if not owned and manifest.is_file():
            prior = json.loads(manifest.read_text())
            owned = prior.get("schema_version") == 1 and prior.get("mesh_frame") == "link_local" and prior.get("pose") == "q_zero"
        if not owned:
            raise ValueError("Refusing to replace an export directory not owned by this runtime")
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / ".urdf_core_owned").write_text("urdf_core_v1\n")
    meshes_dir = export_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    robot = ET.Element("robot", {"name": "generated_object"})
    exports = []
    for link in spec["links"]:
        elem = ET.SubElement(robot, "link", {"name": link["name"]})
        local_vertices = []
        for object_index, name in enumerate(link["objects"]):
            mesh = rig["geometry"][name].data
            mesh.calc_loop_triangles()
            local_vertices.extend([list(vertex.co) for vertex in mesh.vertices])
            groups = {}
            for triangle in mesh.loop_triangles:
                groups.setdefault(triangle.material_index, []).append(triangle)
            for material_index, triangles in sorted(groups.items()):
                filename = f"{link['name']}_{object_index:03d}_{material_index:03d}.obj"
                material = mesh.materials[material_index] if material_index < len(mesh.materials) else None
                info = material_info(material)
                backfaces = backface_source_indices(triangles, material)
                used = write_obj(meshes_dir / filename, mesh, triangles, info["rgba"], backfaces)
                for kind in ("visual", "collision"):
                    component = ET.SubElement(elem, kind)
                    ET.SubElement(component, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
                    geom = ET.SubElement(component, "geometry")
                    ET.SubElement(geom, "mesh", {"filename": "meshes/" + filename, "scale": "1 1 1"})
                    if kind == "visual":
                        mat = ET.SubElement(component, "material", {"name": f"{link['name']}_{object_index}_{material_index}"})
                        ET.SubElement(mat, "color", {"rgba": fmt(info["rgba"])})
                exports.append({"link": link["name"], "object": name, "path": str(meshes_dir / filename),
                                "material_index": material_index,
                                "source_vertex_indices": used, "triangles": len(triangles) + len(backfaces),
                                "backface_source_triangle_indices": backfaces,
                                "source_triangle_vertices": [list(triangle.vertices) for triangle in triangles]})
        # This finite box inertia is a placeholder, not physicalization evidence.
        points = np.asarray(local_vertices)
        size = np.maximum(points.max(0) - points.min(0), 1e-4)
        center = (points.max(0) + points.min(0)) / 2
        inertia = ET.SubElement(elem, "inertial")
        ET.SubElement(inertia, "origin", {"xyz": fmt(center), "rpy": "0 0 0"})
        ET.SubElement(inertia, "mass", {"value": "1"})
        ix, iy, iz = (size[1]**2 + size[2]**2)/12, (size[0]**2 + size[2]**2)/12, (size[0]**2 + size[1]**2)/12
        ET.SubElement(inertia, "inertia", {"ixx": str(ix), "iyy": str(iy), "izz": str(iz), "ixy": "0", "ixz": "0", "iyz": "0"})
    for joint in spec["joints"]:
        elem = ET.SubElement(robot, "joint", {"name": joint["name"], "type": joint["type"]})
        ET.SubElement(elem, "parent", {"link": joint["parent"]})
        ET.SubElement(elem, "child", {"link": joint["child"]})
        ET.SubElement(elem, "origin", {"xyz": fmt(joint["origin"]["xyz"]), "rpy": fmt(joint["origin"]["rpy"])})
        if joint["type"] != "fixed":
            ET.SubElement(elem, "axis", {"xyz": fmt(joint["axis"])})
        if joint["type"] in {"revolute", "prismatic"}:
            ET.SubElement(elem, "limit", {key: str(joint["limit"][key]) for key in ("lower", "upper", "effort", "velocity")})
        elif joint["type"] == "continuous" and joint.get("limit"):
            ET.SubElement(elem, "limit", {key: str(value) for key, value in joint["limit"].items()})
    ET.indent(robot)
    urdf_path = export_dir / "object.urdf"
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)
    write_json(export_dir / "export_manifest.json", {"schema_version": 1, "pose": "q_zero", "mesh_frame": "link_local",
               "visual_origin": "identity", "inertial_policy": "unit_mass_box_placeholder", "collision_policy": "visual_mesh",
               "sidedness_policy": "reverse_faces_for_nonculling_materials",
               "meshes": exports})
    return urdf_path, exports


def validate_runtime(records, rig, spec, urdf_path, exports, scene_diagonal):
    tolerance = max(1e-5, float(scene_diagonal) * 1e-5)
    report = {"schema_version": 1, "passed": True, "tolerance_metres": tolerance,
              "scope": "geometry preservation and sampled kinematic export equivalence; not physical validity",
              "errors": [], "rest_geometry": [], "poses": []}
    set_positions(rig)
    for name, record in records.items():
        actual, expected = actual_world_vertices(rig["geometry"][name]), record["world"]
        error = float(np.linalg.norm(actual - expected, axis=1).max()) if actual.shape == expected.shape else float("inf")
        report["rest_geometry"].append({"object": name, "vertex_count": len(expected), "max_error_metres": error})
        if error > tolerance:
            report["errors"].append(f"Rest geometry drift: {name} ({error})")
    schema_report = validate_urdf(urdf_path, spec=spec)
    report["urdf_validation"] = schema_report
    if not schema_report["passed"]:
        report["errors"].append("Independent URDF validation failed")
    parsed = read_urdf(urdf_path)
    contract = validate_export_geometry_contract(parsed, exports)
    report["export_geometry_contract"] = contract
    if not contract["passed"]:
        report["errors"].extend(contract["errors"])
    disk_meshes = {entry["path"]: read_obj(Path(entry["path"])) for entry in exports}
    for entry in exports:
        faces = disk_meshes[entry["path"]]["faces"]
        source_indices = entry["source_vertex_indices"]
        actual_faces = [[source_indices[index] for index in face] for face in faces]
        mesh = rig["geometry"][entry["object"]].data
        mesh.calc_loop_triangles()
        material_index = entry["material_index"]
        triangles = [triangle for triangle in mesh.loop_triangles if triangle.material_index == material_index]
        material = mesh.materials[material_index] if material_index < len(mesh.materials) else None
        source_faces = [list(triangle.vertices) for triangle in triangles]
        backfaces = backface_source_indices(triangles, material)
        expected_faces = source_faces + [list(reversed(source_faces[index])) for index in backfaces]
        if (entry["source_triangle_vertices"] != source_faces
                or entry["backface_source_triangle_indices"] != backfaces
                or actual_faces != expected_faces):
            raise ValueError(f"Exported OBJ triangles differ from evaluated Blender geometry: {entry['path']}")
    for sample in pose_samples(spec):
        set_positions(rig, sample["positions"])
        independent = urdf_fk(parsed, sample["positions"])
        frame_error = 0.0
        for link, controller in rig["controllers"].items():
            actual = matrix_array(controller.matrix_world)
            expected = independent[link]
            frame_error = max(frame_error, float(np.max(np.abs(actual - expected))))
        geometry_error = 0.0
        actual_vertices = {name: actual_world_vertices(obj) for name, obj in rig["geometry"].items()}
        for entry in exports:
            mesh = disk_meshes[entry["path"]]
            expected = transform_points(independent[entry["link"]], mesh["vertices"])
            actual = actual_vertices[entry["object"]][entry["source_vertex_indices"]]
            if len(mesh["faces"]) != entry["triangles"] or actual.shape != expected.shape:
                raise ValueError(f"Exported mesh topology/vertex count mismatch: {entry['path']}")
            geometry_error = max(geometry_error, float(np.linalg.norm(actual - expected, axis=1).max()))
        item = {"name": sample["name"], "positions": sample["positions"], "max_frame_element_error": frame_error,
                "max_vertex_error_metres": geometry_error}
        report["poses"].append(item)
        if frame_error > max(1e-5, tolerance) or geometry_error > tolerance:
            report["errors"].append(f"Blender versus URDF/OBJ mismatch at {sample['name']}")
    set_positions(rig)
    report["passed"] = not report["errors"]
    return report


def make_animation(rig, spec):
    scene = bpy.context.scene
    movable = [joint for joint in spec["joints"] if joint["type"] != "fixed"]
    schedule = []
    for index, joint in enumerate(movable):
        lower, upper = (-math.pi/2, math.pi/2) if joint["type"] == "continuous" else (joint["limit"]["lower"], joint["limit"]["upper"])
        start = index * 90 + 1
        for frame, q in [(start, 0), (start + 29, lower), (start + 59, upper), (start + 89, 0)]:
            set_positions(rig, {joint["name"]: q})
            for _, (other, obj) in rig["joints"].items():
                if other["type"] == "prismatic":
                    obj.keyframe_insert(data_path="location", frame=frame)
                elif other["type"] in {"revolute", "continuous"}:
                    obj.keyframe_insert(data_path="rotation_axis_angle", frame=frame)
        schedule.append({"joint": joint["name"], "start_frame": start, "end_frame": start + 89,
                         "lower": lower, "upper": upper, "other_joints": "q_zero"})
    for _, obj in rig["joints"].values():
        if obj.animation_data and obj.animation_data.action:
            for curve in obj.animation_data.action.fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR"
    scene.frame_start, scene.frame_end = 1, max(90, len(movable) * 90)
    scene.render.fps, scene.render.fps_base = 30, 1.0
    scene.frame_set(1)
    set_positions(rig)
    return schedule


def export_usd(rig, path):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in rig["objects"]:
        obj.select_set(True)
    result = bpy.ops.wm.usd_export(filepath=str(path), selected_objects_only=True, export_animation=False,
                                   export_materials=True, export_textures=True, relative_paths=True)
    if "FINISHED" not in result or not path.is_file():
        raise RuntimeError("Blender USD export did not produce object.usd")
    bpy.ops.object.select_all(action="DESELECT")


def render_preview(args, rig, schedule):
    scene = bpy.context.scene
    if not scene.camera:
        raise RuntimeError("Accepted geometry baseline has no active camera")
    original = {"percentage": scene.render.resolution_percentage, "filepath": scene.render.filepath,
                "format": scene.render.image_settings.file_format, "threads_mode": scene.render.threads_mode,
                "threads": scene.render.threads}
    samples = []
    try:
        scene.render.resolution_percentage = min(original["percentage"], args.render_resolution_percentage)
        scene.render.threads_mode, scene.render.threads = "FIXED", 4
        if args.render_samples_per_joint == 0:
            scene.render.image_settings.file_format = "FFMPEG"
            scene.render.ffmpeg.format, scene.render.ffmpeg.codec = "MPEG4", "H264"
            scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
            scene.render.filepath = str(args.out_dir / "Res2_Articulation.mp4")
            bpy.ops.render.render(animation=True)
        else:
            frames_dir = args.out_dir / "preview_samples"
            frames_dir.mkdir(exist_ok=True)
            scene.render.image_settings.file_format = "PNG"
            segments = schedule or [{"joint": None, "start_frame": 1, "end_frame": 90}]
            concat = []
            for segment in segments:
                start, end = segment["start_frame"], segment["end_frame"]
                offsets = [0, 29, 59, 89]
                while len(offsets) < min(90, args.render_samples_per_joint):
                    left, right = max(zip(offsets[:-1], offsets[1:]), key=lambda pair: pair[1] - pair[0])
                    offsets = sorted([*offsets, (left + right) // 2])
                for index, offset in enumerate(offsets):
                    frame = start + offset
                    scene.frame_set(frame)
                    bpy.context.view_layer.update()
                    path = frames_dir / f"frame_{frame:06d}.png"
                    scene.render.filepath = str(path)
                    bpy.ops.render.render(write_still=True)
                    if not path.is_file():
                        raise RuntimeError(f"Missing rendered preview sample: {path}")
                    positions = {}
                    for name, (joint, obj) in rig["joints"].items():
                        if joint["type"] == "prismatic":
                            positions[name] = float(obj.location.dot(Vector(joint["axis"])))
                        elif joint["type"] in {"continuous", "revolute"}:
                            positions[name] = float(obj.rotation_axis_angle[0])
                    duration_frames = offsets[index+1] - offset if index+1 < len(offsets) else end - frame + 1
                    samples.append({"joint": segment["joint"], "scene_frame": frame, "positions": positions,
                                    "path": str(path), "video_hold_frames": duration_frames})
                    escaped = str(path).replace("'", "'\\''")
                    concat.extend([f"file '{escaped}'", f"duration {duration_frames/30:.12f}"])
            concat.append(concat[-2])
            concat_path = args.out_dir / "preview.ffconcat"
            concat_path.write_text("\n".join(concat) + "\n", encoding="utf-8")
            command = [args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                       "-i", str(concat_path), "-r", "30", "-frames:v", str(scene.frame_end), "-c:v", "libx264",
                       "-threads", "4", "-pix_fmt", "yuv420p", str(args.out_dir / "Res2_Articulation.mp4")]
            from urdf_core.media_execution import run_ffmpeg
            result = run_ffmpeg(command, progress_path=args.out_dir / 'preview_encode.progress', text=True)
            (args.out_dir / "preview_encode.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError("Preview ffmpeg encoding failed; see preview_encode.log")
        if not (args.out_dir / "Res2_Articulation.mp4").is_file():
            raise RuntimeError("Animation renderer produced no MP4")
        write_json(args.out_dir / "preview_sampling.json", {"fps": 30, "video_frames": scene.frame_end,
                   "method": "all_frames" if args.render_samples_per_joint == 0 else "rendered_samples_held_between_frames",
                   "requested_samples_per_joint": args.render_samples_per_joint,
                   "resolution_percentage": scene.render.resolution_percentage, "samples": samples})
    finally:
        scene.frame_set(1)
        set_positions(rig)
        scene.render.resolution_percentage = original["percentage"]
        scene.render.filepath = original["filepath"]
        scene.render.image_settings.file_format = original["format"]
        scene.render.threads_mode, scene.render.threads = original["threads_mode"], original["threads"]


def build(args, records, inventory):
    spec = validate_spec(json.loads(args.spec.read_text(encoding="utf-8")), inventory)
    write_json(args.out_dir / "kinematic_spec.json", spec)
    records = apply_splits(records, spec)
    rig = build_rig(records, spec)
    urdf_path, exports = export_urdf(rig, spec, args.out_dir)
    report = validate_runtime(records, rig, spec, urdf_path, exports, inventory["scene_diagonal"])
    report["baseline_sha256"] = inventory["baseline_sha256"]
    report["spec_sha256"] = file_hash(args.out_dir / "kinematic_spec.json")
    write_json(args.out_dir / "validation_report.json", report)
    if not report["passed"]:
        raise RuntimeError("Geometry/kinematics validation failed; see validation_report.json")
    schedule = make_animation(rig, spec)
    write_json(args.out_dir / "animation_schedule.json", {"fps": 30, "frames_per_joint": 90, "joints": schedule})
    export_usd(rig, args.out_dir / "urdf_export" / "object.usd")
    scene = bpy.context.scene
    scene.frame_set(1)
    set_positions(rig)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_dir / "pipeline_scene.blend"))
    write_json(args.out_dir/'export_complete.json', {'passed': True, 'files': {
        str(p.relative_to(args.out_dir)): file_hash(p) for p in
        [args.out_dir/'pipeline_scene.blend', args.out_dir/'kinematic_spec.json', *sorted((args.out_dir/'urdf_export').rglob('*'))]
        if p.is_file()}})
    if args.render:
        render_preview(args, rig, schedule)
    scene.frame_set(1)
    set_positions(rig)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out_dir / "pipeline_scene.blend"))
    report["artifacts"] = {"urdf": str(urdf_path), "usd": str(urdf_path.with_suffix(".usd")),
                            "blend": str(args.out_dir / "pipeline_scene.blend"),
                            "video": str(args.out_dir / "Res2_Articulation.mp4") if args.render else None}
    report["usd_scope"] = "static visual geometry at q_zero; no UsdPhysics articulation"
    write_json(args.out_dir / "validation_report.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inspect", "build"))
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-samples-per-joint", type=int, default=7)
    parser.add_argument("--render-resolution-percentage", type=int, default=50)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    args.blend, args.out_dir = args.blend.resolve(), args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.render_samples_per_joint != 0 and args.render_samples_per_joint < 4:
            raise ValueError("--render-samples-per-joint must be 0 or at least 4")
        if not 1 <= args.render_resolution_percentage <= 100:
            raise ValueError("--render-resolution-percentage must be in 1..100")
        if args.command == "build" and args.spec is None:
            raise ValueError("build requires --spec")
        bpy.ops.wm.open_mainfile(filepath=str(args.blend), use_scripts=False)
        records = capture_geometry()
        inventory = inventory_for(records, args.blend)
        if args.command == "inspect":
            save_inspection(records, inventory, args.out_dir)
        else:
            build(args, records, inventory)
        print(json.dumps({"status": "success", "command": args.command, "out_dir": str(args.out_dir)}, ensure_ascii=False))
    except Exception as exc:
        error_path = args.out_dir / "validation_report.json"
        report = json.loads(error_path.read_text()) if error_path.exists() else {"schema_version": 1, "errors": []}
        report["passed"] = False
        report.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}")
        write_json(error_path, report)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
