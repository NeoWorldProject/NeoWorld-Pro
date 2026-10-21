"""Reuse generated geometry and articulation, then obtain new review evidence."""
from pathlib import Path
import json
import shutil

from .run_state import file_hash, write_json

POLICY = "generated_articulation_import.v1"
GEOMETRY_FILES = (
    "geometry_baseline.blend", "Res1_3D.png", "geometry_acceptance.json",
    "geometry_preparation.json", "1_model/Res1_3D_model_selected.py",
    "1_model/model_selection.json",
)


def bind(source, image_hashes, target_reference=None):
    from .asset_delivery import verify_run
    from .generation_delivery import verify_source
    from .run_validation import reject_assisted_provenance
    source = Path(source).resolve()
    manifest_path = source / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = manifest["config"]
    reject_assisted_provenance(source, manifest)
    if config.get("acceptance_policy") != "automatic_isaac_motion_v1":
        raise ValueError("Articulation import requires a generation run")
    if config["source_images"] != image_hashes:
        raise ValueError("Articulation import source images differ")
    prior_context = (config.get("target_reference") or {}).get("context")
    current_context = (target_reference or {}).get("context")
    if prior_context != current_context:
        raise ValueError("Articulation import target identity or scene context differs")
    verify_run(source, {"source_images": image_hashes})
    verify_source(source)
    geometry = manifest["stages"]["geometry"]
    if geometry["status"] != "passed":
        raise ValueError("Articulation import has no completed geometry stage")
    files = {}
    for name in GEOMETRY_FILES:
        path = source / name
        digest = file_hash(path)
        if geometry["artifacts"].get(name) != digest:
            raise ValueError("Imported geometry differs from its checkpoint: " + name)
        files[name] = digest
    for name in ("geometry_degradation.json",):
        if (source / name).exists():
            files[name] = file_hash(source / name)
    files["kinematics.json"] = file_hash(source / "kinematics.json")
    return {
        "policy": POLICY, "source_run": str(source),
        "source_manifest_sha256": file_hash(manifest_path), "files": files,
        "original_model": config["model"], "original_reasoning_effort": config["reasoning_effort"],
        "reuse": ["geometry", "generated_articulation"],
        "new_evaluation": ["blender_critic", "isaac_motion", "isaac_critic", "physics"],
    }


def verify(binding):
    source = Path(binding["source_run"])
    if (binding["policy"] != POLICY
            or file_hash(source / "run_manifest.json") != binding["source_manifest_sha256"]):
        raise ValueError("Articulation import source manifest changed")
    for name, digest in binding["files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or file_hash(path) != digest:
            raise ValueError("Articulation import source changed: " + name)
    return source


def copy_geometry(binding, target):
    source, target = verify(binding), Path(target)
    if target.resolve() == source or target.resolve().is_relative_to(source) or source.is_relative_to(target.resolve()):
        raise ValueError("Articulation import requires a separate output directory")
    paths = []
    for name, digest in binding["files"].items():
        if name == "kinematics.json":
            continue
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, path)
        if file_hash(path) != digest:
            raise ValueError("Articulation import copy changed: " + name)
        paths.append(path)
    receipt = target / "articulation_import.json"
    write_json(receipt, binding)
    verify(binding)
    return [*paths, receipt]


def proposal(binding, target):
    source = verify(binding)
    raw = (source / "kinematics.json").read_text()
    target = Path(target)
    target.write_text(raw)
    receipt = target.with_name("proposal_import.json")
    write_json(receipt, {"policy": POLICY, "source": str(source / "kinematics.json"),
                         "sha256": binding["files"]["kinematics.json"], "model_call": False})
    return raw, [target, receipt]
