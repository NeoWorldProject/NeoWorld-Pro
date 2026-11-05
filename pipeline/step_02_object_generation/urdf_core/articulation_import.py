"""Reuse generated geometry and articulation, then obtain new review evidence."""
from pathlib import Path
import json
import shutil

from .run_state import file_hash, write_json
from .storage import copy_asset, asset_pool

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
    if manifest.get('status') == 'completed':
        verify_run(source, {"source_images": image_hashes})
        verify_source(source)
        proposal_source = 'kinematics.json'
    else:
        from .asset_delivery import verify as verify_delivery, validate_files
        from .runtime_snapshot import verified_runtime
        from .delivery_policy import POLICY as HARNESS_POLICY
        if config.get('harness_policy') != HARNESS_POLICY or config.get('delivery_policy') != 'best-effort':
            raise ValueError('Unsupported generated articulation source')
        if config.get('runtime_snapshot'):
            verified_runtime(config['runtime_snapshot'])
        verify_delivery(source)
        candidates = json.loads((source / 'generation_delivery_candidates.json').read_text())
        candidate = candidates['candidates'][candidates['best']]
        for relative, digest in candidate['files'].items():
            path = (source / relative).resolve()
            if not path.is_relative_to(source) or file_hash(path) != digest:
                raise ValueError('Saved generated articulation changed: ' + relative)
        proposal_source = str(Path(candidate['folder']) / 'kinematic_spec.json')
        if proposal_source not in candidate['files']:
            raise ValueError('Generated articulation has no bound exported specification')
        validate_files(source / candidate['folder'] / 'urdf_export',
                       json.loads((source / proposal_source).read_text()))
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
    files["kinematics.json"] = file_hash(source / proposal_source)
    return {
        "policy": POLICY, "source_run": str(source),
        'proposal_source': proposal_source,
        'source_evaluation_complete': manifest.get('status') == 'completed',
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
        path = (source / (binding.get('proposal_source', name) if name == 'kinematics.json' else name)).resolve()
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
        copy_asset(source / name, path, digest=digest, pool=asset_pool(target))
        if file_hash(path) != digest:
            raise ValueError("Articulation import copy changed: " + name)
        paths.append(path)
    receipt = target / "articulation_import.json"
    write_json(receipt, binding)
    verify(binding)
    return [*paths, receipt]


def proposal(binding, target):
    source = verify(binding)
    source_path = source / binding.get('proposal_source', 'kinematics.json')
    raw = source_path.read_text()
    target = Path(target)
    target.write_text(raw)
    receipt = target.with_name("proposal_import.json")
    write_json(receipt, {"policy": POLICY, "source": str(source_path),
                         "sha256": binding["files"]["kinematics.json"], "model_call": False})
    return raw, [target, receipt]


def bind_geometry(source, image_hashes, target_reference=None):
    """Import only a completed, content-bound geometry stage into a new run."""
    from .run_validation import reject_assisted_provenance
    source = Path(source).resolve()
    manifest_path = source / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    reject_assisted_provenance(source, manifest)
    config = manifest["config"]
    if (config.get("acceptance_policy") != "automatic_isaac_motion_v1"
            or config.get("harness_policy") != "object_pipeline.v1"):
        raise ValueError("Geometry import requires a current generation run")
    if config["source_images"] != image_hashes:
        raise ValueError("Geometry import source images differ")
    if ((config.get("target_reference") or {}).get("context")
            != (target_reference or {}).get("context")):
        raise ValueError("Geometry import target context differs")
    if config.get("runtime_snapshot"):
        from .runtime_snapshot import verified_runtime
        verified_runtime(config["runtime_snapshot"])
    geometry = manifest["stages"]["geometry"]
    if geometry.get("status") != "passed":
        raise ValueError("Geometry source has no completed geometry stage")
    files = {}
    for name in GEOMETRY_FILES:
        digest = file_hash(source / name)
        if geometry["artifacts"].get(name) != digest:
            raise ValueError("Geometry import checkpoint mismatch: " + name)
        files[name] = digest
    if (source / "geometry_degradation.json").exists():
        name = "geometry_degradation.json"
        if geometry["artifacts"].get(name) != file_hash(source / name):
            raise ValueError("Geometry degradation receipt is not bound")
        files[name] = file_hash(source / name)
    return {"policy": POLICY, "source_run": str(source),
            "source_manifest_sha256": file_hash(manifest_path), "files": files,
            "reuse": ["geometry"], "new_evaluation": ["articulation_generation", "critics", "isaac"]}
