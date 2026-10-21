"""Verify published objects in their immutable native runtime, in a fresh process.

Receipts are kept outside the publication. Reuse rehashes every file read by the
verifier, including transitive generation evidence, rather than trusting status.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from scene_refinement.protocol import file_hash, read_json, write_json

STAGE2 = Path(__file__).resolve().parents[2]/'step_02_object_generation'
if str(STAGE2) not in sys.path:
    sys.path.insert(0, str(STAGE2))


def publication_paths(root):
    """Resolve content-bound publication locations without changing their format.

    This lightweight check does not replace the independent frozen verifier.
    It also serves initial-assembly preflight and the independent USD audit.
    """
    root = Path(root).resolve()
    manifest_path = root / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    if manifest.get("config", {}).get("delivery_policy") != "best-effort":
        raise ValueError('Unsupported historical publication; use the current image pipeline')
    folder = (root/'final').resolve()
    if not folder.is_relative_to(root):
        raise ValueError('Delivery escapes object run')
    delivery_path = folder/'delivery_manifest.json'
    delivery = read_json(delivery_path)
    if delivery.get('schema_version') != 'best_effort_delivery.v1' or delivery.get('delivery_status') != 'completed':
        raise ValueError('Object has no technically completed delivery')
    result = {'publication_schema': delivery['schema_version'], 'delivery_manifest_path': str(delivery_path),
              'physics_contract_path': None, 'physics_report_path': None, 'support_rest_pose_path': None}
    for key, name in (('usd_path', 'object.usd'), ('asset_manifest_path', 'asset_manifest.json'),
                      ('kinematics_path', 'kinematics.json'), ('physics_profile_path', 'physics_profile.json')):
        path = folder/name
        if file_hash(path) != delivery['files'].get(name):
            raise ValueError('Published asset content changed: ' + name)
        result[key] = str(path)
    ref = delivery.get('source_report')
    if ref:
        path = Path(ref['path']).resolve()
        if not path.is_relative_to(root) or file_hash(path) != ref['sha256']:
            raise ValueError('Published native report binding changed')
        report = read_json(path)
        result['physics_report_path'] = str(path)
        contract = report.get('inputs', {}).get('contract')
        if contract:
            path = Path(contract['path']).resolve()
            if not path.is_relative_to(root) or file_hash(path) != contract['sha256']:
                raise ValueError('Published contract binding changed')
            result['physics_contract_path'] = str(path)
        rest = report.get('support_rest_pose') or {}
        if rest.get('verified') is True:
            path = Path(rest['path']).resolve()
            if not path.is_relative_to(root) or file_hash(path) != rest['sha256']:
                raise ValueError('Published support-rest state binding changed')
            result['support_rest_pose_path'] = str(path)
    return result


def runtime_binding(root):
    root = Path(root).resolve()
    manifest = read_json(root / "run_manifest.json")
    runtime = manifest.get("config", {}).get("runtime_snapshot")
    if runtime:
        from urdf_core.runtime_snapshot import verified_runtime
        package = verified_runtime(runtime)
        snapshot = read_json(runtime['path'])
        return package, {runtime['path']: runtime['sha256'], **snapshot['snapshot_files']}
    raise ValueError('Current publications require their runtime snapshot')


def publication_runtime_binding(root, reference=None):
    """Bind a publication controller separately from the recorded native worker."""
    root = Path(root).resolve()
    worker, hashes = runtime_binding(root)
    if reference is None:
        return worker, hashes
    paths = {}
    for key in ('snapshot', 'job', 'result', 'publication'):
        ref = reference[key]
        path = Path(ref['path']).resolve()
        if file_hash(path) != ref['sha256']:
            raise ValueError('Publication runtime evidence changed: ' + key)
        paths[key] = path
        hashes[str(path)] = ref['sha256']
    if paths['publication'] != root/'run_manifest.json':
        raise ValueError('Publication runtime refers to another asset')
    snapshot, job, result = (read_json(paths[k]) for k in ('snapshot', 'job', 'result'))
    files = snapshot.get('snapshot_files', {})
    if snapshot.get('kind') != 'physics' or not files:
        raise ValueError('Missing frozen publication controller inventory')
    for name, expected in files.items():
        path = Path(name).resolve()
        if not path.is_relative_to(paths['snapshot'].parent) or file_hash(path) != expected:
            raise ValueError('Frozen publication source changed: ' + name)
    command = job['command']
    agents = [Path(arg).resolve() for arg in command if str(arg).endswith('/generate_object_urdf.py')]
    outputs = [Path(command[i+1]).resolve() for i,arg in enumerate(command[:-1]) if arg in ('--out-dir','--out_dir')]
    if (len(agents) != 1 or str(agents[0]) not in files or outputs != [root]
            or result.get('returncode') != 0 or not job.get('job_id') or result.get('job_id') != job['job_id']):
        raise ValueError('Published controller job identity/output mismatch')
    package = agents[0].parent
    if package != paths['snapshot'].parent/'pipeline/step_02_object_generation':
        raise ValueError('Controller is outside the declared snapshot package')
    return package, {**hashes, **files}


def verify_record(record):
    root = Path(record["object_run_dir"]).resolve()
    receipt_path = record.get("verification_receipt_path")
    if receipt_path:
        if file_hash(receipt_path) != record.get("verification_receipt_sha256"):
            raise ValueError("Asset verification receipt changed")
        receipt = read_json(receipt_path)
        if (receipt.get('schema_version') != 'frozen_scene_asset.v2'
                or receipt.get('run_dir') != str(root)
                or receipt.get('verifier_sha256') != file_hash(__file__)
                or not receipt.get('input_hashes')):
            raise ValueError('Unsupported or changed asset verification receipt')
        if record.get('publication_runtime') and receipt.get('publication_runtime') != record['publication_runtime']:
            raise ValueError('Asset publication controller receipt mismatch')
        for path, expected in receipt["input_hashes"].items():
            if file_hash(path) != expected:
                raise ValueError("Verified object evidence changed: " + path)
        checked = receipt["result"]
    else:
        with tempfile.TemporaryDirectory(prefix="scene_asset_verify_") as folder:
            path = Path(folder) / "receipt.json"
            run_verifier(root, path, record.get('publication_runtime'))
            checked = read_json(path)["result"]
    modern = checked.get("publication_schema") == "best_effort_delivery.v1"
    if modern and receipt_path and receipt.get('schema_version') != 'frozen_scene_asset.v2':
        raise ValueError('Modern deliveries require a v2 verifier receipt')
    if Path(checked["usd_path"]).resolve() != Path(record["usd_path"]).resolve():
        raise ValueError("Asset USD differs from its verified delivery")
    if not modern:
        raise ValueError("Unsupported historical asset publication")
    if modern:
        from urdf_core.asset_delivery import physical_stage_result
        checked = physical_stage_result(root, checked)
    checked = dict(checked, technical_available=True,
                   upstream_quality={k: checked.get(k) for k in
                       ("physics_passed", "passed", "validation_status", "assessment_status", "evaluation_complete", "terminal")})
    return checked


def run_verifier(root, receipt, publication_runtime=None):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(map(str, [Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]/"step_02_object_generation"]))
    command = [sys.executable, "-B", "-m", "scene_refinement.frozen_assets",
               "--run", str(root), "--receipt", str(receipt)]
    if publication_runtime is not None:
        command += ['--publication-runtime-json', json.dumps(publication_runtime)]
    subprocess.run(command, env=environment, check=True)


def audit(root, destination, publication_runtime=None):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.is_relative_to(root) or destination.exists():
        raise ValueError("Use a new receipt outside the immutable publication")
    started = time.monotonic()
    package, hashes = publication_runtime_binding(root, publication_runtime)
    native_worker, _ = runtime_binding(root)
    if destination.is_relative_to(package.parent.parent):
        raise ValueError("Cannot write a receipt inside a frozen runtime")
    observed = set()

    def opened(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            name = os.fsdecode(args[0])
            # Record transitive evidence reads; interpreter packages are not asset evidence.
            if (Path(name).is_absolute() and not name.startswith(("/usr/", "/lib/", "/proc/", "/sys/"))
                    and "/.venv" not in name and not name.endswith((".pyc", ".so"))):
                observed.add(str(Path(name).resolve()))

    sys.addaudithook(opened)
    sys.dont_write_bytecode = True
    for name in list(sys.modules):
        if name == "urdf_core" or name.startswith("urdf_core.") or name == "physics_feedback" or name.startswith("physics_feedback."):
            del sys.modules[name]
    sys.path.insert(0, str(package))
    if publication_runtime is not None:
        from physics_feedback.loop import source_fingerprint, pipeline_code_fingerprint
        if source_fingerprint() != read_json(root/'run_manifest.json').get('config', {}).get('implementation'):
            raise ValueError('Publication controller implementation fingerprint mismatch')
        state_path = root/'research_state.json'
        if state_path.is_file() and pipeline_code_fingerprint() != read_json(state_path).get('implementation'):
            raise ValueError('Publication research implementation fingerprint mismatch')
    if read_json(root/'run_manifest.json').get('config', {}).get('delivery_policy') != 'best-effort':
        raise ValueError('Unsupported historical publication')
    from urdf_core.asset_delivery import verify_run
    result = verify_run(root, require_pass=False)
    for name in list(observed):
        if Path(name).is_file():
            hashes[name] = file_hash(name)
    # Recheck the frozen sources after the independent verifier has run.
    _, final_hashes = publication_runtime_binding(root, publication_runtime)
    if any(hashes.get(p) != h for p, h in final_hashes.items()):
        raise ValueError("Frozen runtime changed during verification")
    write_json(destination, {"schema_version": "frozen_scene_asset.v2", "run_dir": str(root),
        "verifier_sha256": file_hash(__file__), "runtime_package": str(package),
        "native_worker_package": str(native_worker), "publication_runtime": publication_runtime,
        "input_hashes": hashes, "result": result, "elapsed_seconds": time.monotonic() - started,
        "api_calls": 0, "gpu_trials": 0, "read_only_publication": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument('--publication-runtime-json', type=json.loads)
    options = parser.parse_args()
    audit(options.run, options.receipt, options.publication_runtime_json)
