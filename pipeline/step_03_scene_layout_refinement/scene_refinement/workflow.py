"""Content-bound scene orchestration; no hand-authored case corrections."""
from __future__ import annotations

import copy
import fcntl
import html
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from .protocol import SCHEMA_VERSION, load_config, read_json, write_json, digest, file_hash
from .provider import SceneAPI, APIBudgetExhausted, SemanticUnavailable, safe_error

STAGE = Path(__file__).resolve().parents[1]
PROJECT = STAGE.parents[1]
sys.path.insert(0, str(STAGE.parent / "step_01_scene_parsing"))
sys.path.insert(0, str(STAGE.parent / "step_02_object_generation"))


class UpstreamRejected(ValueError):
    pass


class InfrastructureError(RuntimeError):
    pass


def verify_input_layout(path, image_path=None):
    from scene_core.schema import normalize_layout
    from urdf_core.layout_handoff import verify_layout_delivery
    try:
        layout, _ = verify_layout_delivery(path, image_path)
        return normalize_layout(layout)
    except (ValueError, OSError) as exc:
        raise UpstreamRejected(str(exc)) from exc



def source_fingerprint(args, config):
    from .worker_client import native_runtime_binding, source_hashes
    files = [*sorted(STAGE.rglob("*.py")),
             *sorted((STAGE / "resources").glob("*.md")),
             *sorted((STAGE.parent / "step_02_object_generation/physics_feedback").glob("*.py")),
             *sorted((STAGE.parent / "step_02_object_generation/urdf_core").glob("*.py")),
             *sorted((STAGE.parent / "step_01_scene_parsing/scene_core").glob("*.py"))]
    manifest = read_json(args.asset_manifest)
    sources = {**source_hashes(), **{str(p.resolve()): file_hash(p) for p in files}}
    for item in manifest.get("objects", []):
        for key in ("usd_path", "urdf_path", "physics_report_path", "physics_profile_path", "physics_contract_path"):
            if item.get(key) and Path(item[key]).is_file():
                sources[str(Path(item[key]).resolve())] = file_hash(item[key])
    inputs = {str(Path(p).resolve()): file_hash(p) for p in (args.layout_json, args.asset_manifest, args.scene_image)}
    from .input_adapter import referenced_files
    inputs.update({str(p): file_hash(p) for p in referenced_files(args.asset_manifest)})
    for name in ("refine_summary.json", "run_manifest.json"):
        path = args.layout_json.resolve().parent / name
        if path.is_file():
            inputs[str(path)] = file_hash(path)
    for item in manifest.get("objects", []):
        if item.get("object_run_dir"):
            path = Path(item["object_run_dir"]) / "run_manifest.json"
            if path.is_file():
                inputs[str(path.resolve())] = file_hash(path)
    binding = {"schema_version": SCHEMA_VERSION, "config": config, "sources": sources, "inputs": inputs,
               "model": args.model, "reasoning_effort": args.reasoning_effort, "seed": args.seed,
               "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
               "isaac_python": str(args.isaac_python.resolve()), "runtime": args.runtime, "ground_truth_allowed": False}
    binding["native_runtime"] = (getattr(args, "native_runtime_binding", None)
                                 or native_runtime_binding(args.isaac_python))
    return digest(binding), binding



def verify_asset_inputs(layout, manifest):
    from .frozen_assets import verify_record
    records = {r["object_id"]: r for r in manifest.get("objects", [])}
    if len(records) != len(manifest.get("objects", [])):
        raise UpstreamRejected("Asset manifest has duplicate IDs")
    for obj in layout["objects_data"]:
        if obj["is_environment_support"]:
            continue
        record = records.get(obj["object_id"], {})
        if not record.get("object_run_dir"):
            raise UpstreamRejected("Missing a verified dynamic asset for " + obj["object_id"])
        for field in ("object_run_dir", "usd_path"):
            if not record.get(field) or not Path(record[field]).is_absolute():
                raise UpstreamRejected("Official asset manifest paths must be absolute: " + field)
        try:
            checked = verify_record(record)
            if (Path(checked["usd_path"]).resolve() != Path(record["usd_path"]).resolve()):
                raise ValueError("Dynamic USD does not match the verified object delivery")
        except Exception as exc:
            raise UpstreamRejected(f"Object {obj['object_id']} is not a verified physical asset: {safe_error(exc)}") from exc



def search_contract(layout, config, scene=None):
    """Stable BFS sibling blocks; geometric scales are frozen for this attempt."""
    objects = {o["object_id"]: o for o in layout["objects_data"]}
    names = {o["name"]: ident for ident, o in objects.items()}
    children = {names[name]: [names[c] for c in node.get("children", [])]
                for name, node in layout["scene_tree"].items()}
    all_children = {c for group in children.values() for c in group}
    roots = sorted(set(objects) - all_children)
    queue, groups = list(roots), [roots]
    visited = set()
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        group = sorted(children.get(node, []))
        groups.append(group)
        queue.extend(group)
    blocks = []
    for group in groups:
        dynamic = [ident for ident in group if not objects[ident]["is_environment_support"]]
        blocks.extend(dynamic[i:i + config["block_size"]] for i in range(0, len(dynamic), config["block_size"]))
    specs = []
    for ident in [i for b in blocks for i in b]:
        obj = objects[ident]
        if scene is not None:
            from .assets import transformed
            prepared = scene["objects"][ident]
            points = transformed(np.asarray(prepared["rest_points"]) * prepared["initial_scale"],
                                 prepared["initial_transform"])
            basis = np.eye(3)
            binding = scene["bindings"].get(ident)
            if binding:
                parent = scene["objects"][binding["parent_id"]]
                basis = (np.asarray(parent["initial_transform"])[:3, :3] @
                         np.asarray(parent["asset"]["body_rest_transforms"][binding["parent_link"]])[:3, :3] @
                         np.asarray(binding["support_frame_local"]["basis"]))
            dims = np.ptp(points @ basis[:, :2], axis=0)
        elif obj["support_relation"] == "attached":
            dims = sorted(obj["obb"]["size_m"], reverse=True)[:2]
        else:
            from scipy.spatial.transform import Rotation
            rotation = Rotation.from_euler("xyz", obj["obb"]["rotation_xyz_deg"], degrees=True).as_matrix()
            dims = (np.abs(rotation) @ np.asarray(obj["obb"]["size_m"]))[:2]
        length = max(float(np.linalg.norm(dims)), 1e-6)
        sigma = config["translation_std_fraction"] * length
        bound = config["translation_bound_std"] * sigma
        angle_bound = math.radians(config["yaw_bound_deg"])
        specs.append({"object_id": ident, "sigma": [sigma, sigma, math.radians(config["yaw_std_deg"]), config["log_scale_std"]],
                      "bounds": [[-bound, bound], [-bound, bound], [-angle_bound, angle_bound],
                                 [math.log(config["scale_min"]), math.log(config["scale_max"])]]})
    return specs, blocks



def freeze_camera_envelope(scene, specs, config):
    """One conservative frame for the entire bounded search, never auto-follow.

    A legal child has an anchor or projected centroid on its actual parent
    patch. Every vertex is at most one object diagonal from that point.
    Propagate these geometric bounds along the frozen support tree.
    """
    from .assets import transformed
    from .support import world_region
    scene["fixed_render_bounds_m"] = copy.deepcopy(scene["fixed_camera_bounds_m"])
    spec_by_id = {s["object_id"]: s for s in specs}
    envelopes = {}
    for oid in scene["object_order"]:
        obj = scene["objects"][oid]
        points = np.asarray(obj["rest_points"]) * obj["initial_scale"]
        world = transformed(points, obj["initial_transform"])
        if obj["is_environment_support"]:
            lo, hi = world.min(axis=0), world.max(axis=0)
        else:
            diagonal = float(np.linalg.norm(np.ptp(points, axis=0))) * config["scale_max"]
            binding = scene["bindings"].get(oid)
            if binding:
                parent = scene["objects"][obj["parent_id"]]
                if parent["is_environment_support"]:
                    _, _, triangles = world_region(binding["region"], parent,
                        parent["initial_transform"], parent["initial_scale"])
                    vertices = triangles.reshape(-1, 3)
                    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
                else:
                    lo, hi = envelopes[obj["parent_id"]]
                lo, hi = lo - diagonal - .002, hi + diagonal + .002
            else:
                center = np.asarray(obj["layout_object"]["obb"]["center_m"])
                limits = spec_by_id[oid]["bounds"][:2]
                travel = float(np.linalg.norm([max(abs(v) for v in lim) for lim in limits])) * 2.
                lo, hi = center - diagonal - travel, center + diagonal + travel
        envelopes[oid] = (lo, hi)
    scene["fixed_camera_bounds_m"] = {
        "min": np.min([v[0] for v in envelopes.values()], axis=0).tolist(),
        "max": np.max([v[1] for v in envelopes.values()], axis=0).tolist()}
    scene["camera_envelope_policy"] = "frozen_support_geometry_and_maximum_candidate_scale"
    write_json(Path(scene["output_dir"]) / "prepared_scene.json", scene)



def _files_hashes(paths):
    return {str(Path(p).resolve()): file_hash(p) for p in paths if Path(p).is_file() and not str(p).endswith(".tmp")}



def _verify_hashes(hashes):
    return bool(hashes) and all(Path(p).is_file() and file_hash(p) == value for p, value in hashes.items())



def _report_page(root, report):
    parts = ["<!doctype html><meta charset='utf-8'><title>Scene refinement</title>",
             "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto}img{max-width:48%}pre{white-space:pre-wrap}</style>",
             "<h1>Scene layout refinement</h1>",
             f"<p>Status: {html.escape(report['status'])}; accepted: {report.get('accepted', False)}</p>"]
    for attempt in report.get("attempts", []):
        parts.append(f"<h2>Initialization {attempt['index']}</h2>")
        for title, candidate in (("Initial", attempt.get("initial")), ("Selected", attempt.get("selected"))):
            if not candidate:
                continue
            parts.append("<h3>" + title + "</h3>")
            for item in candidate.get("images", candidate.get("physics", {}).get("images", [])):
                relative = os.path.relpath(item["path"], root)
                parts.append('<img src="' + html.escape(relative, quote=True) + '">')
        parts.append("<pre>" + html.escape(str({k: v for k, v in attempt.items() if k not in {"initial", "selected", "search"}})) + "</pre>")
    (root / "index.html").write_text("\n".join(parts), encoding="utf-8")



def run_refinement(args):
    root = args.out_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this scene refinement run") from exc
        return _run_locked(args, root)


def _run_attempt(*, args, config, layout, assets, attempt_dir, fingerprint, api, prompts, report):
    from .composition import prepare_scene
    from .metrics import physics_settings
    from .search import run
    def resolve_binding(request):
        images = []
        if 'diagram_triangles_world' in request:
            from .support_evidence import diagram
            request, evidence = diagram(request, attempt_dir/'support_evidence')
            api.register_support_diagram(evidence, request)
            images = [evidence['path']]
        from .provider import APITechnicalFailure
        try:
            return api.call('support_' + digest(request)[:20], prompts['support'], {'layout': layout, 'catalog': request}, images=images)
        except APITechnicalFailure:
            raise
        except (SemanticUnavailable, APIBudgetExhausted):
            # Content failure leaves the frozen binding unresolved. It is a
            # quality rejection, not a reason to stop before the CEM budget.
            return {}
    started = time.monotonic()
    scene = prepare_scene(layout, assets, attempt_dir/'prepared', resolve_binding=resolve_binding,
                          retain_unresolved=True)
    report['unresolved_support_bindings'] = scene.get('unresolved_bindings', {})
    report.setdefault('timings', {})['preparation_seconds'] = time.monotonic()-started
    scene['physics_settings'] = physics_settings(config)
    scene['support_placement_policy'] = 'component_support_alignment.v1'
    scene['derivative_cache_dir'] = str(Path(args.runtime['cache_dir'])/'scaled')
    specs, blocks = search_contract(layout, config, scene)
    if not specs:
        from .composition import build_candidate
        report.update(initial=build_candidate(scene, {}, attempt_dir/'initial_environment'),
                      stop_reason='environment_only', accepted=False, reconstruction_complete=False)
        return None
    freeze_camera_envelope(scene, specs, config)
    write_json(attempt_dir/'support_bindings.json', scene['bindings'])
    write_json(attempt_dir/'search_contract.json', {'objects': specs, 'blocks': blocks, 'config': config})
    if scene.get('unresolved_bindings'):
        from .composition import build_candidate
        from .terminal_settle import run as terminal_run
        from .worker_pool import scene_worker
        try:
            initial = build_candidate(scene, {}, attempt_dir/'initial_unassessed')
        except Exception as error:
            from .support import SupportUnresolved
            if not isinstance(error, SupportUnresolved): raise
            initial = None
        report.update(initial=initial, stop_reason='unsearchable_support_binding', accepted=False)
        with scene_worker(config=config, work_dir=attempt_dir/'worker', isaac_python=args.isaac_python) as worker:
            result = terminal_run(worker, {}, initial, attempt_dir/'terminal_fallback', config, scene)
        report['terminal_fallback'] = result
        return result['selected']
    return run(args=args, config=config, scene=scene, specs=specs, blocks=blocks, attempt_dir=attempt_dir,
               fingerprint=fingerprint, api=api, prompts=prompts, report=report)


def _recover_delivery_candidates(attempt_dir, attempt):
    from .delivery import select_fallback, terminal_candidate
    terminal = attempt_dir/'terminal_fallback/result.json'
    if terminal.is_file():
        from .terminal_settle import valid_result
        result = read_json(terminal)
        if not valid_result(result):
            raise InfrastructureError('Terminal fallback artifacts changed')
        attempt['terminal_fallback'] = result
        return result['selected'], result['initial']
    checkpoint = attempt_dir/'search_checkpoint.json'
    state = read_json(checkpoint) if checkpoint.is_file() else {}
    selected = select_fallback(state)
    excluded = set()
    while selected:
        try:
            selected = terminal_candidate(selected, attempt_dir/'delivery_states'/selected['candidate_id'])
            break
        except (ValueError, KeyError, FileNotFoundError) as exc:
            excluded.add(selected['selection_key'])
            attempt.setdefault('delivery_ineligible', []).append({
                'candidate_id': selected['candidate_id'], 'reason': safe_error(exc)})
            selected = select_fallback(state, excluded)
    initial = attempt.get('initial') or state.get('candidates', {}).get(state.get('initial_key'))
    if initial is None:
        for job in sorted((attempt_dir/'candidates').glob('*/native_job.json')):
            candidate = read_json(job)['candidate']
            if all(all(v == 0 for v in values) for values in candidate['parameters'].values()):
                initial = candidate
                break
    if initial is None and (attempt_dir/'prepared/prepared_scene.json').exists():
        from .composition import build_candidate
        prepared = read_json(attempt_dir/'prepared/prepared_scene.json')
        from .support import SupportUnresolved
        try:
            initial = build_candidate(prepared, {}, attempt_dir/'initial_unassessed')
        except SupportUnresolved:
            from .terminal_settle import legal_initial
            initial = legal_initial(prepared, attempt_dir/'initial_unassessed_degraded')
    return selected, initial


def _run_locked(args, root):
    entry = time.time()
    manifest_path, report_path = root/'run_manifest.json', root/'scene_refinement_report.json'
    prior = read_json(manifest_path) if manifest_path.exists() else None
    report = {'schema_version': SCHEMA_VERSION, 'status': 'running', 'accepted': False, 'delivered': False,
              'started_unix': prior.get('started_unix', entry) if prior else getattr(args, 'started_unix', entry),
              'attempts': [], 'errors': [], 'ground_truth_allowed': False}
    if getattr(args, 'batch_enqueued_unix', None) is not None:
        report['batch_enqueued_unix'] = args.batch_enqueued_unix
        report['batch_queue_seconds'] = max(0., report['started_unix']-args.batch_enqueued_unix)
    owns_run = False
    selected = initial = None
    attempt_dir = root/'attempts/attempt_00'
    attempt = {'index': 0, 'accepted': False}
    try:
        config = load_config(args.config)
        if shutil.disk_usage(root).free < 1200 * 1024**2:
            raise InfrastructureError('Less than 1200 MiB free for scene startup')
        if read_json(args.asset_manifest).get('schema_version') != 'scene_assets.visual_overlay.v1':
            from .material_preparation import prepare
            start = time.monotonic()
            original_manifest = args.asset_manifest.resolve()
            material_root = Path(args.runtime['cache_dir'])/'visual_materials'/digest(
                {'path': str(original_manifest), 'sha256': file_hash(original_manifest)})
            args.asset_manifest = prepare(original_manifest, material_root, Path(args.runtime['blender']))
            report['material_preparation'] = {'seconds': time.monotonic()-start,
                'source_manifest': str(original_manifest), 'effective_manifest': str(args.asset_manifest)}
        fingerprint, binding = source_fingerprint(args, config)
        if prior:
            if not args.resume or prior.get('schema_version') != SCHEMA_VERSION or prior.get('fingerprint') != fingerprint:
                raise UpstreamRejected('Use a new output: checkpoint version, sources, inputs or settings differ')
            if prior.get('status') == 'completed':
                if not _verify_hashes(prior.get('artifacts', {})):
                    raise UpstreamRejected('Completed evidence changed')
                return read_json(report_path)
        elif any(p.name not in ({'.run.lock', 'invocation.json', 'resources.jsonl'} |
                 ({'scene_refinement_report.json','index.html','delivery.pending','invocation_reports'} if args.resume else set())) for p in root.iterdir()):
            raise UpstreamRejected('Unbound output directory is not empty')
        owns_run = True
        write_json(manifest_path, {'schema_version': SCHEMA_VERSION, 'fingerprint': fingerprint,
            'binding': binding, 'started_unix': report['started_unix'], 'status': 'running'})
        report.update(fingerprint=fingerprint, config=config, runtime=args.runtime)
        original = verify_input_layout(args.layout_json, args.scene_image)
        from urdf_core.layout_handoff import layout_quality
        report['upstream_layout'] = layout_quality(read_json(args.layout_json.parent / 'refine_summary.json'))
        from .input_adapter import load_scene_assets
        assets = load_scene_assets(args.asset_manifest)
        verify_asset_inputs(original, assets)
        if (Path(assets.get('layout_json', '')).resolve() != args.layout_json.resolve()
                or Path(assets.get('scene_image', '')).resolve() != args.scene_image.resolve()):
            raise UpstreamRejected('Assets belong to another delivered layout/image')
        from .hashing import flush_cache
        flush_cache()
        report['input_verification_seconds'] = time.time()-entry
        prompts = {name: (STAGE/'resources'/f'{name}.md').read_text() for name in ('semantic', 'support')}
        attempt_dir.mkdir(parents=True, exist_ok=True)
        write_json(attempt_dir/'input_layout.json', original)
        report['attempts'].append(attempt)
        api = SceneAPI(run_dir=attempt_dir, image_path=args.scene_image,
            fingerprint=digest({'run': fingerprint, 'layout': original}), model=args.model,
            reasoning_effort=args.reasoning_effort, budget=config['api_budget'],
            original_layout_path=args.layout_json)
        try:
            selected = _run_attempt(args=args, config=config, layout=original, assets=assets, attempt_dir=attempt_dir,
                                    fingerprint=api.fingerprint, api=api, prompts=prompts, report=attempt)
        finally:
            attempt['api_attempts'] = api.attempts
        report['accepted'] = bool(attempt.get('accepted'))
        report['status'] = 'accepted' if report['accepted'] else 'search_finished_unaccepted'
        report['stop_reason'] = attempt.get('stop_reason', report['status'])
    except UpstreamRejected as exc:
        report.update(status='input_unavailable', stop_reason='input_unavailable', errors=[safe_error(exc)])
    except APIBudgetExhausted as exc:
        report.update(status='search_finished_unaccepted', stop_reason='api_budget_exhausted', errors=[safe_error(exc)])
    except (InfrastructureError, SemanticUnavailable, OSError, RuntimeError) as exc:
        report.update(status='system_paused', stop_reason='technical_failure', errors=[safe_error(exc)])
    except Exception as exc:
        report.update(status='input_unavailable', stop_reason='technical_input_failure', errors=[safe_error(exc)])
    if not owns_run and prior:
        return {**report, 'report_written': False}
    delivery_started = time.monotonic()
    try:
        if report['accepted']:
            initial = attempt.get('initial')
        else:
            selected, initial = _recover_delivery_candidates(attempt_dir, attempt)
            attempt.update(selected=selected, initial=initial)
        from .delivery import publish, publish_fallback
        if args.resume and (root/'delivery').exists() and prior and prior.get('status') != 'completed':
            history = root/'prior_deliveries'/str(time.time_ns())
            history.parent.mkdir(exist_ok=True)
            (root/'delivery').rename(history)
        if report['accepted']:
            receipt = publish(root, selected, initial, accepted=True, stop_reason=report.get('stop_reason'))
        else:
            receipt = publish_fallback(root, selected, initial, attempt_dir, stop_reason=report.get('stop_reason'))
        report.update(delivered=receipt['delivered'], delivery=receipt,
                      delivery_directory=str(root/'delivery') if receipt['delivered'] else None)
    except Exception as exc:
        report.update(status='system_paused', accepted=False, delivered=False, stop_reason='delivery_failed')
        report['errors'].append(safe_error(exc))
    report['delivery_seconds'] = time.monotonic()-delivery_started
    report['api_attempts'] = attempt.get('api_attempts', 0)
    report['elapsed_seconds'] = time.time()-report['started_unix']
    report['invocation_seconds'] = time.time()-entry
    if attempt_dir.exists():
        write_json(attempt_dir/'attempt_report.json', attempt)
    if args.resume and report_path.exists():
        write_json(root/'invocation_reports'/f'{time.time_ns()}.json', read_json(report_path))
    write_json(report_path, report)
    _report_page(root, report)
    if owns_run:
        manifest = read_json(manifest_path)
        manifest['status'] = 'paused' if report['status'] == 'system_paused' else 'completed'
        hashing_started = time.monotonic()
        manifest['artifacts'] = _files_hashes([p for p in root.rglob('*') if p.is_file()
            and p != manifest_path and p.name != 'resources.jsonl' and not p.name.endswith('.lock')
            and 'worker' not in p.relative_to(root).parts])
        report['evidence_hash_seconds'] = time.monotonic()-hashing_started
        report['elapsed_seconds'] = time.time()-report['started_unix']
        if report['accepted']:
            report.update(first_accepted_seconds=time.time()-report['started_unix'], first_accepted_unix=time.time(),
                          acceptance_evidence_saved_and_hashed=True)
            if report.get('batch_enqueued_unix') is not None:
                report['first_accepted_from_batch_seconds'] = report['first_accepted_unix']-report['batch_enqueued_unix']
        write_json(report_path, report)
        manifest['artifacts'][str(report_path.resolve())] = file_hash(report_path)
        write_json(manifest_path, manifest)
    return report
