"""Object physics: frozen inputs, bounded parameter repairs and complete delivery."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from urdf_core.run_state import RunState, file_hash, fingerprint, write_json
from urdf_core.run_validation import reject_assisted_provenance
from urdf_core.isaac_cache import evidence_files
from .critic import PROMPTS, ask_json, read_json
from .protocol import load_config, validate_profile, make_test_contract
from .physicalize import reconstruct_objects, validate_profile_geometry

GROUND_HOLD_POLICY = "automatic_object_physics_v11"
CONTRACT_POLICIES = {GROUND_HOLD_POLICY: "object_functional_physics.v11"}
MOTION_SCOPE_POLICY = "fixed_full_ground_operable.v1"
MODULE = Path(__file__).resolve().parent
HARNESS_VERSION = 'object_pipeline.v1'

class PhysicsEnvironmentError(RuntimeError): pass
class PhysicsAPIError(RuntimeError): pass
class PhysicsRejectedError(RuntimeError): pass

class PhysicsRunState(RunState):
    def __init__(self, root, fresh, config):
        self.used_stages = set()
        super().__init__(root, fresh, config)

    def current(self, name, key):
        self.used_stages.add(name)
        return super().current(name, key)

    def passed(self, name, key, paths, **details):
        self.used_stages.add(name)
        return super().passed(name, key, paths, **details)

def files_under(root):
    return evidence_files(root)


def source_fingerprint():
    from urdf_core.paths import URDF_CORE, OBJECT_GENERATION
    from .diagnosis import SKILL_PATH
    paths = [SKILL_PATH, *(p for p in MODULE.glob("*.py") if p.name not in {"synthetic_regression.py", "performance_regression.py"}), *PROMPTS.glob("physics_*.md"),
             *URDF_CORE.glob("*.py")]
    return fingerprint({str(p): file_hash(p) for p in sorted(paths)})




def kinematics_fingerprint(args):
    """Physics-only fixes must not force a new geometry/API generation."""
    from urdf_core.paths import URDF_CORE, KINEMATICS_VALIDATION
    paths = [*URDF_CORE.glob("*.py"), *KINEMATICS_VALIDATION.glob("*.py"),
             *args.prompts_dir.glob("*.md")]
    return fingerprint({str(p): file_hash(p) for p in sorted(paths)})




def _audit_content(content):
    return [{"type": item["type"], **({"text": item["text"]} if item["type"] == "input_text"
             else {"image_sha256": fingerprint(item["image_url"])})} for item in content]

def _audited_ask(args, api, root):
    from urdf_core.api_requests import audited_call, logical_stage
    from urdf_core.api_autonomy import assert_autonomous_context
    client = None
    def ask(stage, content):
        nonlocal client
        assert_autonomous_context(content)
        from urdf_core.prompt_payload import prepare_content
        content = prepare_content(content)
        request = {'model': args.model, 'logical_stage': logical_stage(stage),
                   'provider_identity': fingerprint(os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')),
                   'reasoning_effort': os.environ.get('URDF_REASONING_EFFORT', ''),
                   'content': _audit_content(content)}
        def prepare_client():
            nonlocal client
            if client is None:
                kwargs = {'timeout': 300, 'max_retries': 0}
                if os.environ.get('OPENAI_BASE_URL'):
                    kwargs['base_url'] = os.environ['OPENAI_BASE_URL']
                client = api.OpenAI(**kwargs)
        def produce():
            return api.call_model(client, args.model, content, max_retries=1)
        raw, _ = audited_call(Path(root), stage, request, produce, prepare=prepare_client,
                             reuse_root=getattr(args, 'physics_api_cache_from_run', None))
        return raw
    return ask

def _cached_json(state, name, key, path, produce, extra=(), immutable_decision=False):
    saved = state.data['stages'].get(name, {})
    if saved.get('status') == 'passed' and immutable_decision and saved.get('key') != key:
        raise ValueError('Frozen decision inputs changed: ' + name)
    if state.current(name, key):
        return read_json(state.root / saved['result_path'])
    if saved.get('status') == 'passed' and immutable_decision:
        raise ValueError('Frozen decision evidence changed: ' + name)
    state.begin(name, key)
    result = produce()
    write_json(path, result)
    state.passed(name, key, [path, *extra, *files_under(state.root / 'model_calls')],
                 result_path=str(path.relative_to(state.root)))
    return result

def _profile_content(base, images, api, previous_contract=None, *, acceptance_policy=GROUND_HOLD_POLICY,
                     scene_context=None, support_classification=None, physics_free_base=None, target_reference=None):
    inventory = read_json(base / 'geometry_inventory/inventory.json')
    inventory['objects'] = [{k: v for k, v in obj.items() if k != 'components'} for obj in inventory['objects']]
    try:
        measured = reconstruct_objects(base / 'urdf_export', read_json(base / 'kinematics.json'))
    except ValueError as error:
        measured = []
        inventory['physical_surface_error'] = str(error)
    inventory['physical_surface_measurements'] = [
        {'link': entry['link'], 'object': entry['object'], 'watertight': bool(entry['mesh'].is_watertight),
         'surface_area_m2': float(entry['mesh'].area), 'bounds_link_local_m': entry['mesh'].bounds.tolist(),
         'vertices': len(entry['mesh'].vertices), 'triangles': len(entry['mesh'].faces),
         'signed_volume_m3': float(entry['mesh'].volume) if entry['mesh'].is_watertight else None}
        for entry in measured]
    from urdf_core.target_reference import load_reference, reference_content
    if target_reference is None and scene_context:
        target_reference = load_reference(scene_context['context_file']['path'])
    reference = (reference_content(target_reference, api, base) if target_reference else
                 [api.make_image_input(path) for path in images])
    content = [*reference,
        *([api.make_image_input(base / 'Res1_3D.png')] if (base / 'Res1_3D.png').is_file() else []),
        api.make_text_input('Measured geometry inventory:\n' + json.dumps(inventory)),
        api.make_text_input('Validated kinematics:\n' + (base / 'kinematics.json').read_text()),
        api.make_text_input((PROMPTS / 'physics_profile.md').read_text())]
    if scene_context:
        content.append(api.make_text_input(json.dumps({k: scene_context[k] for k in
            ('target', 'parent', 'support_ancestry')}, ensure_ascii=False)))
        content.append(api.make_text_input('Frozen scene support classification; copy stable_required exactly:\n' + json.dumps(support_classification)))
    if previous_contract:
        content.append(api.make_text_input('Frozen test contract:\n' + json.dumps(previous_contract)))
    from .test_modes import free_base_policy
    from .motion_acceptance import RULES
    from .orientation_decision import FUNCTIONAL_GRAVITY_PROMPT
    content.extend([api.make_text_input('Frozen v11 motion rules: ' + json.dumps(RULES)),
        api.make_text_input('All fixed-base motion uses an elevated fixture clearing the full authored range. '
            'Free-base selection: ' + json.dumps(free_base_policy(bool(physics_free_base))) +
            ' Ground and reload rotary initial-state tolerance is 10 degrees. Tail stability and root tilt remain unchanged.'),
        api.make_text_input('Also provide orientation_policy: ' + FUNCTIONAL_GRAVITY_PROMPT)])
    return content

def _validate_initial(proposal, spec, base=None, *, previous_contract=None, require_current=False,
                      require_orientation=False, free_base_policy=None):
    profile = validate_profile(proposal["profile"], spec)
    if require_current and profile.get("schema_version") != 3:
        raise ValueError("New physical acceptance requires profile schema 3")
    if base is not None:
        # Missing approximation fields are a response-contract error, not a
        # fresh opportunity to change the object's physical interpretation.
        profile = validate_profile_geometry(profile, spec, reconstruct_objects(Path(base) / "urdf_export", spec))
    budgets = proposal.get("effort_budgets")
    expected = {j["name"] for j in spec["joints"] if j["type"] != "fixed"}
    if not isinstance(budgets, dict) or set(budgets) != expected:
        raise ValueError("effort_budgets must cover every active joint exactly once")
    # make_test_contract provides the stricter numerical budget checks.
    contract = make_test_contract(spec, budgets, load_config(), 1.0,
                                  profile=profile if profile.get("schema_version") == 3 else None,
                                  orientation_policy=proposal.get("orientation_policy"),
                                  free_base_policy=free_base_policy)
    if profile.get("schema_version") == 3:
        from .diagnosis import validate_parameter_bounds
        controls = {name: {key: values[key] for key in ("kp", "kd", "speed_fraction", "initial_position")}
                    for name, values in contract["joints"].items()}
        validate_parameter_bounds(profile, profile, controls, contract)
    if previous_contract is not None:
        for name, frozen in previous_contract["joints"].items():
            if name not in budgets or budgets[name]["max_effort"] != frozen["max_effort"]:
                raise ValueError(f"{name} effort budget is frozen at {frozen['max_effort']}; repair this response without changing the cap")
    result = {"profile": profile, "effort_budgets": budgets}
    if require_orientation:
        from .protocol import validate_orientation_policy
        result["orientation_policy"] = validate_orientation_policy(proposal.get("orientation_policy"))
    return result


def _run_simulation(args, asset_dir, contract_path, target):
    worker = MODULE / "verify_physics_isaac.py"
    command = [sys.executable, str(worker), "--asset-dir", str(asset_dir),
               "--spec", str(asset_dir / "kinematics.json"),
               "--profile", str(asset_dir / "physics_profile.json"),
               "--contract", str(contract_path), "--out-dir", str(target),
               "--isaac-python", str(args.isaac_python)]
    trial_order = contract_path.parent / 'trial_order.json'
    if trial_order.is_file():
        command.extend(['--trial-order', str(trial_order)])
    representative = contract_path.parent / 'representative_motion_input.json'
    if representative.is_file():
        command.extend(['--representative-motion', str(representative)])
    target.mkdir(parents=True, exist_ok=True)
    session = getattr(args, '_physics_session', None)
    review_config = getattr(args, '_incidental_motion_config', None)
    if review_config is not None:
        from .incidental_motion import prepare_input, INPUT_FILE
        review_input = prepare_input(contract_path.parent / INPUT_FILE, contract_path, **review_config)
        command.extend(['--incidental-motion', str(review_input)])
    command.extend(['--evaluation-policy', getattr(args, '_evaluation_policy', 'first-failure')])
    if getattr(args, '_trial_cache_root', None):
        command.extend(['--trial-cache-root', str(args._trial_cache_root)])
    if session is not None:
        from .verify_physics_isaac import parse_args
        from urdf_core.delivery_policy import BudgetExhausted
        from urdf_core.content_errors import AssetContentError
        try:
            worker_args=parse_args(command[2:])
            worker_args.optimization_short_circuit=worker_args.evaluation_policy == 'grouping'
            return session.run(worker_args)
        except (BudgetExhausted, AssetContentError):
            raise
        except (RuntimeError, ValueError, OSError) as error:
            raise PhysicsEnvironmentError(f'Resident physics worker: {error}') from error
    # The supervisor owns the GPU lock and bounds/cleans its own process group.
    with (target / "orchestration.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    path = target / "report.json"
    if not path.is_file():
        raise PhysicsEnvironmentError(f"Isaac physics produced no report; see {target / 'orchestration.log'}")
    report = read_json(path)
    if report.get('status') == 'asset_rejected' and report.get('asset_error'):
        from urdf_core.content_errors import AssetContentError
        raise AssetContentError(**report['asset_error'])
    if result.returncode not in {0, 2} or report.get("status") not in {"passed", "physics_failed", "grouping_ready"}:
        raise PhysicsEnvironmentError(f"Isaac physics environment/worker error; see {path}")
    return path

def run_physics_pipeline(args, api, kinematics_runner):
    from urdf_core.delivery_policy import POLICY as DELIVERY_POLICY, RunTiming
    from urdf_core.asset_delivery import publish, finish_existing, verify_run
    from urdf_core.generation_delivery import verify_source, import_source
    from urdf_core.environment import runtime_identity
    from urdf_core.run_validation import verify_completed_run
    from .scene_context import RULE_VERSION, load_context, classify
    from .test_modes import resolve_free_base, freeze_selection, free_base_policy, FREE_BASE_VERSION, FIXED_BASE_POLICY
    root = Path(args.out_dir).resolve()
    reject_assisted_provenance(root)
    old_path = root / 'run_manifest.json'
    previous = read_json(old_path).get('config', {}) if old_path.is_file() else None
    if previous is not None and previous.get('harness_policy') != HARNESS_VERSION:
        raise ValueError('Historical runs are read-only; use their frozen verifier or a new output directory')
    selected = getattr(args, 'physics_acceptance_policy', None) or GROUND_HOLD_POLICY
    if selected != GROUND_HOLD_POLICY:
        raise ValueError('New execution uses the frozen v11 numerical rules')
    source_manifest = getattr(args, 'physics_source_manifest', None)
    source = getattr(args, 'physics_from_run', None)
    if source_manifest:
        declared = read_json(source_manifest)
        if declared.get('source_kind') != 'verified_kinematics_run':
            raise ValueError('Physics source must declare a kinematics run')
        if source and Path(source).resolve() != Path(declared['run_dir']).resolve():
            raise ValueError('Conflicting physics sources')
        source = declared['run_dir']
        hashes = declared.get('file_hashes', {})
        if not hashes or any(file_hash(path) != digest for path, digest in hashes.items()):
            raise ValueError('Physics source files changed')
    source = Path(source).resolve() if source else None
    images = [Path(path).resolve() for path in args.image_paths]
    image_hashes = {str(path): file_hash(path) for path in images}
    context = load_context(args.scene_context) if getattr(args, 'scene_context', None) else None
    from urdf_core.target_reference import load_reference
    target_reference = load_reference(getattr(args, 'scene_context', None), args.prompts_dir)
    if context and context['target_crop']['sha256'] not in image_hashes.values():
        raise ValueError('Scene target crop does not belong to this object')
    scene_source = None
    if source:
        if source == root or source.is_relative_to(root) or root.is_relative_to(source):
            raise ValueError('Imported assets require an independent source directory')
        if (source / 'generation_delivery_candidates.json').is_file():
            verify_source(source)
        else:
            verify_completed_run(source)
        if read_json(source / 'run_manifest.json')['config']['source_images'] != image_hashes:
            from urdf_core.campaign_scene_context import verify_physics_scene_source
            scene_source = verify_physics_scene_source(source, source_manifest, getattr(args, 'scene_context', None), image_hashes)
    scope = getattr(args, 'joint_test_scope', None) or (previous or {}).get('joint_test_scope', 'representative')
    if scope not in {'representative', 'all'}:
        raise ValueError('joint_test_scope must be representative or all')
    free_enabled = resolve_free_base(args, selected)
    policy = load_config(getattr(args, 'physics_config', None))
    config = {'harness_policy': HARNESS_VERSION, 'acceptance_policy': selected,
        'delivery_policy': 'best-effort', 'delivery_version': DELIVERY_POLICY,
        'shared_physics_repairs': args.max_physics_repairs, 'geometry_repairs': args.max_generation_repairs,
        'articulation_repairs': getattr(args, 'max_articulation_repairs', 3),
        'source_images': image_hashes, 'model': args.model,
        'reasoning_effort': os.environ.get('URDF_REASONING_EFFORT', ''),
        'provider_identity': fingerprint(os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')),
        'physics_config': policy, 'physics_free_base': free_enabled, 'joint_test_scope': scope,
        'free_base_policy_version': FREE_BASE_VERSION, 'fixed_base_policy': FIXED_BASE_POLICY,
        'motion_scope_policy': MOTION_SCOPE_POLICY, 'ground_initial_rotary_tolerance_deg': 10.,
        'source_run': str(source) if source else None,
        'source_manifest_sha256': file_hash(source / 'run_manifest.json') if source else None,
        'scene_context': context, 'target_reference': target_reference, 'support_applicability_rule': RULE_VERSION,
        'scene_context_source_binding': scene_source, 'implementation': source_fingerprint(),
        'runtime': runtime_identity(args),
        'runtime_snapshot': getattr(args, '_runtime_snapshot', None),
        'isaac_python': str(args.isaac_python), 'blender': str(args.blender)}
    if getattr(args, 'articulation_from_run', None):
        from urdf_core.articulation_import import bind
        config['articulation_import'] = bind(args.articulation_from_run, image_hashes, target_reference)
    if getattr(args, 'physics_api_cache_from_run', None):
        cache_source = Path(args.physics_api_cache_from_run).resolve()
        config['physics_api_cache_source'] = {
            'root': str(cache_source), 'manifest_sha256': file_hash(cache_source / 'run_manifest.json')}
    if previous is not None and previous != config:
        raise ValueError('Frozen physics inputs or implementation changed; use a new output directory')
    state = PhysicsRunState(root, args.fresh, config)
    timing = RunTiming(root)
    ask = _audited_ask(args, api, root)
    semantic_key = fingerprint(config)
    def unchanged():
        if source_fingerprint() != config['implementation']:
            raise PhysicsEnvironmentError('Physics implementation changed during execution')
        if context and load_context(args.scene_context) != context:
            raise ValueError('Frozen scene context changed')
        if load_reference(getattr(args, 'scene_context', None), args.prompts_dir) != target_reference:
            raise ValueError('Frozen target reference changed')
    try:
        ledger = root / 'delivery_research.json'
        if ledger.is_file() and read_json(ledger).get('status') == 'completed':
            result = verify_run(root)
            state.data.update(status='completed', delivery_status='completed', validation_status=result['validation_status'])
            state.save()
            return
        base = root / 'object_revisions' / 'kinematics'
        key = fingerprint({'source': config['source_run'], 'source_manifest': config['source_manifest_sha256'],
                           'generation': kinematics_fingerprint(args)})
        if not state.current('kinematics_revision_000', key):
            state.begin('kinematics_revision_000', key)
            if source:
                if not base.exists():
                    if (source / 'generation_delivery_candidates.json').is_file():
                        import_source(source, base)
                    else:
                        raise ValueError('Physics reruns require a current generation delivery')
            else:
                child = copy.copy(args)
                child.out_dir = base
                child.enable_physics = False
                child.fresh = not base.exists()
                kinematics_runner(child, api)
            if (base / 'generation_delivery_candidates.json').is_file():
                verify_source(base)
            else:
                verify_completed_run(base)
            state.passed('kinematics_revision_000', key, files_under(base), base_run=str(base.relative_to(root)))
        spec = read_json(base / 'kinematics.json')
        if not (root / 'final').exists():
            publish(root, base / 'urdf_export', base / 'urdf_export/object.usd', spec,
                    candidate_id='upstream', validation_status='incomplete', findings=[{'physics': 'not_evaluated'}])
        from urdf_core.delivery_policy import SharedBudget, BudgetExhausted
        from .critic import ask_initial_with_budget
        physics_budget = SharedBudget(root, 'physics', args.max_physics_repairs)
        support = None
        if context:
            support = _cached_json(state, 'scene_support_applicability',
                fingerprint({'context': context['fingerprint'], 'rule': RULE_VERSION}),
                root / 'scene_support_applicability.json', lambda: classify(context, api, ask, repair_budget=physics_budget), immutable_decision=True)
            if support.get('context_fingerprint') != context['fingerprint'] or support.get('status') != 'classified':
                raise BudgetExhausted('Scene support remains unresolved after the physics refinement budget')
        mode = free_base_policy(free_enabled)
        mode_path = freeze_selection(root, free_enabled)
        state.passed('free_base_test_mode', fingerprint(mode), [mode_path], result_path=mode_path.name)
        def validate_initial(value):
            value = _validate_initial(value, spec, base, require_current=True, require_orientation=True,
                                      free_base_policy=mode)
            if support and value['profile']['stable_required'] is not support['decision']['stable_required']:
                raise ValueError('Initial profile changed frozen support applicability')
            from .asset_reuse import check_physical_proposal
            contract = make_test_contract(spec, value['effort_budgets'], policy,
                read_json(base/'geometry_inventory/inventory.json')['scene_diagonal'], profile=value['profile'],
                acceptance_policy=CONTRACT_POLICIES[config['acceptance_policy']],
                orientation_policy=value['orientation_policy'], free_base_policy=mode)
            check_physical_proposal(base, value['profile'], root, contract['config'])
            return value
        initial = _cached_json(state, 'initial_physics_profile', fingerprint({'task': semantic_key, 'stage': 'initial_physics_profile'}),
            root / 'physics_initial_proposal.json', lambda: ask_initial_with_budget(ask, api, 'initial_physics_profile',
                _profile_content(base, images, api, scene_context=context, support_classification=support,
                    target_reference=target_reference,
                                 physics_free_base=free_enabled),
                validate_initial, physics_budget, contract_patches=True),
            immutable_decision=True)
        initial = validate_initial(initial)
        unchanged()
        _run_simple_rounds(args=args, api=api, root=root, base=base, spec=spec, initial=initial, policy=policy,
            config=config, state=state, ask=ask, images=images, run_key=semantic_key, semantic_key=semantic_key,
            ensure_unchanged=unchanged, orientation_policy=initial['orientation_policy'], free_base_policy=mode)
    except BaseException as error:
        from urdf_core.delivery_policy import BudgetExhausted
        from urdf_core.content_errors import AssetContentError
        from urdf_core.execution import terminal_failure
        if terminal_failure(error) or isinstance(error, PhysicsEnvironmentError):
            state.fail(f'{type(error).__name__}: {error}')
            raise
        if isinstance(error, (BudgetExhausted, AssetContentError)) and (root / 'final').exists():
            if (root/'final/asset_manifest.json').is_file():
                finish_existing(root, state, f'{type(error).__name__}: {error}',
                    reason_kind='refinement_budget_exhausted' if isinstance(error, BudgetExhausted) else 'asset_content_rejected')
            else:
                from urdf_core.degraded_asset import finish_physics
                finish_physics(root, base, state, f'{type(error).__name__}: {error}',
                               asset_error=error.evidence if isinstance(error, AssetContentError) else None)
            if isinstance(error, Exception):
                return
        state.fail(f'{type(error).__name__}: {error}')
        raise

def _run_simple_rounds(*, args, api, root, base, spec, initial, policy, config,
                       state, ask, images, run_key, semantic_key, ensure_unchanged,
                       orientation_policy=None, free_base_policy=None):
    from .delivery_harness import run_rounds
    from .resident_worker import ResidentPhysicsSession
    from .incidental_motion import service_request
    workers=getattr(args,'physics_trial_workers',1)
    if workers>1:
        from .parallel_trials import ParallelPhysicsSession
        session=ParallelPhysicsSession(root/'physics_worker',workers=workers,
                    review_handler=lambda path:service_request(path,args,api))
    else:
        session=ResidentPhysicsSession(root/'physics_worker',review_handler=lambda path:service_request(path,args,api))
    with session:
        args._physics_session = session
        args._incidental_motion_config = {'model': args.model,
            'reasoning_effort': os.environ.get('URDF_REASONING_EFFORT', ''), 'images': images}
        try:
            return run_rounds(args=args, api=api, root=root, base=base, spec=spec,
                initial=initial, policy=policy, config=config, state=state, ask=ask, images=images,
                run_key=run_key, semantic_key=semantic_key, ensure_unchanged=ensure_unchanged,
                orientation_policy=orientation_policy, free_base_policy=free_base_policy)
        finally:
            del args._physics_session
            del args._incidental_motion_config
