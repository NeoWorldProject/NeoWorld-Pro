"""One iteration-bounded physics search followed by complete final assessment."""
from __future__ import annotations
import copy
import json
import secrets
import shutil
from pathlib import Path
import time
from urdf_core.run_state import file_hash, fingerprint, write_json
from urdf_core.delivery_policy import SharedBudget, RunTiming, POLICY, BudgetExhausted
from urdf_core.asset_delivery import publish, verify as verify_delivery
from .protocol import make_test_contract
from .control import controller_policy, UNIFORM_PID_POLICY
from .diagnosis import SKILL_PATH, observed_failures, failure_phase, allowed_paths, apply_proposal, validate_parameter_bounds
from .evidence import build_evidence, compare_candidate
from .critic import ask_json, evidence_frames, frame_content
from .verification import verify_simulation_report

def read(p):return json.loads(Path(p).read_text())


def acceptance_summary(report, contract, original):
    """Search scores use the same gates as native acceptance, not diagnostics."""
    value=copy.deepcopy(original);checks=value['checks']
    from .evidence import trial_list
    from .motion_acceptance import enabled
    for trial in trial_list(report):
        identity=trial.get('id',trial.get('kind','unknown'));prefix=identity+'/'
        metrics=trial.get('metrics',{}).get('metrics',{})
        for name,joint in metrics.get('joints',{}).items():
            tolerance=joint.get('initial_q_tolerance')
            deviation=joint.get('max_initial_q_deviation')
            if tolerance and isinstance(deviation,(int,float)):
                checks[prefix+'joint/'+name+'/max_initial_q_deviation']=deviation/tolerance
        if enabled(contract,trial.get('kind')):
            for key in list(checks):
                if key.startswith(prefix) and ('/hold/' in key or key.endswith('/tail_q_span')):
                    del checks[key]
        if trial.get('passed') is True:
            # The recomputed complete trial verdict is authoritative. Diagnostic
            # fields cannot manufacture a regression in a numerically passed trial.
            for key in list(checks):
                if key.startswith(prefix):checks[key]=min(1.,checks[key])
    value['protected_checks']=sorted(k for k,v in checks.items() if v<=1.+1e-6)
    value['normalized_violation']=max((max(0.,v-1.) for v in checks.values()),default=None)
    value['comparison_policy']='native_acceptance_gates.v1'
    return value


def verify_candidate(candidate):
    for path,digest in candidate['files'].items():
        if file_hash(path)!=digest:raise ValueError('Immutable delivery candidate changed: '+path)

def run_rounds(*, args, api, root, base, spec, initial, policy, config, state, ask, images, run_key, semantic_key,
               ensure_unchanged, orientation_policy, free_base_policy):
    from .loop import CONTRACT_POLICIES, _run_simulation
    from .asset_reuse import prepare_asset
    from .physicalize import physicalize_asset
    from .refinement_context import project_context, select_refinement_frames
    from .api_evidence import serialize_refinement_payload
    from . import joint_groups as grouping
    from .representative_motion import prepare_group_input, INPUT_FILE
    root = Path(root)
    from urdf_core.target_reference import load_reference, reference_content
    target_reference = load_reference(getattr(args, 'scene_context', None), getattr(args, 'prompts_dir', None))
    source_content = (reference_content(target_reference, api, root) if target_reference else
                      [api.make_image_input(path) for path in images])
    source_image_count = sum(item['type'] == 'input_image' for item in source_content)
    budget = SharedBudget(root, 'physics', args.max_physics_repairs)
    timing = RunTiming(root)
    ledger_path = root / 'delivery_research.json'
    identity = fingerprint({'config': config, 'initial': initial, 'spec': spec})
    ledger = read(ledger_path) if ledger_path.exists() else {'schema_version': POLICY, 'identity': identity,
        'candidates': [], 'best': None, 'current': None, 'timings': [], 'strategy_reviews': [], 'status': 'running'}
    if ledger['identity'] != identity:
        raise ValueError('Physics search inputs changed; use a new run')
    upstream_path = base / 'validation_report.json'
    upstream_report = read(upstream_path)
    upstream_quality = {'stage': 'upstream_generation', 'path': str(upstream_path),
        'sha256': file_hash(upstream_path), 'passed': upstream_report.get('passed') is True,
        'validation_status': upstream_report.get('validation_status'), 'reason': upstream_report.get('reason')}
    upstream_findings = [] if upstream_quality['passed'] else [upstream_quality]
    def save(): write_json(ledger_path, ledger)
    def candidate(index): return ledger['candidates'][index]
    def controls(contract):
        return {name: {key: joint[key] for key in ('kp', 'kd', 'speed_fraction', 'initial_position')}
                for name, joint in contract['joints'].items()}
    diagonal = read(base / 'geometry_inventory/inventory.json')['scene_diagonal']
    group_root = root / 'joint_groups'
    groups = None
    def make_contract(profile, controller=None, previous=None):
        return make_test_contract(spec, initial['effort_budgets'], policy, diagonal,
            previous=previous, profile=profile, controller=controller,
            acceptance_policy=CONTRACT_POLICIES[config['acceptance_policy']],
            orientation_policy=orientation_policy, free_base_policy=free_base_policy)
    def prepare(profile, contract, folder):
        asset = folder / 'physical_asset'
        if not (asset / 'asset_manifest.json').exists():
            start = time.monotonic()
            prepare_asset(base, profile, asset, config=contract['config'],
                cache_root=root / 'physical_asset_cache', physicalize=physicalize_asset)
            write_json(asset / 'controller_policy.json', controller_policy(contract, UNIFORM_PID_POLICY))
            shutil.copy2(Path(__file__).with_name('control.py'), asset / 'controller_executor.py')
            ledger['timings'].append({'phase': 'physicalize', 'wall_seconds': time.monotonic() - start})
            save()
        return asset
    def create(profile, controller=None, previous=None, proposal=None, repair_id=None):
        contract = make_contract(profile, controller, previous)
        controller = controls(contract)
        validate_parameter_bounds(profile, initial['profile'], controller, contract)
        if groups is not None and grouping.inconsistencies(groups, profile, controller):
            raise ValueError('Candidate changed shared group parameters inconsistently')
        physical_key = fingerprint({'profile': profile, 'controller': controller,
            'contract': {key: value for key, value in contract.items() if key not in ('revision', 'previous_contract_sha256')}})
        if any(item['physical_key'] == physical_key for item in ledger['candidates']):
            raise ValueError('Repeated physical candidate')
        from .asset_reuse import check_physical_proposal
        check_physical_proposal(base, profile, root, contract['config'])
        index = len(ledger['candidates'])
        folder = root / 'delivery_candidates' / f'candidate_{index:03d}'
        snapshot = {'profile': profile, 'controller': controller, 'contract': contract, 'proposal': proposal,
                    'parent': ledger['best'], 'candidate_id': f'candidate_{index:03d}', 'repair_id': repair_id}
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / 'parameters.json', snapshot)
        write_json(folder / 'physics_test_contract.json', contract)
        if groups is not None:
            prepare_group_input(folder / INPUT_FILE, group_root, contract, snapshot['candidate_id'])
        item = {'id': snapshot['candidate_id'], 'index': index, 'folder': str(folder),
                'physical_key': physical_key, 'status': 'pending',
                'files': {str(path): file_hash(path) for path in folder.iterdir() if path.is_file()}}
        ledger['candidates'].append(item)
        ledger['current'] = index
        save()
        return item
    if ledger.get('status') == 'completed':
        verify_delivery(root)
        return
    if not (root / 'final').exists():
        publish(root, base / 'urdf_export', base / 'urdf_export/object.usd', spec,
            candidate_id='upstream', validation_status='incomplete', findings=[{'physics': 'not_evaluated'}])
    termination = 'unknown'
    assessment_complete = False
    try:
        if config.get('joint_test_scope', 'representative') == 'representative':
            preflight_contract = make_contract(initial['profile'])
            request_path = group_root / 'grouping_request.json'
            if not request_path.exists():
                write_json(request_path, {'schema_version': POLICY, 'seed': secrets.randbits(64)})
            contract_path = group_root / 'physics_test_contract.json'
            if not contract_path.exists():
                write_json(contract_path, preflight_contract)
            elif read(contract_path) != preflight_contract:
                raise ValueError('Frozen grouping contract changed')
            if not (group_root / grouping.GROUP_FILE).is_file():
                asset = prepare(initial['profile'], preflight_contract, group_root)
                args._evaluation_policy = 'grouping'
                path = _run_simulation(args, asset, contract_path, group_root / 'simulation')
                if read(path).get('status') != 'grouping_ready':
                    raise ValueError('Native grouping preflight did not complete')
            groups = grouping.load_groups(group_root, preflight_contract)
            if groups is None:
                raise ValueError('Native grouping omitted the frozen membership')
        if ledger['current'] is None:
            profile = copy.deepcopy(initial['profile'])
            controller = controls(make_contract(profile))
            if groups is not None:
                applied = grouping.initialize_from_representatives(groups=groups, profile=profile,
                    initial_profile=initial['profile'], controller=controller, contract=make_contract(profile), spec=spec)
                profile, controller = applied['profile'], applied['controller']
                write_json(group_root / 'initialization.json', applied)
            create(profile, controller)
        while True:
            ensure_unchanged()
            item = candidate(ledger['current'])
            verify_candidate(item)
            folder = Path(item['folder'])
            snap = read(folder / 'parameters.json')
            profile, contract = snap['profile'], snap['contract']
            asset = prepare(profile, contract, folder)
            if item['status'] == 'pending':
                args._evaluation_policy = 'first-failure'
                started = time.monotonic()
                path = Path(_run_simulation(args, asset, folder / 'physics_test_contract.json', folder / 'simulation'))
                checked = verify_simulation_report(path, root=root, contract_path=folder / 'physics_test_contract.json',
                    asset_dir=asset, require_pass=False)
                report = checked['report']
                evidence = build_evidence(path, contract, observed_failures(report, contract=contract), asset_dir=asset)
                evidence_path = folder / 'diagnostic_evidence.json'
                write_json(evidence_path, evidence)
                summary = acceptance_summary(report, contract, evidence['numeric_summary'])
                best = candidate(ledger['best']) if ledger['best'] is not None else None
                comparison = compare_candidate(summary, best['summary'] if best else None)
                item.update(status='measured', report=str(path), evidence=str(evidence_path), summary=summary,
                    comparison=comparison, measurement_policy='first-failure',
                    upstream_passed=upstream_quality['passed'], upstream_quality=upstream_quality)
                item['files'].update({str(p): file_hash(p) for p in [path, evidence_path, *asset.rglob('*')] if p.is_file()})
                if snap.get('repair_id'):
                    budget.complete(snap['repair_id'])
                if comparison['promote']:
                    ledger['best'] = item['index']
                ledger['timings'].append({'phase': 'physics', 'candidate': item['id'],
                    'wall_seconds': time.monotonic() - started, 'trials': len(report['trials'])})
                save()
                if ledger['best'] == item['index']:
                    publish(root, asset, report['delivered_usd_path'], spec, candidate_id=item['id'],
                        validation_status='passed' if report['passed'] and item['upstream_passed'] else 'incomplete',
                        report=path, findings=[*evidence['observed_failures'], *upstream_findings])
            report, evidence = read(item['report']), read(item['evidence'])
            if report['passed']:
                termination = 'physics_passed'
                break
            pending_id = item.get('next_repair_id')
            pending = pending_id in budget.data['entries'] and budget.data['entries'][pending_id]['status'] == 'pending'
            if pending_id and not pending:
                item.pop('next_repair_id', None)
            if budget.remaining == 0 and not pending:
                termination = 'shared_repair_budget_exhausted'
                break
            best = candidate(ledger['best'])
            parent = read(Path(best['folder']) / 'parameters.json')
            failures = evidence['observed_failures']
            phase = failure_phase(failures)
            used = len(budget.data['entries'])
            if used and used % 3 == 0 and used not in [row['after_repairs'] for row in ledger['strategy_reviews']]:
                ledger['strategy_reviews'].append({'after_repairs': used, 'remaining': budget.remaining,
                    'best': best['id'], 'measured_failures': failures, 'decision': 'continue_within_shared_budget'})
            profile, controller, contract = (parent[key] for key in ('profile', 'controller', 'contract'))
            available = budget.remaining + int(pending)
            remaining = {'ground': available, 'motion': available}
            choices = {phase: allowed_paths(profile, controller, phase)}
            if phase != 'ground_settle':
                choices['return_to_ground'] = allowed_paths(profile, controller, 'return_to_ground')
            if phase == 'free_joint':
                choices['return_to_fixed'] = allowed_paths(profile, controller, 'return_to_fixed')
            repair_id = item.get('next_repair_id') or f'physics_repair_{used + 1:03d}'
            item['next_repair_id'] = repair_id
            budget.reserve(repair_id, phase, {'parent': best['id'], 'source': item['id'], 'evidence': file_hash(item['evidence'])})
            save()
            payload = {'phase': phase, 'observed_failures': failures, 'diagnostic_source_candidate_id': item['id'],
                'profile': profile, 'initial_profile': initial['profile'], 'controller': controller, 'contract': contract,
                'allowed_paths': choices, 'remaining': remaining, 'current': {'candidate': item, 'parameters': snap, 'evidence': evidence},
                'best': {'candidate': best, 'parameters': parent, 'evidence': read(best['evidence'])},
                'history': [], 'candidates': ledger['candidates'], 'used_total': {'shared': used}, 'cycle_id': 0,
                'joint_groups': grouping.public_context(groups)}
            payload['tested_candidate_trials'] = [{'candidate_id': row['id'],
                'parent': read(Path(row['folder']) / 'parameters.json')['parent'],
                'implementation': config['implementation'], 'same_implementation_and_policy': True,
                'patches': (read(Path(row['folder']) / 'parameters.json').get('proposal') or {}).get('patches', []),
                'result': {**row['summary'], 'observed_failures': read(row['evidence'])['observed_failures']}}
                for row in ledger['candidates'] if row['status'] == 'measured']
            frames, _ = select_refinement_frames(evidence_frames(item['report']), source_image_count=source_image_count,
                failed_trial_ids={failure['trial_id'] for failure in failures if failure.get('trial_id')})
            content = [api.make_text_input(SKILL_PATH.read_text()), *source_content,
                *frame_content(api, frames), api.make_text_input(serialize_refinement_payload(project_context(payload)))]
            proposal_path = folder / (repair_id + '_proposal.json')
            content.append(api.make_text_input('Prior refinement outcomes for this frozen evidence; use them to revise the next permitted hypothesis:\n'
                + json.dumps(ledger.get('refinement_attempts', []), ensure_ascii=False)))
            def validate(value):
                return apply_proposal(value, profile=profile, initial_profile=initial['profile'], controller=controller,
                    contract=contract, spec=spec, failures=failures, phase=phase, remaining=remaining,
                    evidence_report=report, joint_groups=groups)
            start = time.monotonic()
            try:
                if proposal_path.exists():
                    proposal = read(proposal_path)
                else:
                    proposal = ask_json(ask, api, repair_id, content, lambda value: (validate(value), value)[1], repairs=0)
                    write_json(proposal_path, proposal)
                applied = validate(proposal)
                ledger['timings'].append({'phase': 'model_refinement', 'repair_id': repair_id,
                    'wall_seconds': time.monotonic() - start})
                if applied['verdict'] != 'refine':
                    outcome = {'repair_id': repair_id, 'verdict': applied['verdict'], 'proposal': proposal,
                               'feedback': 'No permitted change proposed; reassess within the remaining parameter-only budget'}
                else:
                    create(applied['profile'], applied['controller'], contract, proposal, repair_id)
                    outcome = {'repair_id': repair_id, 'verdict': 'refine', 'proposal': proposal}
            except (ValueError, KeyError, TypeError) as error:
                outcome = {'repair_id': repair_id, 'verdict': 'invalid_proposal', 'error': str(error)}
            ledger.setdefault('refinement_attempts', []).append(outcome)
            if outcome['verdict'] != 'refine':
                budget.complete(repair_id, status='rejected', outcome=outcome['verdict'])
                item.pop('next_repair_id', None)
            save()
    except BudgetExhausted as error:
        ledger.update(status='refinement_budget_exhausted', termination_reason=str(error))
        save()
        raise  # The outer pipeline finalizes the best already committed artifact.
    except Exception as error:
        ledger.update(status='technical_failure', execution_error=f'{type(error).__name__}: {error}')
        save()
        state.fail(ledger['execution_error'])
        raise
    if ledger['best'] is not None:
        best = candidate(ledger['best'])
        folder = Path(best['folder'])
        try:
            verify_candidate(best)
            contract = read(folder / 'parameters.json')['contract']
            report_path = Path(best['report'])
            report = read(report_path)
            if not report.get('passed'):
                args._evaluation_policy = 'complete'
                args._trial_cache_root = folder / 'simulation'
                start = time.monotonic()
                report_path = Path(_run_simulation(args, folder / 'physical_asset', folder / 'physics_test_contract.json', folder / 'final_evaluation'))
                ledger['timings'].append({'phase': 'final_complete_physics', 'candidate': best['id'],
                    'wall_seconds': time.monotonic() - start})
            checked = verify_simulation_report(report_path, root=root, contract_path=folder / 'physics_test_contract.json',
                asset_dir=folder / 'physical_asset', require_pass=False)
            report = checked['report']
            assessment_complete = bool(report.get('evaluation_complete') and
                (report.get('passed') or report.get('trial_evaluation_policy') == 'all_independent_trials.v1'))
            status = 'passed' if report['passed'] and best['upstream_passed'] else 'failed' if assessment_complete else 'incomplete'
            publish(root, folder / 'physical_asset', report['delivered_usd_path'], spec, candidate_id=best['id'],
                validation_status=status, report=report_path,
                findings=[*observed_failures(report, contract=contract), *upstream_findings],
                lineage=[{'path': str(folder / 'parameters.json'), 'sha256': file_hash(folder / 'parameters.json')},
                         upstream_quality], terminal=True)
            ledger['final_report'] = {'path': str(report_path), 'sha256': file_hash(report_path)}
        except BudgetExhausted as error:
            ledger.update(status='refinement_budget_exhausted', termination_reason=str(error))
            save()
            raise
        except Exception as error:
            ledger.update(status='technical_failure', finalization_error=f'{type(error).__name__}: {error}')
            save()
            state.fail(ledger['finalization_error'])
            raise
        finally:
            if hasattr(args, '_trial_cache_root'):
                del args._trial_cache_root
    result = verify_delivery(root)
    ledger.update(status='completed', termination_reason=termination, delivery_status='completed',
        validation_status=result['validation_status'], assessment_status='complete' if assessment_complete else 'incomplete',
        evaluation_complete=assessment_complete, finished_epoch=time.time(), elapsed_seconds=timing.elapsed)
    save()
    state.data.update(status='completed', delivery_status='completed', validation_status=result['validation_status'],
        assessment_status=ledger['assessment_status'], evaluation_complete=assessment_complete,
        physics_verification=('passed' if report['passed'] else 'failed') if assessment_complete else 'incomplete',
        termination_reason=termination,
        delivery_manifest=str((root / 'final/delivery_manifest.json').resolve()))
    state.data.pop('error', None)
    state.save()
    write_json(root / 'validation_report.json', {'delivery_status': 'completed', 'validation_status': result['validation_status'],
        'assessment_status': ledger['assessment_status'], 'evaluation_complete': assessment_complete,
        'upstream_quality': upstream_quality, 'physics_passed': report['passed'],
        'passed': result['validation_status'] == 'passed', 'termination_reason': termination, 'candidate_id': result['candidate_id']})
    print('Delivered asset: ' + str(root / 'final/object.urdf') + '; validation=' + result['validation_status'], flush=True)
