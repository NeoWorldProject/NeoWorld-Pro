"""Joint RGB -> 2D/3D parsing with three content repairs and independent review.

This is a distinct quality profile, not a shortcut claiming the independent
orientation-v1 gates have run. No scene-specific coordinates are supplied.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re
import time

from . import common as w
from .provider import ModelAPI, aggregate_calls, audit_inputs, provider_config, safe_error, write_json
from .quality import register_images, render_edge_panels, render_review_panels, validate_feedback
from .schema import normalize_layout

PROMPTS = ('24_fast_generate.md', '25_fast_review.md', '26_fast_recheck.md')


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def fast_policy(args):
    limit = getattr(args, 'max_refine_iters', 3)
    if type(limit) is not int or not 0 <= limit <= 3:
        raise ValueError('Scene parsing allows at most three refinements')
    return {'policy': 'scene_refinement_budget.v2', 'max_refinements': limit,
            'technical_attempts': 5, 'max_run_seconds': None}


def validate_review(value, layout):
    if not isinstance(value, dict):
        raise ValueError('critic must be a JSON object')
    if not isinstance(value.get('quality_assessment'), dict):
        raise ValueError('quality_assessment must be an object')
    errors = w.validate_critic(value, layout, require_perception=True)
    if type(value.get('orientation_pass')) is not bool:
        errors.append('orientation_pass must be an explicit boolean')
    if value.get('orientation_pass') is False and (
            value.get('quality_assessment', {}).get('verdict') == 'PASS' or not value.get('issues')):
        errors.append('Failed orientation requires issues and a non-PASS verdict')
    if value.get('quality_assessment', {}).get('verdict') == 'PASS' and value.get('issues'):
        errors.append('PASS cannot contain unresolved issues')
    if errors:
        raise ValueError('; '.join(errors))
    validate_feedback(value, layout)
    return value


def apply_patch(layout, updates, critic):
    """Apply only model-proposed existing-object boxes; never edit identities."""
    if not isinstance(updates, list) or not updates:
        raise ValueError('patch must be a nonempty list')
    result = deepcopy(layout)
    result['inventory_changes'] = []
    rows = {obj['object_id']: obj for obj in result['objects_data']}
    affected = {issue['object_id'] for issue in critic['issues']}
    seen = set()
    for update in updates:
        if not isinstance(update, dict) or set(update) - {'object_id', 'reason', 'obb', '2d_bbox_normalized'}:
            raise ValueError('Patch fields must be object_id, reason, obb and/or 2d_bbox_normalized')
        ident = update.get('object_id')
        if not isinstance(ident, str) or ident not in rows or ident in seen:
            raise ValueError('Patch IDs must be existing, unique object IDs')
        if ident not in affected and not affected.intersection({'Global', 'global'}):
            raise ValueError('Patch cannot change an object absent from the review issues')
        if not isinstance(update.get('reason'), str) or not update['reason'].strip():
            raise ValueError('Patch requires an evidence-based reason')
        fields = set(update).intersection({'obb', '2d_bbox_normalized'})
        if not fields or all(rows[ident].get(key) == update[key] for key in fields):
            raise ValueError('Patch must change a box')
        if 'obb' in fields and (not isinstance(update['obb'], dict) or
                set(update['obb']) != {'center_m', 'size_m', 'rotation_xyz_deg'}):
            raise ValueError('Patch OBB must contain exactly center_m, size_m, rotation_xyz_deg')
        for key in fields:
            rows[ident][key] = deepcopy(update[key])
        if 'obb' in fields:
            rows[ident]['geometry_reasoning'] = update['reason']
        seen.add(ident)
    return normalize_layout(result, previous=layout)


class RefinementExhausted(RuntimeError):
    pass


def refined_layout(value, previous):
    result = normalize_layout(value, previous=previous)
    registry = dict((previous or {}).get('identity_registry', {}))
    for obj in (previous or {}).get('objects_data', []):
        registry[obj['object_id']] = obj['name']
    for obj in result['objects_data']:
        registry[obj['object_id']] = obj['name']
    result['identity_registry'] = registry
    return result


def run_scene(args, api_factory=ModelAPI):
    started = time.monotonic()
    policy = fast_policy(args)
    if getattr(args, 'perception_backend', 'api') != 'api':
        raise ValueError('fast-v1 requires the API backend')
    args.image = args.image.resolve()
    scene_id = args.scene_id or args.image.stem
    if not re.fullmatch(r'[\w.-]+', scene_id) or scene_id in {'.', '..'}:
        raise ValueError('scene_id must be a single safe directory name')
    out = (args.out_dir / scene_id).resolve()
    prompts = {name: (args.prompts_dir / name).read_text() for name in PROMPTS}
    config = provider_config(args.model, args.reasoning_effort, args.thinking_budget)
    config['technical_attempts'] = policy['technical_attempts']
    fingerprint, binding = w._fingerprint(args, config, prompts)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'run_manifest.json'
    if args.force_layout:
        w._archive(out)
    elif manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get('fingerprint') != fingerprint:
            raise RuntimeError('Scene inputs, code, prompts or configuration changed; use a new output or --force_layout.')
        if old.get('status') == 'completed':
            if not w._verify_artifacts(out, old.get('artifacts', {})):
                raise RuntimeError('Completed scene artifacts changed; refusing reuse.')
            summary = json.loads((out / 'refine_summary.json').read_text())
            if summary.get('status') == 'technical_failure':
                w._archive_failed_delivery(out, old)
            else:
                return summary
    elif any(path.name != 'history' for path in out.iterdir()):
        raise RuntimeError('Unbound output exists; use a new output or --force_layout.')
    manifest = {'fingerprint': fingerprint, 'binding': binding, 'status': 'running',
                'inference_inputs': [str(args.image)], 'ground_truth_allowed': False}
    write_json(manifest_path, manifest)
    api = api_factory(config, args.image, out, fingerprint)
    api.max_attempts = policy['technical_attempts']
    calls, refinements = 0, []
    active_stage = 'fast_generate'
    records, candidates, raw_layouts = [], [], []
    degradation = None
    layout = evaluation = None
    selected_round = None
    accepted, failure, terminal = False, None, None

    def reserve(reason):
        if len(refinements) >= policy['max_refinements']:
            raise RefinementExhausted('Scene parsing refinement budget exhausted')
        refinements.append({'refinement': len(refinements) + 1, 'reason': reason})
        write_json(out / 'fast/refinements.json', {'limit': policy['max_refinements'], 'repairs': refinements})
        return len(refinements)

    def call(stage, texts, images):
        nonlocal calls, active_stage
        active_stage = stage
        images = list(dict.fromkeys(images))
        captions = [{'image_index_1based': i + 1, 'source': 'original_rgb' if path == args.image else str(path.relative_to(out))}
                    for i, path in enumerate(images)]
        calls += 1
        return api.call(stage, [*texts, ('prompt', 'Image order: ' + compact(captions))], images)

    def structured_call(stage, texts, images, validate):
        request_stage, request_texts = stage, texts
        while True:
            raw = call(request_stage, request_texts, images)
            try:
                parsed = w.extract_json(raw)
                if isinstance(parsed, dict) and 'objects_data' in parsed:
                    raw_layouts.append(parsed)
                return validate(parsed)
            except (ValueError, TypeError, KeyError) as exc:
                write_json(out / 'fast' / f'{request_stage}_validation.json', {'error': str(exc)})
                number = reserve({'stage': request_stage, 'kind': 'response_contract', 'error': str(exc)})
                request_stage = stage + f'_format_{number}'
                request_texts = [*texts, ('validation_errors', compact({'error': str(exc), 'invalid_response': raw})),
                    ('prompt', 'Correct the response contract only. Return the complete requested JSON; preserve the substantive judgment.')]

    def save_candidate(candidate):
        nonlocal active_stage
        index = len(candidates) + 1
        active_stage = f'prepare_round_{index}'
        candidates.append({'layout': candidate, 'evaluation': None, 'round': index})
        records.append({'round': index, 'structure_valid': True, 'critic_valid': False})
        write_json(out / f'layout_round_{index}.json', candidate)
        write_json(out / 'final_layout.json', candidate)
        write_json(out / 'inventory.json', candidate)
        diagnostics = w.geometry_diagnostics(candidate)
        write_json(out / f'layout_diagnostics_round_{index}.json', diagnostics)
        w.render_layout_visualizations(args.image, candidate, out, index)
        directory = out / 'perception' / f'fast_round_{index}'
        pages = render_review_panels(args.image, candidate['objects_data'], directory, 'inventory')
        pages += render_edge_panels(args.image, candidate['objects_data'], directory)
        register_images(out, args.image, fingerprint, pages)
        images = [args.image, out / f'layout_2d_round_{index}.png', *pages,
                  *[out / f'layout_{view}_round_{index}.png' for view in ('3d', 'top', 'front', 'side')]]
        return index, diagnostics, images

    def proposal_contract(value):
        if not isinstance(value, dict) or set(value) != {'critic', 'patch'} or not isinstance(value['patch'], list):
            raise ValueError('Review must contain exactly critic and patch (list)')
        validate_review(value['critic'], layout)
        return value

    try:
        layout = structured_call('fast_generate', [('prompt', prompts[PROMPTS[0]])], [args.image],
                                 lambda value: refined_layout(value, None))
        while True:
            index, diagnostics, images = save_candidate(layout)
            texts = [('layout', compact(layout)), ('diagnostics', compact(diagnostics))]
            if index == 1:
                proposal = structured_call('fast_review', [('prompt', prompts[PROMPTS[1]]), *texts], images, proposal_contract)
                write_json(out / 'fast/review_proposal.json', proposal)
                evaluation = proposal['critic']
            else:
                proposal = {'patch': []}
                evaluation = structured_call(f'fast_recheck_{index}', [('prompt', prompts[PROMPTS[2]]), *texts],
                                             images, lambda value: validate_review(value, layout))
            candidates[-1]['evaluation'] = evaluation
            write_json(out / f'layout_eval_round_{index}.json', evaluation)
            records[-1].update(critic_valid=True, score=w.get_eval_metrics(evaluation)[0],
                               localization_pass=evaluation['localization_pass'], geometry_pass=evaluation['geometry_pass'],
                               orientation_pass=evaluation['orientation_pass'])
            write_json(out / f'layout_validation_round_{index}.json', records[-1])
            accepted = w.is_pass(evaluation, args.pass_score)
            if accepted:
                break
            number = reserve({'stage': f'round_{index}', 'kind': 'review_rejected', 'critic': evaluation})
            patched = None
            patch_error = None
            if proposal['patch']:
                try:
                    patched = apply_patch(layout, proposal['patch'], evaluation)
                except (ValueError, TypeError, KeyError) as exc:
                    patch_error = str(exc)
                    write_json(out / 'fast' / f'patch_validation_{number}.json', {'error': patch_error})
            if patched is None:
                previous = layout
                layout = structured_call(f'fast_refine_{number}', [('prompt', prompts[PROMPTS[0]]),
                    ('layout', compact(previous)), ('critic', compact(evaluation)),
                    ('validation_errors', compact({'invalid_patch': patch_error})),
                    ('prompt', 'Refine the complete layout using the original RGB and review. You may correct boxes, inventory, identity and support/tree within this parsing stage. '
                               'Preserve IDs/names for unchanged objects. Correct a wrong identity by removing it and adding a new ID; never reuse retired IDs. '
                               'Declare each actual addition/removal in inventory_changes with action, object_id, name, reason. '
                               'Use inventory_changes:[] when membership is unchanged. Return the entire scene_layout.v2 JSON, not a patch.')],
                    images, lambda value: refined_layout(value, previous))
            else:
                layout = patched
    except RefinementExhausted as exc:
        failure = str(exc)
    except Exception as exc:
        accepted = False
        terminal = exc
        failure = safe_error(exc, getattr(api, 'secret', ''))
        write_json(out / 'fast/stop.json', {'stage': active_stage, 'exception_type': type(exc).__name__, 'error': failure})

    # Delivery is the best structurally valid candidate, with its own review.
    # An unreviewed replacement cannot inherit an earlier candidate's approval.
    def rank(candidate):
        critic = candidate['evaluation']
        if critic is None:
            return (False, -1, -1, -len(candidates))
        score, _, issues = w.get_eval_metrics(critic)
        return (w.is_pass(critic, args.pass_score), score, -len(issues), -candidate['round'])
    if not candidates and terminal is None:
        from .recovery import recover_layout
        layout, degradation = recover_layout(raw_layouts)
        write_json(out / 'degradation_receipt.json', degradation)
        write_json(out / 'degraded_layout.json', layout)
        candidates.append({'layout': layout, 'evaluation': None, 'round': None})
    if candidates:
        best = max(candidates, key=rank)
        layout, evaluation, selected_round = best['layout'], best['evaluation'], best['round']
        accepted = bool(terminal is None and evaluation and w.is_pass(evaluation, args.pass_score))
        write_json(out / 'final_layout.json', layout)
        write_json(out / 'inventory.json', layout)
    audit = {'passed': False, 'error': 'Finalization did not complete'}
    try:
        if candidates:
            w.generate_crops_from_layout(args.image, layout, out, skip_environment=not args.include_environment_crops)
        audit = audit_inputs(out, args.image, binding['image_sha256'], fingerprint)
        if not audit['passed']:
            raise RuntimeError('Input provenance audit failed')
    except Exception as exc:
        terminal = terminal or exc
        failure = failure or safe_error(exc, getattr(api, 'secret', ''))
        accepted = False
    write_json(out / 'input_audit.json', audit)
    score, verdict, issues = w.get_eval_metrics(evaluation) if evaluation else (None, 'unreviewed', [])
    status = 'technical_failure' if terminal else 'accepted' if accepted else 'unaccepted'
    summary = {'status': status, 'accepted': accepted, 'selected_round': selected_round,
        'delivery_status': 'completed' if candidates and audit['passed'] else 'unavailable',
        'ready_for_next_step': bool(candidates and audit['passed'] and terminal is None),
        'delivery_mode': (degradation['delivery_mode'] if degradation and terminal is None else
                          'accepted' if accepted else 'best_available' if candidates and terminal is None else 'unavailable'),
        'degradation_receipt': 'degradation_receipt.json' if degradation else None,
        'selected_score': score, 'selected_verdict': verdict, 'remaining_issues': issues,
        'warnings': (evaluation or {}).get('warnings', []), 'model': args.model, 'scene_id': scene_id,
        'quality_profile': 'fast-v1', 'perception_backend': 'api', 'fast_policy': policy,
        'max_refine_iters': policy['max_refinements'], 'refinements_used': len(refinements),
        'termination_reason': 'technical_failure' if terminal else 'accepted' if accepted else 'refinement_budget_exhausted',
        'pass_score': args.pass_score, 'rounds_attempted': len(records), 'rounds': records,
        'elapsed_seconds': time.monotonic() - started, 'api_activity': aggregate_calls(out), 'fingerprint': fingerprint,
        'autonomous_inference': True, 'gt_used_for_inference': False,
        'orientation_pass': bool(evaluation and evaluation.get('orientation_pass')),
        'review_scope': 'Independent joint 2D/3D/orientation review after each candidate; up to three content refinements.',
        'comparison_note': 'Self-critic scores are not cross-model rankings; fast-v1 does not run orientation-v1 gates.'}
    if failure:
        summary['error'] = failure
    write_json(out / 'refine_summary.json', summary)
    manifest['status'] = 'completed'
    manifest['artifacts'] = w._artifacts(out)
    write_json(manifest_path, manifest)
    print(f'[{args.model}/{scene_id}] {status}; selected round={selected_round}; refinements={len(refinements)}; requests={calls}', flush=True)
    if terminal is not None:
        raise terminal
    return summary
