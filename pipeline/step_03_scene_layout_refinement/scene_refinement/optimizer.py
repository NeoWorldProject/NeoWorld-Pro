"""Truncated Gaussian block CEM, stopping at the first independently accepted scene."""
from copy import deepcopy
import math
from pathlib import Path
import random
from statistics import mean

from .scoring import _key, _tuples, _semantic, candidate_reward
from .protocol import write_json, read_json
from .provider import APIBudgetExhausted, SemanticUnavailable, APITechnicalFailure
from .physics import acceptance_ready
from .worker_client import valid_asset_rejection


def sample_population(anchor, group, distributions, specs, count, rng):
    population = []
    for _ in range(count):
        parameters = deepcopy(anchor)
        for oid in group:
            parameters[oid] = []
            for axis, (lo, hi) in enumerate(specs[oid]['bounds']):
                # No clipping: boundary masses must not distort elite moments.
                for attempt in range(100000):
                    value = rng.gauss(distributions[oid]['mu'][axis], distributions[oid]['std'][axis])
                    if lo <= value <= hi:
                        break
                else:
                    raise RuntimeError('Truncated Gaussian sampler exhausted: invalid distribution')
                parameters[oid].append(value)
        population.append(parameters)
    return population


def update_moments(distributions, specs, group, elite_parameters, floor):
    for oid in group:
        for axis in range(4):
            values = [p[oid][axis] for p in elite_parameters]
            mu = mean(values)
            distributions[oid]['mu'][axis] = mu
            distributions[oid]['std'][axis] = max(math.sqrt(mean((v-mu)**2 for v in values)), specs[oid]['sigma'][axis]*floor)


def failure_group(candidate, relations, ids, blocks, generation):
    physics = candidate['physics']
    issues = sorted(physics.get('issues', []), key=lambda i: -i.get('penalty', 0.))
    for issue in issues:
        group = [oid for oid in issue.get('object_ids', [issue.get('object_id')]) if oid in ids]
        if not group:
            continue
        for oid in list(group):
            relation = relations.get(oid, {})
            parent = relation.get('parent_id')
            categories = issue.get('categories', {})
            if parent in ids and (relation.get('relation') == 'inside' or categories.get('target_support', 0) > 0):
                group.append(parent)
        return sorted(set(group))
    # A semantic-only failure may implicate a specific visible relationship.
    for record in candidate.get('semantic_samples', []):
        for item in (record.get('response') or {}).get('evidence', []):
            group = sorted(set(item.get('object_ids', [])) & set(ids))
            if group:
                return group
    return list(blocks[generation % len(blocks)])


def run_search(*, object_specs, evaluate, score, validate, input_fingerprint, settings,
               checkpoint_path, resume=False, blocks=None, relations=None, evaluate_population=None,
               api_attempts=None, propose_initial=None):
    cfg = settings
    specs = {s['object_id']: deepcopy(s) for s in object_specs}
    ids = list(specs)
    initial = {oid: [0.]*4 for oid in ids}
    blocks = blocks or [[oid] for oid in ids]
    path = Path(checkpoint_path)
    identity = {'algorithm': 'scene_cem.v3', 'fingerprint': input_fingerprint,
                'specs': specs, 'config': cfg, 'relations': relations or {}, 'blocks': blocks}
    rng = random.Random(cfg.get('seed', 0))
    if path.exists():
        if not resume:
            raise ValueError('Checkpoint exists; explicit resume required')
        state = read_json(path)
        if state['identity'] != identity:
            raise ValueError('Checkpoint source, inputs, runtime or policy changed')
        if state.get('terminal_technical_failure'):
            raise RuntimeError('Saved terminal technical failure: ' + state['terminal_technical_failure'])
        rng.setstate(_tuples(state['rng_state']))
        for records in state['scores'].values():
            for record in records:
                if record['status'] == 'pending':
                    record.update(status='interrupted', error='No independent judgment available after interruption')
        for record in state['final_validations'].values():
            if record.get('status') == 'pending':
                # Never repeat an ambiguously completed validation of this exact evidence.
                record.update(status='interrupted', accepted=False)
    else:
        state = {'identity': identity, 'format_version': 1, 'status': 'running', 'generation': 0,
                 'candidates': {}, 'scores': {}, 'history': [], 'pending_round': None,
                 'final_validations': {}, 'accepted_key': None, 'initial_key': _key(initial),
                 'distributions': {o: {'mu': initial[o][:], 'std': specs[o]['sigma'][:]} for o in ids},
                 'stagnant_generations': 0}
    candidates = state['candidates']

    def save():
        state['rng_state'] = rng.getstate()
        state['api_attempts'] = api_attempts() if api_attempts else sum(len(r) for r in state['scores'].values())
        write_json(path, state)

    def summary(key):
        return _semantic(state['scores'].get(key, []))

    def reward(key):
        candidate = candidates[key]
        if candidate['physics'].get('status') in ('certified_pruned', 'asset_rejected'):
            raise ValueError('This candidate has no exact physical score')
        distances = [(candidate['parameters'][oid][axis] / max(abs(v) for v in specs[oid]['bounds'][axis]))**2
                     for oid in ids for axis in range(4)]
        distance = min(1., math.sqrt(mean(distances))) if distances else 0.
        return candidate_reward(candidate['physics'], summary(key)['score'], distance,
                                semantic_weight=cfg['semantic_weight'], deviation_weight=cfg['deviation_weight'])

    def usable(key):
        return candidates[key]['physics'].get('status') not in ('certified_pruned', 'invalid_candidate', 'asset_rejected')

    def best():
        keys = [k for k in candidates if usable(k)]
        return min(keys, key=lambda k: (-reward(k), k)) if keys else None

    def record(parameters, candidate):
        key = _key(parameters)
        p = candidate['physics']
        if p.get('status') in ('environment_error', 'worker_error', 'timeout', 'infrastructure_error'):
            raise RuntimeError(str(p.get('errors', p)))
        if p.get('status') == 'asset_rejected':
            if not valid_asset_rejection(p):
                raise ValueError('Malformed native asset rejection')
            if p['asset_scope'] == 'asset_topology':
                state['status'] = 'native_asset_topology_rejected'
        elif p.get('status') != 'certified_pruned':
            if type(p.get('passed')) is not bool or not math.isfinite(p.get('violation', float('nan'))) or p['violation'] < 0:
                raise ValueError('Invalid physical report')
            if p['passed'] and not acceptance_ready(p):
                raise ValueError('Incomplete native/evidence observation cannot pass')
        candidate['parameters'] = deepcopy(parameters)
        candidates[key] = candidate
        state['scores'].setdefault(key, [])
        save()
        return key

    def observe(key):
        records = state['scores'][key]
        if records and not (len(records) == 1 and summary(key)['score'] == 1):
            return
        used = api_attempts() if api_attempts else sum(len(v) for v in state['scores'].values())
        if cfg.get('api_budget') and used >= cfg['api_budget']:
            state['status'] = 'api_budget_exhausted'
            save()
            return
        rec = {'sample_index': len(records), 'status': 'pending', 'api_attempts_before': used}
        records.append(rec)
        save()
        try:
            rec.update(status='done', response=score(deepcopy(candidates[key]), rec['sample_index']))
        except APITechnicalFailure as exc:
            rec.update(status='technical_failure', error=str(exc))
            state['terminal_technical_failure'] = str(exc)
            save()
            raise
        except SemanticUnavailable as exc:
            used_after = api_attempts() if api_attempts else used
            if api_attempts is not None and used_after == used:
                rec.update(status='not_dispatched', error=str(exc), actual_api_attempts=0)
                state.setdefault('pre_dispatch_configuration_failures', []).append({'candidate_key':key,'record':deepcopy(rec)})
                records.pop()
                save()
                raise
            rec.update(status='error', error=f'{type(exc).__name__}: {exc}')
        except APIBudgetExhausted as exc:
            rec.update(status='budget_exhausted', error=str(exc))
            state['status'] = 'api_budget_exhausted'
        except Exception as exc:
            # Only an explicitly classified content error is semantic feedback.
            # Rendering, evidence and implementation failures must stay terminal.
            rec.update(status='technical_failure', error=f'{type(exc).__name__}: {exc}')
            state['terminal_technical_failure'] = rec['error']
            save()
            raise
        rec['actual_api_attempts'] = (api_attempts() - used) if api_attempts else 1
        save()

    def try_accept(key):
        if key in state['final_validations'] or not acceptance_ready(candidates[key]['physics']):
            return False
        if not state['scores'][key]:
            observe(key)
        if len(state['scores'][key]) == 1 and summary(key)['score'] == 1:
            observe(key)
        sem = summary(key)
        if sem['score'] != 1 or not sem['reliable']:
            return False
        finalist = deepcopy(candidates[key])
        finalist.update(semantic=sem, semantic_samples=deepcopy(state['scores'][key]))
        state['final_validations'][key] = {'status': 'pending', 'candidate_id': finalist['candidate_id'], 'accepted': False}
        save()
        try:
            result = validate(finalist)
        except Exception as exc:
            state['final_validations'][key].update(status='technical_failure', error=str(exc))
            state['terminal_technical_failure'] = f'{type(exc).__name__}: {exc}'
            save()
            raise
        if result.get('accepted') and not acceptance_ready(result['physical']):
            raise ValueError('Final validation accepted missing native observations')
        state['final_validations'][key] = result
        if result['accepted']:
            state.update(status='accepted', accepted_key=key)
        elif result.get('physical', {}).get('status') == 'asset_rejected':
            final = deepcopy(result['physical'])
            if not valid_asset_rejection(final):
                raise ValueError('Malformed final native asset rejection')
            candidates[key]['search_physics'] = candidates[key]['physics']
            candidates[key]['physics'] = final
            if final['asset_scope'] == 'asset_topology':
                state['status'] = 'native_asset_topology_rejected'
        elif result.get('physical', {}).get('known_infeasible') or result.get('physical', {}).get('status') == 'physics_failed':
            search = candidates[key]['physics']
            final = deepcopy(result['physical'])
            candidates[key]['search_physics'] = search
            candidates[key]['physics'] = final
            if search['violation'] > final['violation']:
                final.update(violation=search['violation'], physical_residual=search['physical_residual'],
                             reward_terms=deepcopy(search.get('reward_terms', {})))
            final.update(passed=False, status='physics_failed', known_infeasible=True)
        elif result.get('physical', {}).get('status') in ('environment_error', 'infrastructure_error', 'worker_error', 'timeout'):
            save()
            raise RuntimeError('Final validation infrastructure failure')
        save()
        return result['accepted']

    if state['initial_key'] not in candidates:
        record(initial, evaluate(initial))
    if state['status'] == 'running':
        for key in list(candidates):
            if try_accept(key) or state['status'] != 'running':
                break
    if state['status'] == 'running' and propose_initial:
        if 'clearance_initialization' not in state:
            state['clearance_initialization'] = propose_initial(deepcopy(candidates[state['initial_key']]))
            save()
        initialization = state['clearance_initialization']
        parameters = initialization.get('parameters')
        if parameters is not None and not initialization.get('evaluated'):
            if set(parameters) != set(specs) or any(len(parameters[o]) != 4 or
                    any(not math.isfinite(v) or not specs[o]['bounds'][axis][0] <= v <= specs[o]['bounds'][axis][1]
                        for axis, v in enumerate(parameters[o])) for o in specs):
                raise ValueError('Clearance initialization exceeds the frozen search domain')
            key = _key(parameters)
            if key not in candidates:
                record(parameters, evaluate(parameters))
            initialization.update(evaluated=True, candidate_key=key)
            if best() == key:
                for oid in ids:
                    state['distributions'][oid]['mu'] = parameters[oid][:]
            save()
            try_accept(key)
    while state['status'] == 'running' and state['generation'] < cfg['max_iterations']:
        work = state['pending_round']
        if work is None:
            # An invalid initial placement can still have valid neighbors in the
            # frozen search domain. Use its parameters as a sampling anchor;
            # it remains ineligible for acceptance, elites and delivery.
            anchor_key = best() or state['initial_key']
            anchor = candidates[anchor_key]
            annotated = dict(anchor, semantic_samples=state['scores'][anchor_key])
            group = failure_group(annotated, relations or {}, ids, blocks, state['generation'])
            work = {'round': state['generation'], 'block': group, 'anchor_key': anchor_key,
                    'population': sample_population(anchor['parameters'], group, state['distributions'], specs,
                                                    cfg['population_size'], rng), 'tasks': {}, 'completed': []}
            state['pending_round'] = work
            save()

        def witness():
            # Only exact, immutable physical failures in THIS frozen population.
            keys = {_key(p) for p in work['population']} & set(candidates)
            failures = sorted((k for k in keys if usable(k) and not candidates[k]['physics']['passed']
                               and candidates[k]['physics'].get('known_infeasible')), key=lambda k: (candidates[k]['physics']['violation'], k))
            if len(failures) < cfg['elite_count']:
                return None
            return {'generation': work['round'], 'elite_count': cfg['elite_count'],
                    'witnesses': [{'key': k, 'violation': candidates[k]['physics']['violation'],
                                   'known_physical_failure': True} for k in failures[:cfg['elite_count']]]}

        def stage(index, value):
            work['tasks'][str(index)] = value
            save()

        missing = [(i, p) for i, p in enumerate(work['population']) if _key(p) not in candidates]
        def serial():
            for index, p in missing:
                yield index, evaluate(p)
        iterator = evaluate_population(missing, witness, stage) if evaluate_population else serial()
        try:
            # Resume semantic/final service before dispatching further search.
            for p in work['population']:
                key = _key(p)
                if key in candidates and try_accept(key):
                    break
            if state['status'] == 'running':
                for index, candidate in iterator:
                    key = record(work['population'][index], candidate)
                    work['completed'].append(index)
                    stage(index, {'stage': 'complete', 'key': key,
                                  'effective_duration_s': candidate['physics'].get('effective_duration_s'),
                                  'pruning_certificate': candidate['physics'].get('pruning_certificate')})
                    if try_accept(key) or state['status'] != 'running':
                        break
        finally:
            if hasattr(iterator, 'close'):
                iterator.close()
        keys = list(dict.fromkeys(_key(p) for p in work['population'] if _key(p) in candidates))
        if state['status'] != 'running':
            state['history'].append({'round': work['round'], 'block': work['block'], 'mode': 'first_accepted' if state['accepted_key'] else state['status'],
                                     'evaluated_candidates': len(keys), 'planned_candidates': len(work['population'])})
            break
        usable_keys = [k for k in keys if usable(k)]
        if len(usable_keys) < cfg['elite_count']:
            # A population with too few constructible placements gives no
            # valid moment update. Keep the distribution and try the next
            # bounded generation instead of terminating with unused budget.
            state['stagnant_generations'] += 1
            state['history'].append({'round': work['round'], 'block': work['block'],
                'anchor_key': work['anchor_key'],
                'population_keys': [_key(p) for p in work['population']],
                'elite_keys': [], 'elite_rewards': [], 'updated': False,
                'mode': 'insufficient_usable_elites', 'usable_candidates': len(usable_keys),
                'stagnant_generations': state['stagnant_generations'],
                'distributions': deepcopy(state['distributions'])})
            state['generation'] += 1
            state['pending_round'] = None
            save()
            continue
        ranked = sorted(usable_keys, key=lambda k: (-reward(k), work['population'].index(candidates[k]['parameters'])))
        elites = ranked[:cfg['elite_count']]
        rewards = [reward(k) for k in usable_keys]
        # Pruned candidates are strictly worse: they are evidence of non-ties.
        tied = len(usable_keys) == len(keys) and max(rewards)-min(rewards) <= cfg['physical_signal_epsilon']
        if not tied:
            update_moments(state['distributions'], specs, work['block'], [candidates[k]['parameters'] for k in elites], cfg['min_std_fraction'])
        previous_best = work['anchor_key']
        current_best = best()
        improved = current_best is not None and (not usable(previous_best) or
            reward(current_best) > reward(previous_best) + cfg['physical_signal_epsilon'])
        state['stagnant_generations'] = 0 if improved else state['stagnant_generations']+1
        state['history'].append({'round': work['round'], 'block': work['block'], 'anchor_key': previous_best,
            'population_keys': [_key(p) for p in work['population']], 'elite_keys': elites,
            'elite_rewards': [reward(k) for k in elites], 'pruned_count': sum(not usable(k) and candidates[k]['physics']['status']=='certified_pruned' for k in keys),
            'updated': not tied, 'all_tied': tied, 'stagnant_generations': state['stagnant_generations'],
            'distributions': deepcopy(state['distributions'])})
        state['generation'] += 1
        state['pending_round'] = None
        save()
    if state['status'] == 'running':
        state['status'] = 'budget_exhausted'
    save()

    def export(key):
        if key is None:
            return None
        c = deepcopy(candidates[key])
        c.update(semantic=summary(key), semantic_samples=deepcopy(state['scores'][key]),
                 reward=reward(key) if usable(key) else None)
        c['reward_terms'] = {**c['physics'].get('reward_terms', {}), 'mllm': c['semantic']['score']}
        return c
    return {'status': state['status'], 'accepted': bool(state['accepted_key']), 'accepted_candidate': export(state['accepted_key']),
            'initial': export(state['initial_key']), 'best_physical': export(best()),
            'incumbent': export(best()) if best() and candidates[best()]['physics']['passed'] else None,
            'history': state['history'], 'unique_candidates': len(candidates), 'final_validations': state['final_validations'],
            'rounds_completed': state['generation'], 'score_calls': sum(len(v) for v in state['scores'].values()),
            'api_attempts': state['api_attempts'], 'boundary_expansions': 0, 'distributions': state['distributions'],
            'checkpoint_path': str(path), 'stagnant_generations': state['stagnant_generations'],
            'clearance_initialization': state.get('clearance_initialization')}
