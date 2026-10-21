"""First acceptance workflow using the unchanged English semantic review protocol."""
from pathlib import Path
import time

from .protocol import read_json, write_json, file_hash
from .physics import acceptance_ready
from .evaluation import EvaluationPipeline
from .optimizer import run_search
from .clearance import propose as propose_clearance


def run(*, args, config, scene, specs, blocks, attempt_dir, fingerprint, api, prompts, report):
    from .worker_pool import scene_worker
    from .workflow import InfrastructureError
    api.ensure_configured()  # Configuration failures are not uncertain model votes.
    timings = report.setdefault('timings', {})
    with scene_worker(config=config, work_dir=attempt_dir/'worker', isaac_python=args.isaac_python) as worker:
        pipeline = EvaluationPipeline(scene, worker, config, attempt_dir, fingerprint)
        try:
            def evaluate(parameters):
                candidate = pipeline.evaluate(parameters)
                if all(all(v == 0 for v in values) for values in parameters.values()):
                    report['initial'] = candidate
                return candidate

            def score(candidate, index):
                start = time.monotonic()
                with pipeline.worker_lock:
                    rendering = worker.render(candidate, attempt_dir/'renders'/candidate['candidate_id'])
                timings['render_seconds'] = timings.get('render_seconds', 0.) + time.monotonic()-start
                if rendering.get('status') != 'rendered' or rendering.get('render_state_verified') is not True:
                    raise InfrastructureError(str(rendering.get('errors', 'Unverified semantic render')))
                candidate['images'] = rendering['images']
                if rendering.get('semantic_frame_complete') is False:
                    return {'score':None, 'uncertain':True,
                            'evidence':[{'object_ids':[], 'reason':'Candidate extends outside the frozen semantic views'}]}
                start = time.monotonic()
                try:
                    return api.score(candidate, index, prompts['semantic'])
                finally:
                    timings['semantic_seconds'] = timings.get('semantic_seconds', 0.)+time.monotonic()-start

            def validate(candidate):
                start = time.monotonic()
                with pipeline.worker_lock:
                    physical = worker.evaluate(candidate, config['final_duration_s'],
                        out_dir=attempt_dir/'final_validation'/candidate['candidate_id'])
                timings['final_validation_seconds'] = timings.get('final_validation_seconds', 0.)+time.monotonic()-start
                if physical.get('status') in ('environment_error', 'infrastructure_error', 'worker_error', 'timeout'):
                    raise InfrastructureError(str(physical.get('errors', physical['status'])))
                if physical.get('status') == 'asset_rejected':
                    from .worker_client import valid_asset_rejection
                    if not valid_asset_rejection(physical):
                        raise InfrastructureError('Malformed final native asset rejection')
                accepted = acceptance_ready(physical)
                if accepted:
                    # Native return includes a closed writer, encoded video and
                    # content hashes. Verify before publishing acceptance.
                    from .worker_client import valid_cached_evidence
                    if not valid_cached_evidence(physical, config['final_duration_s'], config):
                        raise InfrastructureError('Final evidence is missing, incomplete or changed')
                    if any(file_hash(p) != h for p, h in candidate.get('settled_artifacts', {}).items()):
                        raise InfrastructureError('Settled artifact changed before acceptance')
                return {'candidate_id':candidate['candidate_id'], 'accepted':accepted, 'physical':physical,
                        'semantic':candidate['semantic'], 'accepted_unix':time.time() if accepted else None}

            checkpoint = attempt_dir/'search_checkpoint.json'
            if checkpoint.exists() and args.resume:
                for old in read_json(checkpoint)['candidates'].values():
                    checked = pipeline.cached(old['parameters'])
                    if old['physics']['status'] != 'invalid_candidate' and (checked is None or checked['candidate_id'] != old['candidate_id']):
                        raise InfrastructureError('Checkpoint candidate evidence differs')
            search = run_search(object_specs=specs, blocks=blocks, relations=scene['objects'], evaluate=evaluate,
                evaluate_population=pipeline.population, score=score, validate=validate, input_fingerprint=fingerprint,
                settings={**config, 'seed':args.seed}, checkpoint_path=checkpoint,
                resume=args.resume, api_attempts=lambda:api.attempts,
                propose_initial=lambda candidate: propose_clearance(candidate, scene, specs, config))
            report.update(search=search, initial=search['initial'], stop_reason=search['status'],
                          final_validation=list(search['final_validations'].values()))
            accepted = search['accepted_candidate']
            if accepted:
                receipt = next(r for r in search['final_validations'].values() if r['accepted'])
                selected = {**accepted, 'physics':receipt['physical'], 'images':receipt['physical'].get('images', [])}
                report.update(selected=selected, accepted=True, accepted_unix=receipt['accepted_unix'])
                return selected
            selected = search.get('incumbent') or search['best_physical']
            report.update(selected=selected, accepted=False,
                          failure_type='semantic_unaccepted' if selected and selected['physics']['passed'] else 'physical_layout_failure')
            pipeline.close()
            if search['status'] == 'native_asset_topology_rejected':
                report.update(failure_type='asset_content_rejected',
                    terminal_fallback={'status': 'not_run', 'reason': 'native_asset_topology_rejected'})
                return selected
            from .terminal_settle import run as terminal_run
            result = terminal_run(worker, read_json(checkpoint), report['initial'], attempt_dir/'terminal_fallback', config, scene)
            report['terminal_fallback'] = result
            return result['selected']
        finally:
            pipeline.close()
