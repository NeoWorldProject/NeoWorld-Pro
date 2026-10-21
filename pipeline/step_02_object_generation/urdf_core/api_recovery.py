"""Resume an interrupted frozen request using exactly its recorded model inputs."""
from pathlib import Path
import json
import os
import time

from .api_requests import POLICY, audited_call, logical_stage
from .run_state import file_hash, fingerprint, write_json
from .runtime_snapshot import verified_runtime


def restore_content(request, root, api, *, source_images=None):
    images = {}
    wanted = {item['image_sha256'] for item in request['content'] if item['type'] == 'input_image'}
    sources = []
    for name, digest in (request['source_images'] if source_images is None else source_images).items():
        path = Path(name)
        if file_hash(path) != digest:
            raise ValueError('Original source image changed: ' + name)
        sources.append(path)
    paths = [*sources, *(p for p in sorted(Path(root).rglob('*'))
                        if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'} and p.is_file())]
    for path in paths:
        if not wanted - images.keys():
            break
        value = api.make_image_input(path)
        digest = fingerprint(value['image_url'])
        if digest in wanted:
            images[digest] = (value, {'path': str(path.resolve()), 'sha256': file_hash(path)})
    if wanted - images.keys():
        raise ValueError('Recorded request image bytes are unavailable')
    content = []
    for item in request['content']:
        if item['type'] == 'input_text':
            content.append(api.make_text_input(item['text']))
        elif item['type'] == 'input_image':
            content.append(images[item['image_sha256']][0])
        else:
            raise ValueError('Unsupported recorded request content')
    return content, {digest: binding for digest, (_, binding) in images.items()}


def _bound_sources(config):
    sources = dict(config['source_images'])
    reference = config.get('target_reference') or {}
    contexts = [config.get('scene_context'), reference.get('context')]
    for context in contexts:
        if context is None:
            continue
        for key in ('layout', 'scene_image', 'target_crop', 'context_file'):
            binding = context[key]
            path = Path(binding['path'])
            if not path.is_absolute() or file_hash(path) != binding['sha256']:
                raise ValueError('Frozen scene context evidence changed: ' + key)
            if key in ('scene_image', 'target_crop'):
                if str(path) in sources and sources[str(path)] != binding['sha256']:
                    raise ValueError('Conflicting frozen source image binding')
                sources[str(path)] = binding['sha256']
    if reference and file_hash(reference['prompt_path']) != reference['prompt_sha256']:
        raise ValueError('Frozen target reference prompt changed')
    return sources


def recover_request(ledger, *, execute=False, service_restored=False):
    import generate_object_urdf as api
    ledger = Path(ledger).resolve()
    request_root = ledger.parent.parent
    data = json.loads(ledger.read_text())
    if ledger.parent.name != 'model_calls' or data.get('policy') != POLICY:
        raise ValueError('Select an object API request ledger')
    attempt = data['attempts'][-1]
    confirmed_service = service_restored and data.get('terminal_error') and bool(data.get('service_issue'))
    if attempt['status'] != 'error' or (not confirmed_service and (
            data.get('terminal_error') or not attempt.get('retryable'))):
        raise ValueError('Only interrupted retryable transport requests can be resumed')
    request_path = ledger.parent / attempt['request']
    if request_path.parent != ledger.parent or file_hash(request_path) != attempt['request_sha256']:
        raise ValueError('Recorded request changed')
    request = json.loads(request_path.read_text())
    if fingerprint(request) != attempt['request_key']:
        raise ValueError('Recorded request identity changed')
    prefix = f'{fingerprint({"stage": data["stage"]})}_{attempt["number"]:02d}_'
    if not request_path.name.startswith(prefix):
        raise ValueError('Unexpected request filename')
    stage = request_path.name[len(prefix):-len('.request.json')]
    if logical_stage(stage) != data['stage']:
        raise ValueError('Recorded logical stage changed')
    root = next((path for path in (request_root, *request_root.parents)
                 if (path / 'run_manifest.json').is_file()), None)
    if root is None:
        raise ValueError('API request has no enclosing object run manifest')
    manifest = json.loads((root / 'run_manifest.json').read_text())
    config = manifest['config']
    physics = str(config.get('acceptance_policy', '')).startswith('automatic_object_physics_')
    if request_root != root and not physics:
        raise ValueError('Nested API requests require a physics run')
    runtime = config['runtime_snapshot']
    package = verified_runtime(runtime)
    if file_hash(package / 'generate_object_urdf.py') != file_hash(Path(api.__file__)):
        raise ValueError('Frozen model adapter differs; use its compatible recovery tool')
    endpoint = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')
    provider = fingerprint(endpoint if physics else {'provider': 'openai', 'endpoint': endpoint})
    if provider != request['provider_identity']:
        raise ValueError('Provider endpoint changed')
    fields = ('model', 'reasoning_effort', 'provider_identity')
    if any(request.get(key) != config[key] for key in fields):
        raise ValueError('Request differs from frozen object configuration')
    if physics:
        if request.get('logical_stage') != data['stage'] or 'source_images' in request:
            raise ValueError('Request differs from the frozen physics request schema')
    elif request.get('source_images') != config['source_images']:
        raise ValueError('Request differs from frozen generation configuration')
    content, images = restore_content(request, root, api, source_images=_bound_sources(config))
    receipt = {'policy': 'frozen_transport_recovery.v1', 'ledger': str(ledger),
        'prior_ledger_sha256': file_hash(ledger), 'prior_attempts': len(data['attempts']),
        'request_sha256': file_hash(request_path), 'request_key': attempt['request_key'],
        'runtime_snapshot': runtime, 'object_run_dir': str(root), 'request_root': str(request_root),
        'request_kind': 'physics' if physics else 'generation',
        'stage': stage, 'image_count': len(images), 'images': images,
        'implementation': {str(Path(__file__)): file_hash(Path(__file__)),
            str(Path(__file__).with_name('api_requests.py')): file_hash(Path(__file__).with_name('api_requests.py'))},
        'execute': execute, 'service_restored': service_restored, 'started_epoch': time.time()}
    if not execute:
        return receipt
    os.environ['URDF_REASONING_EFFORT'] = request['reasoning_effort']
    os.environ['NEOWORLD_API_CONCURRENCY'] = str(config['runtime']['api_concurrency'])
    receipt_path = root / 'technical_recovery' / f'{time.time_ns()}.json'
    write_json(receipt_path, receipt)
    try:
        client = api.OpenAI(base_url=endpoint, timeout=300, max_retries=0)
        _, paths = audited_call(request_root, stage, request,
            lambda: api.call_model(client, request['model'], content, max_retries=1),
            service_restored=service_restored)
        receipt.update(status='received', response_path=str(paths[1]), response_sha256=file_hash(paths[1]))
    except BaseException as error:
        receipt.update(status='technical_paused', error_type=type(error).__name__)
        raise
    finally:
        receipt.update(finished_epoch=time.time(), final_ledger_sha256=file_hash(ledger))
        write_json(receipt_path, receipt)
    return receipt


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--service-restored', action='store_true',
        help='Resume the exact failed request after an account/service problem has been resolved')
    return parser


def main(argv=None):
    from .environment import load_project_env
    load_project_env()
    args = build_parser().parse_args(argv)
    result = recover_request(args.ledger, execute=args.execute, service_restored=args.service_restored)
    print(json.dumps({k: v for k, v in result.items() if k != 'images'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
