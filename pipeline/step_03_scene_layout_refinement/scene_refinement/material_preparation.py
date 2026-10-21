"""Create a read-only-source, content-bound appearance overlay for a scene manifest."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

STAGE2 = Path(__file__).resolve().parents[2]/'step_02_object_generation'
sys.path.insert(0, str(STAGE2))
from scene_refinement.input_adapter import load_scene_assets
from scene_refinement.protocol import read_json, write_json, digest, file_hash
from scene_refinement.visual_materials import restore
from scene_refinement.assets import require_physical_file
from urdf_core.content_errors import AppearanceContentError


def run_bake(command, folder):
    """CPU-only material conversion has its own durable five-attempt allowance."""
    from urdf_core.execution import wait_process, TechnicalExecutionError, native_owner_environment
    folder = Path(folder)
    path = folder / 'bake_attempts.json'
    binding = digest(command)
    ledger = read_json(path) if path.exists() else {'binding': binding, 'attempts': []}
    if ledger['binding'] != binding:
        raise RuntimeError('Material conversion command changed')
    rejection_path = folder/'bake/appearance_rejection.json'
    def check_rejection():
        rejected = read_json(rejection_path)
        if not rejected.get('sources') or any(file_hash(p) != h for p, h in rejected['sources'].items()):
            raise RuntimeError('Material content rejection sources changed')
        raise AppearanceContentError(rejected['reason'])
    if ledger.get('status') == 'content_rejected':
        if file_hash(rejection_path) != ledger['receipt_sha256']:
            raise RuntimeError('Material content rejection changed')
        check_rejection()
    if ledger.get('status') == 'completed':
        if file_hash(folder/'bake/bake_receipt.json') != ledger['receipt_sha256']:
            raise RuntimeError('Completed material conversion receipt changed')
        return
    # A completed conversion is reused through the material package receipt.
    # Interrupted attempts still count, so restarting cannot buy a sixth launch.
    while len(ledger['attempts']) < 5:
        attempt = {'attempt': len(ledger['attempts']) + 1, 'status': 'pending'}
        ledger['attempts'].append(attempt)
        write_json(path, ledger)
        log_path = folder / f"bake_attempt_{attempt['attempt']}.log"
        error = None
        with log_path.open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, **native_owner_environment()})
            try:
                code = wait_process(process, progress=lambda: log_path.stat().st_size or None)
                if code:
                    error = f'Blender material conversion exited with {code}; see {log_path}'
            except TechnicalExecutionError as exc:
                error = str(exc)
        attempt.update(status='failed' if error else 'completed', error=error)
        if not error:
            ledger.update(status='completed', receipt_sha256=file_hash(folder/'bake/bake_receipt.json'))
        write_json(path, ledger)
        if error and rejection_path.is_file():
            ledger.update(status='content_rejected', receipt_sha256=file_hash(rejection_path))
            write_json(path, ledger)
            check_rejection()
        if not error:
            return
    raise TechnicalExecutionError('Material conversion exhausted five technical attempts: ' + str(ledger['attempts'][-1].get('error')))


def published_visual_source(source):
    physical_path = Path(source).resolve().parent/'asset_manifest.json'
    require_physical_file(physical_path, 'physical asset_manifest.json')
    physical = read_json(physical_path)
    provenance = physical.get('source', {})
    if not provenance.get('base_run'):
        return None
    origin = Path(provenance['base_run']).resolve()
    manifest = origin/'urdf_export/export_manifest.json'
    blend = origin/'pipeline_scene.blend'
    generation_path = origin/'run_manifest.json'
    generation = read_json(generation_path)
    artifacts = generation.get('stages', {}).get('articulation_export', {}).get('artifacts', {})
    extra_evidence = {}
    required = {'pipeline_scene.blend', 'urdf_export/export_manifest.json'}
    if not required <= artifacts.keys() and generation.get('config', {}).get('delivery_policy') == 'best-effort':
        ledger_path = origin/'generation_delivery_candidates.json'
        delivery_path = origin/'final/delivery_manifest.json'
        ledger, delivery = read_json(ledger_path), read_json(delivery_path)
        selected = ledger['candidates'][ledger['best']]
        folder = (origin/selected['folder']).resolve()
        if (not folder.is_relative_to(origin) or delivery.get('schema_version') != 'best_effort_delivery.v1'
                or delivery.get('delivery_status') != 'completed' or selected not in delivery.get('findings', [])
                or delivery.get('files', {}).get('export_manifest.json') != provenance.get('export_manifest_sha256')):
            raise ValueError('Retained generation publication binding changed: '+str(origin))
        artifacts = {}
        for name in required:
            source_file = (folder/name).resolve()
            if not source_file.is_relative_to(origin):
                raise ValueError('Retained visual source escapes its publication')
            expected = selected['files'].get(str(source_file.relative_to(origin)))
            if not expected or file_hash(source_file) != expected:
                raise ValueError('Retained visual source hash binding changed: '+str(source_file))
            artifacts[name] = expected
            extra_evidence[str(source_file)] = expected
        extra_evidence.update({str(p):file_hash(p) for p in (ledger_path, delivery_path)})
    if (file_hash(manifest) != provenance.get('export_manifest_sha256')
            or file_hash(manifest) != artifacts.get('urdf_export/export_manifest.json')
            or file_hash(blend) != artifacts.get('pipeline_scene.blend')):
        raise ValueError('Published visual source hash binding changed: '+str(origin))
    evidence = {**extra_evidence, **{str(p):file_hash(p) for p in (physical_path, generation_path, manifest, blend)}}
    return blend, manifest, evidence


def prepare(base_manifest, out_dir, blender, resolution=512):
    base_manifest, out_dir = Path(base_manifest).resolve(), Path(out_dir).resolve()
    normalized = load_scene_assets(base_manifest)
    packages = {}; timings = {}; unavailable = []; start = time.monotonic()
    script = STAGE2/'urdf_core/bake_visual_colors.py'
    for record in normalized['objects']:
        source = Path(record['usd_path'])
        try:
            visual_source = published_visual_source(source)
        except FileNotFoundError as error:
            # The published USD remains authoritative; missing optional bake sources do not erase it.
            require_physical_file(source.parent/'asset_manifest.json', 'physical asset_manifest.json')
            unavailable.append({'object_id': record['object_id'], 'reason': str(error), 'appearance': 'published_USD_base_color'})
            visual_source = None
        if visual_source is None:
            timings[record['object_id']] = 0.
            continue
        blend, manifest, provenance = visual_source
        binding = {'usd':file_hash(source), 'blend':file_hash(blend), 'export':file_hash(manifest),
                   'source_evidence':provenance, 'resolver':file_hash(__file__),
                   'baker':file_hash(script), 'converter':file_hash(Path(__file__).parent/'visual_materials.py'),
                   'blender':file_hash(blender), 'resolution':resolution}
        folder = out_dir/'cache'/digest(binding)
        receipt = folder/'usd/material_package.json'
        rejected_path = folder/'appearance_degradation.json'
        elapsed = time.monotonic()
        if rejected_path.is_file():
            rejected = read_json(rejected_path)
            if rejected.get('binding') != binding:
                raise ValueError('Appearance degradation source binding changed')
            unavailable.append(rejected)
            timings[record['object_id']] = time.monotonic()-elapsed
            continue
        if not receipt.exists():
            folder.mkdir(parents=True, exist_ok=True)
            command = [str(blender), '-b', '--factory-startup', '-t', '2', '--python-exit-code', '1',
                       '--python', str(script), '--', '--blend', str(blend), '--export-manifest', str(manifest),
                       '--out-dir', str(folder/'bake'), '--resolution', str(resolution)]
            try:
                run_bake(command, folder)
                restore(source, folder/'bake/bake_receipt.json', folder/'usd')
            except AppearanceContentError as error:
                rejected = {'object_id': record['object_id'], 'reason': str(error),
                    'appearance': 'published_USD_base_color', 'binding': binding}
                write_json(rejected_path, rejected)
                unavailable.append(rejected)
                timings[record['object_id']] = time.monotonic()-elapsed
                continue
            write_json(folder/'binding.json', binding)
        elif read_json(folder/'binding.json') != binding:
            raise ValueError('Material cache binding changed')
        packages[record['object_id']] = {'path':str(receipt), 'sha256':file_hash(receipt)}
        timings[record['object_id']] = time.monotonic()-elapsed
        print(json.dumps({'object_id':record['object_id'], 'material_package':str(receipt), 'seconds':timings[record['object_id']]}), flush=True)
    output = out_dir/'asset_manifest.json'
    write_json(output, {'schema_version':'scene_assets.visual_overlay.v1', 'base_manifest':str(base_manifest),
                        'base_manifest_sha256':file_hash(base_manifest), 'visual_materials':packages})
    write_json(out_dir/'preparation_receipt.json', {'base_manifest':str(base_manifest), 'output_manifest':str(output),
        'output_sha256':file_hash(output), 'elapsed_seconds':time.monotonic()-start, 'object_seconds':timings,
        'source_assets_modified':False, 'missing_visual_sources': unavailable,
        'color_scope':'Baked source base color where available; otherwise published USD appearance.'})
    return output
