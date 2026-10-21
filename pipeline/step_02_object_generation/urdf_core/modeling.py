"""Original free-bpy generation and candidate selection, with isolated execution."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from .generation_research import GenerationResearch
from .delivery_policy import BudgetExhausted
from .run_state import file_hash, fingerprint


def run_blender(cmd, *, cwd, log_path, api, env=None):
    """Run one GPU task with explicit Blender progress and bounded crash recovery."""
    from .gpu_queue import GpuTask, GpuOutOfMemory, oom_evidence
    from .execution import wait_process, TechnicalExecutionError, native_owner_environment
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = log_path.with_suffix('.progress.json')
    child_env = dict(os.environ if env is None else env)
    child_env['NEOWORLD_BLENDER_PROGRESS'] = str(progress_path.resolve())
    child_env.update(native_owner_environment())
    command = list(map(str, cmd))
    source = command[command.index('--python') + 1]
    wrapper = log_path.with_suffix('.watchdog.py')
    wrapper.write_text("import json, pathlib, runpy, sys\n"
        + "sys.path.insert(0," + repr(str(Path(__file__).resolve().parents[1])) + ")\n"
        + "from urdf_core.execution import start_owner_monitor\nstart_owner_monitor()\n"
        + "import bpy\n"
        + "p=pathlib.Path(" + repr(str(progress_path.resolve())) + ")\n"
        + "counter=[0]\n"
        + "def progress(kind):\n    counter[0]+=1\n    t=p.with_suffix('.tmp')\n    t.write_text(json.dumps([kind,counter[0],len(bpy.data.objects)]))\n    t.replace(p)\n"
        + "@bpy.app.handlers.persistent\ndef rendered(*args): progress('render_completed')\n"
        + "@bpy.app.handlers.persistent\ndef frame_changed(*args): progress('frame_changed')\n"
        + "bpy.app.handlers.render_complete.append(rendered)\n"
        + "bpy.app.handlers.frame_change_post.append(frame_changed)\n"
        + "previous=[None]\n"
        + "def geometry_progress():\n    value=(len(bpy.data.objects),len(bpy.data.meshes))\n    if value!=previous[0]:\n        previous[0]=value\n        progress('geometry_changed')\n    return 1.0\n"
        + "bpy.app.timers.register(geometry_progress,persistent=True)\n"
        + "progress('started')\n"
        + "runpy.run_path(" + repr(source) + ",run_name='__main__')\nprogress('completed')\n")
    command[command.index('--python') + 1] = str(wrapper.resolve())
    task = GpuTask(log_path.with_suffix('.gpu.json'),
                   identity={'command': list(map(str, cmd)), 'cwd': str(cwd)})
    def execute(_fd):
        progress_path.unlink(missing_ok=True)
        stdout, stderr = log_path.with_suffix('.stdout'), log_path.with_suffix('.stderr')
        def progress():
            try:
                return json.loads(progress_path.read_text())
            except (OSError, ValueError):
                return None
        with stdout.open('w') as out, stderr.open('w') as err:
            process = subprocess.Popen(command, cwd=cwd, stdout=out, stderr=err,
                                       env=child_env, start_new_session=True)
            task.bind_process(process)
            execution_error = None
            try:
                returncode = wait_process(process, progress=progress)
            except TechnicalExecutionError as error:
                execution_error = error
                returncode = process.poll()
            finally:
                result = subprocess.CompletedProcess(cmd, process.poll(), stdout.read_text(), stderr.read_text())
                api.save_log(result, log_path)
        if returncode and oom_evidence(text=result.stdout + result.stderr):
            archive = log_path.with_suffix('') / 'gpu_attempts'
            archive.mkdir(parents=True, exist_ok=True)
            api.save_log(result, archive / f'{task.state["oom_count"]:03d}.log')
            raise GpuOutOfMemory('Blender GPU OOM; retry the same script')
        if execution_error is not None:
            raise execution_error
        if returncode < 0:
            raise TechnicalExecutionError(f'Blender exited with signal {-returncode}; inspect {log_path}')
        return result
    return task.run(execute)


def validate_geometry_review(value):
    if not isinstance(value, dict) or value.get('verdict') not in {'pass', 'fail', 'uncertain'}:
        raise ValueError('Geometry review needs pass/fail/uncertain')
    score, issues = value.get('score'), value.get('issues')
    if type(score) not in (int, float) or not __import__('math').isfinite(score) or not 0 <= score <= 10:
        raise ValueError('Geometry score must be finite in 0..10')
    if not isinstance(issues, list) or any(not isinstance(item, dict) or
            not all(isinstance(item.get(k), str) and item[k].strip() for k in ('description', 'correction')) for item in issues):
        raise ValueError('Geometry issues need concrete descriptions and corrections')
    return {**value, 'model_verdict': value['verdict'],
            'verdict': 'pass' if score >= 8 else 'fail',
            'acceptance_policy': 'geometry_score_threshold.v1', 'pass_score': 8}


def generate_geometry(args, api, ask, images):
    """Generate initial alternatives once, then refine only the chosen geometry."""
    from concurrent.futures import ThreadPoolExecutor
    from .run_state import write_json
    from .target_reference import modeling_reference
    reference, images = modeling_reference(args, api, images)
    root = args.out_dir
    stage = root / '1_model'
    stage.mkdir(exist_ok=True)
    prompt = (args.prompts_dir / '1_model.md').read_text()
    debug = (args.prompts_dir / 'debug.md').read_text()
    review_prompt = (args.prompts_dir / '1c_model_critic.md').read_text()
    if args.extra_model_instruction.strip():
        prefix = 'Additional instruction:\n' + args.extra_model_instruction.strip() + '\n\n'
        prompt, debug = prefix + prompt, prefix + debug
    research = GenerationResearch(root, 'geometry', baseline={
        'source_images': sorted(file_hash(Path(p)) for p in args.image_paths),
        'model': args.model, 'candidates': args.num_model_candidates,
        'extra_instruction': args.extra_model_instruction, 'target_reference': reference},
        implementation=fingerprint({'modeling': file_hash(Path(__file__)),
            'winding': file_hash(Path(__file__).with_name('blender_mesh_orientation.py')),
            'research': file_hash(Path(__file__).with_name('generation_research.py')),
            'prompt': prompt, 'debug': debug, 'review_prompt': review_prompt}),
        limit=getattr(args, 'max_generation_repairs', 3))
    blender_env = {key: value for key, value in os.environ.items()
                   if key not in {'OPENAI_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY'}}

    def produce_script(token, content, supplied=None):
        work = stage / token
        work.mkdir(exist_ok=True)
        script = work / 'model.py'
        def produce():
            raw = supplied if supplied is not None else ask(token, content)
            script.write_text(api.extract_code(raw))
            return str(script.relative_to(root)), [script]
        return root / research.phase(token, 'proposal', produce)

    def execute(token, script):
        work = script.parent
        rendered, blend = work / 'Res1_3D.png', work / 'geometry.blend'
        preparation, runner, log = work / 'geometry_preparation.json', work / 'run_model.py', work / 'runtime.log'
        def run():
            for stale in (rendered, blend, preparation):
                stale.unlink(missing_ok=True)
            runner.write_text('import os\nfrom urdf_core.blender_mesh_orientation import run_geometry_script\n'
                + f'os.chdir({str(work)!r})\n'
                + f'run_geometry_script({str(script)!r}, {str(rendered)!r}, {str(blend)!r}, {str(preparation)!r})\n')
            cmd = [args.blender, '--background', '--factory-startup', '--disable-autoexec', '--threads', '4',
                   '--python-exit-code', '1', '--python', str(runner)]
            if not script.read_text().strip():
                result = subprocess.CompletedProcess(cmd, 1, 'Empty modeling script; content repair required', '')
                log.write_text(result.stdout)
            else:
                result = run_blender(cmd, cwd=work, log_path=log, api=api, env=blender_env)
            error = api.combined_log(result)
            passed = result.returncode == 0 and all(p.is_file() and p.stat().st_size for p in (rendered, blend))
            if passed:
                try:
                    receipt = json.loads(preparation.read_text())
                    passed = (receipt.get('status') == 'passed'
                        and receipt.get('render_sha256') == file_hash(rendered)
                        and receipt.get('baseline_sha256') == file_hash(blend)
                        and receipt.get('source_script_sha256') == file_hash(script)
                        and receipt.get('implementation_sha256') == file_hash(Path(__file__).with_name('blender_mesh_orientation.py')))
                except (ValueError, FileNotFoundError):
                    passed = False
            value = {'technical_passed': bool(passed), 'error': error or 'Expected render/blend/preparation receipt missing or invalid',
                'script': str(script.relative_to(root)), 'render': str(rendered.relative_to(root)),
                'blend': str(blend.relative_to(root)), 'preparation': str(preparation.relative_to(root)),
                'runner': str(runner.relative_to(root)), 'critic': None}
            return value, [script, runner, log, *[p for p in (rendered, blend, preparation) if p.is_file()]]
        return research.phase(token, 'execution', run)

    def initial(candidate):
        token = f'model_candidate_{candidate}'
        script = produce_script(token, [*images, api.make_text_input(prompt)])
        return dict(execute(token, script), candidate=candidate)

    # Initial candidates get execution evidence, never individual repair loops.
    with ThreadPoolExecutor(max_workers=3) as pool:
        initial_candidates = list(pool.map(initial, range(1, args.num_model_candidates + 1)))
    write_json(stage / 'initial_candidates.json', initial_candidates)
    available = [v for v in initial_candidates if v['technical_passed']] or initial_candidates
    selection_path = stage / 'initial_selection.json'
    def select():
        if len(available) == 1:
            value = {'chosen_candidate': available[0]['candidate'], 'reasoning': 'Only available initial candidate'}
        else:
            content = list(images)
            for item in available:
                content.append(api.make_text_input(f"Candidate {item['candidate']}; execution_passed={item['technical_passed']}"))
                if item['technical_passed']:
                    content.append(api.make_image_input(root / item['render']))
                else:
                    content.append(api.make_text_input('Code:\n' + (root / item['script']).read_text() + '\nExecution error:\n' + item['error'][-12000:]))
            selection_prompt = (args.prompts_dir / '1b_model_selection.md').read_text().format(
                num_candidates=len(available), valid_candidate_ids=', '.join(str(v['candidate']) for v in available))
            content.append(api.make_text_input(selection_prompt))
            raw = ask('model_selection', content)
            try:
                value = api.extract_json(raw)
                if not isinstance(value, dict) or type(value.get('chosen_candidate')) is not int or value['chosen_candidate'] not in {v['candidate'] for v in available}:
                    raise ValueError('Selection must identify an available initial candidate')
            except (ValueError, KeyError, TypeError) as error:
                value = {'chosen_candidate': available[0]['candidate'], 'selection_status': 'unreviewed_fallback',
                         'reasoning': 'Invalid selection response; deterministic initial candidate requires independent geometry acceptance',
                         'error': str(error)}
        write_json(selection_path, value)
        return value, [selection_path]
    selection = research.phase('initial_candidates', 'selection', select)
    chosen = next(v for v in initial_candidates if v['candidate'] == selection['chosen_candidate'])
    best = None
    while True:
        history = list(research.snapshot()['trials'].values())
        last = history[-1] if history else None
        for trial in history:
            item = trial.get('result')
            if item and item['technical_passed']:
                quality = (item.get('critic') or {}).get('score', -1)
                if best is None or (item.get('critic', {}) or {}).get('verdict') == 'pass' or quality > (best.get('critic') or {}).get('score', -1):
                    best = item
        if last and last['status'] == 'passed':
            break
        index = len(history) if last and last['status'] == 'pending' else len(history) + 1
        token = f'selected_geometry_{index}'
        try:
            trial = research.start(token, scope='selected', context=last.get('result') if last else chosen)
        except BudgetExhausted:
            break
        prior = trial['context']
        if index == 1:
            item = dict(chosen)
        else:
            content = [*images, api.make_text_input(debug),
                api.make_text_input('Current complete Blender script:\n' + (root / prior['script']).read_text()),
                api.make_text_input('Execution and independent geometry feedback:\n' + json.dumps(prior, ensure_ascii=False)),
                api.make_text_input('Return the entire corrected modeling script. Correct observed shape, components and proportions as well as execution errors.')]
            if prior['technical_passed']:
                content.append(api.make_image_input(root / prior['render']))
            script = produce_script(token, content)
            item = execute(token, script)
        if item['technical_passed']:
            review_path = stage / token / 'geometry_review.json'
            def review():
                raw = ask(f'{token}_critic', [*images, api.make_image_input(root / item['render']),
                    api.make_text_input(review_prompt)])
                try:
                    value = validate_geometry_review(api.extract_json(raw))
                except (ValueError, TypeError, KeyError) as error:
                    value = {'verdict': 'uncertain', 'score': -1, 'issues': [{'description': 'Invalid geometry review: ' + str(error),
                        'correction': 'Reassess the complete geometry against the original image; no acceptance was established'}],
                        'invalid_response': raw}
                write_json(review_path, value)
                return value, [review_path]
            item['critic'] = research.phase(token, 'review', review)
        passed = item['technical_passed'] and (item.get('critic') or {}).get('verdict') == 'pass'
        research.checked(token, passed=passed, reason='selected_geometry_acceptance',
            paths=[root / item['script']], result=item)
    if best is None:
        from .degraded_asset import proxy_script, proxy_description
        salvage = None
        for path in sorted(stage.glob('*/geometry_preparation.json')):
            receipt = json.loads(path.read_text())
            value = receipt.get('salvage')
            if value and file_hash(value['path']) == value['sha256']:
                salvage = value; break
        if salvage:
            supplied = ('import bpy\nfrom mathutils import Vector\n'
                + f"bpy.ops.wm.open_mainfile(filepath={salvage['path']!r}, use_scripts=False)\n"
                + "s=bpy.context.scene; s.render.engine='BLENDER_WORKBENCH'\n"
                + "s.display.shading.light='STUDIO'; s.display.shading.color_type='MATERIAL'\n"
                + "graph=bpy.context.evaluated_depsgraph_get()\n"
                + "pts=[o.evaluated_get(graph).matrix_world@Vector(c) for o in s.objects if o.type in {'MESH','CURVE','FONT'} and not o.hide_render for c in o.evaluated_get(graph).bound_box]\n"
                + "lo=Vector([min(p[k] for p in pts) for k in range(3)]); hi=Vector([max(p[k] for p in pts) for k in range(3)])\n"
                + "center=(lo+hi)/2; extent=max(hi-lo)\nbpy.ops.object.camera_add(location=center+Vector((2,-3,2))*extent)\n"
                + "cam=bpy.context.object; cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler(); s.camera=cam\n"
                + "s.render.resolution_x=512;s.render.resolution_y=512;s.render.resolution_percentage=100\n"
                + "s.render.image_settings.file_format='PNG';s.render.filepath='Res1_3D.png';bpy.ops.render.render(write_still=True)\n")
            best = execute('salvaged_geometry', produce_script('salvaged_geometry', [], supplied=supplied))
        if not best or not best['technical_passed']:
            best = execute('degraded_geometry', produce_script('degraded_geometry', [], supplied=proxy_script(args)))
            salvage = None
        if not best['technical_passed']:
            from .execution import TechnicalExecutionError
            raise TechnicalExecutionError('Deterministic proxy construction failed: ' + best['error'])
        best['critic'] = {'verdict': 'fail', 'score': -1, 'issues': [{'description': 'All generated models unavailable',
            'correction': 'Explicit primitive proxy delivered after geometry repair exhaustion'}]}
        write_json(root/'geometry_degradation.json', {'salvaged_geometry': salvage,
            'primitive_proxy': proxy_description(args) if salvage is None else None})
    passed = (best.get('critic') or {}).get('verdict') == 'pass'
    acceptance = {'passed': passed, 'verdict': 'pass' if passed else 'fail', 'critic': best.get('critic'),
        'selected_candidate': selection['chosen_candidate'], 'best_script': best['script'],
        'refinements_used': research.snapshot()['used_repairs'],
        'termination_reason': 'accepted' if passed else 'refinement_budget_exhausted'}
    write_json(root / 'geometry_acceptance.json', acceptance)
    selected = stage / 'Res1_3D_model_selected.py'
    baseline, canonical_render = root / 'geometry_baseline.blend', root / 'Res1_3D.png'
    preparation_path = root / 'geometry_preparation.json'
    for source, target in ((best['script'], selected), (best['blend'], baseline), (best['render'], canonical_render),
                           (best['preparation'], preparation_path)):
        shutil.copy2(root / source, target)
    selection['geometry_acceptance'] = acceptance
    selection['geometry_preparation'] = {'path': str(preparation_path.relative_to(root)), 'sha256': file_hash(preparation_path),
        'runner': best['runner'], 'runner_sha256': file_hash(root / best['runner']),
        'source_run': str(root.resolve()), 'runtime_snapshot': getattr(args, '_runtime_snapshot', None),
        'source_replay': 'Run the recorded runner in its frozen object runtime; raw model.py alone omits source winding preparation.'}
    final_selection = stage / 'model_selection.json'
    write_json(final_selection, selection)
    return [selected, baseline, canonical_render, final_selection, preparation_path, root / 'geometry_acceptance.json',
            *[p for p in research.path.parent.rglob('*') if p.is_file()]]
