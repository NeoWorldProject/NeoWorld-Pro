"""Isolated native processes share completed trials, never simulation state.

The ground/fixed/free barriers are strict. Fragments cannot be accepted as an
object result: the ordinary evaluator reloads and validates their checkpoints,
performs delivery reload, and produces the normal complete evidence report.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import threading
import time

from urdf_core.run_state import write_json
from .trial_checkpoint import for_runtime, observe, digest

POLICY='isolated_process_joint_trials.v1'


def selection(report, phase, names):
    value={'policy':POLICY,'inputs':report['inputs'],'source_hashes':report['source_hashes'],
           'phase':phase,'joint_names':list(names)}
    return {**value,'sha256':digest(value)}


def evaluate_partition(runtime,report,contract,path):
    from .representative_motion import runtime_selection
    selected = runtime_selection(runtime, report, contract)
    request=json.loads(Path(runtime.args.trial_selection).read_text())
    if request!=selection(report,request['phase'],request['joint_names']):
        raise ValueError('Parallel trial selection binding changed')
    phase=request['phase'];names=request['joint_names']
    if (phase not in ('ground_settle','fixed_joint','free_joint') or len(names)!=len(set(names))
            or any(n not in contract['joints'] for n in names) or (phase=='ground_settle' and names)):
        raise ValueError('Invalid parallel trial selection')
    if phase != 'ground_settle' and not set(names) <= set(selected[phase]):
        raise ValueError('Partition includes joints not applicable to the frozen selection')
    cache=for_runtime(runtime)
    if cache is None:raise ValueError('Isolated trials require completed-trial checkpoints')
    complete=getattr(runtime.args,'evaluation_policy','first-failure')=='complete'
    stable=contract['stable_required'];support_pose=None
    if phase!='ground_settle':
        if stable and not (complete and phase=='fixed_joint'):
            saved=cache.load('ground_settle',None,runtime)
            if saved is None or saved[0]['passed'] is not True:
                raise ValueError('Parallel motion requires a measured passing ground trial')
            if contract.get('schema_version')==3:
                support_pose=runtime.make_rest_candidate(*saved)
        elif not cache.restore_delivery(runtime):
            raise ValueError('Supported candidate export is missing')
    if phase=='free_joint':
        from .protocol import free_base_required
        if not free_base_required(contract):raise ValueError('Free-base tests are not applicable')
        for name in ([] if complete else selected['fixed_joint']):
            saved=cache.peek('fixed_joint',name)
            if saved is None or saved['trial']['passed'] is not True:
                raise ValueError('All fixed-base joints must pass before free-base dispatch')
    if phase=='ground_settle' and not stable:
        if not cache.restore_delivery(runtime):
            runtime.create_trial(fixed=False);cache.save_delivery(runtime)
    else:
        for name in ([None] if phase=='ground_settle' else names):
            kwargs={'support_pose':support_pose} if phase=='free_joint' and support_pose is not None else {}
            trial,_=observe(runtime,cache,phase,name,reload=phase!='ground_settle',**kwargs)
            report['trials'].append(trial)
            from .verify_physics_isaac import write_report
            write_report(path,report)
            if trial.get('passed') is False and not complete:break
    report.update(status='trial_fragment',passed=False,numerical_passed=False,
                  evaluation_complete=False,acceptance_complete=False,stage='fragment_complete',
                  trial_partition=request,
                  fragment_passed=all(t.get('passed') is True or t.get('status')=='not_applicable_ground_blocked'
                                      for t in report['trials']))


def partition_joints(contract,workers,order=None,required=None):
    """Balance complete trajectories for the frozen selected joint set."""
    from .control import target_schedule, scoped_motion_joint
    required=set(contract['joints']) if required is None else set(required)
    names=[n for n in contract['joints'] if n in required] if order is None else list(order)
    if not required <= set(contract['joints']) or len(names)!=len(set(names)) or set(names)!=required:
        raise ValueError('Parallel plan must cover every joint in the frozen selection exactly once')
    costs={n:target_schedule(scoped_motion_joint(contract,n),contract['config'],contract=contract)[1] for n in names}
    lanes=[[] for _ in range(min(workers,len(names)))];loads=[0.]*len(lanes)
    for name in sorted(names,key=lambda n:-costs[n]):
        index=min(range(len(lanes)),key=lambda i:loads[i])
        lanes[index].append(name);loads[index]+=costs[name]
    return [sorted(lane,key=names.index) for lane in lanes]


class ParallelPhysicsSession:
    def __init__(self,root,*,workers=2,review_handler=None):
        if type(workers) is not int or not 2<=workers<=4:raise ValueError('Use 2..4 isolated physics workers')
        self.root=Path(root);self.workers=workers;self.review_handler=review_handler
        self.sessions=[];self.review_lock=threading.Lock()

    def __enter__(self):return self

    def _review(self,path):
        if self.review_handler is None:raise RuntimeError('Incidental-motion review requires its API owner')
        with self.review_lock:self.review_handler(path)

    def _start(self):
        if self.sessions:return
        from .resident_worker import ResidentPhysicsSession
        self.sessions=[ResidentPhysicsSession(self.root/f'lane_{i}',review_handler=self._review)
                       for i in range(self.workers)]

    def run(self,args):
        if getattr(args,'capture_mode','deferred')!='deferred':
            raise ValueError('Parallel numerical trials require deferred capture')
        from .verify_physics_isaac import initial_report
        from .protocol import free_base_required
        self._start();started=time.monotonic()
        root=Path(args.out_dir);root.mkdir(parents=True,exist_ok=True)
        common=copy.copy(args);common.trial_cache_root=getattr(args,'trial_cache_root',None) or root
        if getattr(args,'optimization_short_circuit',False) or getattr(args,'evaluation_policy',None)=='grouping':
            return self.sessions[0].run(common)
        contract=json.loads(Path(args.contract).read_text())
        audit={'policy':POLICY,'workers':self.workers,'isolation':'independent_process_and_stage',
               'all_samples_retained':True,'phases':[],'complete':False}
        def fragment(index,phase,names):
            child=copy.copy(common);child.out_dir=root/'lanes'/f'lane_{index}'/phase
            child.out_dir.mkdir(parents=True,exist_ok=True)
            child.trial_selection=child.out_dir/'selection.json'
            write_json(child.trial_selection,selection(initial_report(child,'partition'),phase,names))
            path=self.sessions[index].run(child)
            report=json.loads(path.read_text())
            return {'lane':index,'phase':phase,'joint_names':names,'report':str(path),
                    'passed':report['fragment_passed'],'wall_seconds':report['elapsed_seconds']}
        try:
            ground=fragment(0,'ground_settle',[]);audit['phases'].append([ground])
            passed=ground['passed'];ground_passed=passed
            complete=getattr(args,'evaluation_policy','first-failure')=='complete'
            orders={k:list(contract['joints']) for k in ('fixed_joint','free_joint')}
            if getattr(args,'trial_order',None):
                from .trial_order import verify_plan
                orders=verify_plan(json.loads(Path(args.trial_order).read_text()),contract)
            from .representative_motion import selected_joints
            selected=selected_joints(contract,getattr(args,'representative_motion',None))
            orders={phase:[name for name in order if name in selected[phase]] for phase,order in orders.items()}
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for phase in ('fixed_joint','free_joint'):
                    if not complete and not passed:break
                    if phase=='free_joint' and (not free_base_required(contract) or (contract['stable_required'] and not ground_passed)):continue
                    lanes=partition_joints(contract,self.workers,orders[phase],required=selected[phase])
                    futures=[pool.submit(fragment,i,phase,names) for i,names in enumerate(lanes)]
                    results=[future.result() for future in futures]
                    audit['phases'].append(results);passed=all(r['passed'] for r in results)
                    write_json(root/'parallel_execution.json',audit)
            # Ordinary finalizer consumes verified checkpoints and performs reload/media.
            common.trial_selection=None
            path=self.sessions[0].run(common)
            audit.update(complete=True,wall_seconds=time.monotonic()-started)
            write_json(root/'parallel_execution.json',audit)
            report=json.loads(path.read_text());report['parallel_execution']=audit
            report['elapsed_seconds']=audit['wall_seconds'];write_json(path,report)
            return path
        except BaseException:
            audit.update(wall_seconds=time.monotonic()-started)
            write_json(root/'parallel_execution.json',audit)
            self.close(force=True)
            raise

    def close(self,*,force=False):
        for session in self.sessions:session.close(force=force)

    def __exit__(self,*exc):self.close(force=exc[0] is not None)
