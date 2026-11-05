"""Final assessment visits every independent trial without changing thresholds."""
from .protocol import free_base_required
from .trial_checkpoint import for_runtime, observe
from .verify_physics_isaac import write_report

POLICY = 'all_independent_trials.v1'


def evaluate(runtime, report, contract, path):
    from .representative_motion import runtime_selection
    selected = runtime_selection(runtime, report, contract)
    cache=for_runtime(runtime);stable=contract['stable_required'];free=free_base_required(contract)
    report.update(trial_evaluation_policy=POLICY,evaluation_complete=False,acceptance_complete=False,
        blocked_trials=[],stopped_after_phase=None,acceptance_policy=contract['acceptance_policy'],
        support_rest_pose={'status':'pending' if stable else 'not_applicable','verified':False})
    if contract.get('motion_scope_policy'):
        report.update(motion_scope_policy=contract['motion_scope_policy'],inapplicable_ground_operations=[])
    if contract.get('free_base_policy') is not None:report['free_base_policy']=contract['free_base_policy']
    support=None;ground=None
    if not stable and (cache is None or not cache.restore_delivery(runtime)):
        runtime.create_trial(fixed=False)
        if cache:cache.save_delivery(runtime)
    if stable:
        ground,trace=observe(runtime,cache,'ground_settle',None)
        report['trials'].append(ground)
        if ground['passed']:
            support=runtime.make_rest_candidate(ground,trace)
        else:report['support_rest_pose']={'status':'ground_failed','verified':False}
        write_report(path,report)
    for kind in ('fixed_joint','free_joint'):
        if kind=='free_joint' and not free:continue
        if kind=='free_joint' and stable and support is None:
            report['blocked_trials']=[{'kind':kind,'joint_name':name,'status':'blocked_by_prerequisite',
                'blocked_by_phase':'ground_settle','failed_trial_ids':[ground['id']]} for name in selected[kind]]
            continue
        if kind=='free_joint' and support is not None:report['free_motion_initialization_policy']='measured_stable_rest.v1'
        for name in selected[kind]:
            kwargs={'support_pose':support} if kind=='free_joint' and support is not None else {}
            trial,_=observe(runtime,cache,kind,name,reload=True,**kwargs)
            if trial.get('status')=='not_applicable_ground_blocked':
                report['inapplicable_ground_operations'].append(trial)
            else:report['trials'].append(trial)
            write_report(path,report)
    kwargs={'support_pose':support} if support is not None else {}
    trial,_=observe(runtime,cache,'reload' if stable else 'reload_supported',reload=True,**kwargs)
    report['reload_check']={'passed':trial['passed'],'trial':trial,'free_root':True,'skipped_ground':not stable,
        'joint_motion_trial_ids':[t['id'] for t in report['trials'] if t['kind']=='free_joint']}
    if support is not None:
        report['support_rest_pose'].update(verified=trial['passed'],status='verified' if trial['passed'] else 'rejected')
    numerical=all(t['passed'] for t in report['trials']) and not report['blocked_trials']
    passed=numerical and trial['passed']
    report.update(numerical_passed=numerical,passed=passed,status='passed' if passed else 'physics_failed',
        evaluation_complete=True,acceptance_complete=not report['blocked_trials'],stage='complete')
    write_report(path,report)


def validate_coverage(report, expected):
    trials=report['trials'];actual={(t['kind'],t.get('joint_name')) for t in trials}
    blocked=report.get('blocked_trials',[])
    ground=[t for t in trials if t['kind']=='ground_settle' and t.get('passed') is False]
    missing=expected-actual
    if blocked:
        if len(ground)!=1:raise ValueError('Blocked free motion requires failed ground evidence')
        required=[{'kind':kind,'joint_name':name,'status':'blocked_by_prerequisite',
            'blocked_by_phase':'ground_settle','failed_trial_ids':[ground[0]['id']]} for kind,name in sorted(missing)]
        if any(kind!='free_joint' for kind,_ in missing) or sorted(blocked,key=lambda x:x['joint_name'])!=required:
            raise ValueError('Incomplete independent-trial coverage')
    if (len(actual)!=len(trials) or not actual<=expected or (missing and not blocked)
            or report.get('acceptance_complete') is not (not blocked) or report.get('stopped_after_phase') is not None):
        raise ValueError('Final evaluation omitted or duplicated a required independent trial')
    return False
