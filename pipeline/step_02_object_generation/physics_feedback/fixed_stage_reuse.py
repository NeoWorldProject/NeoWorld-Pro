"""Keep a fixed fixture's USD stage; reconstruct PhysX before every trial."""

import numpy as np


def save(runtime, context, rest, scope_placement):
    stage = context['stage']
    prims = list(context['bodies'].values())
    top = stage.GetPrimAtPath(context['root_path'])
    prims.append(top)
    # The asset placement may be above its articulation root/fixture.
    for body in context['bodies'].values():
        parent = body.GetParent()
        while parent and not parent.IsPseudoRoot():
            prims.append(parent)
            parent = parent.GetParent()
    attrs = {}
    for prim in prims:
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if name.startswith(('xformOp', 'physics:velocity', 'physics:angularVelocity', 'state:')):
                attrs[str(attr.GetPath())] = attr.Get()
    return {'context': context, 'attrs': attrs, 'rest': rest,
            'scope_placement': scope_placement.copy(),
            'geometry_binding': runtime.motion_scope_geometry_binding,
            'fixture_height': float(runtime.initial_root[2]),
            'original_clearance_lift': max(0., runtime.fixture_clearance_m - runtime.motion_scope_plan['fixed_min_z_m'])
                if runtime.motion_scope_plan is not None else None}


def restore(runtime, joint_name):
    saved = getattr(runtime, '_fixed_stage', None)
    if saved is None or joint_name is None:
        return None
    import omni.usd
    context = saved['context']
    if omni.usd.get_context().get_stage() != context['stage']:
        return None
    if not runtime.contract.get('motion_scope_policy'):
        return None
    from .motion_scope import plan_joint_motion
    definition = runtime.contract['joints'][joint_name]
    placement = saved['scope_placement']
    plan = plan_joint_motion(runtime.parsed, runtime.cooked, runtime.initial_positions,
                             placement, joint_name, definition['lower'], definition['upper'])
    clearance = max(.005, .01 * runtime.contract['diagonal_m'])
    height = saved['fixture_height'] - saved['original_clearance_lift'] + max(0., clearance - plan['fixed_min_z_m'])
    # Different swept clearances require the full fixture construction path.
    if height != saved['fixture_height']:
        return None
    runtime.close_trial()
    for path, value in saved['attrs'].items():
        attr = context['stage'].GetAttributeAtPath(path)
        if not attr:
            raise RuntimeError('Cached fixed fixture lost an authored state attribute')
        if value is None:
            attr.Clear()
        else:
            attr.Set(value)
    runtime.motion_scope_plan = plan
    runtime.motion_scope_geometry_binding = saved['geometry_binding']
    runtime.fixture_clearance_m = clearance
    runtime.support_pose = None
    runtime.reset_positions = dict(runtime.initial_positions)
    runtime.initial_orientation = np.array([1., 0., 0., 0.])
    runtime.initial_root = context['placement'][:3, 3].copy()
    runtime.initial_root_transform = context['placement'].copy()
    runtime.initial_link_poses = {name: (context['placement'] @ pose).tolist()
                                  for name, pose in saved['rest'].items()}
    return context
