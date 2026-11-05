"""Optional control evidence; never changes effort, parameters or acceptance."""
from __future__ import annotations
import math
import numpy as np


def discrete_control_evidence(mass_matrix, joint_names, joints, dt):
    """Linearize unsaturated explicit PD about the measured configuration.

    This idealized semi-implicit Euler model omits contacts, limits, friction,
    Coriolis forces and configuration changes. It is not the PhysX solver and
    its eigenvalues are neither proof of a physical cause nor an acceptance test.
    For a floating base the native first six coordinates are unactuated; their
    inertial coupling is eliminated with the joint block of the inverse matrix.
    """
    names = list(joint_names)
    if len(names) != len(set(names)) or not names:
        raise ValueError('Need unique native joint names')
    dt = float(dt)
    matrix = np.asarray(mass_matrix, dtype=float)
    n = len(names)
    if (not math.isfinite(dt) or dt <= 0 or matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] not in (n, n+6)
            or not np.isfinite(matrix).all()):
        raise ValueError('Invalid generalized mass matrix or time step')
    scale = max(float(np.linalg.norm(matrix)), np.finfo(float).tiny)
    if np.linalg.norm(matrix-matrix.T) > 1e-5*scale:
        raise ValueError('Generalized mass matrix is not symmetric')
    native_matrix = matrix.tolist()
    matrix = .5*(matrix+matrix.T)
    np.linalg.cholesky(matrix)
    mobility = np.linalg.solve(matrix, np.eye(len(matrix)))[-n:, -n:]
    effective = np.linalg.solve(mobility, np.eye(n))
    kp = np.array([joints[name]['kp'] for name in names], dtype=float)
    kd = np.array([joints[name]['kd'] for name in names], dtype=float)
    if not np.isfinite(kp).all() or not np.isfinite(kd).all() or np.any(kp <= 0) or np.any(kd < 0):
        raise ValueError('Invalid controller gains')
    stiffness_step = dt*dt * mobility @ np.diag(kp)
    damping_step = dt * mobility @ np.diag(kd)
    eye = np.eye(n)
    transition = np.block([[eye-stiffness_step, dt*(eye-damping_step)],
                           [-stiffness_step/dt, eye-damping_step]])
    poles = np.linalg.eigvals(transition)
    if not np.isfinite(poles).all():
        raise ValueError('Nonfinite diagnostic eigenvalues')
    return {'schema_version': 'unconstrained_discrete_pd_evidence.v1',
            'available': True, 'acceptance_gate': False, 'parameters_modified': False,
            'mass_source': 'native generalized mass matrix at trial initialization',
            'joint_names_native_order': names, 'native_matrix_shape': list(matrix.shape),
            'native_generalized_mass_matrix': native_matrix,
            'base_model': 'fixed' if len(matrix)==n else 'unactuated_floating_without_contact',
            'physics_dt_s': dt, 'effective_joint_mass_matrix': effective.tolist(),
            'dt_squared_inverse_mass_kp': stiffness_step.tolist(),
            'dt_inverse_mass_kd': damping_step.tolist(),
            'kp': kp.tolist(), 'kd': kd.tolist(),
            'linearized_poles_real_imag': [[float(z.real),float(z.imag)] for z in poles],
            'spectral_radius': float(np.max(np.abs(poles))),
            'model': 'qdot_next=qdot+dt*M_eff^-1*(-Kp*q-Kd*qdot); q_next=q+dt*qdot_next',
            'assumptions': ['constant measured inertia', 'unsaturated explicit PD',
                            'no contacts, joint limits, passive friction, gravity or Coriolis terms'],
            'interpretation': 'Approximate discrete control model, not a native stability measurement or a root-cause verdict. Compare against measured q, qdot, effort and contact history.'}


def observe_control_evidence(articulation, names, contract):
    if not names:
        return {'available': False, 'reason': 'no_active_joint', 'acceptance_gate': False}
    try:
        matrices = np.asarray(articulation.get_mass_matrices(), dtype=float)
        if matrices.ndim != 3 or matrices.shape[0] != 1:
            raise ValueError('Native generalized mass matrix observation unavailable')
        return discrete_control_evidence(matrices[0], names, contract['joints'], contract['config']['physics_dt_s'])
    except Exception as error:
        # This optional context must not invent a successful diagnostic or turn
        # an otherwise complete native physics test into an acceptance failure.
        return {'available': False, 'reason': type(error).__name__, 'acceptance_gate': False}
