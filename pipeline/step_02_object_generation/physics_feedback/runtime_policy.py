"""Shared PhysX solver and native initialization tolerances."""
import math

SCENE_SOLVER_TYPE = "PGS"
EXTERNAL_FORCES_EVERY_ITERATION = False


def scene_solver_policy():
    return {"solver_type": SCENE_SOLVER_TYPE,
            "enable_external_forces_every_iteration": EXTERNAL_FORCES_EVERY_ITERATION}


def configure_scene_solver(physx):
    policy = scene_solver_policy()
    physx.CreateSolverTypeAttr(policy['solver_type'])
    physx.CreateEnableExternalForcesEveryIterationAttr(policy['enable_external_forces_every_iteration'])
    if (physx.GetSolverTypeAttr().Get() != policy['solver_type']
            or physx.GetEnableExternalForcesEveryIterationAttr().Get() is not policy['enable_external_forces_every_iteration']):
        raise RuntimeError('Native scene attributes differ from the supported solver policy')
    return policy


def initialization_tolerances(diagonal_m):
    diagonal_m = float(diagonal_m)
    if not math.isfinite(diagonal_m) or diagonal_m <= 0:
        raise ValueError('Initialization requires a finite positive object diagonal')
    return {'position_tolerance_m': max(1e-6, diagonal_m * 1e-6),
            'orientation_tolerance_rad': 1e-5}


def initialization_pose_error(reference, measured, diagonal_m):
    import numpy as np
    from scipy.spatial.transform import Rotation
    reference, measured = np.asarray(reference, dtype=float), np.asarray(measured, dtype=float)
    if (reference.shape != (4, 4) or measured.shape != (4, 4)
            or not np.isfinite(reference).all() or not np.isfinite(measured).all()):
        raise ValueError('Invalid initialization link pose')
    limits = initialization_tolerances(diagonal_m)
    position = float(np.linalg.norm(reference[:3, 3] - measured[:3, 3]))
    angle = float(Rotation.from_matrix(reference[:3, :3].T @ measured[:3, :3]).magnitude())
    return {**limits, 'position_error_m': position, 'orientation_error_rad': angle,
            'passed': position <= limits['position_tolerance_m'] and angle <= limits['orientation_tolerance_rad']}


def validate_native_properties(name, expected, mass, com, inertia):
    import numpy as np
    mass, com, inertia = float(mass), np.asarray(com, float), np.asarray(inertia, float)
    if com.shape != (3,) or inertia.shape != (3, 3) or not np.isfinite([mass, *com, *inertia.ravel()]).all():
        raise RuntimeError('Missing/nonfinite native mass properties: ' + name)
    if not np.isclose(mass, expected["mass_kg"], rtol=1e-4, atol=1e-7):
        raise RuntimeError(f"PhysX changed the declared link mass: {name}")
    if "center_of_mass_m" in expected and not np.allclose(com, expected["center_of_mass_m"], rtol=1e-4, atol=1e-6):
        raise RuntimeError(f"PhysX changed the declared center of mass: {name}")
    inertia_error = None
    if "inertia_kg_m2" in expected:
        expected_inertia = np.asarray(expected["inertia_kg_m2"])
        inertia_error = float(np.linalg.norm(inertia - expected_inertia))
        # Principal-axis quaternion roundoff can dominate a nearly
        # zero off-diagonal element. Compare tensor error to the whole
        # tensor scale, not elementwise relative errors at zero.
        allowed_error = max(1e-9, float(np.linalg.norm(expected_inertia)) * 2e-5)
        if inertia_error > allowed_error:
            raise RuntimeError(f"PhysX changed inertia {name}: expected={expected_inertia.tolist()}, "
                               f"actual={inertia.tolist()}, error={inertia_error}, allowed={allowed_error}")
    return {"mass_kg": float(mass), "center_of_mass_m": com.tolist(),
                                 "inertia_kg_m2": inertia.tolist(), "inertia_tensor_error_kg_m2": inertia_error}
