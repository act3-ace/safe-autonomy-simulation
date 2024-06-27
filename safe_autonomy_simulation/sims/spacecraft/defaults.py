"""Default values for spacecraft simulation parameters."""

import numpy as np
import safe_autonomy_simulation.materials as mat


# CWH constants
M_DEFAULT = 12
N_DEFAULT = 0.001027
INERTIA_DEFAULT = 0.0573
INERTIA_MATRIX_DEFAULT = np.matrix(
    [
        [INERTIA_DEFAULT, 0.0, 0.0],
        [0.0, INERTIA_DEFAULT, 0.0],
        [0.0, 0.0, INERTIA_DEFAULT],
    ]
)
INERTIA_WHEEL_DEFAULT = 4.1e-5
ANG_ACC_LIMIT_DEFAULT = 0.017453
ANG_VEL_LIMIT_DEFAULT = 0.034907
ACC_LIMIT_WHEEL_DEFAULT = 181.3
VEL_LIMIT_WHEEL_DEFAULT = 576
THRUST_CONTROL_LIMIT_DEFAULT = 1.0


# Spacecraft material
CWH_MATERIAL = mat.Material(
    specular=np.array([1.0, 1.0, 1.0]),
    diffuse=np.array([0.7, 0, 0]),
    ambient=np.array([0.1, 0, 0]),
    shininess=100,
    reflection=0.5,
)
