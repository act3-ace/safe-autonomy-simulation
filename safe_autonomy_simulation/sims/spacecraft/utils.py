"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Simulation.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains utility functions and constants for spacecraft models.
"""

from typing import Tuple

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


def generate_cwh_matrices(
    m: float, n: float, mode: str = "2d"
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates A and B Matrices from Clohessy-Wiltshire linearized dynamics of dx/dt = Ax + Bu

    Parameters
    ----------
    m : float
        mass in kg of spacecraft
    n : float
        orbital mean motion in rad/s of current Hill's reference frame
    mode : str, optional
        dimensionality of dynamics matrices. '2d' or '3d', by default '2d'

    Returns
    -------
    np.ndarray
        A dynamics matrix
    np.ndarray
        B dynamics matrix
    """
    assert mode in ["2d", "3d"], "mode must be on of ['2d', '3d']"
    if mode == "2d":
        A = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [3 * n**2, 0, 0, 2 * n],
                [0, 0, -2 * n, 0],
            ],
            dtype=np.float64,
        )

        B = np.array(
            [
                [0, 0],
                [0, 0],
                [1 / m, 0],
                [0, 1 / m],
            ],
            dtype=np.float64,
        )
    else:
        A = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [3 * n**2, 0, 0, 0, 2 * n, 0],
                [0, 0, 0, -2 * n, 0, 0],
                [0, 0, -(n**2), 0, 0, 0],
            ],
            dtype=np.float64,
        )

        B = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1 / m, 0, 0],
                [0, 1 / m, 0],
                [0, 0, 1 / m],
            ],
            dtype=np.float64,
        )

    return A, B
