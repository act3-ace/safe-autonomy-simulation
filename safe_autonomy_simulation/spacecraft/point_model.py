"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Simulation.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a 3D point mass spacecraft with Clohessy-Wilshire physics dynamics in non-inertial orbital
Hill's reference frame.
"""

from typing import Union

import numpy as np

from safe_autonomy_simulation.spacecraft.utils import (
    generate_cwh_matrices,
    M_DEFAULT,
    N_DEFAULT,
    CWHMaterial,
)

from safe_autonomy_simulation.entity import PhysicalEntity
from safe_autonomy_simulation.dynamics import LinearODESolverDynamics


class CWHSpacecraft(PhysicalEntity):
    """
    3D point mass spacecraft with +/- xyz thrusters and Clohessy-Wiltshire dynamics in Hill's reference frame.

    States
        x
        y
        z
        x_dot
        y_dot
        z_dot

    Controls
        thrust_x
            range = [-1, 1] Newtons
        thrust_y
            range = [-1, 1] Newtons
        thrust_z
            range = [-1, 1] Newtons

    Parameters
    ----------
    position: np.ndarray, optional
        Initial position of spacecraft in meters, by default np.zeros(3)
    velocity: np.ndarray, optional
        Initial velocity of spacecraft in meters/second, by default np.zeros(3)
    m: float, optional
        Mass of spacecraft in kilograms, by default 12.
    n: float, optional
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027.
    trajectory_samples : int, optional
        number of trajectory samples the generate and store on steps, by default 0
    integration_method: str, optional
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics. By default "RK45"
    material: Material, optional
        Material properties of the spacecraft, by default CWHMaterial()
    parent: Union[PhysicalEntity, None], optional
        Parent entity of spacecraft, by default None
    children: set[PhysicalEntity], optional
        Set of children entities of spacecraft, by default {}
    """

    def __init__(
        self,
        name: str,
        position: np.ndarray = np.zeros(3),
        velocity: np.ndarray = np.zeros(3),
        m=M_DEFAULT,
        n=N_DEFAULT,
        trajectory_samples=0,
        integration_method="RK45",
        material: CWHMaterial = CWHMaterial(),
        parent: Union[PhysicalEntity, None] = None,
        children: set[PhysicalEntity] = {},
    ):
        dynamics = CWHDynamics(
            m=m,
            n=n,
            trajectory_samples=trajectory_samples,
            integration_method=integration_method,
        )

        control_map = {
            "thrust_x": 0,
            "thrust_y": 1,
            "thrust_z": 2,
        }

        super().__init__(
            name=name,
            dynamics=dynamics,
            position=position,
            velocity=velocity,
            orientation=np.ndarray([0, 0, 0, 1]),  # no rotation
            angular_velocity=np.zeros(3),  # no rotation
            control_default=np.zeros((3,)),
            control_min=-1,
            control_max=1,
            control_map=control_map,
            material=material,
            parent=parent,
            children=children,
        )

    @property
    def state(self) -> np.ndarray:
        """State vector of spacecraft

        Returns
        -------
        np.ndarray
            state vector of form [x, y, z, x_dot, y_dot, z_dot]
        """
        return np.concatenate([self.position, self.velocity])

    @state.setter
    def state(self, state: np.ndarray):
        """Set state of spacecraft

        Parameters
        ----------
        state : np.ndarray
            state vector of form [x, y, z, x_dot, y_dot, z_dot]
        """
        assert (
            state.shape == self.state.shape
        ), f"State shape must match {self.state.shape}, got {state.shape}"
        # Internal state is [x, y, z, x_dot, y_dot, z_dot, q0, q1, q2, q3, w_x, w_y, w_z]
        self._state[0:3] = state[0:3]
        self._state[3:6] = state[3:6]


class CWHDynamics(LinearODESolverDynamics):
    """
    State transition implementation of 3D Clohessy-Wiltshire dynamics model.

    Parameters
    ----------
    m: float, optional
        Mass of spacecraft in kilograms, by default 12
    n: float, optional
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027
    trajectory_samples : int, optional
        number of trajectory samples the generate and store on steps, by default 0
    state_min : float or np.ndarray, optional
        Minimum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, -np.inf
    state_max : float or np.ndarray, optional
        Maximum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, np.inf
    state_dot_min : float or np.ndarray, optional
        Minimum allowable value for the state time derivative. State derivative values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, -inf
    state_dot_max : float or np.ndarray, optional
        Maximum allowable value for the state time derivative. State derivative values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, +inf
    angle_wrap_centers: np.ndarray, optional
        Enables circular wrapping of angles. Defines the center of circular wrap such that angles are within
        [center+pi, center-pi].
        When None, no angle wrapping applied.
        When ndarray, each element defines the angle wrap center of the corresponding state element.
        Wrapping not applied when element is NaN.
        By default, None.
    integration_method : string, optional
        Numerical integration method used by dynamics solver. One of ['RK45', 'RK45_JAX', 'Euler'].
        'RK45' is slow but very accurate.
        'RK45_JAX' is very accurate, and fast when JIT compiled but otherwise very slow. 'use_jax' must be set to True.
        'Euler' is fast but very inaccurate.
        By default, 'RK45'.
    use_jax : bool, optional
        True if using jax version of numpy/scipy. By default, False
    """

    def __init__(
        self,
        m=M_DEFAULT,
        n=N_DEFAULT,
        trajectory_samples: int = 0,
        state_min: Union[float, np.ndarray] = -np.inf,
        state_max: Union[float, np.ndarray] = np.inf,
        state_dot_min: Union[float, np.ndarray] = -np.inf,
        state_dot_max: Union[float, np.ndarray] = np.inf,
        angle_wrap_centers: Union[np.ndarray, None] = None,
        integration_method: str = "RK45",
        use_jax: bool = False,
    ):
        self.m = m  # kg
        self.n = n  # rads/s

        A, B = generate_cwh_matrices(self.m, self.n, "3d")

        super().__init__(
            A=A,
            B=B,
            trajectory_samples=trajectory_samples,
            state_min=state_min,
            state_max=state_max,
            state_dot_min=state_dot_min,
            state_dot_max=state_dot_max,
            angle_wrap_centers=angle_wrap_centers,
            integration_method=integration_method,
            use_jax=use_jax,
        )



