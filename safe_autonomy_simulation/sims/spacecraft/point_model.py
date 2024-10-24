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

import typing

import numpy as np

import safe_autonomy_simulation.sims.spacecraft.defaults as defaults

import safe_autonomy_simulation.entities as e
import safe_autonomy_simulation.dynamics as d
import safe_autonomy_simulation.materials as mat
import safe_autonomy_simulation.controls as c


class CWHSpacecraft(e.PhysicalEntity):
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
        Initial absolute position of spacecraft in meters, by default np.zeros(3)
    velocity: np.ndarray, optional
        Initial absolute velocity of spacecraft in meters/second, by default np.zeros(3)
    m: float, optional
        Mass of spacecraft in kilograms, by default 12.
    n: float, optional
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027.
    trajectory_samples : int, optional
        number of trajectory samples the generate and store on steps, by default 0
    integration_method: str, optional
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics. By default "RK45"
    material: Material, optional
        Material properties of the spacecraft, by default CWH_MATERIAL
    parent: Union[PhysicalEntity, None], optional
        Parent entity of spacecraft, by default None
    children: list[PhysicalEntity], optional
        List of children entities of spacecraft, by default []
    control_min: typing.Union[float, np.ndarray, None]
        specify a minimum value that control can be. numbers lower than this will be clipped. (default = -1)
    control_max: typing.Union[float, np.ndarray, None] = 1,
        specify a maximum value that control can be. numbers higher than this will be clipped. (default = 1)
    use_jax : bool, optional
        EXPERIMENTAL: Use JAX to accelerate state transition computation, by default False.
    """

    def __init__(
        self,
        name: str,
        position: np.ndarray = np.zeros(3),
        velocity: np.ndarray = np.zeros(3),
        m=defaults.M_DEFAULT,
        n=defaults.N_DEFAULT,
        trajectory_samples=0,
        integration_method="RK45",
        material: mat.Material = defaults.CWH_MATERIAL,
        parent: typing.Union[e.PhysicalEntity, None] = None,
        children: list[e.PhysicalEntity] = [],
        control_min: typing.Union[float, np.ndarray, None] = -1,
        control_max: typing.Union[float, np.ndarray, None] = 1,
        use_jax: bool = False,
    ):
        dynamics = CWHDynamics(
            m=m,
            n=n,
            trajectory_samples=trajectory_samples,
            integration_method=integration_method,
            use_jax=use_jax,
        )

        control_queue = c.ControlQueue(
            default_control=np.zeros(3),
            control_min=control_min,
            control_max=control_max,
        )

        super().__init__(
            name=name,
            dynamics=dynamics,
            position=position,
            velocity=velocity,
            orientation=np.array([0, 0, 0, 1]),  # no rotation
            angular_velocity=np.zeros(3),  # no rotation
            control_queue=control_queue,
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


class CWHDynamics(d.LinearODEDynamics):
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
    integration_method : string, optional
        Numerical integration method used by dynamics solver. One of ['RK45', 'Euler'].
        'RK45' is slow but very accurate. If jax is available, can be JIT compiled for speed.
        'Euler' is fast but very inaccurate.
        By default, 'RK45'.
    use_jax : bool, optional
        EXPERIMENTAL: Use JAX to accelerate state transition computation, by default False.
    """

    def __init__(
        self,
        m=defaults.M_DEFAULT,
        n=defaults.N_DEFAULT,
        trajectory_samples: int = 0,
        state_min: typing.Union[float, np.ndarray] = -np.inf,
        state_max: typing.Union[float, np.ndarray] = np.inf,
        state_dot_min: typing.Union[float, np.ndarray] = -np.inf,
        state_dot_max: typing.Union[float, np.ndarray] = np.inf,
        integration_method: str = "RK45",
        use_jax: bool = False
    ):
        self.m = m  # kg
        self.n = n  # rads/s

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

        super().__init__(
            A=A,
            B=B,
            trajectory_samples=trajectory_samples,
            state_min=state_min,
            state_max=state_max,
            state_dot_min=state_dot_min,
            state_dot_max=state_dot_max,
            integration_method=integration_method,
            use_jax=use_jax
        )
