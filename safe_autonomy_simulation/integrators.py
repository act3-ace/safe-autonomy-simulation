"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Dynamics.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements 1d, 2d, and 3d point mass integrators.
"""

from typing import Tuple

import numpy as np

from safe_autonomy_simulation.entity import PhysicalEntity
from safe_autonomy_simulation.dynamics import LinearODESolverDynamics

M_DEFAULT = 1
DAMPING_DEFAULT = 0


class PointMassIntegrator1d(PhysicalEntity):
    """
    1d point mass integrator simulation entity

    States
        x
        x_dot

    Controls
        thrust_x
            default range = [-1, 1]

    Parameters
    ----------
    name: str
        name of entity
    position: np.ndarray, optional
        initial position, by default np.array([0])
    velocity: np.ndarray, optional
        initial velocity, by default np.array([0])
    m: float, optional
        Mass of integrator, by default 1.
    trajectory_samples : int, optional
        number of trajectory samples the generate and store on steps, by default 0
    integration_method: str, optional
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics. By default "RK45".
    """

    def __init__(
        self,
        name: str,
        position: np.ndarray = np.array([0]),
        velocity: np.ndarray = np.array([0]),
        m=M_DEFAULT,
        damping=DAMPING_DEFAULT,
        trajectory_samples=0,
        integration_method="RK45",
    ):
        assert position.shape == (1,), f"Position must be 1D vector, got {position}"
        assert velocity.shape == (1,), f"Velocity must be 1D vector, got {velocity}"

        dynamics = PointMassIntegratorDynamics(
            m=m,
            damping=damping,
            mode="1d",
            trajectory_samples=trajectory_samples,
            integration_method=integration_method,
        )

        control_map = {
            "thrust_x": 0,
        }

        super().__init__(
            name=name,
            dynamics=dynamics,
            position=np.concatenate([position, np.array([0, 0])]),
            velocity=np.concatenate([velocity, np.array([0, 0])]),
            control_default=np.zeros((1,)),
            control_min=-1,
            control_max=1,
            control_map=control_map,
        )

    @property
    def state(self) -> np.ndarray:
        """1d point mass integrator state vector

        State vector is [x, x_dot]

        Returns
        -------
        np.ndarray
            state vector of form [x, x_dot]
        """
        return np.array([self.x, self.x_dot])

    @state.setter
    def state(self, state: np.ndarray):
        """Set state of 1d point mass integrator

        State vector is [x, x_dot]

        Parameters
        ----------
        state : np.ndarray
            state vector of form [x, x_dot]
        """
        assert (
            state.shape == self.state.shape
        ), f"State must be of shape {self.state.shape}, got {state.shape}"
        self._state[0] = state[0]
        self._state[3] = state[1]


class PointMassIntegrator2d(PhysicalEntity):
    """
    2d point mass integrator simulation entity

    States
        x
        y
        x_dot
        y_dot

    Controls
        thrust_x
            default range = [-1, 1]
        thrust_y
            default range = [-1, 1]

    Parameters
    ----------
    name: str
        name of entity
    position: np.ndarray, optional
        initial 2d position, by default np.array([0, 0])
    velocity: np.ndarray, optional
        initial 2d velocity, by default np.array([0, 0])
    m: float, optional
        Mass of integrator, by default 1.
    trajectory_samples : int, optional
        number of trajectory samples the generate and store on steps, by default 0
    integration_method: str, optional
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics. By default "RK45".
    """

    def __init__(
        self,
        name: str,
        position: np.ndarray = np.array([0, 0]),
        velocity: np.ndarray = np.array([0, 0]),
        m=M_DEFAULT,
        damping=DAMPING_DEFAULT,
        trajectory_samples=0,
        integration_method="RK45",
    ):
        assert position.shape == (2,), f"Position must be 2D vector, got {position}"
        assert velocity.shape == (2,), f"Velocity must be 2D vector, got {velocity}"

        dynamics = PointMassIntegratorDynamics(
            m=m,
            damping=damping,
            mode="2d",
            trajectory_samples=trajectory_samples,
            integration_method=integration_method,
        )

        control_map = {
            "thrust_x": 0,
            "thrust_y": 0,
        }

        super().__init__(
            name=name,
            position=np.concatenate([position, np.array([0])]),
            velocity=np.concatenate([velocity, np.array([0])]),
            dynamics=dynamics,
            control_default=np.zeros((2,)),
            control_min=-1,
            control_max=1,
            control_map=control_map,
        )

    @property
    def state(self) -> np.ndarray:
        """2d point mass integrator state vector

        State vector is [x, y, x_dot, y_dot]

        Returns
        -------
        np.ndarray
            state vector of form [x, y, x_dot, y_dot]
        """
        return np.array([self.x, self.y, self.x_dot, self.y_dot])

    @state.setter
    def state(self, state: np.ndarray):
        """Set state of 2d point mass integrator

        State vector is [x, y, x_dot, y_dot]

        Parameters
        ----------
        state : np.ndarray
            state vector of form [x, y, x_dot, y_dot]
        """
        assert (
            state.shape == self.state.shape
        ), f"State must be of shape {self.state.shape}, got {state.shape}"
        self._state[0] = state[0]
        self._state[1] = state[1]
        self._state[3] = state[2]
        self._state[4] = state[3]


class PointMassIntegrator3d(PhysicalEntity):
    """
    3d point mass integrator simulation entity

    States
        x
        y
        z
        x_dot
        y_dot
        z_dot

    Controls
        thrust_x
            default range = [-1, 1]
        thrust_y
            default range = [-1, 1]
        thrust_z
            default range = [-1, 1]

    Parameters
    ----------
    name: str
        name of entity
    position: np.ndarray, optional
        initial 3d position, by default np.array([0, 0, 0])
    velocity: np.ndarray, optional
        initial 3d velocity, by default np.array([0, 0, 0])
    m: float
        Mass of integrator, by default 1.
    trajectory_samples : int
        number of trajectory samples the generate and store on steps
    integration_method: str
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics.
    """

    def __init__(
        self,
        name: str,
        position: np.ndarray = np.array([0, 0, 0]),
        velocity: np.ndarray = np.array([0, 0, 0]),
        m=M_DEFAULT,
        damping=DAMPING_DEFAULT,
        trajectory_samples=0,
        integration_method="RK45",
    ):
        assert position.shape == (3,), "Position must be 3D"
        assert velocity.shape == (3,), "Velocity must be 3D"

        dynamics = PointMassIntegratorDynamics(
            m=m,
            damping=damping,
            mode="3d",
            trajectory_samples=trajectory_samples,
            integration_method=integration_method,
        )

        control_map = {
            "thrust_x": 0,
            "thrust_y": 0,
            "thrust_z": 0,
        }

        super().__init__(
            name=name,
            position=position,
            velocity=velocity,
            dynamics=dynamics,
            control_default=np.zeros((3,)),
            control_min=-1,
            control_max=1,
            control_map=control_map,
        )

    @property
    def state(self) -> np.ndarray:
        """3d point mass integrator state vector

        State vector is [x, y, z, x_dot, y_dot, z_dot]

        Returns
        -------
        np.ndarray
            state vector of form [x, y, z, x_dot, y_dot, z_dot]
        """
        return np.array([self.x, self.y, self.z, self.x_dot, self.y_dot, self.z_dot])

    @state.setter
    def state(self, state: np.ndarray):
        """Set state of 3d point mass integrator

        State vector is [x, y, z, x_dot, y_dot, z_dot]

        Parameters
        ----------
        state : np.ndarray
            state vector of form [x, y, z, x_dot, y_dot, z_dot]
        """
        assert (
            state.shape == self.state.shape
        ), f"State must be of shape {self.state.shape}, got {state.shape}"
        self._state[0] = state[0]
        self._state[1] = state[1]
        self._state[2] = state[2]
        self._state[3] = state[3]
        self._state[4] = state[4]
        self._state[5] = state[5]


class PointMassIntegratorDynamics(LinearODESolverDynamics):
    """
    State transition implementation of 3D integrator dynamics model.

    Parameters
    ----------
    m: float, optional
        Mass of object, by default 1
    damping: float, optional
        linear velocity damper, by default 0
    mode : str, optional
        dimensionality of dynamics matrices. '1d', '2d', or '3d', by default '1d'
    """

    def __init__(self, m=M_DEFAULT, damping=DAMPING_DEFAULT, mode="1d", **kwargs):
        self.m = m
        self.damping = damping
        A, B = generate_dynamics_matrices(self.m, self.damping, mode)
        super().__init__(A=A, B=B, **kwargs)


def generate_dynamics_matrices(
    m: float, damping: float = 0, mode: str = "1d"
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates A and B Matrices for linearized dynamics of dx/dt = Ax + Bu

    Parameters
    ----------
    m : float
        mass of object
    damping : float, optional
        linear velocity damper. Default is zero
    mode : str, optional
        dimensionality of dynamics matrices. '1d', '2d', or '3d', by default '1d'

    Returns
    -------
    np.ndarray
        A dynamics matrix
    np.ndarray
        B dynamics matrix
    """
    assert mode in ["1d", "2d", "3d"], "mode must be one of ['1d', '2d', '3d']"
    if mode == "1d":
        A = np.array(
            [
                [0, 1],
                [0, -damping],
            ],
            dtype=np.float64,
        )

        B = np.array(
            [
                [0],
                [1 / m],
            ],
            dtype=np.float64,
        )
    elif mode == "2d":
        A = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, -damping, 0],
                [0, 0, 0, -damping],
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
                [0, 0, 0, -damping, 0, 0],
                [0, 0, 0, 0, -damping, 0],
                [0, 0, 0, 0, 0, -damping],
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
