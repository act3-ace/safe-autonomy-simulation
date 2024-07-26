"""1D, 2D, and 3D point mass integrator entities"""

import typing

import numpy as np

import safe_autonomy_simulation.entities as e
import safe_autonomy_simulation.dynamics as d
import safe_autonomy_simulation.materials as materials
import safe_autonomy_simulation.controls as c

M_DEFAULT = 1  # default mass in kg
DAMPING_DEFAULT = 0  # default damping coefficient


class PointMassIntegrator1d(e.PhysicalEntity):
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
        initial absolute position relative to origin, by default np.array([0])
    velocity: np.ndarray, optional
        initial absolute velocity, by default np.array([0])
    m: float, optional
        Mass of integrator in kg, by default 1.
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

        control_queue = c.ControlQueue(
            default_control=np.zeros((1,)), control_min=-1, control_max=1
        )

        super().__init__(
            name=name,
            dynamics=dynamics,
            position=np.concatenate([position, np.array([0, 0])]),
            velocity=np.concatenate([velocity, np.array([0, 0])]),
            orientation=np.array([0, 0, 0, 1]),
            angular_velocity=np.array([0, 0, 0]),
            control_queue=control_queue,
            material=materials.BLACK,
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


class PointMassIntegrator2d(e.PhysicalEntity):
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
        initial absolute 2d position relative to origin, by default np.array([0, 0])
    velocity: np.ndarray, optional
        initial absolute 2d velocity, by default np.array([0, 0])
    m: float, optional
        Mass of integrator in kg, by default 1.
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

        control_queue = c.ControlQueue(
            default_control=np.zeros((2,)), control_min=-1, control_max=1
        )

        super().__init__(
            name=name,
            position=np.concatenate([position, np.array([0])]),
            velocity=np.concatenate([velocity, np.array([0])]),
            orientation=np.array([0, 0, 0, 1]),
            angular_velocity=np.array([0, 0, 0]),
            dynamics=dynamics,
            control_queue=control_queue,
            material=materials.BLACK,
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


class PointMassIntegrator3d(e.PhysicalEntity):
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
        initial absolute 3d position relative to origin, by default np.array([0, 0, 0])
    velocity: np.ndarray, optional
        initial absolute 3d velocity, by default np.array([0, 0, 0])
    m: float
        Mass of integrator in kg, by default 1.
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

        control_queue = c.ControlQueue(
            default_control=np.zeros((3,)), control_min=-1, control_max=1
        )

        super().__init__(
            name=name,
            position=position,
            velocity=velocity,
            orientation=np.array([0, 0, 0, 1]),
            angular_velocity=np.array([0, 0, 0]),
            dynamics=dynamics,
            control_queue=control_queue,
            material=materials.BLACK,
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


class PointMassIntegratorDynamics(d.LinearODEDynamics):
    """
    State transition implementation of 3D integrator dynamics model.

    Parameters
    ----------
    m: float, optional
        Mass of object in kg, by default 1
    damping: float, optional
        linear velocity damper, by default 0
    mode : str, optional
        dimensionality of dynamics matrices. '1d', '2d', or '3d', by default '1d'
    trajectory_samples : int, optional
        number of trajectory samples the generate and store on steps, by default 0
    state_min : Union[float, np.ndarray], optional
        minimum state values, by default -np.inf
    state_max : Union[float, np.ndarray], optional
        maximum state values, by default np.inf
    state_dot_min : Union[float, np.ndarray], optional
        minimum state derivative values, by default -np.inf
    state_dot_max : Union[float, np.ndarray], optional
        maximum state derivative values, by default np.inf
    integration_method : str, optional
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics. By default "RK45"
    """

    def __init__(
        self,
        m=M_DEFAULT,
        damping=DAMPING_DEFAULT,
        mode="1d",
        trajectory_samples: int = 0,
        state_min: typing.Union[float, np.ndarray] = -np.inf,
        state_max: typing.Union[float, np.ndarray] = np.inf,
        state_dot_min: typing.Union[float, np.ndarray] = -np.inf,
        state_dot_max: typing.Union[float, np.ndarray] = np.inf,
        integration_method: str = "RK45",
    ):
        self.m = m
        self.damping = damping
        A, B = self.generate_dynamics_matrices(mode=mode)
        super().__init__(
            A=A,
            B=B,
            trajectory_samples=trajectory_samples,
            state_min=state_min,
            state_max=state_max,
            state_dot_min=state_dot_min,
            state_dot_max=state_dot_max,
            integration_method=integration_method,
        )

    def generate_dynamics_matrices(
        self, mode: str = "1d"
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Generates A and B Matrices for linearized dynamics of dx/dt = Ax + Bu

        Parameters
        ----------
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
                    [0, -self.damping],
                ],
                dtype=np.float64,
            )

            B = np.array(
                [
                    [0],
                    [1 / self.m],
                ],
                dtype=np.float64,
            )
        elif mode == "2d":
            A = np.array(
                [
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, -self.damping, 0],
                    [0, 0, 0, -self.damping],
                ],
                dtype=np.float64,
            )

            B = np.array(
                [
                    [0, 0],
                    [0, 0],
                    [1 / self.m, 0],
                    [0, 1 / self.m],
                ],
                dtype=np.float64,
            )
        else:
            A = np.array(
                [
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, -self.damping, 0, 0],
                    [0, 0, 0, 0, -self.damping, 0],
                    [0, 0, 0, 0, 0, -self.damping],
                ],
                dtype=np.float64,
            )

            B = np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1 / self.m, 0, 0],
                    [0, 1 / self.m, 0],
                    [0, 0, 1 / self.m],
                ],
                dtype=np.float64,
            )

        return A, B
