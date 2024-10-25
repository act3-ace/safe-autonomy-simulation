"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Dynamics.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a spacecraft with Clohessy-Wilshire physics dynamics in non-inertial orbital
Hill's reference frame along with rotational dynamics. 2D scenario models in-plane (x-y) translational
motion and rotation about the z axis. 3D scenario is pending.
"""

import typing

import pint
import scipy.spatial.transform as transform
import numpy as np

import safe_autonomy_simulation.entities as e
import safe_autonomy_simulation.dynamics as d
import safe_autonomy_simulation.materials as mat
import safe_autonomy_simulation.controls as c
import safe_autonomy_simulation.sims.spacecraft.defaults as defaults


class CWHRotation2dSpacecraft(e.PhysicalEntity):  # pylint: disable=too-many-public-methods
    """
    Spacecraft with 2D translational Clohessy-Wiltshire dynamics in Hill's reference frame.
    In-plane motion (x,y) using +/- x thruster rotated to desired direction

    1D rotational dynamics (about z) using a +/- z reaction wheel

    States
        x
        y
        theta
        x_dot
        y_dot
        theta_dot

    Controls
        thrust_x
            range = [-1, 1] Newtons
        thrust_y
            range = [-1, 1] Newtons
        moment_z
            range = [-0.001, 0.001] Newton-Meters

    Parameters
    ----------
    name: str
        Name of spacecraft
    position: np.ndarray, optional
        Initial absolute position, by default np.zeros(2)
    velocity: np.ndarray, optional
        Initial absolulte velocity, by default np.zeros(2)
    theta: float or pint.Quantity, optional
        Initial absolute rotation angle value, by default 0
    wz: float or pint.Quantity, optional
        Initial absolute rotation rate value, by default 0
    m: float, optional
        Mass of spacecraft in kilograms, by default 12.
    inertia: float, optional
        Inertia of spacecraft in kg*m^2, by default 0.0573
    ang_acc_limit: float, optional
        Angular acceleration limit in rad/s^2, by default 0.017453
    ang_vel_limit: float, optional
        Angular velocity limit in rad/s, by default 0.034907
    inertia_wheel: float, optional
        Inertia of reaction wheel in kg*m^2, by default 4.1e-5
    acc_limit_wheel: float, optional
         Acceleration limit of reaction wheel in rad/s^2, by default 181.3
    vel_limit_wheel: float, optional
         Velocity limit of reaction wheel in rad/s, by default 576
    n: float, optional
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027.
    trajectory_samples : int, optional
        number of trajectory samples the generate and store on steps, by default 0
    integration_method: str, optional
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics. By default "RK45".
    material: Material, optional
        Material properties of the spacecraft, by default CWH_MATERIAL
    parent: Union[PhysicalEntity, None], optional
        Parent entity of spacecraft, by default None
    children: list[PhysicalEntity], optional
        List of children entities of spacecraft, by default []
    control_min: typing.Union[float, np.ndarray, None]
        specify a minimum value that control can be. numbers lower than this will be clipped. (default = None).
        If this value is None, control_min will be [-1, -1, -ang_acc_limit * self.inertia]
    control_max: typing.Union[float, np.ndarray, None] = 1,
        specify a maximum value that control can be. numbers higher than this will be clipped. (default = None)
        If this value is None, control_min will be [1, 1, ang_acc_limit * self.inertia]
    use_jax : bool, optional
        EXPERIMENTAL: Use JAX to accelerate state transition computation, by default False.
    """

    def __init__(
        self,
        name: str,
        position: np.ndarray = np.zeros(2),
        velocity: np.ndarray = np.zeros(2),
        theta: typing.Union[float, pint.Quantity] = 0,
        wz: typing.Union[float, pint.Quantity] = 0,
        m=defaults.M_DEFAULT,
        inertia=defaults.INERTIA_DEFAULT,
        ang_acc_limit=defaults.ANG_ACC_LIMIT_DEFAULT,
        ang_vel_limit=defaults.ANG_VEL_LIMIT_DEFAULT,
        inertia_wheel=defaults.INERTIA_WHEEL_DEFAULT,
        acc_limit_wheel=defaults.ACC_LIMIT_WHEEL_DEFAULT,
        vel_limit_wheel=defaults.VEL_LIMIT_WHEEL_DEFAULT,
        n=defaults.N_DEFAULT,
        trajectory_samples=0,
        integration_method="RK45",
        material: mat.Material = defaults.CWH_MATERIAL,
        parent: typing.Union[e.PhysicalEntity, None] = None,
        children: list[e.PhysicalEntity] = [],
        control_min: typing.Union[float, np.ndarray, None] = None,
        control_max: typing.Union[float, np.ndarray, None] = None,
        use_jax: bool = False,
    ):
        assert position.shape == (2,), f"Position must be 2D. Instead got {position}"
        assert velocity.shape == (2,), f"Velocity must be 2D. Instead got {velocity}"

        self.m = m  # kg
        self.inertia = inertia  # kg*m^2
        self.ang_acc_limit = ang_acc_limit  # rad/s^2
        self.ang_vel_limit = ang_vel_limit  # rad/s
        self.inertia_wheel = inertia_wheel  # kg*m^2
        self.acc_limit_wheel = acc_limit_wheel  # rad/s^2
        self.vel_limit_wheel = vel_limit_wheel  # rad/s
        self.n = n  # rads/s

        # Define limits for angular acceleration, angular velocity, and control inputs
        ang_acc_limit = min(
            self.ang_acc_limit, self.inertia_wheel * self.acc_limit_wheel / self.inertia
        )
        ang_vel_limit = min(
            self.ang_vel_limit, self.inertia_wheel * self.vel_limit_wheel / self.inertia
        )

        if not control_min:
            control_min = np.array([-1, -1, -ang_acc_limit * self.inertia])
        if not control_max:
            control_max = np.array([1, 1, ang_acc_limit * self.inertia])
        control_queue = c.ControlQueue(
            default_control=np.zeros(3),
            control_min=control_min,
            control_max=control_max,
        )

        dynamics = CWHRotation2dDynamics(
            m=m,
            inertia=inertia,
            ang_acc_limit=ang_acc_limit,
            ang_vel_limit=ang_vel_limit,
            n=n,
            trajectory_samples=trajectory_samples,
            integration_method=integration_method,
            use_jax=use_jax,
        )

        super().__init__(
            name=name,
            dynamics=dynamics,
            position=np.concatenate([position, np.array([0])]),  # z=0
            velocity=np.concatenate([velocity, np.array([0])]),  # z_dot=0
            orientation=transform.Rotation.from_euler("ZYX", [theta, 0, 0]).as_quat(),
            angular_velocity=np.array([0, 0, wz]),
            control_queue=control_queue,
            material=material,
            parent=parent,
            children=children,
        )

    @property
    def theta(self) -> float:
        """Rotation angle about z axis

        Returns
        -------
        float
            Rotation angle about z axis
        """
        return transform.Rotation.from_quat(self.orientation).as_euler("ZYX")[0]

    @property
    def theta_with_units(self) -> pint.Quantity:
        """Rotation angle about z axis as a pint.Quantity with units

        Returns
        -------
        pint.Quantity
            Rotation angle about z axis with units
        """
        return self._ureg.Quantity(self.theta, self.base_units.angle)

    @property
    def state(self) -> np.ndarray:
        """State vector of spacecraft

        Returns
        -------
        np.ndarray
            state vector of form [x, y, x_dot, y_dot, theta, wz]
        """
        return np.concatenate(
            [
                self.position[:2],
                self.velocity[:2],
                np.array([self.theta]),
                np.array([self.wz]),
            ]
        )

    @state.setter
    def state(self, state: np.ndarray):
        """Set state of spacecraft

        Parameters
        ----------
        state : np.ndarray
            state vector of form [x, y, x_dot, y_dot, theta, wz]
        """
        assert (
            state.shape == self.state.shape
        ), f"State shape must match {self.state.shape}, got {state.shape}"
        # Internal state is [x, y, z, x_dot, y_dot, z_dot, q0, q1, q2, q3, wx, wy, wz]
        self._state[0:2] = state[0:2]
        self._state[3:4] = state[2:4]
        self._state[6:10] = transform.Rotation.from_euler(
            "ZYX", [state[4], 0, 0]
        ).as_quat()
        self._state[12] = state[5]


class CWHRotation2dDynamics(d.ControlAffineODEDynamics):
    """
    State transition implementation of a Clohessy-Wiltshire dynamics model
    with 2D translational motion and 1D rotational motion.

    Parameters
    ----------
    m: float, optional
        Mass of spacecraft in kilograms, by default 12
    inertia: float, optional
        Inertia of spacecraft in kg*m^2, by default 0.0573
    ang_acc_limit: float, optional
         Angular acceleration limit in rad/s^2, by default 0.017453
    ang_vel_limit: float, optional
         Angular velocity limit in rad/s, by default 0.034907
    n: float, optional
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027
    trajectory_samples : int, optional
        number of trajectory samples the generate and store on steps, by default 0
    state_min : float or np.ndarray, optional
        Minimum state values, by default None
    state_max : float or np.ndarray, optional
        Maximum state values, by default None
    state_dot_min : float or np.ndarray, optional
        Minimum state derivative values, by default None
    state_dot_max : float or np.ndarray, optional
        Maximum state derivative values, by default None
    integration_method: str, optional
        Numerical integration method, by default "RK45"
    use_jax : bool, optional
        EXPERIMENTAL: Use JAX to accelerate state transition computation, by default False.
    """

    def __init__(
        self,
        m: float = defaults.M_DEFAULT,
        inertia: float = defaults.INERTIA_DEFAULT,
        ang_acc_limit: float = defaults.ANG_ACC_LIMIT_DEFAULT,
        ang_vel_limit: float = defaults.ANG_VEL_LIMIT_DEFAULT,
        n: float = defaults.N_DEFAULT,
        trajectory_samples: int = 0,
        state_min: typing.Union[float, np.ndarray, None] = None,
        state_max: typing.Union[float, np.ndarray, None] = None,
        state_dot_min: typing.Union[float, np.ndarray, None] = None,
        state_dot_max: typing.Union[float, np.ndarray, None] = None,
        integration_method: str = "RK45",
        use_jax: bool = False,
    ):
        self.m = m  # kg
        self.inertia = inertia  # kg*m^2
        self.ang_acc_limit = ang_acc_limit  # rad/s^2
        self.ang_vel_limit = ang_vel_limit  # rad/s
        self.n = n  # rads/s

        if state_min is None:
            state_min = np.array(
                [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -self.ang_vel_limit]
            )
        if state_max is None:
            state_max = np.array(
                [np.inf, np.inf, np.inf, np.inf, np.inf, self.ang_vel_limit]
            )

        if state_dot_min is None:
            state_dot_min = np.array(
                [
                    -np.inf,
                    -np.inf,
                    -self.ang_vel_limit,
                    -np.inf,
                    -np.inf,
                    -self.ang_acc_limit,
                ]
            )
        if state_dot_max is None:
            state_dot_max = np.array(
                [np.inf, np.inf, self.ang_vel_limit, np.inf, np.inf, self.ang_acc_limit]
            )

        super().__init__(
            trajectory_samples=trajectory_samples,
            state_min=state_min,
            state_max=state_max,
            state_dot_min=state_dot_min,
            state_dot_max=state_dot_max,
            integration_method=integration_method,
            use_jax=use_jax,
        )

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

        self.A = self.np.copy(A)
        self.B = self.np.copy(B)

    def state_transition_system(self, state: np.ndarray) -> np.ndarray:
        x, y, x_dot, y_dot, theta, wz = state
        # Form separate state vector for translational state
        pos_vel_state_vec = self.np.array([x, y, x_dot, y_dot], self.np.float32)
        # Compute derivatives
        pos_vel_derivative = self.A @ pos_vel_state_vec

        # Form array of state derivatives
        state_derivative = self.np.array(
            [
                pos_vel_derivative[0],
                pos_vel_derivative[1],
                wz,
                pos_vel_derivative[2],
                pos_vel_derivative[3],
                0,
            ],
            dtype=self.np.float32,
        )
        return state_derivative

    def state_transition_input(self, state: np.ndarray) -> np.ndarray:
        theta = state[4]

        g = self.np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [np.cos(theta) / self.m, -np.sin(theta) / self.m, 0],
                [np.sin(theta) / self.m, np.cos(theta) / self.m, 0],
                [0, 0, 1 / self.inertia],
            ]
        )
        return g
