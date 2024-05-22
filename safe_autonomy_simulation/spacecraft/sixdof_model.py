"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Simulation.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a spacecraft with 3D Clohessy-Wilshire physics dynamics in non-inertial orbital
Hill's reference frame along with 3D rotational dynamics using quaternions for attitude representation.
"""

from typing import Union

import numpy as np

from safe_autonomy_simulation.spacecraft.utils import (
    generate_cwh_matrices,
    M_DEFAULT,
    N_DEFAULT,
    INERTIA_MATRIX_DEFAULT,
    INERTIA_WHEEL_DEFAULT,
    ANG_ACC_LIMIT_DEFAULT,
    ANG_VEL_LIMIT_DEFAULT,
    ACC_LIMIT_WHEEL_DEFAULT,
    VEL_LIMIT_WHEEL_DEFAULT,
    THRUST_CONTROL_LIMIT_DEFAULT,
    CWH_MATERIAL,
)
from safe_autonomy_simulation.utils import number_list_to_np
from safe_autonomy_simulation.entity import Entity, PhysicalEntity, ControlQueue
from safe_autonomy_simulation.dynamics import ControlAffineODESolverDynamics
from safe_autonomy_simulation.material import Material


class SixDOFSpacecraft(PhysicalEntity):  # pylint: disable=too-many-public-methods
    """
    Spacecraft with 3D Clohessy-Wiltshire translational dynamics, in Hill's frame and 3D rotational dynamics

    States
        x, y, z
        q1, q2, q3, q4
        x_dot, y_dot, z_dot
        wx, wy, wz

    Controls
        thrust_x
            range = [-1, 1] Newtons
        thrust_y
            range = [-1, 1] Newtons
        thrust_z
            range = [-1, 1] Newtons
        moment_x
            range = [-0.001, 0.001] Newton-Meters
        moment_y
            range = [-0.001, 0.001] Newton-Meters
        moment_z
            range = [-0.001, 0.001] Newton-Meters

    Parameters
    ----------
    name: str
        Name of the entity
    position: np.ndarray, optional
        Initial position of spacecraft in meters, by default np.zeros(3)
    velocity: np.ndarray, optional
        Initial velocity of spacecraft in meters/second, by default np.zeros(3)
    orientation: np.ndarray, optional
        Initial orientation of spacecraft as quaternion, by default np.array([0, 0, 0, 1])
    angular_velocity: np.ndarray, optional
        Initial angular velocity of spacecraft in rad/s, by default np.zeros(3)
    m: float, optional
        Mass of spacecraft in kilograms, by default 12
    inertia_matrix: float, optional
        Inertia matrix of spacecraft (3x3) in kg*m^2, by default np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
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
    thrust_control_limit: float, optional
        Thrust control limit in N, by default 1.0
    body_frame_thrust: bool, optional
        Flag indicating the reference frame for the control thrust vector: True- Body frame, False - Hill's frame
        by default, True
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
    children: set[PhysicalEntity], optional
        Set of children entities of spacecraft, by default {}
    """

    def __init__(
        self,
        name: str,
        position: np.ndarray = np.zeros(3),
        velocity: np.ndarray = np.zeros(3),
        orientation: np.ndarray = np.array([0, 0, 0, 1]),
        angular_velocity: np.ndarray = np.zeros(3),
        m=M_DEFAULT,
        inertia_matrix=INERTIA_MATRIX_DEFAULT,
        ang_acc_limit=ANG_ACC_LIMIT_DEFAULT,
        ang_vel_limit=ANG_VEL_LIMIT_DEFAULT,
        inertia_wheel=INERTIA_WHEEL_DEFAULT,
        acc_limit_wheel=ACC_LIMIT_WHEEL_DEFAULT,
        vel_limit_wheel=VEL_LIMIT_WHEEL_DEFAULT,
        thrust_control_limit=THRUST_CONTROL_LIMIT_DEFAULT,
        body_frame_thrust=True,
        n=N_DEFAULT,
        trajectory_samples=0,
        integration_method="RK45",
        material: Material = CWH_MATERIAL,
        parent: Union[PhysicalEntity, None] = None,
        children: set[PhysicalEntity] = {},
    ):
        # Define limits for angular acceleration, angular velocity, and control inputs
        ang_acc_limit = number_list_to_np(ang_acc_limit, shape=(3,))  # rad/s^2
        ang_vel_limit = number_list_to_np(ang_vel_limit, shape=(3,))  # rad/s

        acc_limit_combined = np.zeros((3,))
        vel_limit_combined = np.zeros((3,))
        control_limit = np.zeros((6,))
        for i in range(3):
            acc_limit_combined[i] = min(
                ang_acc_limit[i], inertia_wheel * acc_limit_wheel / inertia_matrix[i, i]
            )
            vel_limit_combined[i] = min(
                ang_vel_limit[i], inertia_wheel * vel_limit_wheel / inertia_matrix[i, i]
            )
            control_limit[i] = thrust_control_limit
            control_limit[i + 3] = acc_limit_combined[i] * inertia_matrix[i, i]

        control_queue = ControlQueue(
            default_control=np.zeros(6),
            control_map={
                "thrust_x": 0,
                "thrust_y": 1,
                "thrust_z": 2,
                "moment_x": 3,
                "moment_y": 4,
                "moment_z": 5,
            },
            control_min=-control_limit,
            control_max=control_limit,
        )

        dynamics = SixDOFDynamics(
            m=m,
            inertia_matrix=inertia_matrix,
            ang_acc_limit=acc_limit_combined,
            ang_vel_limit=vel_limit_combined,
            n=n,
            body_frame_thrust=body_frame_thrust,
            trajectory_samples=trajectory_samples,
            integration_method=integration_method,
        )
        self.lead = None

        super().__init__(
            name=name,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            dynamics=dynamics,
            control_queue=control_queue,
            material=material,
            parent=parent,
            children=children,
        )

    def register_lead(self, lead: Entity):
        """
        Register another entity as this entity's lead. Defines line of communication between entities.

        Parameters
        ----------
        lead: Entity
            Entity with line of communication to this entity.

        Returns
        -------
        None
        """
        self.lead = lead

    @property
    def state(self) -> np.ndarray:
        """State vector of spacecraft

        Returns
        -------
        np.ndarray
            state vector of form [x, y, z, x_dot, y_dot, z_dot, q1, q2, q3, q4, wx, wy, wz]
        """
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        """Set state of spacecraft

        Parameters
        ----------
        state : np.ndarray
            state vector of form [x, y, z, x_dot, y_dot, z_dot, q1, q2, q3, q4, wx, wy, wz]
        """
        assert (
            state.shape == self.state.shape
        ), f"State shape must match {self.state.shape}, got {state.shape}"
        # Internal state is [x, y, z, x_dot, y_dot, z_dot, q1, q2, q3, q4, wx, wy, wz]
        self._state = state


class SixDOFDynamics(ControlAffineODESolverDynamics):
    """
    State transition implementation of 3D Clohessy-Wiltshire dynamics model and 3D rotational dynamics model.

    Parameters
    ----------
    m: float, optional
        Mass of spacecraft in kilograms, by default 12
    inertia_matrix: float, optional
        Inertia matrix of spacecraft (3x3) in kg*m^2, by default np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ang_acc_limit: float, list, np.ndarray, optional
        Angular acceleration limit in rad/s^2. If array_like, applied to x, y, z elementwise, by default 0.017453
    ang_vel_limit: float, list, np.ndarray, optional
        Angular velocity limit in rad/s. If array_like, applied to x, y, z elementwise, by default 0.034907
    thrust_control_limit: float, optional
        Thrust control limit in N, by default 1.0
    n: float, optional
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027
    body_frame_thrust: bool, optional
        Flag indicating the reference frame for the control thrust vector: True- Body frame, False - Hill's frame
        by default, True
    trajectory_samples : int, optional
        number of trajectory samples the generate and store on steps, by default 0
    state_max : float or np.ndarray, optional
        Maximum state values, by default None
    state_min : float or np.ndarray, optional
        Minimum state values, by default None
    state_dot_max : float or np.ndarray, optional
        Maximum state derivative values, by default None
    state_dot_min : float or np.ndarray, optional
        Minimum state derivative values, by default None
    angle_wrap_centers : np.ndarray, optional
        Centers for angle wrapping, by default None
    integration_method: str, optional
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics. By default "RK45"
    use_jax: bool, optional
        Flag to use JAX for numerical integration, by default False
    """

    def __init__(
        self,
        m=M_DEFAULT,
        inertia_matrix=INERTIA_MATRIX_DEFAULT,
        ang_acc_limit=ANG_ACC_LIMIT_DEFAULT,
        ang_vel_limit=ANG_VEL_LIMIT_DEFAULT,
        n=N_DEFAULT,
        body_frame_thrust=True,
        trajectory_samples: int = 0,
        state_max: Union[float, np.ndarray] = None,
        state_min: Union[float, np.ndarray] = None,
        state_dot_max: Union[float, np.ndarray] = None,
        state_dot_min: Union[float, np.ndarray] = None,
        angle_wrap_centers: np.ndarray = None,
        integration_method="RK45",
        use_jax=False,
    ):
        self.m = m  # kg
        self.inertia_matrix = inertia_matrix  # kg*m^2
        self.n = n  # rads/s
        self.body_frame_thrust = body_frame_thrust
        self.control_thrust_Hill = np.zeros(
            3,
        )

        ang_acc_limit = number_list_to_np(ang_acc_limit, shape=(3,))  # rad/s^2
        ang_vel_limit = number_list_to_np(ang_vel_limit, shape=(3,))  # rad/s

        A, B = generate_cwh_matrices(self.m, self.n, "3d")

        assert (
            len(A.shape) == 2
        ), f"A must be square matrix. Instead got shape {A.shape}"
        assert (
            len(B.shape) == 2
        ), f"A must be square matrix. Instead got shape {B.shape}"
        assert (
            A.shape[0] == A.shape[1]
        ), f"A must be a square matrix, not dimension {A.shape}"
        assert A.shape[1] == B.shape[0], (
            "number of columns in A must match the number of rows in B."
            + f" However, got shapes {A.shape} for A and {B.shape} for B"
        )

        self.A = np.copy(A)
        self.B = np.copy(B)

        if state_min is None:
            state_min = np.array(
                [
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -ang_vel_limit[0],
                    -ang_vel_limit[1],
                    -ang_vel_limit[2],
                ]
            )

        if state_max is None:
            state_max = np.array(
                [
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    ang_vel_limit[0],
                    ang_vel_limit[1],
                    ang_vel_limit[2],
                ]
            )

        super().__init__(
            trajectory_samples=trajectory_samples,
            state_min=state_min,
            state_max=state_max,
            state_dot_min=state_dot_min,
            state_dot_max=state_dot_max,
            angle_wrap_centers=angle_wrap_centers,
            integration_method=integration_method,
            use_jax=use_jax,
        )

    def state_transition_system(self, state: np.ndarray) -> np.ndarray:
        x, y, z, q1, q2, q3, q4, x_dot, y_dot, z_dot, wx, wy, wz = state

        # Compute translational derivatives
        pos_vel_state_vec = np.array([x, y, z, x_dot, y_dot, z_dot], dtype=np.float64)
        pos_vel_derivative = self.A @ pos_vel_state_vec

        # Compute rotational derivatives
        q_derivative = np.zeros((4,))
        w_derivative = np.zeros((3,))
        q_derivative[0] = 0.5 * (q4 * wx - q3 * wy + q2 * wz)
        q_derivative[1] = 0.5 * (q3 * wx + q4 * wy - q1 * wz)
        q_derivative[2] = 0.5 * (-q2 * wx + q1 * wy + q4 * wz)
        q_derivative[3] = 0.5 * (-q1 * wx - q2 * wy - q3 * wz)
        w_derivative[0] = (
            1
            / self.inertia_matrix[0, 0]
            * ((self.inertia_matrix[1, 1] - self.inertia_matrix[2, 2]) * wy * wz)
        )
        w_derivative[1] = (
            1
            / self.inertia_matrix[1, 1]
            * ((self.inertia_matrix[2, 2] - self.inertia_matrix[0, 0]) * wx * wz)
        )
        w_derivative[2] = (
            1
            / self.inertia_matrix[2, 2]
            * ((self.inertia_matrix[0, 0] - self.inertia_matrix[1, 1]) * wx * wy)
        )

        # Form derivative array
        state_derivative = np.array(
            [
                pos_vel_derivative[0],
                pos_vel_derivative[1],
                pos_vel_derivative[2],
                q_derivative[0],
                q_derivative[1],
                q_derivative[2],
                q_derivative[3],
                pos_vel_derivative[3],
                pos_vel_derivative[4],
                pos_vel_derivative[5],
                w_derivative[0],
                w_derivative[1],
                w_derivative[2],
            ],
            dtype=np.float32,
        )
        return state_derivative

    def state_transition_input(self, state: np.ndarray) -> np.ndarray:
        quat = state[3:7]

        w_derivative = np.array(
            [
                [1 / self.inertia_matrix[0, 0], 0, 0],
                [0, 1 / self.inertia_matrix[1, 1], 0],
                [0, 0, 1 / self.inertia_matrix[2, 2]],
            ]
        )

        # Convert the control thrust to Hill's frame prior to application in the CWH equations
        if self.body_frame_thrust:
            r1 = 1 / self.m * self.apply_quat(np.array([1, 0, 0]), quat)
            r2 = 1 / self.m * self.apply_quat(np.array([0, 1, 0]), quat)
            r3 = 1 / self.m * self.apply_quat(np.array([0, 0, 1]), quat)
            vel_derivative = np.array(
                [[r1[0], r2[0], r3[0]], [r1[1], r2[1], r3[1]], [r1[2], r2[2], r3[2]]]
            )
        else:
            vel_derivative = self.B[3:6, :]

        g = np.vstack(
            (
                np.zeros((7, 6)),
                np.hstack((vel_derivative, np.zeros(vel_derivative.shape))),
                np.hstack((np.zeros(w_derivative.shape), w_derivative)),
            )
        )

        return g

    def apply_quat(self, x: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """
        Apply quaternion rotation to 3d vector

        Parameters
        ----------
        x : np.ndarray
            vector of length 3
        quat : np.ndarray
            quaternion vector of form [x, y, z, w]

        Returns
        -------
        np.ndarray
            rotated vector of length 3
        """
        p = np.insert(x, 0, 0, axis=0)
        r = np.array([quat[3], quat[0], quat[1], quat[2]])
        r_p = np.array([quat[3], -quat[0], -quat[1], -quat[2]])
        rotated_x = self.hamilton_product(self.hamilton_product(r, p), r_p)[1:]
        return rotated_x

    def hamilton_product(self, r, q):
        """Hamilton product between 2 vectors"""
        return np.array(
            [
                r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
                r[0] * q[1] + r[1] * q[0] + r[2] * q[3] - r[3] * q[2],
                r[0] * q[2] - r[1] * q[3] + r[2] * q[0] + r[3] * q[1],
                r[0] * q[3] + r[1] * q[2] - r[2] * q[1] + r[3] * q[0],
            ]
        )
