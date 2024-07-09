import numpy as np
import typing

import safe_autonomy_simulation.entities as e
import safe_autonomy_simulation.sims.spacecraft.defaults as defaults
import safe_autonomy_simulation.sims.spacecraft as spacecraft
import safe_autonomy_simulation.sims.inspection.inspection_points as p
import safe_autonomy_simulation.materials as mat


class Target(spacecraft.CWHSpacecraft):
    """Inspection target spacecraft with inspection points.

    Parameters
    ----------
    name : str
        name of the entity
    num_points : int
        Number of inspection points
    radius : float
        Radius of inspection point cloud in meters
    priority_vector : np.ndarray, optional
        Priority vector of inspection points, by default np.zeros(3)
    position : np.ndarray, optional
        Initial absolute position of spacecraft in meters, by default np.zeros(3)
    velocity : np.ndarray, optional
        Initial absolute velocity of spacecraft in meters/second, by default np.zeros(3)
    m : float, optional
        Mass of spacecraft in kilograms, by default 12.
    n : float, optional
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027.
    trajectory_samples : int, optional
        number of trajectory samples the generate and store on steps, by default 0
    integration_method : str, optional
        Numerical integration method passed to dynamics model. See ODESolverDynamics. By default "RK45"
    material : Material, optional
        Material properties of the spacecraft, by default CWH_MATERIAL
    parent : Union[PhysicalEntity, None], optional
        Parent entity of spacecraft, by default None
    """

    def __init__(
        self,
        name: str,
        num_points: int,
        radius: float,
        priority_vector: np.ndarray = np.array([1, 0, 0]),
        position: np.ndarray = np.zeros(3),
        velocity: np.ndarray = np.zeros(3),
        m: float = defaults.M_DEFAULT,
        n: float = defaults.N_DEFAULT,
        trajectory_samples: int = 0,
        integration_method: str = "RK45",
        material: mat.Material = defaults.CWH_MATERIAL,
        parent: typing.Union[e.PhysicalEntity, None] = None,
    ):
        super().__init__(
            name=name,
            position=position,
            velocity=velocity,
            m=m,
            n=n,
            trajectory_samples=trajectory_samples,
            integration_method=integration_method,
            material=material,
            parent=parent,
        )
        self._inspection_points = p.InspectionPointSet(
            name="inspection_points",
            parent=self,
            num_points=num_points,
            radius=radius,
            priority_vector=priority_vector,
        )

    @property
    def inspection_points(self) -> p.InspectionPointSet:
        """Inspection points of the target.

        Returns
        -------
        InspectionPoints
            Inspection points of the target
        """
        return self._inspection_points


class SixDOFTarget(spacecraft.SixDOFSpacecraft):
    """SixDOF Inspection target spacecraft with inspection points.

    Parameters
    ----------
    name : str
        name of the entity
    num_points : int
        Number of inspection points
    radius : float
        Radius of inspection point cloud in meters
    priority_vector : np.ndarray, optional
        Priority vector of inspection points, by default np.zeros(3)
    position : np.ndarray, optional
        Initial absolute position of spacecraft in meters, by default np.zeros(3)
    velocity : np.ndarray, optional
        Initial absolute velocity of spacecraft in meters/second, by default np.zeros(3)
    orientation : np.ndarray, optional
        Initial orientation of spacecraft as quaternion, by default np.array([0, 0, 0, 1])
    angular_velocity : np.ndarray, optional
        Initial angular velocity of spacecraft in rad/s, by default np.zeros(3)
    m : float, optional
        Mass of spacecraft in kilograms, by default 12.
    n : float, optional
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027.
    inertia_matrix : np.matrix, optional
        Inertia matrix of spacecraft in kg*m^2, by default diagonal matrix of 0.0573
    ang_acc_limit : float, optional
        Maximum angular acceleration of spacecraft in rad/s^2, by default 0.017453
    ang_vel_limit : float, optional
        Maximum angular velocity of spacecraft in rad/s, by default 0.034907
    inertia_wheel : float, optional
        Inertia of reaction wheel in kg*m^2, by default 4.1e-5
    acc_limit_wheel : float, optional
        Maximum acceleration of reaction wheel in rad/s^2, by default 181.3
    vel_limit_wheel : float, optional
        Maximum velocity of reaction wheel in rad/s, by default 576
    thrust_control_limit : float, optional
        Maximum thrust control input in Newtons, by default 1.0
    body_frame_thrust : bool, optional
        Whether thrust control is in body frame or Hill's frame, by default True
    trajectory_samples : int, optional
        number of trajectory samples the generate and store on steps, by default 0
    integration_method : str, optional
        Numerical integration method passed to dynamics model. See ODESolverDynamics. By default "RK45"
    material : Material, optional
        Material properties of the spacecraft, by default CWH_MATERIAL
    parent : Union[PhysicalEntity, None], optional
        Parent entity of spacecraft, by default None
    children : list[PhysicalEntity], optional
        List of children entities of spacecraft, by default []
    """

    def __init__(
        self,
        name: str,
        num_points: int,
        radius: float,
        priority_vector: np.ndarray = np.zeros(3),
        position: np.ndarray = np.zeros(3),
        velocity: np.ndarray = np.zeros(3),
        orientation: np.ndarray = np.array([0, 0, 0, 1]),
        angular_velocity: np.ndarray = np.zeros(3),
        m: float = defaults.M_DEFAULT,
        n: float = defaults.N_DEFAULT,
        inertia_matrix: np.matrix = defaults.INERTIA_MATRIX_DEFAULT,
        ang_acc_limit: float = defaults.ANG_ACC_LIMIT_DEFAULT,
        ang_vel_limit: float = defaults.ANG_VEL_LIMIT_DEFAULT,
        inertia_wheel: float = defaults.INERTIA_WHEEL_DEFAULT,
        acc_limit_wheel: float = defaults.ACC_LIMIT_WHEEL_DEFAULT,
        vel_limit_wheel: float = defaults.VEL_LIMIT_WHEEL_DEFAULT,
        thrust_control_limit: float = defaults.THRUST_CONTROL_LIMIT_DEFAULT,
        body_frame_thrust: bool = True,
        trajectory_samples: int = 0,
        integration_method: str = "RK45",
        material: mat.Material = defaults.CWH_MATERIAL,
        parent: typing.Union[e.PhysicalEntity, None] = None,
        children: list[e.PhysicalEntity] = [],
    ):
        super().__init__(
            name=name,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            m=m,
            n=n,
            inertia_matrix=inertia_matrix,
            ang_acc_limit=ang_acc_limit,
            ang_vel_limit=ang_vel_limit,
            inertia_wheel=inertia_wheel,
            acc_limit_wheel=acc_limit_wheel,
            vel_limit_wheel=vel_limit_wheel,
            thrust_control_limit=thrust_control_limit,
            body_frame_thrust=body_frame_thrust,
            trajectory_samples=trajectory_samples,
            integration_method=integration_method,
            material=material,
            parent=parent,
            children=children,
        )
        self._inspection_points = p.InspectionPointSet(
            name="inspection_points",
            parent=self,
            num_points=num_points,
            radius=radius,
            priority_vector=priority_vector,
        )

    @property
    def inspection_points(self) -> p.InspectionPointSet:
        """Inspection points of the target.

        Returns
        -------
        InspectionPoints
            Inspection points of the target
        """
        return self._inspection_points
