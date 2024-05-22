import numpy as np
from typing import Union

from safe_autonomy_simulation.entity import PhysicalEntity
from safe_autonomy_simulation.inspection.camera import Camera
from safe_autonomy_simulation.spacecraft.utils import (
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
from safe_autonomy_simulation.spacecraft.point_model import CWHSpacecraft
from safe_autonomy_simulation.spacecraft.sixdof_model import SixDOFSpacecraft
from safe_autonomy_simulation.material import Material


class Inspector(CWHSpacecraft):
    """Inspector spacecraft with a camera.

    Parameters
    ----------
    name : str
        name of the entity
    camera : Camera
        Inspector camera sensor
    position : np.ndarray, optional
        Initial position of spacecraft in meters, by default np.zeros(3)
    velocity : np.ndarray, optional
        Initial velocity of spacecraft in meters/second, by default np.zeros(3)
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
    children : set[PhysicalEntity], optional
        Set of children entities of spacecraft, by default {}
    """

    def __init__(
        self,
        name: str,
        camera: Camera,
        position: np.ndarray = np.zeros(3),
        velocity: np.ndarray = np.zeros(3),
        m: float = M_DEFAULT,
        n: float = N_DEFAULT,
        trajectory_samples: int = 0,
        integration_method: str = "RK45",
        material: Material = CWH_MATERIAL,
        parent: Union[PhysicalEntity, None] = None,
        children: set[PhysicalEntity] = {},
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
            children=children,
        )
        self._camera = camera
        self.add_child(camera)

    @property
    def camera(self) -> Camera:
        """Inspector camera sensor

        Returns
        -------
        Camera
            Inspector camera sensor
        """
        return self._camera


class SixDOFInspector(SixDOFSpacecraft):
    """SixDOF Inspector spacecraft with a camera.

    Parameters
    ----------
    name : str
        name of the entity
    camera : Camera
        Inspector camera sensor
    position : np.ndarray, optional
        Initial position of spacecraft in meters, by default np.zeros(3)
    velocity : np.ndarray, optional
        Initial velocity of spacecraft in meters/second, by default np.zeros(3)
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
    children : set[PhysicalEntity], optional
        Set of children entities of spacecraft, by default {}
    """

    def __init__(
        self,
        name: str,
        camera: Camera,
        position: np.ndarray = np.zeros(3),
        velocity: np.ndarray = np.zeros(3),
        orientation: np.ndarray = np.array([0, 0, 0, 1]),
        angular_velocity: np.ndarray = np.zeros(3),
        m: float = M_DEFAULT,
        n: float = N_DEFAULT,
        inertia_matrix: np.matrix = INERTIA_MATRIX_DEFAULT,
        ang_acc_limit: float = ANG_ACC_LIMIT_DEFAULT,
        ang_vel_limit: float = ANG_VEL_LIMIT_DEFAULT,
        inertia_wheel: float = INERTIA_WHEEL_DEFAULT,
        acc_limit_wheel: float = ACC_LIMIT_WHEEL_DEFAULT,
        vel_limit_wheel: float = VEL_LIMIT_WHEEL_DEFAULT,
        thrust_control_limit: float = THRUST_CONTROL_LIMIT_DEFAULT,
        body_frame_thrust: bool = True,
        trajectory_samples: int = 0,
        integration_method: str = "RK45",
        material: Material = CWH_MATERIAL,
        parent: Union[PhysicalEntity, None] = None,
        children: set[PhysicalEntity] = {},
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
        self._camera = camera
        self.add_child(camera)

    @property
    def camera(self) -> Camera:
        """Inspector camera sensor

        Returns
        -------
        Camera
            Inspector camera sensor
        """
        return self._camera
