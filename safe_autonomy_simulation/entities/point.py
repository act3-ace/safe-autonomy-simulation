import typing
import numpy as np
import scipy.spatial.transform as transform
import safe_autonomy_simulation.entities as e
import safe_autonomy_simulation.dynamics as d
import safe_autonomy_simulation.materials as m
import safe_autonomy_simulation.controls as c


class Point(e.PhysicalEntity):
    """A point entity with three degrees of freedom.

    Parameters
    ----------
    name : str
        Name of the entity
    position : np.ndarray
        Initial absolute position of the entity.
    velocity : np.ndarray, optional
        Initial absolute velocity of the entity. By default [0, 0, 0]
    dynamics : Dynamics, optional
        Dynamics object for computing state transitions. By default PassThroughDynamics()
    control_queue : ControlQueue, optional
        Queue of control vectors to be applied to the entity. By default NoControl()
    parent : Union[Entity, None], optional
        Optional parent entity of the entity. By default None.
    children : list[Entity], optional
        Optional list of child entities of the entity. By default [].
    material : Material, optional
        Material properties of the entity. By default BLACK.
    """

    def __init__(
        self,
        name: str,
        position: np.ndarray,
        velocity: np.ndarray = np.array([0, 0, 0]),
        dynamics: d.Dynamics = d.PassThroughDynamics(),
        control_queue: c.ControlQueue = c.NoControl(),
        parent: typing.Union[e.Entity, None] = None,
        children: list[e.Entity] = [],
        material: m.Material = m.BLACK,
    ):
        self._initial_position = position
        self._initial_velocity = velocity

        super().__init__(
            name=name,
            dynamics=dynamics,
            position=position,
            velocity=velocity,
            orientation=transform.Rotation.from_euler("ZYX", [0, 0, 0]).as_quat(),
            angular_velocity=np.zeros(3),
            control_queue=control_queue,
            parent=parent,
            children=children,
            material=material,
        )

    @property
    def state(self) -> np.ndarray:
        """Point entity state vector

        Point state vector is [x, y, z, vx, vy, vz]

        Returns
        -------
        np.ndarray
            state vector [x, y, z, vx, vy, vz]
        """
        return np.concatenate((self.position, self.velocity))

    @state.setter
    def state(self, state: np.ndarray):
        """Set the point entity state vector

        Parameters
        ----------
        state : np.ndarray
            New state vector [x, y, z, vx, vy, vz]
        """
        assert (
            state.shape == self.state.shape
        ), f"State shape must be {self.state.shape}, got {state.shape}"
        self._state[0:3] = state[0:3]
        self._state[3:6] = state[3:6]
