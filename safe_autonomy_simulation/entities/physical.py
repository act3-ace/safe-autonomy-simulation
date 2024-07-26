"""Physical entities with applied kinematics and state transition dynamics"""

import typing
import numpy as np
import pint
import safe_autonomy_simulation.entities as e
import safe_autonomy_simulation.dynamics as d
import safe_autonomy_simulation.controls as c
import safe_autonomy_simulation.materials as m


class BaseUnits:
    """Provides unit system definitions for physical entities

    Parameters
    ----------
    distance : Union[str, pint.Unit]
        Distance unit definition
    time : Union[str, pint.Unit]
        Time unit definition
    angle : Union[str, pint.Unit]
        Angle unit definition

    Attributes
    ----------
    distance : pint.Unit
        Distance unit definition
    time : pint.Unit
        Time unit definition
    angle : pint.Unit
        Angle unit definition
    velocity : pint.Unit
        Velocity unit definition
    angular_velocity : pint.Unit
        Angular velocity unit definition
    acceleration : pint.Unit
        Acceleration unit definition
    angular_acceleration : pint.Unit
        Angular acceleration unit definition
    """

    def __init__(
        self,
        distance: typing.Union[str, pint.Unit],
        time: typing.Union[str, pint.Unit],
        angle: typing.Union[str, pint.Unit],
    ):
        self.distance: pint.Unit = pint.Unit(distance)
        self.time: pint.Unit = pint.Unit(time)
        self.angle: pint.Unit = pint.Unit(angle)

        self.velocity: pint.Unit = self.distance / self.time
        self.angular_velocity: pint.Unit = self.angle / self.time

        self.acceleration: pint.Unit = self.distance / (self.time**2)
        self.angular_acceleration: pint.Unit = self.angle / (self.time**2)


class PhysicalEntity(e.Entity):
    """
    A dynamics controlled kinematic entity within the simulation.

    Physical entities have position, velocity, orientation, and angular velocity properties.

    Maintains an internal state vector of the form [position, velocity, orientation, angular_velocity].

    All physical entities must implement the following properties:
    - state: **External** (client-facing) state vector of the entity
    - state.setter: Set the **external** state vector of the entity

    Note: the internal state vector is not directly accessible to clients. The state property should be
    used to access the entity's state.

    Parameters
    ----------
    name : str
        Name of the entity
    position : np.ndarray
        Initial absolute position of the entity
    velocity : np.ndarray
        Initial absolute velocity of the entity
    orientation : np.ndarray
        Initial absolute orientation quaternion of the entity
    angular_velocity : np.ndarray
        Initial absolute angular velocity of the entity
    control_queue : ControlQueue
        Queue of control vectors to be applied to the entity
    dynamics : Dynamics
        Dynamics object for computing state transitions
    material : Material
        Material properties of the entity
    parent : Union[Entity, None], optional
        Optional parent entity of the entity. By default None.
    children : list[Entity], optional
        Optional list of child entities of the entity. By default [].
    """

    base_units = BaseUnits("meters", "seconds", "radians")

    def __init__(
        self,
        name: str,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation: np.ndarray,
        angular_velocity: np.ndarray,
        control_queue: c.ControlQueue,
        dynamics: d.Dynamics,
        material: m.Material,
        parent: typing.Union[e.Entity, None] = None,
        children: list[e.Entity] = [],
    ):
        assert position.shape == (
            3,
        ), f"Position must be a 3D vector of shape (3,), got shape {position.shape}"
        assert velocity.shape == (
            3,
        ), f"Velocity must be a 3D vector of shape (3,), got shape {velocity.shape}"
        assert orientation.shape == (
            4,
        ), f"Orientation must be a quaternion of shape (4,), got shape {orientation.shape}"
        assert angular_velocity.shape == (
            3,
        ), f"Angular velocity must be a 3D vector of shape (3,), got shape {angular_velocity.shape}"

        self._initial_position = position
        self._initial_velocity = velocity
        self._initial_orientation = orientation
        self._initial_angular_velocity = angular_velocity
        super().__init__(
            name=name,
            dynamics=dynamics,
            control_queue=control_queue,
            parent=parent,
            children=children,
            material=material,
        )
        self._ureg: pint.UnitRegistry = pint.get_application_registry()

    def __eq__(self, other):
        if isinstance(other, type(self)):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = eq and (self.orientation == other.orientation).all()
            eq = eq and (self.angular_velocity == other.angular_velocity).all()
            return eq
        return False

    def build_initial_state(self) -> np.ndarray:
        return np.concatenate(
            (
                self._initial_position,
                self._initial_velocity,
                self._initial_orientation,
                self._initial_angular_velocity,
            )
        )

    @property
    def state(self) -> np.ndarray:
        """State vector of the entity

        state = [position, velocity, orientation, angular_velocity]

        Returns
        -------
        np.ndarray
            state vector of the entity
        """
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        """Set state of the entity

        Parameters
        ----------
        state : np.ndarray
            state vector of the entity
        """
        assert (
            state.shape == self.state.shape
        ), f"State shape must match {self.state.shape}, got {state.shape}"
        state = state.astype(self._state.dtype)
        self._state = state

    @property
    def x(self) -> float:
        """absolute X position of entity

        Returns
        -------
        float
            absolute X position
        """
        return self.position[0]

    @property
    def x_with_units(self) -> typing.Annotated[pint.Quantity, float]:
        """absolute X position of entity with units

        Returns
        -------
        pint.Quantity
            absolute X position with units
        """
        return self._ureg.Quantity(self.x, self.base_units.distance)

    @property
    def y(self) -> float:
        """absolute Y position of entity

        Returns
        -------
        float
            absolute Y position
        """
        return self.position[1]

    @property
    def y_with_units(self) -> typing.Annotated[pint.Quantity, float]:
        """absolute Y position of entity with units

        Returns
        -------
        pint.Quantity
            absolute Y position with units
        """
        return self._ureg.Quantity(self.y, self.base_units.distance)

    @property
    def z(self) -> float:
        """absolute Z position of entity

        Returns
        -------
        float
            absolute Z position
        """
        return self.position[2]

    @property
    def z_with_units(self) -> typing.Annotated[pint.Quantity, float]:
        """absolute Z position of entity with units

        Returns
        -------
        pint.Quantity
            absolute Z position with units
        """
        return self._ureg.Quantity(self.z, self.base_units.distance)

    @property
    def position(self) -> np.ndarray:
        """Entity absolute position vector

        Returns
        -------
        np.ndarray
            3D absolute position vector
        """
        return self._state[:3]

    @property
    def position_with_units(self) -> typing.Annotated[pint.Quantity, np.ndarray[float]]:
        """Entity absolute position vector with units

        Returns
        -------
        pint.Quantity
            3D absolute position vector with units
        """
        return self._ureg.Quantity(self.position, self.base_units.distance)

    @property
    def x_dot(self) -> float:
        """absolute X velocity component

        Returns
        -------
        float
            absolute X velocity component
        """
        return self.velocity[0]

    @property
    def x_dot_with_units(self) -> typing.Annotated[pint.Quantity, float]:
        """absolute X velocity component with units

        Returns
        -------
        pint.Quantity
            absolute X velocity component with units
        """
        return self._ureg.Quantity(self.x_dot, self.base_units.velocity)

    @property
    def y_dot(self) -> float:
        """absolute Y velocity component

        Returns
        -------
        float
            absolute Y velocity component
        """
        return self.velocity[1]

    @property
    def y_dot_with_units(self) -> typing.Annotated[pint.Quantity, float]:
        """absolute Y velocity component with units

        Returns
        -------
        pint.Quantity
            absolute Y velocity component with units
        """
        return self._ureg.Quantity(self.y_dot, self.base_units.velocity)

    @property
    def z_dot(self) -> float:
        """absolute Z velocity component

        Returns
        -------
        float
            absolute Z velocity component
        """
        return self.velocity[2]

    @property
    def z_dot_with_units(self) -> typing.Annotated[pint.Quantity, float]:
        """absolute Z velocity component with units

        Returns
        -------
        pint.Quantity
            absolute Z velocity component with units
        """
        return self._ureg.Quantity(self.z_dot, self.base_units.velocity)

    @property
    def velocity(self) -> np.ndarray:
        """Entity absolute velocity vector

        Returns
        -------
        np.ndarray
            absolute 3D velocity vector
        """
        return self._state[3:6]

    @property
    def velocity_with_units(self) -> typing.Annotated[pint.Quantity, np.ndarray[float]]:
        """Entity absolute velocity vector with units

        Returns
        -------
        pint.Quantity
            absolute 3D velocity vector with units
        """
        return self._ureg.Quantity(self.velocity, self.base_units.velocity)

    @property
    def orientation(self) -> np.ndarray:
        """Entity absolute orientation quaternion

        Returns
        -------
        np.ndarray
            absolute orientation quaternion
        """
        return self._state[6:10]

    @property
    def wx(self) -> float:
        """Wx, the absolute angular velocity component about the local body frame x axis

        Returns
        -------
        float
            Wx
        """
        return self.angular_velocity[0]

    @property
    def wx_with_units(self) -> typing.Annotated[pint.Quantity, float]:
        """Wx, the absolute angular velocity component about the local body frame x axis with units

        Returns
        -------
        pint.Quantity
            Wx with units
        """
        return self._ureg.Quantity(self.wx, self.base_units.angular_velocity)

    @property
    def wy(self) -> float:
        """Wy, the absolute angular velocity component about the local body frame y axis

        Returns
        -------
        float
            Wy
        """
        return self.angular_velocity[1]

    @property
    def wy_with_units(self) -> typing.Annotated[pint.Quantity, float]:
        """Wy, the absolute angular velocity component about the local body frame y axis with units

        Returns
        -------
        pint.Quantity
            Wy with units
        """
        return self._ureg.Quantity(self.wy, self.base_units.angular_velocity)

    @property
    def wz(self) -> float:
        """Wz, the absolute angular velocity component about the local body frame z axis

        Returns
        -------
        float
            Wz
        """
        return self.angular_velocity[2]

    @property
    def wz_with_units(self) -> typing.Annotated[pint.Quantity, float]:
        """Wz, the absolute angular velocity component about the local body frame z axis with units

        Returns
        -------
        pint.Quantity
            Wz with units
        """
        return self._ureg.Quantity(self.wz, self.base_units.angular_velocity)

    @property
    def angular_velocity(self) -> np.ndarray:
        """Entity absolute angular velocity vector

        Angular velocity is wrapped to [0, 2 * pi]

        Returns
        -------
        np.ndarray
            3D angular velocity vector
        """
        return self._state[10:13]

    @property
    def angular_velocity_with_units(
        self,
    ) -> typing.Annotated[pint.Quantity, np.ndarray[float]]:
        """Entity absolute angular velocity vector with units

        Returns
        -------
        pint.Quantity
            3D angular velocity vector with units
        """
        return self._ureg.Quantity(
            self.angular_velocity, self.base_units.angular_velocity
        )
