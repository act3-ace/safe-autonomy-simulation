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
        Initial position of the entity
    velocity : np.ndarray
        Initial velocity of the entity
    orientation : np.ndarray
        Initial orientation quaternion of the entity
    angular_velocity : np.ndarray
        Initial angular velocity of the entity
    control_queue : ControlQueue
        Queue of control vectors to be applied to the entity
    dynamics : Dynamics
        Dynamics object for computing state transitions
    material : Material
        Material properties of the entity
    parent : Union[Entity, None], optional
        Optional parent entity of the entity. By default None.
    children : set[Entity], optional
        Optional set of child entities of the entity. By default {}.
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
        children: set[e.Entity] = {},
    ):
        assert position.shape == (
            3,
        ), f"Position must be a 3D vector of shape (3,), got shape {position.shape}"
        assert velocity.shape == (
            3,
        ), f"Velocity must be a 3D vector of shape (3,), got shape {velocity.shape}"
        assert (
            orientation.shape == (4,)
        ), f"Orientation must be a quaternion of shape (4,), got shape {orientation.shape}"
        assert (
            angular_velocity.shape == (3,)
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
    def x(self) -> float:
        """X position of entity

        Returns
        -------
        float
            X position
        """
        return self.position[0]

    @property
    def x_with_units(self) -> pint.Quantity:
        """X position of entity with units

        Returns
        -------
        pint.Quantity
            X position with units
        """
        return self._ureg.Quantity(self.x, self.base_units.distance)

    @property
    def y(self) -> float:
        """Y position of entity

        Returns
        -------
        float
            Y position
        """
        return self.position[1]

    @property
    def y_with_units(self) -> pint.Quantity:
        """Y position of entity with units

        Returns
        -------
        pint.Quantity
            Y position with units
        """
        return self._ureg.Quantity(self.y, self.base_units.distance)

    @property
    def z(self) -> float:
        """Z position of entity

        Returns
        -------
        float
            Z position
        """
        return self.position[2]

    @property
    def z_with_units(self) -> pint.Quantity:
        """Z position of entity with units

        Returns
        -------
        pint.Quantity
            Z position with units
        """
        return self._ureg.Quantity(self.z, self.base_units.distance)

    @property
    def position(self) -> np.ndarray:
        """Entity position vector

        Returns
        -------
        np.ndarray
            3D position vector
        """
        return self._state[:3]

    @property
    def position_with_units(self) -> pint.Quantity[np.ndarray]:
        """Entity position vector with units

        Returns
        -------
        pint.Quantity
            3D position vector with units
        """
        return self._ureg.Quantity(self.position, self.base_units.distance)

    @property
    def x_dot(self) -> float:
        """X velocity component

        Returns
        -------
        float
            X velocity component
        """
        return self.velocity[0]

    @property
    def x_dot_with_units(self) -> pint.Quantity:
        """X velocity component with units

        Returns
        -------
        pint.Quantity
            X velocity component with units
        """
        return self._ureg.Quantity(self.x_dot, self.base_units.velocity)

    @property
    def y_dot(self) -> float:
        """Y velocity component

        Returns
        -------
        float
            Y velocity component
        """
        return self.velocity[1]

    @property
    def y_dot_with_units(self) -> pint.Quantity:
        """Y velocity component with units

        Returns
        -------
        pint.Quantity
            Y velocity component with units
        """
        return self._ureg.Quantity(self.y_dot, self.base_units.velocity)

    @property
    def z_dot(self) -> float:
        """Z velocity component

        Returns
        -------
        float
            Z velocity component
        """
        return self.velocity[2]

    @property
    def z_dot_with_units(self) -> pint.Quantity:
        """Z velocity component with units

        Returns
        -------
        pint.Quantity
            Z velocity component with units
        """
        return self._ureg.Quantity(self.z_dot, self.base_units.velocity)

    @property
    def velocity(self) -> np.ndarray:
        """Entity velocity vector

        Returns
        -------
        np.ndarray
            3D velocity vector
        """
        return self._state[3:6]

    @property
    def velocity_with_units(self) -> pint.Quantity[np.ndarray]:
        """Entity velocity vector with units

        Returns
        -------
        pint.Quantity
            3D velocity vector with units
        """
        return self._ureg.Quantity(self.velocity, self.base_units.velocity)

    @property
    def orientation(self) -> np.ndarray:
        """Entity orientation quaternion

        Returns
        -------
        np.ndarray
            Orientation quaternion
        """
        return self._state[6:10]

    @property
    def q1(self) -> float:
        """First element of entity's orientation quaternion

        Returns
        -------
        float
            First element of orientation quaternion
        """
        return self.orientation[0]

    @property
    def q2(self) -> float:
        """Second element of entity's orientation quaternion

        Returns
        -------
        float
            Second element of orientation quaternion
        """
        return self.orientation[1]

    @property
    def q3(self) -> float:
        """Third element of entity's orientation quaternion

        Returns
        -------
        float
            Third element of orientation quaternion
        """
        return self.orientation[2]

    @property
    def q4(self) -> float:
        """Fourth element of entity's orientation quaternion

        Returns
        -------
        float
            Fourth element of orientation quaternion
        """
        return self.orientation[3]

    @property
    def wx(self) -> float:
        """Wx, the angular velocity component about the local body frame x axis

        Returns
        -------
        float
            Wx
        """
        return self.angular_velocity[0]

    @property
    def wx_with_unit(self) -> pint.Quantity:
        """Wx, the angular velocity component about the local body frame x axis with units

        Returns
        -------
        pint.Quantity
            Wx with units
        """
        return self._ureg.Quantity(self.wx, self.base_units.angular_velocity)

    @property
    def wy(self) -> float:
        """Wy, the angular velocity component about the local body frame y axis

        Returns
        -------
        float
            Wy
        """
        return self.angular_velocity[1]

    @property
    def wy_with_unit(self) -> pint.Quantity:
        """Wy, the angular velocity component about the local body frame y axis with units

        Returns
        -------
        pint.Quantity
            Wy with units
        """
        return self._ureg.Quantity(self.wy, self.base_units.angular_velocity)

    @property
    def wz(self) -> float:
        """Wz, the angular velocity component about the local body frame z axis

        Returns
        -------
        float
            Wz
        """
        return self.angular_velocity[2]

    @property
    def wz_with_unit(self) -> pint.Quantity:
        """Wz, the angular velocity component about the local body frame z axis with units

        Returns
        -------
        pint.Quantity
            Wz with units
        """
        return self._ureg.Quantity(self.wz, self.base_units.angular_velocity)

    @property
    def angular_velocity(self) -> np.ndarray:
        """Entity angular velocity vector

        Returns
        -------
        np.ndarray
            3D angular velocity vector
        """
        self._state[9:12]

    @property
    def angular_velocity_with_units(self) -> pint.Quantity[np.ndarray]:
        """Entity angular velocity vector with units

        Returns
        -------
        pint.Quantity
            3D angular velocity vector with units
        """
        return self._ureg.Quantity(
            self.angular_velocity, self.base_units.angular_velocity
        )
