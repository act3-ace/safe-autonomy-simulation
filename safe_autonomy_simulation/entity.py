"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Simulation.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module provides base implementations for simulation entities
"""

from __future__ import annotations

import abc
import warnings
from queue import SimpleQueue
from typing import TYPE_CHECKING, Union

import numpy as np
import pint
from scipy.spatial.transform import Rotation
from pint import _typing as pintt

from safe_autonomy_simulation.dynamics import Dynamics, PassThroughDynamics
from safe_autonomy_simulation.material import Material, BLACK

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    from jax.experimental.ode import odeint
else:
    try:
        import jax
        import jax.numpy as jnp
        from jax.experimental.ode import odeint
    except ImportError:
        jax = None
        jnp = None
        odeint = None


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
        distance: Union[str, pint.Unit],
        time: Union[str, pint.Unit],
        angle: Union[str, pint.Unit],
    ):
        self.distance: pint.Unit = pint.Unit(distance)
        self.time: pint.Unit = pint.Unit(time)
        self.angle: pint.Unit = pint.Unit(angle)

        self.velocity: pint.Unit = self.distance / self.time
        self.angular_velocity: pint.Unit = self.angle / self.time

        self.acceleration: pint.Unit = self.distance / (self.time**2)
        self.angular_acceleration: pint.Unit = self.angle / (self.time**2)


class ControlQueue:
    """
    A queue of entity controls to be applied to an Entity.

    Parameters
    ----------
    default_control : np.ndarray
        Default control vector used when the control queue is empty. Typically 0 or neutral for each actuator.
    control_min : Union[float, None], optional
        Optional minimum allowable control vector values. Control vectors that exceed this limit are clipped. By default None.
    control_max : Union[float, None], optional
        Optional maximum allowable control vector values. Control vectors that exceed this limit are clipped. By default None.
    control_map : Union[dict, None], optional
        Optional mapping for actuator names to their indices in the state vector.
        Allows dictionary action inputs in add_control(). By default None.

    Attributes
    ----------
    controls : SimpleQueue
        Queue of control vectors to be applied to the entity.
    default_control : np.ndarray
        Default control vector used when the control queue is empty. Typically 0 or neutral for each actuator.
    control_min : Union[np.ndarray, None], optional
        Minimum allowable control vector values. Control vectors that exceed this limit are clipped. By default None.
    control_max : Union[np.ndarray, None], optional
        Maximum allowable control vector values. Control vectors that exceed this limit are clipped. By default None.
    control_map : Union[dict, None], optional
        Mapping for actuator names to their indices in the state vector.
        Required for dictionary action inputs in add_control(). By default None.
    """

    def __init__(
        self,
        default_control: np.ndarray,
        control_min: Union[float, None] = None,
        control_max: Union[float, None] = None,
        control_map: Union[dict, None] = None,
    ):
        self.controls = SimpleQueue()
        self.default_control = default_control
        self.control_min = control_min
        self.control_max = control_max
        self.control_map = control_map

    def reset(self):
        """Clears the control queue"""
        while not self.controls.empty():
            self.controls.get()

    def empty(self) -> bool:
        """
        Check if the control queue is empty.

        Returns
        -------
        bool
            True if the control queue is empty, False otherwise.
        """
        return self.controls.empty()

    def next_control(self) -> Union[np.ndarray, jnp.ndarray]:
        """Removes and returns the next control in the control queue.

        If control queue is empty, returns the default control.

        Returns
        -------
        Union[np.ndarray, dict]
            Next control in the control queue or default control.
        """
        if self.empty():
            control = self.default_control
        else:
            control = self.controls.get()

        return control

    def _vectorize_control(self, control: dict) -> np.ndarray:
        control_vector = np.zeros(len(self.default_control))
        for k, v in control.items():
            if k not in self.control_map:
                raise KeyError(
                    f"action '{k}' not found in entity's control_map, "
                    f"please use one of: {self.control_map.keys()}"
                )

            control_vector[self.control_map[k]] = v
        if jnp is not None:
            control_vector = jnp.array(control_vector)
        return control_vector

    def add_control(self, control: Union[np.ndarray, dict, list, jnp.ndarray]):
        """Adds a control to the end of the control queue.

        Parameters
        ----------
        control : Union[np.ndarray, dict, list, jnp.ndarray]
            Control vector to be added to the control queue.
        """
        if isinstance(control, dict):
            assert (
                self.control_map is not None
            ), "Cannot use dict-type action without a control_map "
            control = self._vectorize_control(control)
        elif isinstance(control, list):
            control = np.array(control, dtype=np.float32)
        elif isinstance(control, np.ndarray):
            control = control.copy()
        elif jnp is not None and isinstance(  # pylint: disable=used-before-assignment
            control, jnp.ndarray
        ):  # pylint: disable=used-before-assignment
            control = control.copy()
        else:
            raise ValueError(
                "action must be type dict, list, np.ndarray or jnp.ndarray"
            )

        # enforce control bounds (if any)
        if (self.control_min or self.control_max) and (
            np.any(control < self.control_min) or np.any(control > self.control_max)
        ):
            warnings.warn(
                f"Control input exceeded limits. Clipping to range ({self.control_min}, {self.control_max})"
            )
            control = np.clip(control, self.control_min, self.control_max)

        self.controls.put(control)


class DefaultControlQueue(ControlQueue):
    """A control queue with an empty default control vector."""
    def __init__(self):
        super().__init__(default_control=np.empty(0))


class Entity(abc.ABC):
    """
    Base implementation of a dynamics controlled entity within the simulation.

    All entities must implement the following methods and properties:
    - build_initial_state: Builds the initial **internal** state vector for the entity
    - state: **External** (client-facing) state vector of the entity
    - state.setter: Set the **external** state vector of the entity

    Note: the internal state vector is not directly accessible to clients. The state property should be
    used to access the entity's state.

    Parameters
    ----------
    name : str
        Name of the entity
    dynamics : Dynamics, optional
        Dynamics object for computing state transitions, by default PassThroughDynamics()
    control_queue : ControlQueue, optional
        Queue of control vectors to be applied to the entity, by default DefaultControlQueue()
    parent : Union[Entity, None], optional
        Optional parent entity of the entity. By default None.
    children : set[Entity], optional
        Optional set of child entities of the entity. By default {}.
    material : Material, optional
        Material properties of the entity. By default BLACK.
    """

    def __init__(
        self,
        name: str,
        dynamics: Dynamics = PassThroughDynamics(),
        control_queue: ControlQueue = DefaultControlQueue(),
        parent: Union[Entity, None] = None,
        children: set[Entity] = {},
        material: Material = BLACK(),
    ):
        self.name = name

        self._state = self.build_initial_state()

        self._dynamics = dynamics

        self._control_queue = control_queue
        self._last_control = None
        self._state_dot = np.zeros_like(self.state)

        # Register parent and children
        self.parent = parent
        self._children = children

        # Set material
        self._material = material

    @abc.abstractmethod
    def build_initial_state(self) -> np.ndarray:
        """Builds the initial internal state vector for the entity.

        Returns
        -------
        np.ndarray
            Initial state vector for the entity
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the entity to an initial state
        """
        self._state = self.build_initial_state()
        self._last_control = None
        self._state_dot = np.zeros_like(self.state)
        self.control_queue.reset()
        for child in self.children:
            child.reset()

    def _pre_step(self, step_size: float):
        """
        Pre-step hook for entity dynamics.

        This method is called before the entity's state is updated in the step() method.
        It can be used to perform any necessary pre-step computations or updates to the entity's state.

        Parameters
        ----------
        step_size : float
            Duration of simulation step in seconds.
        """
        pass

    def _post_step(self, step_size: float):
        """
        Post-step hook for entity dynamics.

        This method is called after the entity's state is updated in the step() method.
        It can be used to perform any necessary post-step computations or updates to the entity's state.

        Parameters
        ----------
        step_size : float
            Duration of simulation step in seconds.
        """
        pass

    def step(self, step_size: float):
        """
        Executes a state transition simulation step for the entity and its children.

        This method should only update the `state`, `state_dot`, and `last_control` vectors
        of the entity and its children and should not modify any other entity attributes.
        The entity's state vector is updated according to its dynamics, the previous state,
        and the next control vector in the control queue.

        Entity children are stepped after the parent entity to ensure that the parent state
        is updated before the children are stepped.

        Note: This method should not be overridden by subclasses. To customize entity behavior,
        override the `_pre_step` and `_post_step` methods.

        Parameters
        ----------
        step_size : float
            Duration of simulation step in seconds.
        """
        self._pre_step(step_size)
        self._last_control = self.control_queue.next_control()
        self.state, self._state_dot = self.dynamics.step(
            step_size, self.state, self.last_control
        )
        for child in self.children:
            child.step(step_size)
        self._post_step(step_size)

    def add_control(self, control: Union[np.ndarray, list, jnp.ndarray, dict]):
        """Add a control to the entity's control queue

        Parameters
        ----------
        control : np.ndarray
            Control vector to be added to the control queue
        """
        self.control_queue.add_control(control)

    def add_child(self, child: Entity):
        """
        Adds a child entity to the entity

        Parameters
        ----------
        child : Entity
            Child entity to be added
        """
        assert child is not self, "Entity cannot be its own child"
        assert (
            child is not self.parent
        ), f"Entity cannot be a parent of its parent {child}"
        assert child.parent is None, "Entity already has a parent"
        self._children.add(child)
        child.parent = self

    def remove_child(self, child: Entity):
        """
        Removes a child entity from the entity

        Parameters
        ----------
        child : Entity
            Child entity to be removed
        """
        self._children.remove(child)
        child.parent = None

    @property
    @abc.abstractmethod
    def state(self) -> np.ndarray:
        """
        Entity state vector

        Returns
        -------
        np.ndarray
            state vector
        """
        raise NotImplementedError

    @state.setter
    @abc.abstractmethod
    def state(self, state: np.ndarray):
        """
        Set the entity state vector

        Parameters
        ----------
        state : np.ndarray
            New state vector
        """
        raise NotImplementedError

    @property
    def last_control(self) -> Union[np.ndarray, None]:
        """
        Last control vector applied to the entity

        Returns
        -------
        Union[np.ndarray, None]
            Last control vector applied to the entity
        """
        return self._last_control

    @property
    def parent(self) -> Union[Entity, None]:
        """
        Parent entity of the entity

        Returns
        -------
        Union[Entity, None]
            Parent entity of the entity
        """
        return self._parent
    
    @parent.setter
    def parent(self, parent: Entity):
        """
        Set the parent entity of the entity

        Parameters
        ----------
        parent : Entity
            Parent entity of the entity
        """
        assert parent is not self, "Entity cannot be its own parent"
        assert (
            parent not in self.children
        ), f"Entity cannot be a child of its child {parent}"
        if self._parent is not None:
            self._parent.remove_child(self)
        self._parent = parent
        self._parent.add_child(self)

    @property
    def children(self) -> set[Entity]:
        """
        Set of child entities of the entity

        Returns
        -------
        set[Entity]
            Set of child entities of the entity
        """
        return self._children

    @property
    def material(self) -> Material:
        """
        Material properties of the entity

        Returns
        -------
        Material
            Material properties of the entity
        """
        return self._material

    @property
    def dynamics(self) -> Dynamics:
        """
        Dynamics object for computing state transitions

        Returns
        -------
        Dynamics
            Dynamics object for computing state transitions
        """
        return self._dynamics

    @property
    def control_queue(self) -> ControlQueue:
        """
        Queue of control vectors to be applied to the entity

        Returns
        -------
        ControlQueue
            Queue of control vectors to be applied to the entity
        """
        return self._control_queue

    @property
    def state_dot(self) -> np.ndarray:
        """Time derivative of the entity state vector
        
        Returns
        -------
        np.ndarray
            Time derivative of the entity state vector
        """
        return self._state_dot


class PhysicalEntity(Entity):
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
    control_queue : ControlQueue, optional
        Queue of control vectors to be applied to the entity, by default DefaultControlQueue()
    dynamics : Dynamics, optional
        Dynamics object for computing state transitions. By default PassThroughDynamics()
    parent : Union[Entity, None], optional
        Optional parent entity of the entity. By default None.
    children : set[Entity], optional
        Optional set of child entities of the entity. By default {}.
    material : Material, optional
        Material properties of the entity. By default BLACK.
    """

    base_units = BaseUnits("meters", "seconds", "radians")

    def __init__(
        self,
        name: str,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation: np.ndarray,
        angular_velocity: np.ndarray,
        control_queue: ControlQueue = DefaultControlQueue(),
        dynamics: Dynamics = PassThroughDynamics(),
        parent: Union[Entity, None] = None,
        children: set[Entity] = {},
        material: Material = BLACK(),
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
            eq = (
                eq
                and (
                    self.orientation.as_euler("ZYX")
                    == other.orientation.as_euler("ZYX")
                ).all()
            )
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
        return self._ureg.Quantity(self.x, self.base_units.length)

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
        return self._ureg.Quantity(self.y, self.base_units.length)

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
        return self._ureg.Quantity(self.z, self.base_units.length)

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
    def position_with_units(self) -> pintt.Quantity[np.ndarray]:
        """Entity position vector with units

        Returns
        -------
        pint.Quantity
            3D position vector with units
        """
        return self._ureg.Quantity(self.position, self.base_units.length)

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
    def velocity_with_units(self) -> pintt.Quantity[np.ndarray]:
        """Entity velocity vector with units

        Returns
        -------
        pint.Quantity
            3D velocity vector with units
        """
        return self._ureg.Quantity(self.velocity, self.base_units.velocity)

    @property
    def orientation(self) -> Rotation:
        """Entity orientation vector

        Returns
        -------
        scipy.spatial.transform.Rotation
            Rotation transformation of the entity's local reference frame basis vectors in the global reference frame.
            i.e. applying this rotation to [1, 0, 0] yields the entity's local x-axis in the global frame.
        """
        return Rotation.from_quat(self.quaternion)

    @property
    def q1(self) -> float:
        """First element of entity's orientation quaternion

        Returns
        -------
        float
            First element of orientation quaternion
        """
        return self.quaternion[0]

    @property
    def q2(self) -> float:
        """Second element of entity's orientation quaternion

        Returns
        -------
        float
            Second element of orientation quaternion
        """
        return self.quaternion[1]

    @property
    def q3(self) -> float:
        """Third element of entity's orientation quaternion

        Returns
        -------
        float
            Third element of orientation quaternion
        """
        return self.quaternion[2]

    @property
    def q4(self) -> float:
        """Fourth element of entity's orientation quaternion

        Returns
        -------
        float
            Fourth element of orientation quaternion
        """
        return self.quaternion[3]

    @property
    def quaternion(self) -> np.ndarray:
        """Entity orientation quaternion

        Returns
        -------
        np.ndarray
            Orientation quaternion
        """
        return self._state[6:10]

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
    def angular_velocity_with_units(self) -> pintt.Quantity[np.ndarray]:
        """Entity angular velocity vector with units

        Returns
        -------
        pint.Quantity
            3D angular velocity vector with units
        """
        return self._ureg.Quantity(
            self.angular_velocity, self.base_units.angular_velocity
        )


class Point(PhysicalEntity):
    """A point entity with pass-through dynamics and three degrees of freedom.

    Parameters
    ----------
    name : str
        Name of the entity
    position : np.ndarray
        Initial position of the entity.
    velocity : np.ndarray, optional
        Initial velocity of the entity. By default [0, 0, 0]
    dynamics : Dynamics, optional
        Dynamics object for computing state transitions. By default PassThroughDynamics()
    control_queue : ControlQueue, optional
        Queue of control vectors to be applied to the entity. By default DefaultControlQueue()
    parent : Union[Entity, None], optional
        Optional parent entity of the entity. By default None.
    children : set[Entity], optional
        Optional set of child entities of the entity. By default {}.
    material : Material, optional
        Material properties of the entity. By default BLACK.
    """

    def __init__(
        self,
        name: str,
        position: np.ndarray,
        velocity: np.ndarray = np.array([0, 0, 0]),
        dynamics: Dynamics = PassThroughDynamics(),
        control_queue: ControlQueue = DefaultControlQueue(),
        parent: Union[Entity, None] = None,
        children: set[Entity] = {},
        material: Material = BLACK(),
    ):
        self._initial_position = position
        self._initial_velocity = velocity

        super().__init__(
            name=name,
            dynamics=dynamics,
            position=position,
            velocity=velocity,
            orientation=Rotation.from_euler("ZYX", [0, 0, 0]).as_quat(),
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
