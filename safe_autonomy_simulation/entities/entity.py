"""A controllable entity with state transition dynamics"""

import typing
import typing_extensions
import safe_autonomy_simulation.dynamics as d
import safe_autonomy_simulation.materials as materials
import safe_autonomy_simulation.controls.control_queue as controls
import safe_autonomy_simulation.utils.sets as sets

try:
    import jax.numpy as np
except ImportError:
    import numpy as np


class Entity:
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
    dynamics : Dynamics
        Dynamics object for computing state transitions
    control_queue : ControlQueue
        Queue of control vectors to be applied to the entity
    material : Material
        Material properties of the entity
    parent : Union[Entity, None], optional
        Optional parent entity of the entity. By default None.
    children : list[Entity], optional
        Optional list of child entities of the entity. By default []
    """

    def __init__(
        self,
        name: str,
        dynamics: d.Dynamics,
        control_queue: controls.ControlQueue,
        material: materials.Material,
        parent: typing.Union[typing_extensions.Self, None] = None,
        children: list[typing_extensions.Self] = [],
    ):
        self.name: str = name

        self._state: np.ndarray = self.build_initial_state()

        self._dynamics: d.Dynamics = dynamics

        self._control_queue: controls.ControlQueue = control_queue
        self._last_control: np.ndarray | None = None
        self._state_dot: np.ndarray = np.zeros_like(self.state)

        # Register parent and children
        self._children: sets.TypedSet = sets.TypedSet[Entity](type=Entity)
        self._parent: Entity | None = None
        if parent is not None:
            parent.add_child(self)
        for child in children:
            self.add_child(child)

        # Set material
        self._material = material

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

    def add_control(self, control: typing.Union[np.ndarray, list]):
        """Add a control to the entity's control queue

        Parameters
        ----------
        control : np.ndarray
            Control vector to be added to the control queue
        """
        self.control_queue.add_control(control)

    def _is_descendant(self, entity: typing_extensions.Self) -> bool:
        """
        Check if descendant of entity

        Returns True if self is a descendant of entity (including self), False otherwise

        Parameters
        ----------
        entity : Entity
            Entity to check if descendant of

        Returns
        -------
        bool
            True if descendant of entity, False otherwise
        """
        if self is entity:
            return True
        if self.parent is None:
            return False
        return self.parent._is_descendant(entity)

    def add_child(self, child: typing_extensions.Self):
        """
        Adds a child entity to the entity

        Parameters
        ----------
        child : Entity
            Child entity to be added
        """
        assert child is not self, "Entity cannot be its own child"
        assert not self._is_descendant(
            child
        ), f"New child {child} cannot be an ancestor of self {self}"
        if child.parent is not None:
            child.parent.remove_child(child)
        child._parent = self
        self._children.add(child)

    def remove_child(self, child: typing_extensions.Self):
        """
        Removes a child entity from the entity

        Parameters
        ----------
        child : Entity
            Child entity to be removed
        """
        assert child in self._children, f"Child {child} not found in children"
        self._children.remove(child)
        child._parent = None

    @property
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
    def last_control(self) -> typing.Union[np.ndarray, None]:
        """
        Last control vector applied to the entity

        Returns
        -------
        Union[np.ndarray, None]
            Last control vector applied to the entity
        """
        return self._last_control

    @property
    def parent(self) -> typing.Union[typing_extensions.Self, None]:
        """
        Parent entity

        Returns
        -------
        Union[Entity, None]
            Parent entity
        """
        return self._parent

    @property
    def children(self) -> sets.TypedSet[typing_extensions.Self]:
        """
        Set of child entities of the entity

        Returns
        -------
        set[Entity]
            Set of child entities of the entity
        """
        return self._children

    @property
    def material(self) -> materials.Material:
        """
        Material properties of the entity

        Returns
        -------
        Material
            Material properties of the entity
        """
        return self._material

    @property
    def dynamics(self) -> d.Dynamics:
        """
        Dynamics object for computing state transitions

        Returns
        -------
        Dynamics
            Dynamics object for computing state transitions
        """
        return self._dynamics

    @property
    def control_queue(self) -> controls.ControlQueue:
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
