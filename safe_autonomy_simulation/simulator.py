"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Simulation.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains a base simulator class for building new simulations.
"""

import typing
import numpy as np

from safe_autonomy_simulation.entity import Entity


class Simulator:
    """Simulator base class for building simulations

    Each simulator contains a set of entities that are updated at each time step.

    Simulations update in two stages:
    1. Entities step forward in time, updating their state vectors via their individual step() methods.
    Each entity is responsible for updating its own state vector.
    2. The simulation state is updated based on the new entity state vectors via the update() method.
    During this stage entity *state vectors* can be read but **not modified**. Entity object *attributes*
    can be modified during this stage. A common use case for the update() method is to handle entity interactions. 

    Parameters
    ----------
    frame_rate : float
        simulation frame rate in hertz (Hz)
    entities : list
        list of top-level simulation entities (no child entities)
    """

    def __init__(self, frame_rate: float, entities: list[Entity]):
        assert frame_rate > 0, "Frame rate must be greater than 0"
        for entity in entities:
            assert entity.parent is None, "Entities must be top-level entities with no parent"
        self._frame_rate = frame_rate
        self._sim_time = 0
        self._entities = {entity.name: entity for entity in entities}

    def reset(self):
        """
        Reset the simulation to an initial state
        """
        self._sim_time = 0
        for _, entity in self.entities.items():
            entity.reset()

    def step(self):
        """
        Move the simulation forward one frame

        Simulations update in two stages:
        1.  Entities step forward in time, updating their state vectors via their individual step() methods.
        2.  The simulation state is updated based on the new entity state vectors via the update() method.
            - During this stage entity *state vectors* can be read but **not modified**.
            - Entity object *attributes* can be modified during this stage.
            - A common use case for the update() method is to handle entity interactions.
        """
        step_size = 1 / self.frame_rate
        self._sim_time += step_size
        for _, entity in self.entities.items():
            entity.step(step_size=step_size)
        self.update()

    def update(self):
        """
        Update the simulation state based on the entity states.

        This method is called after all entities have stepped forward in time.
        It is used to update the simulation state based on the new entity states
        and can be used to handle entity interactions.

        This method can modify entity *attributes* and read entity state vectors
        but should **not** modify any entity state vectors.
        """
        pass

    def add_controls(
        self, control_dict: typing.Dict[str, typing.Union[np.ndarray, dict]]
    ):
        """Add controls to the control queues of the simulation entities

        Parameters
        ----------
        control_dict : typing.Dict[str, np.ndarray]
            dictionary of controls to be added to the control queues of the form {entity_name: control}
        """
        for e_name, e_control in control_dict.items():
            self.entities[e_name].add_control(e_control)

    @property
    def frame_rate(self) -> float:
        """Simulation frame rate in Hz

        Returns
        -------
        float
            simulation frame rate
        """
        return self._frame_rate

    @property
    def sim_time(self) -> float:
        """Current simulation time in seconds

        Equal to the number of frames that have passed divided by the frame rate

        Returns
        -------
        float
            current simulation time
        """
        return self._sim_time

    @property
    def entities(self) -> typing.Dict[str, Entity]:
        """Set of top-level simulator entities

        Returns
        -------
        dict
            top-level simulation entities dict of the form {entity_name: entity}
        """
        return self._entities
