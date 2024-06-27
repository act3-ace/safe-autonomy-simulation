"""Simulator base class for building simulations"""

import safe_autonomy_simulation.entities as e
import safe_autonomy_simulation.utils as utils


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

    def __init__(self, frame_rate: float, entities: list[e.Entity]):
        assert frame_rate > 0, "Frame rate must be greater than 0"
        for entity in entities:
            assert (
                entity.parent is None
            ), "Entities must be top-level entities with no parent"
        self._frame_rate = frame_rate
        self._sim_time = 0
        self._entities = utils.TypedSet(type=e.Entity, elements=entities)

    def reset(self):
        """
        Reset the simulation to an initial state
        """
        self._sim_time = 0
        for entity in self.entities:
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
        for entity in self.entities:
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
    def entities(self) -> utils.TypedSet[e.Entity]:
        """Set of top-level simulator entities

        Returns
        -------
        set
            top-level simulation entities set
        """
        return self._entities
