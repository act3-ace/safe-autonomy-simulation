"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Simulation.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains base simulator classes for building new simulations.
"""

import typing
from abc import ABC, abstractmethod
from pydantic import BaseModel

from safe_autonomy_simulation.base_models import BaseEntity

class SimulatorValidator(BaseModel):
    """A configuration validator for the Simulator class"""
    frame_rate: float


class Simulator(ABC):
    """An abstract simulator class for building new simulations"""
    def __init__(self, **kwargs):
        self.config = self.get_sim_validator(**kwargs)
        self._sim_time = 0

    @property
    def get_sim_validator(self):
        """
        Returns a pydantic model used for validating the simulator configuration options.

        Returns:
            SimulatorValidator
        """
        return SimulatorValidator

    def reset(self):
        """
        Reset the simulation to an initial state.
        """
        self._sim_time = 0

    def step(self):
        """
        Move the simulation forward one time step.
        """
        self._sim_time += 1 / self.frame_rate

    @abstractmethod
    def info(self):
        """
        Return info about the current state of the simulation.
        """

    @property
    def frame_rate(self):
        """
        Simulation frame rate in Hz
        """
        return self.config.frame_rate

    @property
    def sim_time(self):
        """
        Current simulation time
        """
        return self._sim_time



class DiscreteSimulatorValidator(SimulatorValidator):
    entities: typing.Dict[str, BaseEntity]

    class Config:
        arbitrary_types_allowed = True


class DiscreteSimulator(Simulator):
    @property
    def get_sim_validator(self):
        return DiscreteSimulatorValidator

    def reset(self):
        for _, entity in self.entities.items():
            entity.reset()
        super().reset()

    def step(self):
        step_size = 1 / self.frame_rate
        for _, entity in self.entities.items():
            entity.step(step_size=step_size)
        super().step()

    def info(self):
        entity_states = {entity.name: entity.state for _, entity in self.entities.items()}
        return entity_states

    @property
    def entities(self):
        """
        Set of simulator entities
        """
        return self.config.entities


class ControlledDiscreteSimulator(DiscreteSimulator):
    def add_controls(self, control_dict: dict):
        """
        Add controls to the sim entities control queues.
        Expects a dict of entity_name: control_to_add items.
        """
        for e_name, e_control in control_dict.items():
            self.entities[e_name].add_control(e_control)
