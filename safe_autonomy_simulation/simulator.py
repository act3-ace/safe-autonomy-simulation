"""An abstract simulator class for building new simulations."""

from abc import ABC, abstractmethod
from pydantic import BaseModel


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
