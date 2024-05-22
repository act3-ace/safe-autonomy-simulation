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
import numpy as np
from abc import ABC, abstractmethod

from safe_autonomy_simulation.entity import Entity


class Simulator(ABC):
    """An abstract simulator class

    Parameters
    ----------
    frame_rate : float
        simulation frame rate in hertz (Hz)

    """

    def __init__(self, frame_rate: float):
        self._frame_rate = frame_rate
        self._sim_time = 0

    def reset(self):
        """
        Reset the simulation to an initial state
        """
        self._sim_time = 0

    def step(self):
        """
        Move the simulation forward one frame
        """
        self._sim_time += 1 / self.frame_rate

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


class ContinuousSimulator(Simulator):
    """A class for building continuous simulations using incremental time progression

    See https://en.wikipedia.org/wiki/Continuous_simulation for more information
    on continuous simulations.

    Parameters
    ----------
    frame_rate : float
        simulation frame rate in hertz (Hz)
    entities : dict
        simulation entities dict of the form {entity_name: entity_object}
    """

    def __init__(self, frame_rate: float, entities: typing.Dict[str, Entity]):
        super().__init__(frame_rate=frame_rate)
        self._entities = entities

    def reset(self):
        for _, entity in self.entities.items():
            entity.reset()
        super().reset()

    def step(self):
        step_size = 1 / self.frame_rate
        for _, entity in self.entities.items():
            entity.step(step_size=step_size)
        super().step()

    @property
    def entities(self) -> typing.Dict[str, Entity]:
        """Set of simulator entities

        Returns
        -------
        dict
            simulation entities dict of the form {entity_name: entity_class}
        """
        return self._entities


class ControlledContinuousSimulator(ContinuousSimulator):
    """
    A class for building continuous simulations where
    user controls can be applied at any time step

    Parameters
    ----------
    frame_rate : float
        simulation frame rate
    entities : dict
        simulation entities dict of the form {entity_name: entity_object}
    """

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
