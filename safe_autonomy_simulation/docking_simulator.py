"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Simulation.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a simulator for a spacecraft docking environment
using a 3D CWH point mass spacecraft model.
"""

import typing

from safe_autonomy_simulation.simulator import (
    ControlledDiscreteSimulator,
    DiscreteSimulatorValidator,
)
from safe_autonomy_simulation.spacecraft.point_model import CWHSpacecraft


class DockingSimulatorValidator(DiscreteSimulatorValidator):
    """A configuration validator for DockingSimulator"""

    entities: typing.Dict[str, CWHSpacecraft]

    class Config:
        """Allows arbitrary parameter types"""

        arbitrary_types_allowed = True


class DockingSimulator(ControlledDiscreteSimulator):
    """
    A discrete simulation of spacecraft docking using
    3D CWH point mass spacecraft models.
    """

    @property
    def get_sim_validator(self):
        return DockingSimulatorValidator
