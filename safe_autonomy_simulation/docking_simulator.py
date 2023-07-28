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

from safe_autonomy_simulation.simulator import DiscreteSimulator, DiscreteSimulatorValidator
from safe_autonomy_simulation.spacecraft.point_model import CWHSpacecraft


class DockingSimulatorValidator(DiscreteSimulatorValidator):
    entities: typing.Dict[str, CWHSpacecraft]

    class Config:
        arbitrary_types_allowed = True


class DockingSimulator(DiscreteSimulator):
    @property
    def get_sim_validator(self):
        return DockingSimulatorValidator
