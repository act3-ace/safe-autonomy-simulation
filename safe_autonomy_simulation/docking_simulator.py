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

from safe_autonomy_simulation.simulator import ControlledDiscreteSimulator
from safe_autonomy_simulation.spacecraft.point_model import CWHSpacecraft


class DockingSimulator(ControlledDiscreteSimulator):
    """
    A discrete spacecraft docking simulation using 3D CWH point mass spacecraft models.

    Parameters
    ----------
    frame_rate : float
        simulation frame rate
    entities : dict[str, CWHSpacecraft]
        simulation entities dict of the form {entity_name: entity_class}

    Attributes
    ----------
    sim_time : float
        current simulation time
    frame_rate : float
        simulation frame rate
    entities : dict[str, CWHSpacecraft]
        simulation entities dict of the form {entity_name: entity_class}
    """

    def __init__(self, frame_rate: float, entities: typing.Dict[str, CWHSpacecraft]):
        super().__init__(frame_rate=frame_rate, entities=entities)
