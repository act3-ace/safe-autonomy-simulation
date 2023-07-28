"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Simulation.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This package implements spacecraft models for use in discrete space
simulations.
"""

from safe_autonomy_simulation.spacecraft.point_model import (
    CWHSpacecraft,
    CWHSpacecraftValidator,
)
from safe_autonomy_simulation.spacecraft.sixdof_model import (
    SixDOFSpacecraft,
    SixDOFSpacecraftValidator,
)
