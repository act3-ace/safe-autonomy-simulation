"""This package implements spacecraft models for use in discrete space simulations."""

from safe_autonomy_simulation.sims.spacecraft.point_model import CWHSpacecraft
from safe_autonomy_simulation.sims.spacecraft.sixdof_model import SixDOFSpacecraft
from safe_autonomy_simulation.sims.spacecraft.rotational_model import (
    CWHRotation2dSpacecraft,
)


__all__ = [
    "CWHSpacecraft",  # spacecraft model with three degrees of freedom
    "SixDOFSpacecraft",  # spacecraft model with six degrees of freedom
    "CWHRotation2dSpacecraft",  # spacecraft model with 2d translation and 1d rotation
]
