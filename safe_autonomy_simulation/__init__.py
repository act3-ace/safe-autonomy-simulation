"""
The `safe-autonomy-sims` package contains various modules designed to help implement
continuous simulations in python.
"""

from safe_autonomy_simulation import (
    dynamics,
    entities,
    materials,
    controls,
    sims,
    utils,
    jax,
)
from safe_autonomy_simulation.dynamics import Dynamics
from safe_autonomy_simulation.entities import Entity
from safe_autonomy_simulation.materials import Material
from safe_autonomy_simulation.controls import ControlQueue
from safe_autonomy_simulation.simulator import Simulator


__all__ = [
    # core classes
    "Dynamics",  # base dynamics class
    "Entity",  # base entity class
    "Material",  # base material class
    "ControlQueue",  # base control queue class
    "Simulator",  # base simulator class
    # modules
    "dynamics",
    "entities",
    "materials",
    "controls",
    "sims",
    "utils",
    "jax",
]
