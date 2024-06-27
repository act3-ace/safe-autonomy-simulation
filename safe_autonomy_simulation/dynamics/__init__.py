"""This package implements various dynamics models.

Dynamics models are used to define the state transition function of an entity in the simulation.
They can accept control inputs and the current state of the entity, and return the next state
of the entity after a simulation step.

All dynamics models are instances of the Dynamics class, which is an abstract base class that
defines the interface for all dynamics models in the simulation.
"""

from safe_autonomy_simulation.dynamics.dynamics import Dynamics
from safe_autonomy_simulation.dynamics.passthrough import PassThroughDynamics
from safe_autonomy_simulation.dynamics.ode import (
    ODEDynamics,
    ControlAffineODEDynamics,
    LinearODEDynamics,
)

__all__ = [
    "Dynamics",  # base dynamics model
    "PassThroughDynamics",  # dynamics model that passes through control inputs
    "ODEDynamics",  # base dynamics model that integrates ordinary differential equations
    "ControlAffineODEDynamics",  # control-affine ODE dynamics model
    "LinearODEDynamics",  # linear ODE dynamics model
]
