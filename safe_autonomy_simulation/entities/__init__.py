"""This package implements various entities.

Entities are objects that exist in the simulation environment. Each entity has a state
vector that is unique to that entity. Entity states are updated at each simulation step
according to the entity's dynamics and control inputs.

All entities are instances of the Entity class, which is an abstract base class that defines the
interface for all entities in the simulation.
"""

from safe_autonomy_simulation.entities.entity import Entity
from safe_autonomy_simulation.entities.physical import PhysicalEntity
from safe_autonomy_simulation.entities.point import Point
import safe_autonomy_simulation.entities.integrator as integrator


__all__ = [
    "integrator",  # integrator entities
    "Entity",  # base entity
    "PhysicalEntity",  # physical entity with dynamics and controls
    "Point",  # point entity with three degrees of freedom
]
