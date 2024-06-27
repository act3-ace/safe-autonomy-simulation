"""This package implements various materials.

Materials are used to define the material properties of an object in the simulation.
They can be used to define the appearance of an object, such as its color, shininess,
and reflectivity.

All materials are instances of the Material class, which is a dataclass that stores
the material properties of an object.
"""

from safe_autonomy_simulation.materials.material import Material
from safe_autonomy_simulation.materials.common import BLACK, LIGHT, METALLIC_GREY


__all__ = [
    "Material",  # base material class
    # common materials
    "BLACK",  # black material
    "LIGHT",  # light material
    "METALLIC_GREY",  # metallic grey material
]
