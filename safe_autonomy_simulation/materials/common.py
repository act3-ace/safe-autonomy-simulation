"""Common material properties"""

import numpy as np
import safe_autonomy_simulation.materials.material as material


# Black material
BLACK = material.Material(
    specular=np.array([0.0, 0.0, 0.0]),
    diffuse=np.array([0.0, 0.0, 0.0]),
    ambient=np.array([0.0, 0.0, 0.0]),
    shininess=0.0,
    reflection=0.0,
)


# Light material (point light source)
LIGHT = material.Material(
    specular=np.array([1.0, 1.0, 1.0]),
    diffuse=np.array([1.0, 1.0, 1.0]),
    ambient=np.array([1.0, 1.0, 1.0]),
    shininess=0.0,
    reflection=0.0,
)


# Metallic grey material
METALLIC_GREY = material.Material(
    specular=np.array([1, 1, 1]),
    diffuse=np.array([0.1, 0.1, 0.1]),
    ambient=np.array([0.4, 0.4, 0.4]),
    shininess=100.0,
    reflection=0.0,
)
