"""
This module contains the Material class, which is used to define the material properties
of an object in the simulation.
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """Material properties of a simulation entity

    Parameters
    ----------
    specular: np.ndarray
        RGB specular reflection coefficients
    diffuse: np.ndarray
        RGB diffuse reflection coefficients
    ambient: np.ndarray
        RGB ambient reflection coefficients
    shininess: float
        shininess coefficient
    reflection: float
        reflection coefficient
    """

    specular: np.ndarray
    diffuse: np.ndarray
    ambient: np.ndarray
    shininess: float
    reflection: float


# Black material
BLACK = Material(
    specular=np.array([0.0, 0.0, 0.0]),
    diffuse=np.array([0.0, 0.0, 0.0]),
    ambient=np.array([0.0, 0.0, 0.0]),
    shininess=0.0,
    reflection=0.0,
)


# Light material (point light source)
LIGHT = Material(
    specular=np.array([1.0, 1.0, 1.0]),
    diffuse=np.array([1.0, 1.0, 1.0]),
    ambient=np.array([1.0, 1.0, 1.0]),
    shininess=0.0,
    reflection=0.0,
)


# Metallic grey material
METALLIC_GREY = Material(
    specular=np.array([1, 1, 1]),
    diffuse=np.array([0.1, 0.1, 0.1]),
    ambient=np.array([0.4, 0.4, 0.4]),
    shininess=100.0,
    reflection=0.0,
)
