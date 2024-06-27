"""Entity material properties"""

import numpy as np
import dataclasses


@dataclasses.dataclass(frozen=True)
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
