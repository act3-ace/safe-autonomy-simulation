"""
This module contains the Material class, which is used to define the material properties
of an object in the simulation.
"""

import numpy as np


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
    def __init__(
        self,
        specular: np.ndarray,
        diffuse: np.ndarray,
        ambient: np.ndarray,
        shininess: float,
        reflection: float,
    ):
        self._specular = specular
        self._diffuse = diffuse
        self._ambient = ambient
        self._shininess = shininess
        self._reflection = reflection

    @property
    def specular(self) -> float:
        """Specular reflection coefficients
        
        Returns
        -------
        np.ndarray
            RGB specular reflection coefficients
        """
        return self._specular

    @property
    def diffuse(self) -> float:
        """Diffuse reflection coefficients
        
        Returns
        -------
        np.ndarray
            RGB diffuse reflection coefficients
        """
        return self._diffuse

    @property
    def ambient(self) -> float:
        """Ambient reflection coefficients
        
        Returns
        -------
        np.ndarray
            RGB ambient reflection coefficients
        """
        return self._ambient

    @property
    def shininess(self) -> float:
        """Shininess coefficient
        
        Returns
        -------
        float
            shininess coefficient
        """
        return self._shininess

    @property
    def reflection(self) -> float:
        """Reflection coefficient
        
        Returns
        -------
        float
            reflection coefficient
        """
        return self._reflection


class Black(Material):
    """Black material properties"""
    def __init__(self):
        super().__init__(
            specular=np.array([0.0, 0.0, 0.0]),
            diffuse=np.array([0.0, 0.0, 0.0]),
            ambient=np.array([0.0, 0.0, 0.0]),
            shininess=0.0,
            reflection=0.0,
        )

class Light(Material):
    """Point light material properties"""
    def __init__(self):
        super().__init__(
            specular=np.array([1.0, 1.0, 1.0]),
            diffuse=np.array([1.0, 1.0, 1.0]),
            ambient=np.array([1.0, 1.0, 1.0]),
            shininess=0.0,
            reflection=0.0,
        )
