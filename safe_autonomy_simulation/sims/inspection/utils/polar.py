"""Polar coordinate utility functions"""

import numpy as np
import math
import typing


def get_vec_az_elev(azimuth: float, elevation: float) -> np.ndarray:
    """Get vector from azimuth and elevation angles

    Parameters
    ----------
    azimuth: float
        azimuth angle in radians
    elevation: float
        elevation angle in radians

    Returns
    -------
    np.ndarray
        vector from azimuth and elevation angles
    """
    v = np.array(
        [
            np.cos(azimuth) * np.cos(elevation),
            np.sin(azimuth) * np.cos(elevation),
            np.sin(elevation),
        ]
    )
    return v


def sample_az_elev() -> typing.Tuple[float, float]:
    """Sample azimuth and elevation angles from uniform distribution

    azimuth: [0, 2pi]
    elevation: [-pi/2, pi/2]

    Returns
    -------
    Tuple[float, float]
        tuple of azimuth and elevation angles
    """
    azimuth = np.random.uniform(0, 2 * math.pi)
    elevation = np.random.uniform(-math.pi / 2, math.pi / 2)
    return azimuth, elevation
