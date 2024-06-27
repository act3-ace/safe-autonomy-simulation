""" Helper functions for vector operations """

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    """
    Helper function to normalize a vector

    $v_n = v / ||v||$

    Parameters
    ----------
    vector: np.ndarray
        vector to normalize

    Returns
    -------
    np.ndarray
        normalized vector
    """
    return vector / np.linalg.norm(vector)
