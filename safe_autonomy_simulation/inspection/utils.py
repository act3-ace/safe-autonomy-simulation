"""
General utility functions for the inspection module
"""

import math
import numpy as np
from typing import Union, Tuple
from safe_autonomy_simulation.inspection.inspection_points import Point
from safe_autonomy_simulation.entity import PhysicalEntity


AVG_EARTH_TO_SUN_DIST = 150000000000  # meters


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


def sample_az_elev() -> Tuple[float, float]:
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


def points_on_sphere_fibonacci(self, num_points: int, radius: float) -> list:
    """
    Generate a set of equidistant points on sphere using the
    Fibonacci Sphere algorithm: https://arxiv.org/pdf/0912.4540.pdf

    Parameters
    ----------
    num_points: int
        number of points to attempt to place on a sphere
    radius: float
        radius of the sphere

    Returns
    -------
    points: list
        Set of equidistant points on sphere in cartesian coordinates
    """
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        r = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * r
        z = math.sin(theta) * r

        points.append(radius * np.array([x, y, z]))

    return points


def points_on_sphere_cmu(self, num_points: int, radius: float) -> list:
    """
    Generate a set of equidistant points on a sphere using the algorithm
    in https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf. Number
    of points may not be exact.

    Mostly the same as CMU algorithm, most important tweak is that the constant "a" should not depend on r
    (Paper assumed r = 1)

    Parameters
    ----------
    num_points: int
        number of points to attempt to place on a sphere
    radius: float
        radius of the sphere

    Returns
    -------
    points: list
        Set of equidistant points on sphere in cartesian coordinates
    """
    points = []

    a = 4.0 * math.pi * (1 / num_points)
    d = math.sqrt(a)
    m_theta = int(round(math.pi / d))
    d_theta = math.pi / m_theta
    d_phi = a / d_theta

    for m in range(0, m_theta):
        theta = math.pi * (m + 0.5) / m_theta
        m_phi = int(round(2.0 * math.pi * math.sin(theta) / d_phi))
        for n in range(0, m_phi):
            phi = 2.0 * math.pi * n / m_phi

            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)

            points.append(np.array([x, y, z]))

    return points


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


def sphere_intersect(
    center: np.ndarray, radius: float, ray_origin: np.ndarray, ray_direction: np.ndarray
) -> Union[float, None]:
    """
    Get distance to between ray origin and intersection between ray and sphere. If no intersection, return None.

    Parameters
    ----------
    center: np.ndarray
        center of sphere
    radius: float
        radius of sphere
    ray_origin: np.ndarray
        origin of ray
    ray_direction: np.ndarray
        direction of ray

    Returns
    -------
    Union[float, None]
        distance to intersection point or None if no intersection
    """
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius**2
    delta = b**2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    # No intersection
    return None


def is_illuminated(point: Point, light: PhysicalEntity, radius: float) -> bool:
    """
    Check illumination status of a point on a spacecraft situated at the origin (CWH dynamics)

    Parameters
    ----------
    point: Point
        point on origin-situated spacecraft to check for illumination
    light: PhysicalEntity
        light entity
    radius: float
        radius of spacecraft in meters

    Returns
    -------
    bool
        point illumination status, True if illuminated, False if not
    """

    # Spacecraft position is origin [cwh dynamics]
    center = [0, 0, 0]
    normal_to_surface = normalize(point.position)
    # Get a point slightly off the surface of the sphere so don't detect surface as an intersection
    shifted_point = point.position + 1e-5 * normal_to_surface
    light_position = light.position
    intersection_to_light = normalize(light_position - shifted_point)

    intersect_var = sphere_intersect(
        center, radius, shifted_point, intersection_to_light
    )

    bool_val = False
    # No intersection means that the point in question is illuminated in some capacity
    # (i.e. the point on the chief is not blocked by the chief itself)
    if intersect_var is None:
        bool_val = True

    return bool_val


def is_illuminated_rgb(
    rgb: np.ndarray,
    rgb_min: np.ndarray = np.array([0.8, 0.8, 0.8]),
    rgb_max: np.ndarray = np.array([0.12, np.inf, np.inf]),
) -> bool:
    """
    Determine if given RGB value is sufficient for illumination
    given minimum and maximum RGB values.

    Parameters
    ----------
    rgb: np.ndarray
        3x1 array containing RGB value
    rgb_min: np.ndarray, optional
        minimum RGB value for illumination, by default [0.8, 0.8, 0.8]
    rgb_max: np.ndarray, optional
        maximum RGB value for illumination, by default [0.12, np.inf, np.inf]

    Returns
    -------
    bool:
        whether RGB value is sufficient for illumination, True if sufficient, False if not
    """
    rgb_bool = np.greater(rgb, rgb_min).all() and np.less(rgb, rgb_max).all()
    return rgb_bool
