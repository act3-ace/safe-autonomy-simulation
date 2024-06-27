""" Sphere utility functions for inspection simulation. """

import math
import typing
import numpy as np


def points_on_sphere_fibonacci(num_points: int, radius: float) -> list:
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
    assert num_points > 1, "Number of points must be greater than 1"
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


def points_on_sphere_cmu(num_points: int, radius: float) -> list:
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
    assert num_points > 1, "Number of points must be greater than 1"
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


def sphere_intersect(
    center: np.ndarray, radius: float, ray_origin: np.ndarray, ray_direction: np.ndarray
) -> typing.Union[float, None]:
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
