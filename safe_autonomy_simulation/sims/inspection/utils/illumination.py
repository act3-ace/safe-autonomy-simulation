import numpy as np
import safe_autonomy_simulation.entities as e
import safe_autonomy_simulation.sims.inspection.utils.sphere as sphere
import safe_autonomy_simulation.sims.inspection.utils.vector as vector


def is_illuminated(point: e.Point, light: e.PhysicalEntity, radius: float) -> bool:
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
    normal_to_surface = vector.normalize(point.position)
    # Get a point slightly off the surface of the sphere so don't detect surface as an intersection
    shifted_point = point.position + 1e-5 * normal_to_surface
    light_position = light.position
    intersection_to_light = vector.normalize(light_position - shifted_point)

    intersect_var = sphere.sphere_intersect(
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
    rgb_max: np.ndarray = np.array([0.9, np.inf, np.inf]),
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
        maximum RGB value for illumination, by default [0.9, np.inf, np.inf]

    Returns
    -------
    bool:
        whether RGB value is sufficient for illumination, True if sufficient, False if not
    """
    rgb_bool = np.greater(rgb, rgb_min).all() and np.less(rgb, rgb_max).all()
    return rgb_bool
