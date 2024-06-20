import pytest
import numpy as np
import safe_autonomy_simulation


@pytest.mark.parametrize(
    "num_points, radius",
    [
        (2, 1),
        (10, 1),
        (100, 1),
        (1000, 1),
        (2, 10),
        (10, 10),
        (100, 10),
        (1000, 10),
    ],
)
def test_points_on_sphere_fibonacci(num_points, radius):
    points = safe_autonomy_simulation.sims.inspection.utils.sphere.points_on_sphere_fibonacci(
        num_points=num_points, radius=radius
    )
    assert len(points) == num_points
    for point in points:
        assert np.allclose(np.linalg.norm(point), radius)


@pytest.mark.parametrize(
    "num_points, radius",
    [
        (2, 1),
        (10, 1),
        (100, 1),
        (1000, 1),
        (2, 10),
        (10, 10),
        (100, 10),
        (1000, 10),
    ],
)
def test_points_on_sphere_cmu(num_points, radius):
    points = safe_autonomy_simulation.sims.inspection.utils.sphere.points_on_sphere_cmu(
        num_points=num_points, radius=radius
    )
    for point in points:
        assert np.allclose(np.linalg.norm(point), radius)


# TODO: Fix sphere_intersect test
# @pytest.mark.parametrize(
#     "center, radius, ray_origin, ray_direction, expected",
#     [
#         (np.array([0, 0, 0]), 1, np.array([2, 0, 0]), np.array([0, 0, 0]), 1),
#         (np.array([0, 0, 0]), 1, np.array([0, 2, 0]), np.array([0, 0, 0]), 1),
#         (np.array([0, 0, 0]), 1, np.array([0, 0, 2]), np.array([0, 0, 0]), 1),
#         (np.array([0, 0, 0]), 1, np.array([1, 1, 1]), np.array([0, 0, 0]), np.sqrt(3) - 1),
#     ],
# )
# def test_sphere_intersect(center, radius, ray_origin, ray_direction, expected):
#     distance = safe_autonomy_simulation.sims.inspection.utils.sphere.sphere_intersect(
#         center=center, radius=radius, ray_origin=ray_origin, ray_direction=ray_direction
#     )
#     assert distance == expected
#     # Test no intersection
#     distance = safe_autonomy_simulation.sims.inspection.utils.sphere.sphere_intersect(
#         center=center,
#         radius=radius,
#         ray_origin=ray_origin,
#         ray_direction=-ray_direction,
#     )
#     assert distance is None
