import math
import pytest
import numpy as np
import safe_autonomy_simulation
import safe_autonomy_simulation.sims.spacecraft.defaults


@pytest.mark.parametrize(
    "name, num_points, radius",
    [
        ("target", 10, 1),
    ],
)
def test_init_default(name, num_points, radius):
    target = safe_autonomy_simulation.sims.inspection.SixDOFTarget(
        name=name, num_points=num_points, radius=radius
    )
    assert target.name == name
    assert target.inspection_points.radius == radius
    assert np.all(target.inspection_points.priority_vector == np.zeros(3))
    assert len(target.children) == 1
    for child in target.children:
        assert isinstance(
            child, safe_autonomy_simulation.sims.inspection.InspectionPointSet
        )
    assert np.all(target.position == np.zeros(3))
    assert np.all(target.velocity == np.zeros(3))
    assert np.all(target.orientation == np.array([0, 0, 0, 1]))
    assert np.all(target.angular_velocity == np.zeros(3))
    assert (
        target.dynamics.m == safe_autonomy_simulation.sims.spacecraft.defaults.M_DEFAULT
    )
    assert (
        target.dynamics.n == safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT
    )
    assert target.dynamics.trajectory_samples == 0
    assert target.dynamics.integration_method == "RK45"
    assert (
        target.material
        == safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL
    )
    assert target.parent is None


@pytest.mark.parametrize(
    "name, num_points, radius, priority_vector, position, velocity, orientation, angular_velocity, m, n, inertia_matrix, ang_acc_limit, ang_vel_limit, inertia_wheel, acc_limit_wheel, vel_limit_wheel, thrust_control_limit, body_frame_thrust, trajectory_samples, integration_method, material, parent, half_weighted",
    [
        (
            "target",
            10,
            1,
            np.array([1, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0]),
            1,
            1,
            np.identity(3),
            0.01,
            0.03,
            1e-5,
            180,
            576,
            1.0,
            True,
            10,
            "RK45",
            safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL,
            None,
            False,
        ),
        (
            "target",
            10,
            1,
            np.array([1, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0]),
            1,
            1,
            np.identity(3),
            0.01,
            0.03,
            1e-5,
            180,
            576,
            1.0,
            True,
            10,
            "RK45",
            safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL,
            None,
            True,
        ),
    ],
)
def test_init_args(
    name,
    num_points,
    radius,
    priority_vector,
    position,
    velocity,
    orientation,
    angular_velocity,
    m,
    n,
    inertia_matrix,
    ang_acc_limit,
    ang_vel_limit,
    inertia_wheel,
    acc_limit_wheel,
    vel_limit_wheel,
    thrust_control_limit,
    body_frame_thrust,
    trajectory_samples,
    integration_method,
    material,
    parent,
    half_weighted,
):
    target = safe_autonomy_simulation.sims.inspection.SixDOFTarget(
        name=name,
        num_points=num_points,
        radius=radius,
        priority_vector=priority_vector,
        position=position,
        velocity=velocity,
        orientation=orientation,
        angular_velocity=angular_velocity,
        m=m,
        n=n,
        inertia_matrix=inertia_matrix,
        ang_acc_limit=ang_acc_limit,
        ang_vel_limit=ang_vel_limit,
        inertia_wheel=inertia_wheel,
        acc_limit_wheel=acc_limit_wheel,
        vel_limit_wheel=vel_limit_wheel,
        thrust_control_limit=thrust_control_limit,
        body_frame_thrust=body_frame_thrust,
        trajectory_samples=trajectory_samples,
        integration_method=integration_method,
        material=material,
        parent=parent,
        half_weighted=half_weighted
    )
    assert target.name == name
    assert target.inspection_points.radius == radius
    assert np.all(target.inspection_points.priority_vector == priority_vector)
    assert len(target.children) == 1
    for child in target.children:
        assert isinstance(
            child, safe_autonomy_simulation.sims.inspection.InspectionPointSet
        )
    assert np.all(target.position == position)
    assert np.all(target.velocity == velocity)
    assert np.all(target.orientation == orientation)
    assert np.all(target.angular_velocity == angular_velocity)
    assert target.dynamics.m == m
    assert target.dynamics.n == n
    assert target.dynamics.trajectory_samples == trajectory_samples
    assert target.dynamics.integration_method == integration_method
    assert target.material == material
    assert target.parent == parent

    weighted_point_count = 0

    for point in target.inspection_points.points.values():
        if point.weight > 0.0:
            weighted_point_count += 1

    max_weighted_count = math.ceil(target.inspection_points.num_points / 2) + 2 if half_weighted else target.inspection_points.num_points
    min_weighted_count = math.ceil(target.inspection_points.num_points / 2) - 2 if half_weighted else target.inspection_points.num_points

    # There is some wiggle room in number of points and weighting, so exact numbers cannot be used
    assert min_weighted_count <= weighted_point_count <= max_weighted_count
