import pytest
import numpy as np
import safe_autonomy_simulation
import safe_autonomy_simulation.sims.spacecraft.defaults
import safe_autonomy_simulation.sims.spacecraft.sixdof_model


def test_init_default():
    spacecraft = safe_autonomy_simulation.sims.spacecraft.SixDOFSpacecraft(name="test")
    assert spacecraft.name == "test"
    assert np.all(spacecraft.position == np.zeros(3))
    assert np.all(spacecraft.velocity == np.zeros(3))
    assert np.all(spacecraft.orientation == np.array([0, 0, 0, 1]))
    assert np.all(spacecraft.angular_velocity == np.zeros(3))
    assert np.all(spacecraft.state == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
    assert (
        spacecraft.dynamics.m
        == safe_autonomy_simulation.sims.spacecraft.defaults.M_DEFAULT
    )
    assert np.all(
        spacecraft.dynamics.inertia_matrix
        == safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_MATRIX_DEFAULT
    )
    assert (
        spacecraft.dynamics.n
        == safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT
    )
    expected_ang_acc_limit = np.zeros((3,))
    expected_control_limit = np.zeros((6,))
    for i in range(3):
        expected_ang_acc_limit[i] = min(
            safe_autonomy_simulation.sims.spacecraft.sixdof_model.number_list_to_np(
                safe_autonomy_simulation.sims.spacecraft.defaults.ANG_ACC_LIMIT_DEFAULT,
                shape=(3,),
            )[i],
            safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_WHEEL_DEFAULT
            * safe_autonomy_simulation.sims.spacecraft.defaults.ACC_LIMIT_WHEEL_DEFAULT
            / safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_MATRIX_DEFAULT[
                i, i
            ],
        )
        expected_control_limit[i] = (
            safe_autonomy_simulation.sims.spacecraft.defaults.THRUST_CONTROL_LIMIT_DEFAULT
        )
        expected_control_limit[i + 3] = (
            expected_ang_acc_limit[i]
            * safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_MATRIX_DEFAULT[
                i, i
            ]
        )
    assert spacecraft.dynamics.trajectory_samples == 0
    assert spacecraft.dynamics.integration_method == "RK45"
    assert spacecraft.dynamics.body_frame_thrust
    assert isinstance(
        spacecraft.control_queue, safe_autonomy_simulation.controls.ControlQueue
    )
    assert np.all(spacecraft.control_queue.default_control == np.zeros(6))
    assert np.all(spacecraft.control_queue.control_min == -expected_control_limit)
    assert np.all(spacecraft.control_queue.control_max == expected_control_limit)
    assert spacecraft.parent is None
    assert len(spacecraft.children) == 0
    assert (
        spacecraft.material
        == safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL
    )


@pytest.mark.parametrize(
    "name, position, velocity, orientation, angular_velocity, m, inertia_matrix, n, ang_acc_limit, ang_vel_limit, trajectory_samples, integration_method, material, parent, children, body_frame_thrust, thrust_control_limit, inertia_wheel, acc_limit_wheel, vel_limit_wheel",
    [
        (
            "test",
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([0, 0, 0, 1]),
            np.array([7, 8, 9]),
            10,
            np.array([[11, 0, 0], [0, 12, 0], [0, 0, 13]]),
            14,
            np.array([15.0, 16.0, 17.0]),
            np.array([18.0, 19.0, 20.0]),
            21,
            "RK45",
            safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL,
            None,
            [],
            True,
            22,
            23,
            24,
            25,
        ),
    ],
)
def test_init_args(
    name,
    position,
    velocity,
    orientation,
    angular_velocity,
    m,
    inertia_matrix,
    n,
    ang_acc_limit,
    ang_vel_limit,
    trajectory_samples,
    integration_method,
    material,
    parent,
    children,
    body_frame_thrust,
    thrust_control_limit,
    inertia_wheel,
    acc_limit_wheel,
    vel_limit_wheel,
):
    spacecraft = safe_autonomy_simulation.sims.spacecraft.SixDOFSpacecraft(
        name=name,
        position=position,
        velocity=velocity,
        orientation=orientation,
        angular_velocity=angular_velocity,
        m=m,
        inertia_matrix=inertia_matrix,
        n=n,
        ang_acc_limit=ang_acc_limit,
        ang_vel_limit=ang_vel_limit,
        trajectory_samples=trajectory_samples,
        integration_method=integration_method,
        material=material,
        parent=parent,
        children=children,
        body_frame_thrust=body_frame_thrust,
        thrust_control_limit=thrust_control_limit,
        inertia_wheel=inertia_wheel,
        acc_limit_wheel=acc_limit_wheel,
        vel_limit_wheel=vel_limit_wheel,
    )
    assert spacecraft.name == name
    assert np.all(spacecraft.position == position)
    assert np.all(spacecraft.velocity == velocity)
    assert np.all(spacecraft.orientation == orientation)
    assert np.all(spacecraft.angular_velocity == angular_velocity)
    assert np.all(
        spacecraft.state
        == np.concatenate(
            (position, velocity, orientation, angular_velocity)
        )
    )
    assert spacecraft.dynamics.m == m
    assert np.all(spacecraft.dynamics.inertia_matrix == inertia_matrix)
    assert spacecraft.dynamics.n == n
    expected_ang_acc_limit = np.zeros((3,))
    expected_control_limit = np.zeros((6,))
    for i in range(3):
        expected_ang_acc_limit[i] = min(
            safe_autonomy_simulation.sims.spacecraft.sixdof_model.number_list_to_np(
                ang_acc_limit,
                shape=(3,),
            )[i],
            inertia_wheel * acc_limit_wheel / inertia_matrix[i, i],
        )
        expected_control_limit[i] = thrust_control_limit
        expected_control_limit[i + 3] = expected_ang_acc_limit[i] * inertia_matrix[i, i]
    assert spacecraft.dynamics.trajectory_samples == trajectory_samples
    assert spacecraft.dynamics.integration_method == integration_method
    assert spacecraft.dynamics.body_frame_thrust == body_frame_thrust
    assert spacecraft.material == material
    assert spacecraft.parent == parent
    assert len(spacecraft.children) == len(children)
    for child in children:
        assert child in spacecraft.children
    assert isinstance(
        spacecraft.control_queue, safe_autonomy_simulation.controls.ControlQueue
    )
    assert np.all(spacecraft.control_queue.default_control == np.zeros(6))
    assert np.all(spacecraft.control_queue.control_min == -expected_control_limit)
    assert np.all(spacecraft.control_queue.control_max == expected_control_limit)


@pytest.mark.parametrize(
    "lead",
    [
        (safe_autonomy_simulation.sims.spacecraft.SixDOFSpacecraft(name="test")),
        None,
    ],
)
def test_lead(lead):
    spacecraft = safe_autonomy_simulation.sims.spacecraft.SixDOFSpacecraft(name="test")
    spacecraft.lead = lead
    assert spacecraft.lead == lead
