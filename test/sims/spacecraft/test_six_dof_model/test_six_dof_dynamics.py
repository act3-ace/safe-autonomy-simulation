import pytest
import numpy as np
import safe_autonomy_simulation
import scipy.spatial.transform as transform
import safe_autonomy_simulation.sims.spacecraft.sixdof_model
import safe_autonomy_simulation.sims.spacecraft.defaults


def test_init_default():
    dynamics = safe_autonomy_simulation.sims.spacecraft.sixdof_model.SixDOFDynamics()
    assert dynamics.m == safe_autonomy_simulation.sims.spacecraft.defaults.M_DEFAULT
    assert np.all(
        dynamics.inertia_matrix
        == safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_MATRIX_DEFAULT
    )
    assert dynamics.n == safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT
    assert dynamics.trajectory_samples == 0
    assert dynamics.integration_method == "RK45"
    assert dynamics.body_frame_thrust
    assert not dynamics.use_jax
    ang_vel_limit = (
        safe_autonomy_simulation.sims.spacecraft.sixdof_model.number_list_to_np(
            safe_autonomy_simulation.sims.spacecraft.defaults.ANG_VEL_LIMIT_DEFAULT,
            shape=(3,),
        )
    )
    assert np.all(
        dynamics.state_min
        == np.array(
            [
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -ang_vel_limit[0],
                -ang_vel_limit[1],
                -ang_vel_limit[2],
            ]
        )
    )
    assert np.all(
        dynamics.state_max
        == np.array(
            [
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                ang_vel_limit[0],
                ang_vel_limit[1],
                ang_vel_limit[2],
            ]
        )
    )
    n = safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT
    assert np.all(
        dynamics.A
        == np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [3 * n**2, 0, 0, 0, 2 * n, 0],
                [0, 0, 0, -2 * n, 0, 0],
                [0, 0, -(n**2), 0, 0, 0],
            ],
            dtype=np.float64,
        )
    )
    m = safe_autonomy_simulation.sims.spacecraft.defaults.M_DEFAULT
    assert np.all(
        dynamics.B
        == np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1 / m, 0, 0],
                [0, 1 / m, 0],
                [0, 0, 1 / m],
            ],
            dtype=np.float64,
        )
    )


@pytest.mark.parametrize(
    "m, inertia_matrix, ang_acc_limit, ang_vel_limit, n, body_frame_thrust, trajectory_samples, state_max, state_min, state_dot_max, state_dot_min, integration_method, use_jax",
    [
        (
            1,
            np.ones((3, 3)),
            [1, 1, 1],
            [1, 1, 1],
            1,
            True,
            1,
            3,
            2,
            3,
            2,
            "RK45",
            True,
        ),
    ],
)
def test_init_args(
    m,
    inertia_matrix,
    ang_acc_limit,
    ang_vel_limit,
    n,
    body_frame_thrust,
    trajectory_samples,
    state_max,
    state_min,
    state_dot_max,
    state_dot_min,
    integration_method,
    use_jax,
):
    dynamics = safe_autonomy_simulation.sims.spacecraft.sixdof_model.SixDOFDynamics(
        m=m,
        inertia_matrix=inertia_matrix,
        ang_acc_limit=ang_acc_limit,
        ang_vel_limit=ang_vel_limit,
        n=n,
        body_frame_thrust=body_frame_thrust,
        trajectory_samples=trajectory_samples,
        state_max=state_max,
        state_min=state_min,
        state_dot_max=state_dot_max,
        state_dot_min=state_dot_min,
        integration_method=integration_method,
        use_jax=use_jax,
    )
    assert dynamics.m == m
    assert np.all(dynamics.inertia_matrix == inertia_matrix)
    assert dynamics.n == n
    assert dynamics.body_frame_thrust == body_frame_thrust
    assert dynamics.trajectory_samples == trajectory_samples
    assert np.all(dynamics.state_max == state_max)
    assert np.all(dynamics.state_min == state_min)
    assert np.all(dynamics.state_dot_max == state_dot_max)
    assert np.all(dynamics.state_dot_min == state_dot_min)
    assert dynamics.integration_method == integration_method
    assert dynamics.use_jax == use_jax
    ang_vel_limit = (
        safe_autonomy_simulation.sims.spacecraft.sixdof_model.number_list_to_np(
            ang_vel_limit, shape=(3,), dtype=np.float64
        )
    )
    assert np.all(
        dynamics.A
        == np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [3 * n**2, 0, 0, 0, 2 * n, 0],
                [0, 0, 0, -2 * n, 0, 0],
                [0, 0, -(n**2), 0, 0, 0],
            ],
            dtype=np.float64,
        )
    )
    assert np.all(
        dynamics.B
        == np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1 / m, 0, 0],
                [0, 1 / m, 0],
                [0, 0, 1 / m],
            ],
            dtype=np.float64,
        )
    )


@pytest.mark.parametrize(
    "state",
    [
        np.array([1, 2, 3, 4, 5, 6, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64),
    ],
)
def test_state_transition_system(state):
    dynamics = safe_autonomy_simulation.sims.spacecraft.sixdof_model.SixDOFDynamics()
    state_derivative = dynamics.state_transition_system(state)
    assert state_derivative.shape == state.shape
    x, y, z, x_dot, y_dot, z_dot, q1, q2, q3, q4, wx, wy, wz = state
    pos_vel_state_vec = np.array([x, y, z, x_dot, y_dot, z_dot], dtype=np.float64)
    pos_vel_derivative = dynamics.A @ pos_vel_state_vec
    q_derivative = np.zeros((4,))
    w_derivative = np.zeros((3,))
    q_derivative[0] = 0.5 * (q4 * wx - q3 * wy + q2 * wz)
    q_derivative[1] = 0.5 * (q3 * wx + q4 * wy - q1 * wz)
    q_derivative[2] = 0.5 * (-q2 * wx + q1 * wy + q4 * wz)
    q_derivative[3] = 0.5 * (-q1 * wx - q2 * wy - q3 * wz)
    w_derivative[0] = (
        1
        / dynamics.inertia_matrix[0, 0]
        * ((dynamics.inertia_matrix[1, 1] - dynamics.inertia_matrix[2, 2]) * wy * wz)
    )
    w_derivative[1] = (
        1
        / dynamics.inertia_matrix[1, 1]
        * ((dynamics.inertia_matrix[2, 2] - dynamics.inertia_matrix[0, 0]) * wx * wz)
    )
    w_derivative[2] = (
        1
        / dynamics.inertia_matrix[2, 2]
        * ((dynamics.inertia_matrix[0, 0] - dynamics.inertia_matrix[1, 1]) * wx * wy)
    )

    # Form derivative array
    expected_state_derivative = np.array(
        [
            pos_vel_derivative[0],
            pos_vel_derivative[1],
            pos_vel_derivative[2],
            q_derivative[0],
            q_derivative[1],
            q_derivative[2],
            q_derivative[3],
            pos_vel_derivative[3],
            pos_vel_derivative[4],
            pos_vel_derivative[5],
            w_derivative[0],
            w_derivative[1],
            w_derivative[2],
        ],
        dtype=np.float32,
    )
    assert np.allclose(state_derivative, expected_state_derivative)


@pytest.mark.parametrize(
    "state",
    [
        np.array([1, 2, 3, 4, 5, 6, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64),
    ],
)
def test_state_transition_input(state):
    dynamics = safe_autonomy_simulation.sims.spacecraft.sixdof_model.SixDOFDynamics()
    st_input = dynamics.state_transition_input(state)
    quat = state[6:10]
    w_derivative = np.array(
        [
            [1 / dynamics.inertia_matrix[0, 0], 0, 0],
            [0, 1 / dynamics.inertia_matrix[1, 1], 0],
            [0, 0, 1 / dynamics.inertia_matrix[2, 2]],
        ]
    )
    if dynamics.body_frame_thrust:
        r1 = 1 / dynamics.m * dynamics.apply_quat(np.array([1, 0, 0]), quat)
        r2 = 1 / dynamics.m * dynamics.apply_quat(np.array([0, 1, 0]), quat)
        r3 = 1 / dynamics.m * dynamics.apply_quat(np.array([0, 0, 1]), quat)
        vel_derivative = np.array(
            [[r1[0], r2[0], r3[0]], [r1[1], r2[1], r3[1]], [r1[2], r2[2], r3[2]]]
        )
    else:
        vel_derivative = dynamics.B[3:6, :]
    expected_st_input = np.vstack(
        (
            np.zeros((7, 6)),
            np.hstack((vel_derivative, np.zeros(vel_derivative.shape))),
            np.hstack((np.zeros(w_derivative.shape), w_derivative)),
        )
    )
    assert np.allclose(st_input, expected_st_input)


@pytest.mark.parametrize(
    "x, quat",
    [
        (np.array([1, 2, 3]), np.array([0, 0, 0, 1])),
    ],
)
def test_apply_quat(x, quat):
    dynamics = safe_autonomy_simulation.sims.spacecraft.sixdof_model.SixDOFDynamics()
    result = dynamics.apply_quat(x, quat)
    assert result.shape == x.shape
    assert np.allclose(result, transform.Rotation.from_quat(quat).apply(x))


@pytest.mark.parametrize(
    "r, q",
    [
        (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])),
    ],
)
def test_hamilton_product(r, q):
    dynamics = safe_autonomy_simulation.sims.spacecraft.sixdof_model.SixDOFDynamics()
    result = dynamics.hamilton_product(r, q)
    assert result.shape == r.shape
    assert np.all(
        result
        == np.array(
            [
                r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
                r[0] * q[1] + r[1] * q[0] + r[2] * q[3] - r[3] * q[2],
                r[0] * q[2] - r[1] * q[3] + r[2] * q[0] + r[3] * q[1],
                r[0] * q[3] + r[1] * q[2] - r[2] * q[1] + r[3] * q[0],
            ]
        )
    )
