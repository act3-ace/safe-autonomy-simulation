import pytest
import numpy as np
import safe_autonomy_simulation
import safe_autonomy_simulation.sims.spacecraft.defaults
import safe_autonomy_simulation.sims.spacecraft.rotational_model


def test_init_default():
    dynamics = (
        safe_autonomy_simulation.sims.spacecraft.rotational_model.CWHRotation2dDynamics()
    )
    assert dynamics.m == safe_autonomy_simulation.sims.spacecraft.defaults.M_DEFAULT
    assert dynamics.n == safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT
    assert (
        dynamics.inertia
        == safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_DEFAULT
    )
    assert (
        dynamics.ang_acc_limit
        == safe_autonomy_simulation.sims.spacecraft.defaults.ANG_ACC_LIMIT_DEFAULT
    )
    assert (
        dynamics.ang_vel_limit
        == safe_autonomy_simulation.sims.spacecraft.defaults.ANG_VEL_LIMIT_DEFAULT
    )
    assert dynamics.trajectory_samples == 0
    assert np.all(
        dynamics.state_min
        == np.array(
            [
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -safe_autonomy_simulation.sims.spacecraft.defaults.ANG_VEL_LIMIT_DEFAULT,
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
                safe_autonomy_simulation.sims.spacecraft.defaults.ANG_VEL_LIMIT_DEFAULT,
            ]
        )
    )
    assert np.all(
        dynamics.state_dot_min
        == np.array(
            [
                -np.inf,
                -np.inf,
                -safe_autonomy_simulation.sims.spacecraft.defaults.ANG_VEL_LIMIT_DEFAULT,
                -np.inf,
                -np.inf,
                -safe_autonomy_simulation.sims.spacecraft.defaults.ANG_ACC_LIMIT_DEFAULT,
            ]
        )
    )
    assert np.all(
        dynamics.state_dot_max
        == np.array(
            [
                np.inf,
                np.inf,
                safe_autonomy_simulation.sims.spacecraft.defaults.ANG_VEL_LIMIT_DEFAULT,
                np.inf,
                np.inf,
                safe_autonomy_simulation.sims.spacecraft.defaults.ANG_ACC_LIMIT_DEFAULT,
            ]
        )
    )
    n = safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT
    assert np.all(
        dynamics.A
        == np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [3 * n**2, 0, 0, 2 * n],
                [0, 0, -2 * n, 0],
            ],
            dtype=np.float64,
        )
    )
    m = safe_autonomy_simulation.sims.spacecraft.defaults.M_DEFAULT
    assert np.all(
        dynamics.B
        == np.array(
            [
                [0, 0],
                [0, 0],
                [1 / m, 0],
                [0, 1 / m],
            ],
            dtype=np.float64,
        )
    )


@pytest.mark.parametrize(
    "m, inertia, ang_acc_limit, ang_vel_limit, n, trajectory_samples, state_min, state_max, state_dot_min, state_dot_max, integration_method",
    [
        (
            1,
            1,
            1,
            1,
            1,
            1,
            np.zeros(6),
            np.ones(6),
            np.zeros(6),
            np.ones(6),
            "Euler",
        ),
        (
            2,
            2,
            2,
            2,
            2,
            2,
            np.ones(6),
            np.zeros(6),
            np.ones(6),
            np.zeros(6),
            "RK45",
        ),
    ],
)
def test_init_args(
    m,
    inertia,
    ang_acc_limit,
    ang_vel_limit,
    n,
    trajectory_samples,
    state_min,
    state_max,
    state_dot_min,
    state_dot_max,
    integration_method,
):
    dynamics = (
        safe_autonomy_simulation.sims.spacecraft.rotational_model.CWHRotation2dDynamics(
            m=m,
            n=n,
            inertia=inertia,
            ang_acc_limit=ang_acc_limit,
            ang_vel_limit=ang_vel_limit,
            trajectory_samples=trajectory_samples,
            state_min=state_min,
            state_max=state_max,
            state_dot_min=state_dot_min,
            state_dot_max=state_dot_max,
            integration_method=integration_method,
        )
    )
    assert dynamics.m == m
    assert dynamics.n == n
    assert dynamics.inertia == inertia
    assert dynamics.ang_acc_limit == ang_acc_limit
    assert dynamics.ang_vel_limit == ang_vel_limit
    assert dynamics.trajectory_samples == trajectory_samples
    assert np.all(dynamics.state_min == state_min)
    assert np.all(dynamics.state_max == state_max)
    assert np.all(dynamics.state_dot_min == state_dot_min)
    assert np.all(dynamics.state_dot_max == state_dot_max)
    assert dynamics.integration_method == integration_method
    assert np.all(
        dynamics.A
        == np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [3 * n**2, 0, 0, 2 * n],
                [0, 0, -2 * n, 0],
            ],
            dtype=np.float64,
        )
    )
    assert np.all(
        dynamics.B
        == np.array(
            [
                [0, 0],
                [0, 0],
                [1 / m, 0],
                [0, 1 / m],
            ],
            dtype=np.float64,
        )
    )


@pytest.mark.parametrize(
    "state",
    [
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([1, 2, 3, 4, 5, 6]),
        np.array([-1, -2, -3, -4, -5, -6]),
        np.array([1, -2, 3, -4, 5, -6]),
    ],
)
def test_state_transition_system(state):
    dynamics = (
        safe_autonomy_simulation.sims.spacecraft.rotational_model.CWHRotation2dDynamics()
    )
    state_dot = dynamics.state_transition_system(state)
    assert state_dot.shape == (6,)
    x, y, xdot, ydot, theta, wz = state
    expected_state_dot = dynamics.A @ np.array([x, y, xdot, ydot])
    expected_state_dot = np.array(
        [
            expected_state_dot[0],
            expected_state_dot[1],
            wz,
            expected_state_dot[2],
            expected_state_dot[3],
            0,
        ],
        dtype=np.float32,
    )
    assert np.allclose(state_dot, expected_state_dot)


@pytest.mark.parametrize(
    "state",
    [
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([1, 2, 3, 4, 5, 6]),
        np.array([-1, -2, -3, -4, -5, -6]),
        np.array([1, -2, 3, -4, 5, -6]),
    ],
)
def test_state_transition_input(state):
    dynamics = (
        safe_autonomy_simulation.sims.spacecraft.rotational_model.CWHRotation2dDynamics()
    )
    st_input = dynamics.state_transition_input(state)
    x, y, xdot, ydot, theta, wz = state
    expected_st_input = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [np.cos(theta) / dynamics.m, -np.sin(theta) / dynamics.m, 0],
            [np.sin(theta) / dynamics.m, np.cos(theta) / dynamics.m, 0],
            [0, 0, 1 / dynamics.inertia],
        ],
        dtype=np.float32,
    )
    assert np.allclose(st_input, expected_st_input)
