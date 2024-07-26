import re
import pytest
import numpy as np
import safe_autonomy_simulation


def test_init_no_args():
    dynamics = (
        safe_autonomy_simulation.entities.integrator.PointMassIntegratorDynamics()
    )
    assert dynamics.m == safe_autonomy_simulation.entities.integrator.M_DEFAULT
    assert (
        dynamics.damping == safe_autonomy_simulation.entities.integrator.DAMPING_DEFAULT
    )
    assert dynamics.trajectory_samples == 0
    assert dynamics.integration_method == "RK45"
    assert np.all(dynamics.state_min == -np.inf)
    assert np.all(dynamics.state_max == np.inf)
    assert np.all(dynamics.state_dot_min == -np.inf)
    assert np.all(dynamics.state_dot_max == np.inf)

    A_1d = np.array([[0, 1], [0, -dynamics.damping]], dtype=np.float64)
    B_1d = np.array([[0], [1 / dynamics.m]], dtype=np.float64)
    assert np.all(dynamics.A == A_1d)
    assert np.all(dynamics.B == B_1d)


@pytest.mark.parametrize(
    "m, damping, mode, trajectory_samples, state_min, state_max, state_dot_min, state_dot_max, integration_method, expected_A, expected_B",
    [
        (
            1,
            2,
            "1d",
            3,
            np.array([-1]),
            np.array([1]),
            np.array([-1]),
            np.array([1]),
            "Euler",
            np.array([[0, 1], [0, -2]], dtype=np.float64),
            np.array([[0], [1]], dtype=np.float64),
        ),
        (
            1,
            2,
            "2d",
            3,
            np.array([-1, -1]),
            np.array([1, 1]),
            np.array([-1, -1]),
            np.array([1, 1]),
            "RK45",
            np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -2, 0], [0, 0, 0, -2]]),
            np.array([[0, 0], [0, 0], [1, 0], [0, 1]]),
        ),
        (
            1,
            2,
            "3d",
            3,
            np.array([-1, -1, -1]),
            np.array([1, 1, 1]),
            np.array([-1, -1, -1]),
            np.array([1, 1, 1]),
            "RK45",
            np.array(
                [
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, -2, 0, 0],
                    [0, 0, 0, 0, -2, 0],
                    [0, 0, 0, 0, 0, -2],
                ]
            ),
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            ),
        ),
    ],
)
def test_init_args(
    m,
    damping,
    mode,
    trajectory_samples,
    state_min,
    state_max,
    state_dot_min,
    state_dot_max,
    integration_method,
    expected_A,
    expected_B,
):
    dynamics = safe_autonomy_simulation.entities.integrator.PointMassIntegratorDynamics(
        m=m,
        damping=damping,
        mode=mode,
        trajectory_samples=trajectory_samples,
        state_min=state_min,
        state_max=state_max,
        state_dot_min=state_dot_min,
        state_dot_max=state_dot_max,
        integration_method=integration_method,
    )
    assert dynamics.m == m
    assert dynamics.damping == damping
    assert dynamics.trajectory_samples == trajectory_samples
    assert dynamics.integration_method == integration_method
    assert np.all(dynamics.state_min == state_min)
    assert np.all(dynamics.state_max == state_max)
    assert np.all(dynamics.state_dot_min == state_dot_min)
    assert np.all(dynamics.state_dot_max == state_dot_max)

    assert np.all(dynamics.A == expected_A)
    assert np.all(dynamics.B == expected_B)


def test_init_invalid_mode():
    with pytest.raises(
        AssertionError, match=re.escape("mode must be one of ['1d', '2d', '3d']")
    ):
        safe_autonomy_simulation.entities.integrator.PointMassIntegratorDynamics(
            mode="4d"
        )


@pytest.mark.parametrize(
    "mode, m, damping, expected_A, expected_B",
    [
        (
            "1d",
            1,
            2,
            np.array([[0, 1], [0, -2]], dtype=np.float64),
            np.array([[0], [1]], dtype=np.float64),
        ),
        (
            "2d",
            1,
            2,
            np.array(
                [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -2, 0], [0, 0, 0, -2]],
                dtype=np.float64,
            ),
            np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=np.float64),
        ),
        (
            "3d",
            1,
            2,
            np.array(
                [
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, -2, 0, 0],
                    [0, 0, 0, 0, -2, 0],
                    [0, 0, 0, 0, 0, -2],
                ],
                dtype=np.float64,
            ),
            np.array(
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                dtype=np.float64,
            ),
        ),
    ],
)
def test_generate_dynamics_matrices(mode, m, damping, expected_A, expected_B):
    dynamics = safe_autonomy_simulation.entities.integrator.PointMassIntegratorDynamics(
        m=m, damping=damping, mode=mode
    )
    A, B = dynamics.generate_dynamics_matrices(mode)
    assert np.all(A == expected_A)
    assert np.all(B == expected_B)
