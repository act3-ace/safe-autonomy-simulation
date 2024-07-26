import pytest
import numpy as np
import safe_autonomy_simulation
import safe_autonomy_simulation.sims.spacecraft.defaults
import safe_autonomy_simulation.sims.spacecraft.point_model


def test_init_default():
    dynamics = safe_autonomy_simulation.sims.spacecraft.point_model.CWHDynamics()
    assert dynamics.m == safe_autonomy_simulation.sims.spacecraft.defaults.M_DEFAULT
    assert dynamics.n == safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT
    assert dynamics.state_min == -np.inf
    assert dynamics.state_max == np.inf
    assert dynamics.state_dot_min == -np.inf
    assert dynamics.state_dot_max == np.inf
    assert dynamics.integration_method == "RK45"
    assert dynamics.trajectory_samples == 0
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
    "m, n, trajectory_samples, state_min, state_max, state_dot_min, state_dot_max, integration_method",
    [
        (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            "RK45",
        ),
    ],
)
def test_init(
    m,
    n,
    trajectory_samples,
    state_min,
    state_max,
    state_dot_min,
    state_dot_max,
    integration_method,
):
    dynamics = safe_autonomy_simulation.sims.spacecraft.point_model.CWHDynamics(
        m=m,
        n=n,
        trajectory_samples=trajectory_samples,
        state_min=state_min,
        state_max=state_max,
        state_dot_min=state_dot_min,
        state_dot_max=state_dot_max,
        integration_method=integration_method,
    )
    assert dynamics.m == m
    assert dynamics.n == n
    assert dynamics.trajectory_samples == trajectory_samples
    assert dynamics.state_min == state_min
    assert dynamics.state_max == state_max
    assert dynamics.state_dot_min == state_dot_min
    assert dynamics.state_dot_max == state_dot_max
    assert dynamics.integration_method == integration_method
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
