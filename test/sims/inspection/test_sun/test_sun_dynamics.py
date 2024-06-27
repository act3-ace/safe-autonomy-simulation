import pytest
import numpy as np
import safe_autonomy_simulation


def test_init_default():
    dynamics = safe_autonomy_simulation.sims.inspection.SunDynamics()
    assert not dynamics.use_jax
    assert dynamics.integration_method == "RK45"
    assert dynamics.n == safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT


@pytest.mark.parametrize(
    "n, integration_method, use_jax",
    [
        (10, "RK45", False),
    ],
)
def test_init_args(n, integration_method, use_jax):
    dynamics = safe_autonomy_simulation.sims.inspection.SunDynamics(
        n=n, integration_method=integration_method, use_jax=use_jax
    )
    assert dynamics.n == n
    assert dynamics.integration_method == integration_method
    assert dynamics.use_jax == use_jax


@pytest.mark.parametrize(
    "n",
    [10],
)
def test__compute_state_dot(n):
    dynamics = safe_autonomy_simulation.sims.inspection.SunDynamics(n=n)
    state = np.array([0])
    control = np.array([0])
    assert np.all(dynamics._compute_state_dot(0, state, control) == np.array([n]))
