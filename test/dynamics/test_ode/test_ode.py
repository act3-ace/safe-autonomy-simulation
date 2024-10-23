import re
import pytest
import numpy as np
import safe_autonomy_simulation


def test_init_no_args():
    ode = safe_autonomy_simulation.dynamics.ODEDynamics()
    assert ode.integration_method == "RK45"
    assert ode.state_dot_min == -np.inf
    assert ode.state_dot_max == np.inf
    assert ode.state_min == -np.inf
    assert ode.state_max == np.inf
    assert ode.trajectory is None
    assert ode.trajectory_t is None
    assert ode.trajectory_samples == 0


@pytest.mark.parametrize(
    "trajectory_samples, state_min, state_max, state_dot_min, state_dot_max, integration_method",
    [
        (100, -10, 10, -5, 5, "RK45"),
        (10, -5, 5, -2, 2, "RK45"),
        (100, -10, 10, -5, 5, "Euler"),
        (10, np.array([-5, -1, -0.2]), np.array([5, 1, 0.2]), -2, 2, "RK45"),
    ],
)
def test_init_args(
    trajectory_samples,
    state_min,
    state_max,
    state_dot_min,
    state_dot_max,
    integration_method,
):
    ode = safe_autonomy_simulation.dynamics.ODEDynamics(
        trajectory_samples=trajectory_samples,
        state_min=state_min,
        state_max=state_max,
        state_dot_min=state_dot_min,
        state_dot_max=state_dot_max,
        integration_method=integration_method,
    )
    assert ode.integration_method == integration_method
    assert np.all(ode.state_dot_min == state_dot_min)
    assert np.all(ode.state_dot_max == state_dot_max)
    assert np.all(ode.state_min == state_min)
    assert np.all(ode.state_max == state_max)
    assert ode.trajectory is None
    assert ode.trajectory_t is None
    assert ode.trajectory_samples == trajectory_samples


@pytest.mark.parametrize(
    "integration_method",
    [
        1,
        45,
        0.2,
        "RK4",
        "RK45_JAX_JAX",
        "Euler_JAX",
        "RK45_JAX_JAX_JAX",
        "Euler_JAX_JAX",
        "RK45_JAX_JAX_JAX_JAX",
        "Euler_JAX_JAX_JAX",
        "RK45_JAX_JAX_JAX_JAX_JAX",
        "Euler_JAX_JAX_JAX_JAX",
        "RK45_JAX_JAX_JAX_JAX_JAX_JAX",
        "Euler_JAX_JAX_JAX_JAX_JAX",
        "RK45_JAX_JAX_JAX_JAX_JAX_JAX_JAX",
        "Euler_JAX_JAX_JAX_JAX_JAX",
        True,
        False,
        dict(),
        list(),
        set(),
        tuple(),
    ],
)
def test_init_integration_method_error(integration_method):
    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"invalid integration method {integration_method}, must be one of 'RK45', 'Euler'"
        ),
    ):
        safe_autonomy_simulation.dynamics.ODEDynamics(
            integration_method=integration_method
        )


@pytest.mark.parametrize(
    "samples", [0.1, -10.5, 1.5, -0.5, 0.5, str(), dict(), list(), set(), tuple()]
)
def test_init_trajectory_sample_int_error(samples):
    with pytest.raises(
        AssertionError,
        match=re.escape(f"trajectory_samples must be an integer, got {samples}"),
    ):
        safe_autonomy_simulation.dynamics.ODEDynamics(trajectory_samples=samples)


@pytest.mark.parametrize("samples", [-10, -1, -5, -100, -1000, -10000, -1000000])
def test_init_trajectory_sample_negative_error(samples):
    with pytest.raises(
        AssertionError,
        match=re.escape(f"trajectory_samples must be non-negative, got {samples}"),
    ):
        safe_autonomy_simulation.dynamics.ODEDynamics(trajectory_samples=samples)


def test_not_implemented_errors():
    ode = safe_autonomy_simulation.dynamics.ODEDynamics()
    with pytest.raises(NotImplementedError):
        ode.step(0, np.zeros(2), np.zeros(2))
    with pytest.raises(NotImplementedError):
        ode._step(0, np.zeros(2), np.zeros(2))
    with pytest.raises(NotImplementedError):
        ode.compute_state_dot(0, np.zeros(2), np.zeros(2))
    with pytest.raises(NotImplementedError):
        ode._compute_state_dot(0, np.zeros(2), np.zeros(2))
