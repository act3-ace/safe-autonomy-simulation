import pytest
import numpy as np
import jax.numpy as jnp
import safe_autonomy_simulation


def test_init_no_args():
    dynamics = safe_autonomy_simulation.Dynamics()
    assert dynamics.state_min == -np.inf
    assert dynamics.state_max == np.inf
    assert dynamics.np is np


@pytest.mark.parametrize(
    "state_min, state_max, use_jax",
    [
        (-np.inf, np.inf, False),
        (-np.inf, np.inf, True),
        (np.array([-1, -2, -3]), np.array([1, 2, 3]), False),
        (np.array([-1, -2, -3]), np.array([1, 2, 3]), True),
    ]
)
def test_init_args(state_min, state_max, use_jax):
    dynamics = safe_autonomy_simulation.Dynamics(
        state_min=state_min, state_max=state_max, use_jax=use_jax
    )
    assert np.all(dynamics.state_min == state_min)
    assert np.all(dynamics.state_max == state_max)
    assert dynamics.use_jax == dynamics.use_jax
    if use_jax:
        assert isinstance(dynamics.state_min, jnp.ndarray)
        assert isinstance(dynamics.state_max, jnp.ndarray)
        assert dynamics.np is jnp
    else:
        assert isinstance(dynamics.state_min, np.ndarray)
        assert isinstance(dynamics.state_max, np.ndarray)
        assert dynamics.np is np


def test_not_implemented_errors():
    dynamics = safe_autonomy_simulation.Dynamics()
    with pytest.raises(NotImplementedError):
        dynamics.step(0, np.zeros(2), np.zeros(2))
    with pytest.raises(NotImplementedError):
        dynamics._step(0, np.zeros(2), np.zeros(2))
