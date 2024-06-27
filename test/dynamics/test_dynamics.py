import pytest
import numpy as np
import jax.numpy as jnp
import safe_autonomy_simulation


def test_init_no_jax():
    dynamics = safe_autonomy_simulation.Dynamics()
    assert dynamics.state_min == -np.inf
    assert dynamics.state_max == np.inf
    assert not dynamics.use_jax
    assert dynamics.np == np


def test_init_jax():
    dynamics = safe_autonomy_simulation.Dynamics(use_jax=True)
    assert dynamics.state_min == -np.inf
    assert dynamics.state_max == np.inf
    assert dynamics.use_jax
    assert dynamics.np == jnp


def test_not_implemented_errors():
    dynamics = safe_autonomy_simulation.Dynamics()
    with pytest.raises(NotImplementedError):
        dynamics.step(0, np.zeros(2), np.zeros(2))
    with pytest.raises(NotImplementedError):
        dynamics._step(0, np.zeros(2), np.zeros(2))
