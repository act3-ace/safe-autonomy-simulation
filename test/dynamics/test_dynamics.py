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
        dynamics._step(0, np.zeros(2), np.zeros(2))


@pytest.mark.parametrize(
    "state, expected_state",
    [
        (np.array([0]), np.array([0])),
        (np.array([np.pi / 4]), np.array([np.pi / 4])),
        (np.array([np.pi / 2]), np.array([np.pi / 2])),
        (np.array([3 * np.pi / 4]), np.array([3 * np.pi / 4])),
        (np.array([np.pi]), np.array([np.pi])),
        (np.array([5 * np.pi / 4]), np.array([5 * np.pi / 4])),
        (np.array([3 * np.pi / 2]), np.array([3 * np.pi / 2])),
        (np.array([7 * np.pi / 4]), np.array([7 * np.pi / 4])),
        (np.array([2 * np.pi]), np.array([0])),
        (np.array([2 * np.pi + np.pi / 4]), np.array([np.pi / 4])),
        (np.array([2 * np.pi + np.pi / 2]), np.array([np.pi / 2])),
        (np.array([2 * np.pi + 3 * np.pi / 4]), np.array([3 * np.pi / 4])),
        (np.array([2 * np.pi + np.pi]), np.array([np.pi])),
        (np.array([2 * np.pi + 5 * np.pi / 4]), np.array([5 * np.pi / 4])),
        (np.array([2 * np.pi + 3 * np.pi / 2]), np.array([3 * np.pi / 2])),
        (np.array([2 * np.pi + 7 * np.pi / 4]), np.array([7 * np.pi / 4])),
        (np.array([2 * np.pi + 2 * np.pi]), np.array([0])),
        (np.array([-np.pi / 4]), np.array([np.pi / 4])),
        (np.array([-np.pi / 2]), np.array([np.pi / 2])),
        (np.array([-3 * np.pi / 4]), np.array([3 * np.pi / 4])),
        (np.array([-np.pi]), np.array([np.pi])),
        (np.array([-5 * np.pi / 4]), np.array([5 * np.pi / 4])),
        (np.array([-3 * np.pi / 2]), np.array([3 * np.pi / 2])),
        (np.array([-7 * np.pi / 4]), np.array([7 * np.pi / 4])),
        (np.array([-2 * np.pi]), np.array([0])),
        (np.array([-2 * np.pi - np.pi / 4]), np.array([np.pi / 4])),
        (np.array([-2 * np.pi - np.pi / 2]), np.array([np.pi / 2])),
        (np.array([-2 * np.pi - 3 * np.pi / 4]), np.array([3 * np.pi / 4])),
        (np.array([-2 * np.pi - np.pi]), np.array([np.pi])),
        (np.array([-2 * np.pi - 5 * np.pi / 4]), np.array([5 * np.pi / 4])),
        (np.array([-2 * np.pi - 3 * np.pi / 2]), np.array([3 * np.pi / 2])),
        (np.array([-2 * np.pi - 7 * np.pi / 4]), np.array([7 * np.pi / 4])),
        (np.array([-2 * np.pi - 2 * np.pi]), np.array([0])),
    ]
)
def test_wrap_angles_no_jax(state, expected_state):
    dynamics = safe_autonomy_simulation.Dynamics()
    wrapped_state = dynamics._wrap_angles(state)
    assert np.allclose(wrapped_state, expected_state, equal_nan=True)


@pytest.mark.parametrize(
    "state, expected_state",
    [
        (jnp.array([0]), jnp.array([0])),
        (jnp.array([jnp.pi / 4]), jnp.array([jnp.pi / 4])),
        (jnp.array([jnp.pi / 2]), jnp.array([jnp.pi / 2])),
        (jnp.array([3 * jnp.pi / 4]), jnp.array([3 * jnp.pi / 4])),
        (jnp.array([jnp.pi]), jnp.array([jnp.pi])),
        (jnp.array([5 * jnp.pi / 4]), jnp.array([5 * jnp.pi / 4])),
        (jnp.array([3 * jnp.pi / 2]), jnp.array([3 * jnp.pi / 2])),
        (jnp.array([7 * jnp.pi / 4]), jnp.array([7 * jnp.pi / 4])),
        (jnp.array([2 * jnp.pi]), jnp.array([0])),
        (jnp.array([2 * jnp.pi + jnp.pi / 4]), jnp.array([jnp.pi / 4])),
        (jnp.array([2 * jnp.pi + jnp.pi / 2]), jnp.array([jnp.pi / 2])),
        (jnp.array([2 * jnp.pi + 3 * jnp.pi / 4]), jnp.array([3 * jnp.pi / 4])),
        (jnp.array([2 * jnp.pi + jnp.pi]), jnp.array([jnp.pi])),
        (jnp.array([2 * jnp.pi + 5 * jnp.pi / 4]), jnp.array([5 * jnp.pi / 4])),
        (jnp.array([2 * jnp.pi + 3 * jnp.pi / 2]), jnp.array([3 * jnp.pi / 2])),
        (jnp.array([2 * jnp.pi + 7 * jnp.pi / 4]), jnp.array([7 * jnp.pi / 4])),
        (jnp.array([2 * jnp.pi + 2 * jnp.pi]), jnp.array([0])),
        (jnp.array([-jnp.pi / 4]), jnp.array([jnp.pi / 4])),
        (jnp.array([-jnp.pi / 2]), jnp.array([jnp.pi / 2])),
        (jnp.array([-3 * jnp.pi / 4]), jnp.array([3 * jnp.pi / 4])),
        (jnp.array([-jnp.pi]), jnp.array([jnp.pi])),
        (jnp.array([-5 * jnp.pi / 4]), jnp.array([5 * jnp.pi / 4])),
        (jnp.array([-3 * jnp.pi / 2]), jnp.array([3 * jnp.pi / 2])),
        (jnp.array([-7 * jnp.pi / 4]), jnp.array([7 * jnp.pi / 4])),
        (jnp.array([-2 * jnp.pi]), jnp.array([0])),
        (jnp.array([-2 * jnp.pi - jnp.pi / 4]), jnp.array([jnp.pi / 4])),
        (jnp.array([-2 * jnp.pi - jnp.pi / 2]), jnp.array([jnp.pi / 2])),
        (jnp.array([-2 * jnp.pi - 3 * jnp.pi / 4]), jnp.array([3 * jnp.pi / 4])),
        (jnp.array([-2 * jnp.pi - jnp.pi]), jnp.array([jnp.pi])),
        (jnp.array([-2 * jnp.pi - 5 * jnp.pi / 4]), jnp.array([5 * jnp.pi / 4])),
        (jnp.array([-2 * jnp.pi - 3 * jnp.pi / 2]), jnp.array([3 * jnp.pi / 2])),
        (jnp.array([-2 * jnp.pi - 7 * jnp.pi / 4]), jnp.array([7 * jnp.pi / 4])),
        (jnp.array([-2 * jnp.pi - 2 * jnp.pi]), jnp.array([0])), 
    ]
)
def test_wrap_angles_jax(state, expected_state):
    dynamics = safe_autonomy_simulation.Dynamics(use_jax=True)
    wrapped_state = dynamics._wrap_angles(state)
    assert jnp.allclose(wrapped_state, expected_state, equal_nan=True)
