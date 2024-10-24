import pytest
import numpy as np
import jax.numpy as jnp
import safe_autonomy_simulation


def test_not_implemented_error():
    ode = safe_autonomy_simulation.dynamics.ControlAffineODEDynamics()
    with pytest.raises(NotImplementedError):
        ode.step(0, np.zeros(2), np.zeros(2))
    with pytest.raises(NotImplementedError):
        ode._step(0, np.zeros(2), np.zeros(2))
    with pytest.raises(NotImplementedError):
        ode.compute_state_dot(0, np.zeros(2), np.zeros(2))
    with pytest.raises(NotImplementedError):
        ode._compute_state_dot(0, np.zeros(2), np.zeros(2))
    with pytest.raises(NotImplementedError):
        ode.state_transition_system(np.zeros(2))
    with pytest.raises(NotImplementedError):
        ode.state_transition_input(np.zeros(2))


@pytest.mark.parametrize(
    "t, state, control, use_jax, expected",
    [
        (1, np.array([1, 2, 3]), np.array([0, 0, 0]), False, np.array([1, 2, 3])),
        (1, np.array([1, 2, 3]), np.array([0, 0, 0]), True, np.array([1, 2, 3])),
    ],
)
def test__compute_state_dot(mocker, t, state, control, use_jax, expected):
    ode = safe_autonomy_simulation.dynamics.ControlAffineODEDynamics(use_jax=use_jax)

    # mock f(state) = state and g(state) = np.ones(len(state), len(control))
    # tested function should return f(state) + g(state) @ control
    st_sys_return = np.array(state) if not use_jax else jnp.array(state)
    st_input_return = (
        np.ones((state.shape[0], control.shape[0]))
        if not use_jax
        else jnp.ones((state.shape[0], control.shape[0]))
    )
    mocker.patch.object(ode, "state_transition_system", return_value=st_sys_return)
    mocker.patch.object(ode, "state_transition_input", return_value=st_input_return)

    # test function
    if use_jax:
        state = jnp.array(state)
        control = jnp.array(control)
    state_dot = ode._compute_state_dot(t=t, state=state, control=control)
    if use_jax:
        assert isinstance(state_dot, jnp.ndarray)
    else:
        assert isinstance(state_dot, np.ndarray)
    assert np.all(state_dot == expected)
