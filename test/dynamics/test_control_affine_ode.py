import pytest
import numpy as np
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
        ode.compute_state_dot_jax(0, np.zeros(2), np.zeros(2))
    with pytest.raises(NotImplementedError):
        ode.state_transition_system(np.zeros(2))
    with pytest.raises(NotImplementedError):
        ode.state_transition_input(np.zeros(2))
