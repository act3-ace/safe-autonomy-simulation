import pytest
import numpy as np
import safe_autonomy_simulation


@pytest.mark.parametrize(
    "step_size, state, control",
    [
        (0.1, np.array([0, 0]), np.array([0, 0])),
        (0.1, np.array([1, 2]), np.array([3, 4])),
        (0.1, np.array([5, 6, 7]), np.array([8, 9, 10])),
    ],
)
def test__step(step_size, state, control):
    pass_through_dynamics = safe_autonomy_simulation.dynamics.PassThroughDynamics()
    next_state, state_derivative = pass_through_dynamics._step(
        step_size, state, control
    )
    assert np.all(next_state == state)
    assert np.all(state_derivative == control)


@pytest.mark.parametrize(
    "step_size, state, control",
    [
        (0.1, np.array([0, 0]), np.array([0, 0])),
        (0.1, np.array([1, 2]), np.array([3, 4])),
        (0.1, np.array([5, 6, 7]), np.array([8, 9, 10])),
    ],
)
def test_step(step_size, state, control):
    pass_through_dynamics = safe_autonomy_simulation.dynamics.PassThroughDynamics()
    next_state, state_derivative = pass_through_dynamics.step(step_size, state, control)
    assert np.all(next_state == state)
    assert np.all(state_derivative == control)
