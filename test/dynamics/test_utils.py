import pytest
import numpy as np
import safe_autonomy_simulation


@pytest.mark.parametrize(
    "state_dot, s_min, s_max, expected",
    [
        (np.array([1, 2, 3]), -np.inf, np.inf, np.array([1, 2, 3])),
        (np.array([1, 2, 3]), -np.inf, 2, np.array([1, 2, 2])),
        (np.array([1, 2, 3]), -np.inf, 1, np.array([1, 1, 1])),
        (np.array([1, 2, 3]), 0, np.inf, np.array([1, 2, 3])),
        (np.array([1, 2, 3]), 2, np.inf, np.array([2, 2, 3])),
        (np.array([1, 2, 3]), 3, np.inf, np.array([3, 3, 3])),
        (np.array([1, 2, 3]), 2, 3, np.array([2, 2, 3])),
        (np.array([1, 2, 3]), 3, 3, np.array([3, 3, 3])),
        (np.array([1, 2, 3]), 4, 5, np.array([4, 4, 4])),
        (np.array([4, 5, 6]), 2, 3, np.array([3, 3, 3])),
        (
            np.array([1, 2, 3]),
            np.array([-1, -2, -3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
        ),
        (
            np.array([1, 2, 3]),
            np.array([-1, -2, -3]),
            np.array([1, 1, 3]),
            np.array([1, 1, 3]),
        ),
        (
            np.array([1, 2, 3]),
            np.array([-1, -2, -3]),
            np.array([1, 1, 2]),
            np.array([1, 1, 2]),
        ),
        (
            np.array([1, 2, 3]),
            np.array([-1, -2, -3]),
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
        ),
        (
            np.array([1, 2, 3]),
            np.array([-1, -2, -3]),
            np.array([0, 1, 1]),
            np.array([0, 1, 1]),
        ),
        (
            np.array([1, 2, 3]),
            np.array([-1, -2, -3]),
            np.array([0, 0, 1]),
            np.array([0, 0, 1]),
        ),
        (
            np.array([1, 2, 3]),
            np.array([-1, -2, -3]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
        ),
        (
            np.array([1, 2, 3]),
            np.array([-1, -2, -3]),
            np.array([1, 0, 0]),
            np.array([1, 0, 0]),
        ),
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([1, 2, 3]),
        ),
        (
            np.array([3, 4, 5]),
            np.array([1, 2, 3]),
            np.array([2, 5, 6]),
            np.array([2, 4, 5]),
        ),
        (
            np.array([3, 4, 5]),
            np.array([1, 2, 3]),
            np.array([2, 3, 6]),
            np.array([2, 3, 5]),
        ),
    ],
)
def test_clip_state_dot(state_dot, s_min, s_max, expected):
    assert np.all(
        safe_autonomy_simulation.dynamics.utils.clip_state_dot(
            state_dot=state_dot, s_min=s_min, s_max=s_max
        )
        == expected
    )


@pytest.mark.parametrize(
    "state, state_dot, state_min, state_max, expected",
    [
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            -np.inf,
            np.inf,
            np.array([1, 2, 3]),
        ),
        (np.array([1, 2, 3]), np.array([1, 2, 3]), -np.inf, 2, np.array([1, 0, 0])),
        (np.array([1, 2, 3]), np.array([1, 2, 3]), -np.inf, 1, np.array([0, 0, 0])),
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 0, np.inf, np.array([1, 2, 3])),
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 2, np.inf, np.array([1, 2, 3])),
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 3, np.inf, np.array([1, 2, 3])),
        (
            np.array([1, 2, 3]),
            np.array([-1, 2, -3]),
            np.array([3, 2, 1]),
            np.array([4, 3, 2]),
            np.array([0, 2, -3]),
        ),
    ],
)
def test_clip_state_dot_at_state_limits(
    state, state_dot, state_min, state_max, expected
):
    assert np.all(
        safe_autonomy_simulation.dynamics.utils.clip_state_dot_at_state_limits(
            state, state_dot, state_min, state_max
        )
        == expected
    )
