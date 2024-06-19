import re
import pytest
import numpy as np
import safe_autonomy_simulation


@pytest.mark.parametrize(
    "A, B",
    [
        (
            np.array(
                [
                    [0, 0],
                    [0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0],
                    [0, 0],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
            np.array(
                [
                    [5, 6],
                    [7, 8],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ),
            np.array(
                [
                    [10, 11, 12],
                    [13, 14, 15],
                    [16, 17, 18],
                ]
            ),
        ),
    ],
)
def test_init(A, B):
    linear_ode = safe_autonomy_simulation.dynamics.LinearODEDynamics(A, B)
    assert np.all(linear_ode.A == A)
    assert np.all(linear_ode.B == B)


@pytest.mark.parametrize(
    "A",
    [
        np.array(
            [0, 0],
        ),
        np.array(
            [
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ]
        ),
        np.array([]),
    ],
)
def test_init_2d_error_A(A):
    with pytest.raises(
        AssertionError,
        match=re.escape(f"A must be a 2D matrix. Instead got shape {A.shape}"),
    ):
        B = np.zeros((2, 2))
        safe_autonomy_simulation.dynamics.LinearODEDynamics(A, B)


@pytest.mark.parametrize(
    "B",
    [
        np.array(
            [0, 0],
        ),
        np.array(
            [
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ]
        ),
        np.array([]),
    ],
)
def test_init_2d_error_B(B):
    with pytest.raises(
        AssertionError,
        match=re.escape(f"B must be a 2D matrix. Instead got shape {B.shape}"),
    ):
        A = np.zeros((2, 2))
        safe_autonomy_simulation.dynamics.LinearODEDynamics(A, B)


@pytest.mark.parametrize(
    "A",
    [
        np.array(
            [
                [0, 0],
            ]
        ),
        np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
    ],
)
def test_init_square_error(A):
    with pytest.raises(
        AssertionError,
        match=re.escape(f"A must be a square matrix, not shape {A.shape}"),
    ):
        B = np.zeros((3, 2))
        safe_autonomy_simulation.dynamics.LinearODEDynamics(A, B)


@pytest.mark.parametrize(
    "A, B",
    [
        (
            np.array(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
            np.array(
                [
                    [5, 6],
                    [7, 8],
                    [9, 10],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ),
            np.array(
                [
                    [10, 11],
                    [13, 14],
                ]
            ),
        ),
    ],
)
def test_init_shape_error(A, B):
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "number of columns in A must match the number of rows in B."
            + f" However, got shapes {A.shape} for A and {B.shape} for B"
        ),
    ):
        safe_autonomy_simulation.dynamics.LinearODEDynamics(A, B)
