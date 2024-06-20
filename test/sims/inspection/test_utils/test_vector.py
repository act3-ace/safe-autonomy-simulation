import pytest
import numpy as np
import safe_autonomy_simulation


@pytest.mark.parametrize(
    "vector, expected",
    [([1, 0, 0], [1, 0, 0]), ([0, 1, 0], [0, 1, 0]), ([0, 0, 1], [0, 0, 1])],
)
def test_normalize(vector, expected):
    v = safe_autonomy_simulation.sims.inspection.utils.vector.normalize(
        np.array(vector)
    )
    assert np.allclose(v, expected)
