import pytest
import numpy as np
import safe_autonomy_simulation
import safe_autonomy_simulation.sims.spacecraft.sixdof_model


@pytest.mark.parametrize(
    "input_val, shape, dtype, expected_output",
    [
        (1, (1,), np.float64, np.array([1.0])),
        (1, (2, 2), np.float64, np.array([[1.0, 1.0], [1.0, 1.0]])),
        ([1, 2, 3], (3,), np.float64, np.array([1.0, 2.0, 3.0])),
    ],
)
def test_number_list_to_np(input_val, shape, dtype, expected_output):
    output = safe_autonomy_simulation.sims.spacecraft.sixdof_model.number_list_to_np(
        input_val, shape, dtype
    )
    assert np.all(output == expected_output)
