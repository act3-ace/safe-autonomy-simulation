import pytest
import numpy as np
import safe_autonomy_simulation

if safe_autonomy_simulation.jax_available():
    import jax.numpy as jnp


cases = [(np.array([0, 0, 0]), np.array([0, 0, 0]), False)]

if safe_autonomy_simulation.jax_available():
    jax_cases = [
        (np.array([0, 0, 0]), jnp.array[0, 0, 0], True),
    ]
    cases = cases + jax_cases


@pytest.mark.parametrize("in_arr, out_arr, use_jax", cases)
def test_cast_jax(in_arr, out_arr, use_jax):
    cast_arr = safe_autonomy_simulation.utils.cast_jax(in_arr, use_jax=use_jax)
    assert np.all(cast_arr == out_arr)
