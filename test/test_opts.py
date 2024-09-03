import os
import pytest
import safe_autonomy_simulation


@pytest.mark.parametrize(
    "USE_JAX, expected",
    [
        ("1", True),
        ("0", False)
    ]
)
def test_use_jax(USE_JAX, expected):
    os.environ["USE_JAX"] = USE_JAX
    assert safe_autonomy_simulation.use_jax() == expected
