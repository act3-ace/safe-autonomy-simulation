import pytest
import safe_autonomy_simulation

if safe_autonomy_simulation.use_jax():
    import jax.numpy as np
else:
    import numpy as np


def test_init_no_jax():
    dynamics = safe_autonomy_simulation.Dynamics()
    assert dynamics.state_min == -np.inf
    assert dynamics.state_max == np.inf


def test_not_implemented_errors():
    dynamics = safe_autonomy_simulation.Dynamics()
    with pytest.raises(NotImplementedError):
        dynamics.step(0, np.zeros(2), np.zeros(2))
    with pytest.raises(NotImplementedError):
        dynamics._step(0, np.zeros(2), np.zeros(2))
