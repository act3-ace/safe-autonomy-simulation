import pytest
import numpy as np
import safe_autonomy_simulation
import safe_autonomy_simulation.sims.spacecraft.defaults


def test_init_default():
    sun = safe_autonomy_simulation.sims.inspection.Sun()
    assert sun.theta == 0
    assert sun.n == safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT
    assert np.all(sun.position == np.array([0, 0, 0]))
    assert not sun.dynamics.use_jax
    assert sun.dynamics.integration_method == "RK45"
    assert sun.dynamics.n == safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT


@pytest.mark.parametrize(
    "name, theta, n, integration_method, use_jax, material",
    [
        (
            "sun",
            0,
            10,
            "RK45",
            False,
            safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL,
        ),
    ],
)
def test_init_args(name, theta, n, integration_method, use_jax, material):
    sun = safe_autonomy_simulation.sims.inspection.Sun(
        name=name,
        theta=theta,
        n=n,
        integration_method=integration_method,
        use_jax=use_jax,
        material=material,
    )
    assert sun.name == name
    assert sun.theta == theta
    assert sun.n == n
    assert sun.dynamics.integration_method == integration_method
    assert sun.dynamics.use_jax == use_jax
    assert sun.material == material
    assert np.all(sun.position == np.array([0, 0, 0]))
    assert sun.dynamics.n == n