import pytest
import numpy as np
import safe_autonomy_simulation
import safe_autonomy_simulation.sims.spacecraft.defaults


def test_init_default():
    sun = safe_autonomy_simulation.sims.inspection.Sun()
    assert sun.theta == 0
    assert sun.n == safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT
    assert np.all(
        sun.position
        == np.array([safe_autonomy_simulation.sims.inspection.Sun.SUN_DISTANCE, 0, 0])
    )
    assert sun.dynamics.integration_method == "RK45"
    assert sun.dynamics.n == safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT


@pytest.mark.parametrize(
    "name, theta, n, integration_method, material",
    [
        (
            "sun",
            0,
            10,
            "RK45",
            safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL,
        ),
    ],
)
def test_init_args(name, theta, n, integration_method, material):
    sun = safe_autonomy_simulation.sims.inspection.Sun(
        name=name,
        theta=theta,
        n=n,
        integration_method=integration_method,
        material=material,
    )
    assert sun.name == name
    assert sun.theta == theta
    assert sun.n == n
    assert sun.dynamics.integration_method == integration_method
    assert sun.material == material
    assert np.all(
        sun.position
        == np.array(
            [
                safe_autonomy_simulation.sims.inspection.Sun.SUN_DISTANCE
                * np.cos(theta),
                -safe_autonomy_simulation.sims.inspection.Sun.SUN_DISTANCE
                * np.sin(theta),
                0,
            ]
        )
    )
    assert sun.dynamics.n == n


@pytest.mark.parametrize(
    "theta, step_size",
    [
        (0, 1),
        (np.pi, 1),
        (0, 100),
    ]
)
def test_step(theta, step_size):
    sun = safe_autonomy_simulation.sims.inspection.Sun(theta=theta)
    sun.step(step_size=step_size)
    assert sun.theta - (theta + (sun.n * step_size)) < 0.0000001  # epsilon comparison accounts for rounding error
