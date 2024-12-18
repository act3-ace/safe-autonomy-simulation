import numpy as np
import safe_autonomy_simulation


def test_defaults():
    assert safe_autonomy_simulation.sims.spacecraft.defaults.M_DEFAULT == 12
    assert safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT == 0.001027
    assert safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_DEFAULT == 0.0573
    assert np.all(
        safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_MATRIX_DEFAULT
        == np.array(
            [
                [
                    safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_DEFAULT,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_DEFAULT,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_DEFAULT,
                ],
            ]
        )
    )
    assert (
        safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_WHEEL_DEFAULT
        == 4.1e-5
    )
    assert (
        safe_autonomy_simulation.sims.spacecraft.defaults.ANG_ACC_LIMIT_DEFAULT
        == 0.017453
    )
    assert (
        safe_autonomy_simulation.sims.spacecraft.defaults.ANG_VEL_LIMIT_DEFAULT
        == 0.034907
    )
    assert (
        safe_autonomy_simulation.sims.spacecraft.defaults.ACC_LIMIT_WHEEL_DEFAULT
        == 181.3
    )
    assert (
        safe_autonomy_simulation.sims.spacecraft.defaults.VEL_LIMIT_WHEEL_DEFAULT == 576
    )
    assert (
        safe_autonomy_simulation.sims.spacecraft.defaults.THRUST_CONTROL_LIMIT_DEFAULT
        == 1.0
    )
    assert np.all(
        safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL.specular
        == np.array([1.0, 1.0, 1.0])
    )
    assert np.all(
        safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL.diffuse
        == np.array([0.7, 0, 0])
    )
    assert np.all(
        safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL.ambient
        == np.array([0.1, 0, 0])
    )
    assert (
        safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL.shininess == 100
    )
    assert (
        safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL.reflection == 0.5
    )
