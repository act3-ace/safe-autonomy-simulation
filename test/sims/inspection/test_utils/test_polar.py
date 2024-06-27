import pytest
import numpy as np
import safe_autonomy_simulation


@pytest.mark.parametrize(
    "azimuth, elevation, expected",
    [
        (0, 0, np.array([1, 0, 0])),
        (0, np.pi / 2, np.array([0, 0, 1])),
        (np.pi / 2, 0, np.array([0, 1, 0])),
        (np.pi, 0, np.array([-1, 0, 0])),
        (3 * np.pi / 2, 0, np.array([0, -1, 0])),
        (0, -np.pi / 2, np.array([0, 0, -1])),
    ],
)
def test_get_vec_az_elev(azimuth, elevation, expected):
    v = safe_autonomy_simulation.sims.inspection.utils.polar.get_vec_az_elev(
        azimuth=azimuth, elevation=elevation
    )
    assert np.allclose(v, expected)


def test_sample_az_elev():
    azimuth, elevation = (
        safe_autonomy_simulation.sims.inspection.utils.polar.sample_az_elev()
    )
    assert 0 <= azimuth < 2 * np.pi
    assert -np.pi / 2 <= elevation <= np.pi / 2
