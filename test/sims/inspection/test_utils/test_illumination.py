import pytest
import numpy as np
import safe_autonomy_simulation


@pytest.mark.parametrize(
    "point, light, radius, expected",
    [
        (
            safe_autonomy_simulation.entities.Point(
                name="point", position=np.array([0, 0, 0])
            ),
            safe_autonomy_simulation.entities.PhysicalEntity(
                name="light",
                position=np.array([0, 0, 1]),
                velocity=np.array([0, 0, 0]),
                orientation=np.array([0, 0, 0, 1]),
                angular_velocity=np.array([0, 0, 0]),
                material=safe_autonomy_simulation.materials.LIGHT,
                control_queue=safe_autonomy_simulation.controls.NoControl(),
                dynamics=safe_autonomy_simulation.dynamics.PassThroughDynamics(),
            ),
            1,
            True,
        ),
    ],
)
def test_illumination(point, light, radius, expected):
    assert (
        safe_autonomy_simulation.sims.inspection.utils.illumination.is_illuminated(
            point, light, radius
        )
        == expected
    )


@pytest.mark.parametrize(
    "rgb, rgb_min, rgb_max, expected",
    [
        (
            np.array([0, 0, 0]),
            np.array([0.8, 0.8, 0.8]),
            np.array([0.9, 0.9, 0.9]),
            False,
        ),
        (
            np.array([0.8, 0.8, 0.8]),
            np.array([0.08, 0.08, 0.08]),
            np.array([0.12, 0.12, 0.12]),
            False,
        ),
        (
            np.array([0.8, 0.8, 0.8]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.5]),
            False,
        ),
        (
            np.array([0.5, 0.55, 0.8]),
            np.array([0.0, 0.5, 0.2]),
            np.array([0.8, 0.6, 0.9]),
            True,
        ),
    ],
)
def test_illumination_rgb(rgb, rgb_min, rgb_max, expected):
    assert (
        safe_autonomy_simulation.sims.inspection.utils.illumination.is_illuminated_rgb(
            rgb, rgb_min, rgb_max
        )
        == expected
    )
