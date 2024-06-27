import pytest
import numpy as np
import safe_autonomy_simulation
import scipy.spatial.transform as transform


@pytest.mark.parametrize(
    "default_position, parent",
    [
        (
            np.array([1, 0, 0]),
            safe_autonomy_simulation.sims.inspection.InspectionPointSet(
                name="set",
                parent=safe_autonomy_simulation.entities.Point(
                    "parent", position=np.array([0, 0, 0])
                ),
                num_points=10,
                radius=1,
                priority_vector=np.array([1, 0, 0]),
            ),
        ),
        (
            np.array([0, 1, 0]),
            safe_autonomy_simulation.sims.inspection.InspectionPointSet(
                name="set",
                parent=safe_autonomy_simulation.entities.Point(
                    "parent", position=np.array([0, 0, 0])
                ),
                num_points=10,
                radius=1,
                priority_vector=np.array([1, 0, 0]),
            ),
        ),
        (
            np.array([0, 0, 1]),
            safe_autonomy_simulation.sims.inspection.InspectionPointSet(
                name="set",
                parent=safe_autonomy_simulation.entities.Point(
                    "parent", position=np.array([0, 0, 0])
                ),
                num_points=10,
                radius=1,
                priority_vector=np.array([1, 0, 0]),
            ),
        ),
    ],
)
def test_init(default_position, parent):
    dynamics = safe_autonomy_simulation.sims.inspection.InspectionPointDynamics(
        default_position=default_position, parent=parent
    )
    assert np.all(dynamics.default_position == default_position)
    assert dynamics.parent == parent


@pytest.mark.parametrize(
    "default_position, parent",
    [
        (
            np.array([1, 0, 0]),
            safe_autonomy_simulation.sims.inspection.InspectionPointSet(
                name="set",
                parent=safe_autonomy_simulation.entities.Point(
                    "parent", position=np.array([0, 0, 0])
                ),
                num_points=10,
                radius=1,
                priority_vector=np.array([1, 0, 0]),
            ),
        ),
        (
            np.array([0, 1, 0]),
            safe_autonomy_simulation.sims.inspection.InspectionPointSet(
                name="set",
                parent=safe_autonomy_simulation.entities.Point(
                    "parent", position=np.array([0, 0, 0])
                ),
                num_points=10,
                radius=1,
                priority_vector=np.array([1, 0, 0]),
            ),
        ),
        (
            np.array([0, 0, 1]),
            safe_autonomy_simulation.sims.inspection.InspectionPointSet(
                name="set",
                parent=safe_autonomy_simulation.entities.Point(
                    "parent", position=np.array([0, 0, 0])
                ),
                num_points=10,
                radius=1,
                priority_vector=np.array([1, 0, 0]),
            ),
        ),
    ],
)
def test_step(default_position, parent):
    dynamics = safe_autonomy_simulation.sims.inspection.InspectionPointDynamics(
        default_position=default_position, parent=parent
    )
    next_state, state_dot = dynamics.step(
        step_size=1, state=np.array([0, 0, 0, 0, 0, 0]), control=np.array([0, 0, 0])
    )
    new_position = (
        transform.Rotation.from_quat(parent.orientation).apply(default_position)
        + parent.position
    )
    assert np.all(next_state[:3] == new_position)
    assert np.all(state_dot == np.array([0, 0, 0]))
