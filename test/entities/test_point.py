import pytest
import numpy as np
import safe_autonomy_simulation


def test_init_no_args():
    point = safe_autonomy_simulation.entities.Point(
        name="point", position=np.array([0, 0, 0])
    )
    assert point.name == "point"
    assert np.all(point.position == np.array([0, 0, 0]))
    assert np.all(point.velocity == np.array([0, 0, 0]))
    assert np.all(point.orientation == np.array([0, 0, 0, 1]))
    assert np.all(point.angular_velocity == np.array([0, 0, 0]))
    assert point.material == safe_autonomy_simulation.materials.BLACK
    assert point.parent is None
    assert len(point.children) == 0


@pytest.mark.parametrize(
    "name, position, velocity, dynamics, control_queue, parent, children, material",
    [
        (
            "point",
            np.array([0, 0, 0]),
            np.array([1, 2, 3]),
            safe_autonomy_simulation.dynamics.PassThroughDynamics(),
            safe_autonomy_simulation.controls.NoControl(),
            None,
            [],
            safe_autonomy_simulation.materials.BLACK,
        ),
        (
            "point",
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            safe_autonomy_simulation.dynamics.PassThroughDynamics(),
            safe_autonomy_simulation.controls.NoControl(),
            safe_autonomy_simulation.entities.Point(
                name="parent", position=np.array([0, 0, 0])
            ),
            [
                safe_autonomy_simulation.entities.Point(
                    name="child1", position=np.array([0, 0, 0])
                ),
                safe_autonomy_simulation.entities.Point(
                    name="child2", position=np.array([0, 0, 0])
                ),
            ],
            safe_autonomy_simulation.materials.LIGHT,
        ),
    ],
)
def test_init_with_args(
    name, position, velocity, dynamics, control_queue, parent, children, material
):
    point = safe_autonomy_simulation.entities.Point(
        name=name,
        position=position,
        velocity=velocity,
        dynamics=dynamics,
        control_queue=control_queue,
        parent=parent,
        children=children,
        material=material,
    )
    assert point.name == name
    assert np.all(point.position == position)
    assert np.all(point.velocity == velocity)
    assert np.all(point.orientation == np.array([0, 0, 0, 1]))
    assert np.all(point.angular_velocity == np.array([0, 0, 0]))
    assert point.material == material
    assert point.parent == parent
    assert len(point.children) == len(children)
    for child in children:
        assert child in point.children


def test_state():
    point = safe_autonomy_simulation.entities.Point(
        name="point", position=np.array([0, 0, 0])
    )
    assert np.all(point.state == np.array([0, 0, 0, 0, 0, 0]))
    point.state = np.array([1, 2, 3, 4, 5, 6])
    assert np.all(point.state == np.array([1, 2, 3, 4, 5, 6]))
