import pytest
import numpy as np
import safe_autonomy_simulation


def test_init_default():
    spacecraft = safe_autonomy_simulation.sims.spacecraft.CWHSpacecraft(name="test")
    assert spacecraft.name == "test"
    assert np.all(spacecraft.position == np.zeros(3))
    assert np.all(spacecraft.velocity == np.zeros(3))
    assert (
        spacecraft.dynamics.m
        == safe_autonomy_simulation.sims.spacecraft.defaults.M_DEFAULT
    )
    assert (
        spacecraft.dynamics.n
        == safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT
    )
    assert spacecraft.dynamics.trajectory_samples == 0
    assert spacecraft.dynamics.integration_method == "RK45"
    assert (
        spacecraft.material
        == safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL
    )
    assert spacecraft.parent is None
    assert len(spacecraft.children) == 0
    assert np.all(spacecraft.control_queue.default_control == np.zeros(3))
    assert spacecraft.control_queue.control_min == -1
    assert spacecraft.control_queue.control_max == 1
    assert np.all(spacecraft.orientation == np.array([0, 0, 0, 1]))
    assert np.all(spacecraft.angular_velocity == np.zeros(3))
    assert np.all(spacecraft.state == np.zeros(6))


@pytest.mark.parametrize(
    "name, position, velocity, m, n, trajectory_samples, integration_method, material, parent, children",
    [
        (
            "test",
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            1,
            2,
            3,
            "RK45",
            safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL,
            safe_autonomy_simulation.entities.Point(
                "parent", position=np.array([0, 0, 0])
            ),
            [
                safe_autonomy_simulation.entities.Point(
                    "child", position=np.array([0, 0, 0])
                )
            ],
        ),
    ],
)
def test_init(
    name,
    position,
    velocity,
    m,
    n,
    trajectory_samples,
    integration_method,
    material,
    parent,
    children,
):
    spacecraft = safe_autonomy_simulation.sims.spacecraft.CWHSpacecraft(
        name=name,
        position=position,
        velocity=velocity,
        m=m,
        n=n,
        trajectory_samples=trajectory_samples,
        integration_method=integration_method,
        material=material,
        parent=parent,
        children=children,
    )
    assert spacecraft.name == name
    assert np.all(spacecraft.position == position)
    assert np.all(spacecraft.velocity == velocity)
    assert spacecraft.dynamics.m == m
    assert spacecraft.dynamics.n == n
    assert spacecraft.dynamics.trajectory_samples == trajectory_samples
    assert spacecraft.dynamics.integration_method == integration_method
    assert spacecraft.material == material
    assert spacecraft.parent == parent
    assert len(spacecraft.children) == len(children)
    for child in children:
        assert child in spacecraft.children
    assert np.all(spacecraft.control_queue.default_control == np.zeros(3))
    assert spacecraft.control_queue.control_min == -1
    assert spacecraft.control_queue.control_max == 1
    assert np.all(spacecraft.orientation == np.array([0, 0, 0, 1]))
    assert np.all(spacecraft.angular_velocity == np.zeros(3))
    assert np.all(spacecraft.state == np.concatenate([position, velocity]))


@pytest.mark.parametrize(
    "state",
    [
        np.array([1, 2, 3, 4, 5, 6]),
        np.array([7, 8, 9, 10, 11, 12]),
    ],
)
def test_state(state):
    spacecraft = safe_autonomy_simulation.sims.spacecraft.CWHSpacecraft(name="test")
    assert np.all(spacecraft.state == np.zeros(6))
    spacecraft.state = state
    assert np.all(spacecraft.position == state[0:3])
    assert np.all(spacecraft.velocity == state[3:6])
    assert np.all(spacecraft.state == state)
