import re
import pint
import pytest
import numpy as np
import safe_autonomy_simulation
import test.entities.test_physical.utils as utils


@pytest.mark.parametrize(
    "name, position, velocity, orientation, angular_velocity, control_queue, dynamics, material",
    [
        (
            "test_entity",
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0]),
            safe_autonomy_simulation.controls.ControlQueue(
                default_control=np.array([])
            ),
            safe_autonomy_simulation.dynamics.PassThroughDynamics(),
            safe_autonomy_simulation.materials.BLACK,
        ),
        (
            "test_entity",
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9, 10]),
            np.array([11, 12, 13]),
            safe_autonomy_simulation.controls.ControlQueue(
                default_control=np.array([])
            ),
            safe_autonomy_simulation.dynamics.PassThroughDynamics(),
            safe_autonomy_simulation.materials.BLACK,
        ),
        (
            "test_entity",
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9, 10]),
            np.array([11, 12, 13]),
            safe_autonomy_simulation.controls.NoControl(),
            safe_autonomy_simulation.dynamics.PassThroughDynamics(),
            safe_autonomy_simulation.materials.BLACK,
        ),
        (
            str(),
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9, 10]),
            np.array([11, 12, 13]),
            safe_autonomy_simulation.controls.ControlQueue(
                default_control=np.array([])
            ),
            safe_autonomy_simulation.dynamics.PassThroughDynamics(),
            safe_autonomy_simulation.materials.BLACK,
        ),
    ],
)
def test_init(
    name,
    position,
    velocity,
    orientation,
    angular_velocity,
    control_queue,
    dynamics,
    material,
):
    entity = safe_autonomy_simulation.entities.PhysicalEntity(
        name=name,
        position=position,
        velocity=velocity,
        orientation=orientation,
        angular_velocity=angular_velocity,
        control_queue=control_queue,
        dynamics=dynamics,
        material=material,
    )
    assert entity.name == name
    assert np.all(entity.position == position)
    assert np.all(entity.velocity == velocity)
    assert np.all(entity.orientation == orientation)
    assert np.all(entity.angular_velocity == angular_velocity)
    assert entity.control_queue == control_queue
    assert entity.dynamics == dynamics
    assert entity.material == material
    assert entity.last_control is None
    assert np.all(entity.state_dot == np.zeros_like(entity.state))
    assert len(entity.children) == 0
    assert entity.parent is None


def test_init_parent():
    parent = safe_autonomy_simulation.entities.PhysicalEntity(
        name="parent",
        position=np.array([0, 0, 0]),
        velocity=np.array([0, 0, 0]),
        orientation=np.array([0, 0, 0, 1]),
        angular_velocity=np.array([0, 0, 0]),
        control_queue=safe_autonomy_simulation.controls.NoControl(),
        dynamics=safe_autonomy_simulation.dynamics.PassThroughDynamics(),
        material=safe_autonomy_simulation.materials.BLACK,
    )
    entity = safe_autonomy_simulation.entities.PhysicalEntity(
        name="entity",
        position=np.array([0, 0, 0]),
        velocity=np.array([0, 0, 0]),
        orientation=np.array([0, 0, 0, 1]),
        angular_velocity=np.array([0, 0, 0]),
        control_queue=safe_autonomy_simulation.controls.NoControl(),
        dynamics=safe_autonomy_simulation.dynamics.PassThroughDynamics(),
        material=safe_autonomy_simulation.materials.BLACK,
        parent=parent,
    )
    assert entity.parent == parent
    assert entity in parent.children


def test_init_children():
    child1 = safe_autonomy_simulation.entities.PhysicalEntity(
        name="child1",
        position=np.array([0, 0, 0]),
        velocity=np.array([0, 0, 0]),
        orientation=np.array([0, 0, 0, 1]),
        angular_velocity=np.array([0, 0, 0]),
        control_queue=safe_autonomy_simulation.controls.NoControl(),
        dynamics=safe_autonomy_simulation.dynamics.PassThroughDynamics(),
        material=safe_autonomy_simulation.materials.BLACK,
    )
    child2 = safe_autonomy_simulation.entities.PhysicalEntity(
        name="child2",
        position=np.array([0, 0, 0]),
        velocity=np.array([0, 0, 0]),
        orientation=np.array([0, 0, 0, 1]),
        angular_velocity=np.array([0, 0, 0]),
        control_queue=safe_autonomy_simulation.controls.NoControl(),
        dynamics=safe_autonomy_simulation.dynamics.PassThroughDynamics(),
        material=safe_autonomy_simulation.materials.BLACK,
    )
    entity = safe_autonomy_simulation.entities.PhysicalEntity(
        name="entity",
        position=np.array([0, 0, 0]),
        velocity=np.array([0, 0, 0]),
        orientation=np.array([0, 0, 0, 1]),
        angular_velocity=np.array([0, 0, 0]),
        control_queue=safe_autonomy_simulation.controls.NoControl(),
        dynamics=safe_autonomy_simulation.dynamics.PassThroughDynamics(),
        material=safe_autonomy_simulation.materials.BLACK,
        children=[child1, child2],
    )
    assert child1 in entity.children
    assert child2 in entity.children
    assert child1.parent == entity
    assert child2.parent == entity


@pytest.mark.parametrize(
    "position, velocity, orientation, angular_velocity, expected_error",
    [
        (
            np.array([0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0]),
            "Position must be a 3D vector of shape (3,), got shape (2,)",
        ),
        (
            np.array([0, 0, 0]),
            np.array([0, 0]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0]),
            "Velocity must be a 3D vector of shape (3,), got shape (2,)",
        ),
        (
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0]),
            np.array([0, 0, 0]),
            "Orientation must be a quaternion of shape (4,), got shape (2,)",
        ),
        (
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0]),
            "Angular velocity must be a 3D vector of shape (3,), got shape (2,)",
        ),
    ],
)
def test_init_error(position, velocity, orientation, angular_velocity, expected_error):
    with pytest.raises(AssertionError, match=re.escape(expected_error)):
        _ = safe_autonomy_simulation.entities.PhysicalEntity(
            name="test_entity",
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            control_queue=safe_autonomy_simulation.controls.NoControl(),
            dynamics=safe_autonomy_simulation.dynamics.PassThroughDynamics(),
            material=safe_autonomy_simulation.materials.BLACK,
        )


def split_zip(lst, num_splits):
    """Split a list into a given number of equal-length sublists and zip them together."""
    return list(zip(*[iter(lst)] * num_splits))


@pytest.mark.parametrize(
    "entity1, entity2, entity3",
    [(a, a, b) for a, b in split_zip(utils.TEST_ENTITIES, 2)],
)
def test__eq__(entity1, entity2, entity3):
    assert entity1 == entity2
    assert not entity1 == entity3


@pytest.mark.parametrize("initial_state", utils.TEST_STATES)
def test_build_initial_state(initial_state):
    entity = safe_autonomy_simulation.entities.PhysicalEntity(
        name="test_entity",
        position=initial_state[:3],
        velocity=initial_state[3:6],
        orientation=initial_state[6:10],
        angular_velocity=initial_state[10:],
        control_queue=safe_autonomy_simulation.controls.NoControl(),
        dynamics=safe_autonomy_simulation.dynamics.PassThroughDynamics(),
        material=safe_autonomy_simulation.materials.BLACK,
    )
    assert np.all(
        entity.build_initial_state()
        == np.concatenate(
            (
                entity.position,
                entity.velocity,
                entity.orientation,
                entity.angular_velocity,
            )
        )
    )


@pytest.mark.parametrize(
    "initial_state, state",
    [
        (a, b)
        for a, b in zip(
            utils.TEST_STATES[: len(utils.TEST_STATES) // 2],
            utils.TEST_STATES[len(utils.TEST_STATES) // 2 :],
        )
    ],
)
def test_state(initial_state, state):
    entity = safe_autonomy_simulation.entities.PhysicalEntity(
        name="test_entity",
        position=initial_state[:3],
        velocity=initial_state[3:6],
        orientation=initial_state[6:10],
        angular_velocity=initial_state[10:],
        control_queue=safe_autonomy_simulation.controls.NoControl(),
        dynamics=safe_autonomy_simulation.dynamics.PassThroughDynamics(),
        material=safe_autonomy_simulation.materials.BLACK,
    )
    assert np.all(
        entity.state
        == np.concatenate(
            (
                entity.position,
                entity.velocity,
                entity.orientation,
                entity.angular_velocity,
            )
        )
    )
    entity.state = state
    assert np.all(entity.position == state[:3])
    assert np.all(entity.velocity == state[3:6])
    assert np.all(entity.orientation == state[6:10])
    assert np.all(entity.angular_velocity == state[10:13])


@pytest.mark.parametrize(
    "entity, position",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_POSITIONS)],
)
def test_x(entity, position):
    assert entity.x == position[0]


@pytest.mark.parametrize(
    "entity, position",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_POSITIONS)],
)
def test_x_with_units(entity, position):
    ureg = pint.get_application_registry()
    expected = ureg.Quantity(position[0], entity.base_units.distance)
    assert entity.x_with_units == expected


@pytest.mark.parametrize(
    "entity, position",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_POSITIONS)],
)
def test_y(entity, position):
    assert entity.y == position[1]


@pytest.mark.parametrize(
    "entity, position",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_POSITIONS)],
)
def test_y_with_units(entity, position):
    ureg = pint.get_application_registry()
    expected = ureg.Quantity(position[1], entity.base_units.distance)
    assert entity.y_with_units == expected


@pytest.mark.parametrize(
    "entity, position",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_POSITIONS)],
)
def test_z(entity, position):
    assert entity.z == position[2]


@pytest.mark.parametrize(
    "entity, position",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_POSITIONS)],
)
def test_z_with_units(entity, position):
    ureg = pint.get_application_registry()
    expected = ureg.Quantity(position[2], entity.base_units.distance)
    assert entity.z_with_units == expected


@pytest.mark.parametrize(
    "entity, position",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_POSITIONS)],
)
def test_position(entity, position):
    assert np.all(entity.position == position)


@pytest.mark.parametrize(
    "entity, position",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_POSITIONS)],
)
def test_position_with_units(entity, position):
    ureg = pint.get_application_registry()
    expected = ureg.Quantity(position, entity.base_units.distance)
    assert np.all(entity.position_with_units == expected)


@pytest.mark.parametrize(
    "entity, velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_VELOCITIES)],
)
def test_x_dot(entity, velocity):
    assert entity.x_dot == velocity[0]


@pytest.mark.parametrize(
    "entity, velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_VELOCITIES)],
)
def test_x_dot_with_units(entity, velocity):
    ureg = pint.get_application_registry()
    expected = ureg.Quantity(velocity[0], entity.base_units.velocity)
    assert entity.x_dot_with_units == expected


@pytest.mark.parametrize(
    "entity, velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_VELOCITIES)],
)
def test_y_dot(entity, velocity):
    assert entity.y_dot == velocity[1]


@pytest.mark.parametrize(
    "entity, velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_VELOCITIES)],
)
def test_y_dot_with_units(entity, velocity):
    ureg = pint.get_application_registry()
    expected = ureg.Quantity(velocity[1], entity.base_units.velocity)
    assert entity.y_dot_with_units == expected


@pytest.mark.parametrize(
    "entity, velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_VELOCITIES)],
)
def test_z_dot(entity, velocity):
    assert entity.z_dot == velocity[2]


@pytest.mark.parametrize(
    "entity, velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_VELOCITIES)],
)
def test_z_dot_with_units(entity, velocity):
    ureg = pint.get_application_registry()
    expected = ureg.Quantity(velocity[2], entity.base_units.velocity)
    assert entity.z_dot_with_units == expected


@pytest.mark.parametrize(
    "entity, velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_VELOCITIES)],
)
def test_velocity(entity, velocity):
    assert np.all(entity.velocity == velocity)


@pytest.mark.parametrize(
    "entity, velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_VELOCITIES)],
)
def test_velocity_with_units(entity, velocity):
    ureg = pint.get_application_registry()
    expected = ureg.Quantity(velocity, entity.base_units.velocity)
    assert np.all(entity.velocity_with_units == expected)


@pytest.mark.parametrize(
    "entity, orientation",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_ORIENTATIONS)],
)
def test_orientation(entity, orientation):
    assert np.all(entity.orientation == orientation)


@pytest.mark.parametrize(
    "entity, angular_velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_ANGULAR_VELOCITIES)],
)
def test_wx(entity, angular_velocity):
    assert entity.wx == angular_velocity[0]


@pytest.mark.parametrize(
    "entity, angular_velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_ANGULAR_VELOCITIES)],
)
def test_wx_with_units(entity, angular_velocity):
    ureg = pint.get_application_registry()
    expected = ureg.Quantity(
        angular_velocity[0], entity.base_units.angular_velocity
    )
    assert entity.wx_with_units == expected


@pytest.mark.parametrize(
    "entity, angular_velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_ANGULAR_VELOCITIES)],
)
def test_wy(entity, angular_velocity):
    assert entity.wy == angular_velocity[1]


@pytest.mark.parametrize(
    "entity, angular_velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_ANGULAR_VELOCITIES)],
)
def test_wy_with_units(entity, angular_velocity):
    ureg = pint.get_application_registry()
    expected = ureg.Quantity(
        angular_velocity[1], entity.base_units.angular_velocity
    )
    assert entity.wy_with_units == expected


@pytest.mark.parametrize(
    "entity, angular_velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_ANGULAR_VELOCITIES)],
)
def test_wz(entity, angular_velocity):
    assert entity.wz == angular_velocity[2]


@pytest.mark.parametrize(
    "entity, angular_velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_ANGULAR_VELOCITIES)],
)
def test_wz_with_units(entity, angular_velocity):
    ureg = pint.get_application_registry()
    expected = ureg.Quantity(
        angular_velocity[2], entity.base_units.angular_velocity
    )
    assert entity.wz_with_units == expected


@pytest.mark.parametrize(
    "entity, angular_velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_ANGULAR_VELOCITIES)],
)
def test_angular_velocity(entity, angular_velocity):
    assert np.all(entity.angular_velocity == angular_velocity)


@pytest.mark.parametrize(
    "entity, angular_velocity",
    [(a, b) for a, b in zip(utils.TEST_ENTITIES, utils.TEST_ANGULAR_VELOCITIES)],
)
def test_angular_velocity_with_units(entity, angular_velocity):
    ureg = pint.get_application_registry()
    expected = ureg.Quantity(
        angular_velocity, entity.base_units.angular_velocity
    )
    assert np.all(entity.angular_velocity_with_units == expected)
