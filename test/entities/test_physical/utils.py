import numpy as np
import safe_autonomy_simulation
import scipy.spatial.transform as transform


test_values = [-1, 0, 1]

TEST_POSITIONS = [
    np.array([x / 4, y / 4, z / 4])
    for x in test_values
    for y in test_values
    for z in test_values
]
TEST_VELOCITIES = [
    np.array([x_dot / 4, y_dot / 4, z_dot / 4])
    for x_dot in test_values
    for y_dot in test_values
    for z_dot in test_values
]
TEST_ORIENTATIONS = [
    transform.Rotation.from_euler(
        "XYZ", np.array([theta_x / 4, theta_y / 4, theta_z / 4])
    ).as_quat()
    for theta_x in test_values
    for theta_y in test_values
    for theta_z in test_values
]
TEST_ANGULAR_VELOCITIES = [
    np.array([w_x / 4, w_y / 4, w_z / 4])
    for w_x in test_values
    for w_y in test_values
    for w_z in test_values
]
TEST_STATES = [
    np.concatenate(
        [
            position,
            velocity,
            orientation,
            angular_velocity,
        ]
    )
    for position, velocity, orientation, angular_velocity in zip(
        TEST_POSITIONS, TEST_VELOCITIES, TEST_ORIENTATIONS, TEST_ANGULAR_VELOCITIES
    )
]

TEST_ENTITIES = [
    safe_autonomy_simulation.entities.PhysicalEntity(
        name="entity",
        position=position,
        velocity=velocity,
        orientation=orientation,
        angular_velocity=angular_velocity,
        control_queue=safe_autonomy_simulation.controls.NoControl(),
        dynamics=safe_autonomy_simulation.dynamics.PassThroughDynamics(),
        material=safe_autonomy_simulation.materials.BLACK,
    )
    for position, velocity, orientation, angular_velocity in zip(
        TEST_POSITIONS, TEST_VELOCITIES, TEST_ORIENTATIONS, TEST_ANGULAR_VELOCITIES
    )
]
