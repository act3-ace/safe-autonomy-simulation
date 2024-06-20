import pytest
import numpy as np
import safe_autonomy_simulation
import safe_autonomy_simulation.sims.spacecraft.defaults


@pytest.mark.parametrize(
    "name, camera",
    [
        (
            "inspector",
            safe_autonomy_simulation.sims.inspection.Camera(
                name="camera",
                fov=np.pi / 2,
                resolution=[1920, 1080],
                focal_length=1,
                pixel_pitch=1e-3,
                position=np.array([0, 0, 0]),
                velocity=np.array([0, 0, 0]),
                orientation=np.array([0, 0, 0, 1]),
                angular_velocity=np.array([0, 0, 0]),
                parent=None,
                children=[],
            ),
        ),
    ],
)
def test_init_default(name, camera):
    inspector = safe_autonomy_simulation.sims.inspection.SixDOFInspector(
        name=name, camera=camera
    )
    assert inspector.name == name
    assert inspector.camera == camera
    assert camera in inspector.children
    assert np.all(inspector.position == np.zeros(3))
    assert np.all(inspector.velocity == np.zeros(3))
    assert np.all(inspector.orientation == np.array([0, 0, 0, 1]))
    assert np.all(inspector.angular_velocity == np.zeros(3))
    assert (
        inspector.dynamics.m
        == safe_autonomy_simulation.sims.spacecraft.defaults.M_DEFAULT
    )
    assert (
        inspector.dynamics.n
        == safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT
    )
    assert inspector.dynamics.trajectory_samples == 0
    assert inspector.dynamics.integration_method == "RK45"
    assert (
        inspector.material
        == safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL
    )
    assert inspector.parent is None


@pytest.mark.parametrize(
    "name, camera, position, velocity, orientation, angular_velocity, m, n, inertia_matrix, ang_acc_limit, ang_vel_limit, inertia_wheel, acc_limit_wheel, vel_limit_wheel, thrust_control_limit, body_frame_thrust, trajectory_samples, integration_method, material, parent",
    [
        (
            "inspector",
            safe_autonomy_simulation.sims.inspection.Camera(
                name="camera",
                fov=np.pi / 2,
                resolution=[1920, 1080],
                focal_length=1,
                pixel_pitch=1e-3,
                position=np.array([0, 0, 0]),
                velocity=np.array([0, 0, 0]),
                orientation=np.array([0, 0, 0, 1]),
                angular_velocity=np.array([0, 0, 0]),
                parent=None,
                children=[],
            ),
            np.array([1, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0]),
            1,
            1,
            np.identity(3),
            0.01,
            0.03,
            1e-5,
            180,
            576,
            1.0,
            True,
            10,
            "RK45",
            safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL,
            None,
        ),
    ],
)
def test_init_args(
    name,
    camera,
    position,
    velocity,
    orientation,
    angular_velocity,
    m,
    n,
    inertia_matrix,
    ang_acc_limit,
    ang_vel_limit,
    inertia_wheel,
    acc_limit_wheel,
    vel_limit_wheel,
    thrust_control_limit,
    body_frame_thrust,
    trajectory_samples,
    integration_method,
    material,
    parent,
):
    inspector = safe_autonomy_simulation.sims.inspection.SixDOFInspector(
        name=name,
        camera=camera,
        position=position,
        velocity=velocity,
        orientation=orientation,
        angular_velocity=angular_velocity,
        m=m,
        n=n,
        inertia_matrix=inertia_matrix,
        ang_acc_limit=ang_acc_limit,
        ang_vel_limit=ang_vel_limit,
        inertia_wheel=inertia_wheel,
        acc_limit_wheel=acc_limit_wheel,
        vel_limit_wheel=vel_limit_wheel,
        thrust_control_limit=thrust_control_limit,
        body_frame_thrust=body_frame_thrust,
        trajectory_samples=trajectory_samples,
        integration_method=integration_method,
        material=material,
        parent=parent,
    )
    assert inspector.name == name
    assert inspector.camera == camera
    assert camera in inspector.children
    assert np.all(inspector.position == position)
    assert np.all(inspector.velocity == velocity)
    assert np.all(inspector.orientation == orientation)
    assert np.all(inspector.angular_velocity == angular_velocity % (2 * np.pi))
    assert inspector.dynamics.m == m
    assert inspector.dynamics.n == n
    assert inspector.dynamics.trajectory_samples == trajectory_samples
    assert inspector.dynamics.integration_method == integration_method
    assert inspector.material == material
    assert inspector.parent == parent
