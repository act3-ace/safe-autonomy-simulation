import pytest
import numpy as np
import safe_autonomy_simulation
import safe_autonomy_simulation.sims.spacecraft.defaults


@pytest.mark.parametrize("name", ["inspector"])
def test_init_default(name):
    inspector = safe_autonomy_simulation.sims.inspection.Inspector(name=name)
    assert inspector.name == name
    assert inspector.camera.name == f"{name}_camera"
    assert inspector.camera.fov == np.pi / 2
    assert inspector.camera.resolution == (640, 480)
    assert inspector.camera.focal_length == 0.01
    assert inspector.camera.pixel_pitch == 1.12e-6
    assert len(inspector.children) == 1
    assert inspector.camera in inspector.children
    assert np.all(inspector.position == np.zeros(3))
    assert np.all(inspector.velocity == np.zeros(3))
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
    "name, fov, resolution, focal_length, pixel_pitch, position, velocity, m, n, trajectory_samples, integration_method, material, parent",
    [
        (
            "inspector",
            np.pi,
            (1920, 1080),
            1,
            1e-3,
            np.array([1, 0, 0]),
            np.array([0, 0, 0]),
            1,
            1,
            10,
            "RK45",
            safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL,
            None,
        ),
    ],
)
def test_init_args(
    name,
    fov,
    resolution,
    focal_length,
    pixel_pitch,
    position,
    velocity,
    m,
    n,
    trajectory_samples,
    integration_method,
    material,
    parent,
):
    inspector = safe_autonomy_simulation.sims.inspection.Inspector(
        name=name,
        fov=fov,
        resolution=resolution,
        focal_length=focal_length,
        pixel_pitch=pixel_pitch,
        position=position,
        velocity=velocity,
        m=m,
        n=n,
        trajectory_samples=trajectory_samples,
        integration_method=integration_method,
        material=material,
        parent=parent,
    )
    assert inspector.name == name
    assert inspector.camera.name == f"{name}_camera"
    assert inspector.camera.fov == fov
    assert inspector.camera.resolution == resolution
    assert inspector.camera.focal_length == focal_length
    assert inspector.camera.pixel_pitch == pixel_pitch
    assert len(inspector.children) == 1
    assert inspector.camera in inspector.children
    assert np.all(inspector.position == position)
    assert np.all(inspector.velocity == velocity)
    assert inspector.dynamics.m == m
    assert inspector.dynamics.n == n
    assert inspector.dynamics.trajectory_samples == trajectory_samples
    assert inspector.dynamics.integration_method == integration_method
    assert inspector.material == material
    assert inspector.parent == parent


def test__post_step():
    inspector = safe_autonomy_simulation.sims.inspection.Inspector(name="inspector")
    inspector._post_step(0.1)
    assert np.all(
        inspector.camera.state
        == np.concatenate(
            (
                inspector.position,
                inspector.velocity,
                inspector.orientation,
                inspector.angular_velocity,
            )
        )
    )


def test_reset():
    inspector = safe_autonomy_simulation.sims.inspection.Inspector(name="inspector")
    inspector.reset()
    assert np.all(
        inspector.camera.state
        == np.concatenate(
            (
                inspector.position,
                inspector.velocity,
                inspector.orientation,
                inspector.angular_velocity,
            )
        )
    )
