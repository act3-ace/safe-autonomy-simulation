import pytest
import numpy as np
import safe_autonomy_simulation
import safe_autonomy_simulation.entities.integrator


def test_init_no_args():
    integrator = safe_autonomy_simulation.entities.integrator.PointMassIntegrator1d(
        name="integrator"
    )
    assert integrator.material is safe_autonomy_simulation.materials.BLACK
    assert integrator.name == "integrator"
    assert np.all(integrator.state == np.array([0, 0]))
    assert np.all(integrator.x == np.array([0]))
    assert np.all(integrator.y == np.array([0]))
    assert np.all(integrator.z == np.array([0]))
    assert np.all(integrator.x_dot == np.array([0]))
    assert np.all(integrator.y_dot == np.array([0]))
    assert np.all(integrator.z_dot == np.array([0]))
    assert np.all(integrator.wx == np.array([0]))
    assert np.all(integrator.wy == np.array([0]))
    assert np.all(integrator.wz == np.array([0]))
    assert np.all(integrator.position == np.array([0, 0, 0]))
    assert np.all(integrator.velocity == np.array([0, 0, 0]))
    assert np.all(integrator.orientation == np.array([0, 0, 0, 1]))
    assert np.all(integrator.angular_velocity == np.array([0, 0, 0]))
    assert (
        integrator.dynamics.m == safe_autonomy_simulation.entities.integrator.M_DEFAULT
    )
    assert (
        integrator.dynamics.damping
        == safe_autonomy_simulation.entities.integrator.DAMPING_DEFAULT
    )
    assert integrator.dynamics.trajectory_samples == 0
    assert integrator.dynamics.integration_method == "RK45"


@pytest.mark.parametrize(
    "name, position, velocity, m, damping, trajectory_samples, integration_method",
    [
        ("integrator", np.array([1]), np.array([2]), 3, 4, 5, "Euler"),
        ("integrator", np.array([1]), np.array([2]), 3, 4, 5, "RK45"),
    ],
)
def test_init_with_args(
    name, position, velocity, m, damping, trajectory_samples, integration_method
):
    integrator = safe_autonomy_simulation.entities.integrator.PointMassIntegrator1d(
        name=name,
        position=position,
        velocity=velocity,
        m=m,
        damping=damping,
        trajectory_samples=trajectory_samples,
        integration_method=integration_method,
    )
    assert integrator.material is safe_autonomy_simulation.materials.BLACK
    assert integrator.name == name
    assert np.all(integrator.state == np.concatenate((position, velocity)))
    assert np.all(integrator.x == position)
    assert np.all(integrator.y == np.array([0]))
    assert np.all(integrator.z == np.array([0]))
    assert np.all(integrator.x_dot == velocity)
    assert np.all(integrator.y_dot == np.array([0]))
    assert np.all(integrator.z_dot == np.array([0]))
    assert np.all(integrator.wx == np.array([0]))
    assert np.all(integrator.wy == np.array([0]))
    assert np.all(integrator.wz == np.array([0]))
    assert np.all(integrator.position == np.concatenate((position, np.array([0, 0]))))
    assert np.all(integrator.velocity == np.concatenate((velocity, np.array([0, 0]))))
    assert np.all(integrator.orientation == np.array([0, 0, 0, 1]))
    assert np.all(integrator.angular_velocity == np.array([0, 0, 0]))
    assert integrator.dynamics.m == m
    assert integrator.dynamics.damping == damping
    assert integrator.dynamics.trajectory_samples == trajectory_samples
    assert integrator.dynamics.integration_method == integration_method


def test_state():
    integrator = safe_autonomy_simulation.entities.integrator.PointMassIntegrator1d(
        name="integrator"
    )
    assert np.all(integrator.state == np.array([0, 0]))
    integrator.state = np.array([1, 2])
    assert np.all(integrator.state == np.array([1, 2]))
