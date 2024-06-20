import pytest
import numpy as np
import safe_autonomy_simulation
import safe_autonomy_simulation.sims.spacecraft.defaults


@pytest.mark.parametrize(
    "name, inspection_points",
    [
        (
            "target",
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
def test_init_default(name, inspection_points):
    target = safe_autonomy_simulation.sims.inspection.Target(
        name=name, inspection_points=inspection_points
    )
    assert target.name == name
    assert target.inspection_points == inspection_points
    assert inspection_points in target.children
    assert np.all(target.position == np.zeros(3))
    assert np.all(target.velocity == np.zeros(3))
    assert (
        target.dynamics.m
        == safe_autonomy_simulation.sims.spacecraft.defaults.M_DEFAULT
    )
    assert (
        target.dynamics.n
        == safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT
    )
    assert target.dynamics.trajectory_samples == 0
    assert target.dynamics.integration_method == "RK45"
    assert (
        target.material
        == safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL
    )
    assert target.parent is None


@pytest.mark.parametrize(
    "name, inspection_points, position, velocity, m, n, trajectory_samples, integration_method, material, parent",
    [
        (
            "target",
            safe_autonomy_simulation.sims.inspection.InspectionPointSet(
                name="set",
                parent=safe_autonomy_simulation.entities.Point(
                    "parent", position=np.array([0, 0, 0])
                ),
                num_points=10,
                radius=1,
                priority_vector=np.array([1, 0, 0]),
            ),
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
    inspection_points,
    position,
    velocity,
    m,
    n,
    trajectory_samples,
    integration_method,
    material,
    parent,
):
    target = safe_autonomy_simulation.sims.inspection.Target(
        name=name,
        inspection_points=inspection_points,
        position=position,
        velocity=velocity,
        m=m,
        n=n,
        trajectory_samples=trajectory_samples,
        integration_method=integration_method,
        material=material,
        parent=parent,
    )
    assert target.name == name
    assert target.inspection_points == inspection_points
    assert inspection_points in target.children
    assert np.all(target.position == position)
    assert np.all(target.velocity == velocity)
    assert target.dynamics.m == m
    assert target.dynamics.n == n
    assert target.dynamics.trajectory_samples == trajectory_samples
    assert target.dynamics.integration_method == integration_method
    assert target.material == material
    assert target.parent == parent
