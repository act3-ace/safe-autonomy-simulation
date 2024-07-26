import pytest
import numpy as np
import safe_autonomy_simulation
import scipy.spatial.transform as transform

import safe_autonomy_simulation.sims.spacecraft.defaults


def test_init_default():
    spacecraft = safe_autonomy_simulation.sims.spacecraft.CWHRotation2dSpacecraft(
        name="test"
    )
    assert spacecraft.name == "test"
    assert np.all(spacecraft.position == np.zeros(3))
    assert np.all(spacecraft.velocity == np.zeros(3))
    assert np.all(
        spacecraft.orientation
        == transform.Rotation.from_euler("zyx", [0, 0, 0]).as_quat()
    )
    assert np.all(spacecraft.angular_velocity == np.array([0, 0, 0]))
    assert np.all(spacecraft.state == np.array([0, 0, 0, 0, 0, 0]))
    assert spacecraft.m == safe_autonomy_simulation.sims.spacecraft.defaults.M_DEFAULT
    assert spacecraft.n == safe_autonomy_simulation.sims.spacecraft.defaults.N_DEFAULT
    assert (
        spacecraft.inertia
        == safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_DEFAULT
    )
    assert (
        spacecraft.ang_acc_limit
        == safe_autonomy_simulation.sims.spacecraft.defaults.ANG_ACC_LIMIT_DEFAULT
    )
    assert (
        spacecraft.ang_vel_limit
        == safe_autonomy_simulation.sims.spacecraft.defaults.ANG_VEL_LIMIT_DEFAULT
    )
    assert (
        spacecraft.acc_limit_wheel
        == safe_autonomy_simulation.sims.spacecraft.defaults.ACC_LIMIT_WHEEL_DEFAULT
    )
    assert (
        spacecraft.vel_limit_wheel
        == safe_autonomy_simulation.sims.spacecraft.defaults.VEL_LIMIT_WHEEL_DEFAULT
    )
    assert (
        spacecraft.inertia_wheel
        == safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_WHEEL_DEFAULT
    )
    assert isinstance(
        spacecraft.control_queue, safe_autonomy_simulation.controls.ControlQueue
    )
    assert np.all(spacecraft.control_queue.default_control == np.zeros(3))
    assert np.all(
        spacecraft.control_queue.control_min
        == np.array(
            [
                -1,
                -1,
                -min(
                    safe_autonomy_simulation.sims.spacecraft.defaults.ANG_ACC_LIMIT_DEFAULT,
                    safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_WHEEL_DEFAULT
                    * safe_autonomy_simulation.sims.spacecraft.defaults.ACC_LIMIT_WHEEL_DEFAULT
                    / safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_DEFAULT,
                )
                * safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_DEFAULT,
            ]
        )
    )
    assert np.all(
        spacecraft.control_queue.control_max
        == np.array(
            [
                1,
                1,
                min(
                    safe_autonomy_simulation.sims.spacecraft.defaults.ANG_ACC_LIMIT_DEFAULT,
                    safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_WHEEL_DEFAULT
                    * safe_autonomy_simulation.sims.spacecraft.defaults.ACC_LIMIT_WHEEL_DEFAULT
                    / safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_DEFAULT,
                )
                * safe_autonomy_simulation.sims.spacecraft.defaults.INERTIA_DEFAULT,
            ]
        )
    )
    assert spacecraft.parent is None
    assert len(spacecraft.children) == 0
    assert (
        spacecraft.material
        == safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL
    )


@pytest.mark.parametrize(
    "name, position, velocity, theta, wz, m, n, inertia, ang_acc_limit, ang_vel_limit, inertia_wheel, acc_limit_wheel, vel_limit_wheel, trajectory_samples, integration_method, material, parent, children",
    [
        (
            "test",
            np.array([1, 2]),
            np.array([4, 5]),
            1,
            2,
            3,
            4,
            5,
            8,
            9,
            10,
            11,
            12,
            13,
            "RK45",
            safe_autonomy_simulation.sims.spacecraft.defaults.CWH_MATERIAL,
            None,
            [],
        ),
        (
            "test",
            np.array([1, 2]),
            np.array([4, 5]),
            1,
            2,
            3,
            4,
            5,
            8,
            9,
            10,
            11,
            12,
            13,
            "Euler",
            safe_autonomy_simulation.materials.BLACK,
            safe_autonomy_simulation.entities.Point(
                name="parent", position=np.array([1, 2, 3])
            ),
            [
                safe_autonomy_simulation.entities.Point(
                    name="child", position=np.array([1, 2, 3])
                )
            ],
        ),
    ],
)
def test_init_args(
    name,
    position,
    velocity,
    theta,
    wz,
    m,
    n,
    inertia,
    ang_acc_limit,
    ang_vel_limit,
    inertia_wheel,
    acc_limit_wheel,
    vel_limit_wheel,
    trajectory_samples,
    integration_method,
    material,
    parent,
    children,
):
    spacecraft = safe_autonomy_simulation.sims.spacecraft.CWHRotation2dSpacecraft(
        name=name,
        position=position,
        velocity=velocity,
        theta=theta,
        wz=wz,
        m=m,
        n=n,
        inertia=inertia,
        ang_acc_limit=ang_acc_limit,
        ang_vel_limit=ang_vel_limit,
        inertia_wheel=inertia_wheel,
        acc_limit_wheel=acc_limit_wheel,
        vel_limit_wheel=vel_limit_wheel,
        trajectory_samples=trajectory_samples,
        integration_method=integration_method,
        material=material,
        parent=parent,
        children=children,
    )
    assert spacecraft.name == name
    assert np.all(spacecraft.position == np.concatenate([position, [0]]))
    assert np.all(spacecraft.velocity == np.concatenate([velocity, [0]]))
    assert np.all(
        spacecraft.orientation
        == transform.Rotation.from_euler("zyx", [theta, 0, 0]).as_quat()
    )
    assert np.all(spacecraft.angular_velocity == np.array([0, 0, wz]))
    assert np.all(spacecraft.state == np.concatenate([position, velocity, [theta, wz]]))
    assert spacecraft.m == m
    assert spacecraft.n == n
    assert spacecraft.inertia == inertia
    assert spacecraft.ang_acc_limit == ang_acc_limit
    assert spacecraft.ang_vel_limit == ang_vel_limit
    assert spacecraft.acc_limit_wheel == acc_limit_wheel
    assert spacecraft.vel_limit_wheel == vel_limit_wheel
    assert spacecraft.inertia_wheel == inertia_wheel
    assert isinstance(
        spacecraft.control_queue, safe_autonomy_simulation.controls.ControlQueue
    )
    assert np.all(spacecraft.control_queue.default_control == np.zeros(3))
    assert np.all(
        spacecraft.control_queue.control_min
        == np.array(
            [
                -1,
                -1,
                -min(ang_acc_limit, inertia_wheel * acc_limit_wheel / inertia)
                * inertia,
            ]
        )
    )
    assert np.all(
        spacecraft.control_queue.control_max
        == np.array(
            [
                1,
                1,
                min(ang_acc_limit, inertia_wheel * acc_limit_wheel / inertia) * inertia,
            ]
        )
    )
    assert spacecraft.parent == parent
    assert len(spacecraft.children) == len(children)
    for child in children:
        assert child in spacecraft.children
    assert spacecraft.material == material
