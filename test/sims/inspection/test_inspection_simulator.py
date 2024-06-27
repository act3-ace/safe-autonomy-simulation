import typing
import pytest
import numpy as np
import safe_autonomy_simulation


@pytest.mark.parametrize(
    "frame_rate, inspectors, targets",
    [
        (
            1,
            [
                safe_autonomy_simulation.sims.inspection.Inspector(
                    name="inspector",
                    position=np.array([0, 0, 0]),
                    camera=safe_autonomy_simulation.sims.inspection.Camera(
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
            [
                safe_autonomy_simulation.sims.inspection.Target(
                    name="target",
                    position=np.array([1, 0, 0]),
                    num_points=10,
                    radius=1,
                    priority_vector=np.array([1, 0, 0]),
                )
            ],
        )
    ],
)
def test_init_default(
    frame_rate: float,
    inspectors: typing.List[safe_autonomy_simulation.sims.inspection.Inspector],
    targets: typing.List[safe_autonomy_simulation.sims.inspection.Target],
):
    inspection_sim = safe_autonomy_simulation.sims.inspection.InspectionSimulator(
        frame_rate=frame_rate, inspectors=inspectors, targets=targets
    )
    assert inspection_sim.frame_rate == frame_rate
    assert inspection_sim.inspectors == inspectors
    assert inspection_sim.targets == targets
    assert inspection_sim.sim_time == 0
    entities = inspectors + targets
    assert len(inspection_sim.entities) == len(entities)
    for e in entities:
        assert e in inspection_sim.entities
    assert inspection_sim.sun is None
    assert not inspection_sim.binary_ray


@pytest.mark.parametrize(
    "frame_rate, inspectors, targets, sun, binary_ray",
    [
        (
            1,
            [
                safe_autonomy_simulation.sims.inspection.Inspector(
                    name="inspector",
                    position=np.array([0, 0, 0]),
                    camera=safe_autonomy_simulation.sims.inspection.Camera(
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
            [
                safe_autonomy_simulation.sims.inspection.Target(
                    name="target",
                    position=np.array([1, 0, 0]),
                    num_points=10,
                    radius=1,
                    priority_vector=np.array([1, 0, 0]),
                )
            ],
            safe_autonomy_simulation.sims.inspection.Sun(),
            True,
        )
    ],
)
def test_init_args(frame_rate, inspectors, targets, sun, binary_ray):
    inspection_sim = safe_autonomy_simulation.sims.inspection.InspectionSimulator(
        frame_rate=frame_rate,
        inspectors=inspectors,
        targets=targets,
        sun=sun,
        binary_ray=binary_ray,
    )
    assert inspection_sim.frame_rate == frame_rate
    assert inspection_sim.inspectors == inspectors
    assert inspection_sim.targets == targets
    assert inspection_sim.sim_time == 0
    entities = inspectors + targets + [sun]
    assert len(inspection_sim.entities) == len(entities)
    for e in entities:
        assert e in inspection_sim.entities
    assert inspection_sim.sun == sun
    assert inspection_sim.binary_ray == binary_ray


@pytest.mark.parametrize(
    "inspection_sim",
    [
        safe_autonomy_simulation.sims.inspection.InspectionSimulator(
            frame_rate=1,
            inspectors=[
                safe_autonomy_simulation.sims.inspection.Inspector(
                    name="inspector",
                    position=np.array([0, 0, 0]),
                    camera=safe_autonomy_simulation.sims.inspection.Camera(
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
            targets=[
                safe_autonomy_simulation.sims.inspection.Target(
                    name="target",
                    position=np.array([1, 0, 0]),
                    num_points=10,
                    radius=1,
                    priority_vector=np.array([1, 0, 0]),
                )
            ],
            sun=safe_autonomy_simulation.sims.inspection.Sun(),
            binary_ray=True,
        ),
    ],
)
def test_reset(
    inspection_sim: safe_autonomy_simulation.sims.inspection.InspectionSimulator,
):
    for inspector in inspection_sim.inspectors:
        for target in inspection_sim.targets:
            target.inspection_points.update_points_inspection_status(
                camera=inspector.camera,
                sun=inspection_sim.sun,
                binary_ray=inspection_sim.binary_ray,
            )
    expected_inspection_point_states = {
        target.name: target.inspection_points.points
        for target in inspection_sim.targets
    }
    inspection_sim.reset()
    assert inspection_sim.sim_time == 0
    for inspector in inspection_sim.inspectors:
        for target in inspection_sim.targets:
            for id, point in target.inspection_points.points.items():
                expected_point = expected_inspection_point_states[target.name][id]
                assert np.all(point.state == expected_point.state)


@pytest.mark.parametrize(
    "inspection_sim",
    [
        safe_autonomy_simulation.sims.inspection.InspectionSimulator(
            frame_rate=1,
            inspectors=[
                safe_autonomy_simulation.sims.inspection.Inspector(
                    name="inspector",
                    position=np.array([0, 0, 0]),
                    camera=safe_autonomy_simulation.sims.inspection.Camera(
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
            targets=[
                safe_autonomy_simulation.sims.inspection.Target(
                    name="target",
                    position=np.array([1, 0, 0]),
                    num_points=10,
                    radius=1,
                    priority_vector=np.array([1, 0, 0]),
                )
            ],
            sun=safe_autonomy_simulation.sims.inspection.Sun(),
            binary_ray=True,
        ),
    ],
)
def test_post_step(
    inspection_sim: safe_autonomy_simulation.sims.inspection.InspectionSimulator,
):
    for inspector in inspection_sim.inspectors:
        for target in inspection_sim.targets:
            target.inspection_points.update_points_inspection_status(
                camera=inspector.camera,
                sun=inspection_sim.sun,
                binary_ray=inspection_sim.binary_ray,
            )
    expected_inspection_point_states = {
        target.name: target.inspection_points.points
        for target in inspection_sim.targets
    }
    inspection_sim._post_step()
    for inspector in inspection_sim.inspectors:
        for target in inspection_sim.targets:
            for id, point in target.inspection_points.points.items():
                expected_point = expected_inspection_point_states[target.name][id]
                assert np.all(point.state == expected_point.state)
