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
    assert inspection_sim.binary_ray


@pytest.mark.parametrize(
    "frame_rate, inspectors, targets, sun, binary_ray",
    [
        (
            1,
            [
                safe_autonomy_simulation.sims.inspection.Inspector(
                    name="inspector",
                    position=np.array([0, 0, 0]),
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
def test_update(
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
    inspection_sim.update()
    for inspector in inspection_sim.inspectors:
        for target in inspection_sim.targets:
            for id, point in target.inspection_points.points.items():
                expected_point = expected_inspection_point_states[target.name][id]
                assert np.all(point.state == expected_point.state)


@pytest.mark.parametrize(
    "frame_rate, inspectors, targets",
    [
        (
            1,
            [
                safe_autonomy_simulation.sims.inspection.Inspector(
                    name="inspector",
                    position=np.array([0, 0, 0]),
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
def test_step(frame_rate, inspectors, targets):
    inspection_sim = safe_autonomy_simulation.sims.inspection.InspectionSimulator(
        frame_rate=frame_rate, inspectors=inspectors, targets=targets
    )
    inspection_sim.reset()

    # check initial sim time
    assert inspection_sim.sim_time == 0

    # save the initial state of the entities
    initial_states = {id(entity): entity.state for entity in inspection_sim.entities}

    inspection_sim.step()

    # check that the simulation time has been updated
    assert inspection_sim.sim_time == 1 / frame_rate

    # check that the entities' state has updated
    for entity in inspection_sim.entities:
        old_state = initial_states[id(entity)]
        default_control = entity.control_queue.default_control
        new_state, state_dot = entity.dynamics.step(
            step_size=1 / frame_rate, state=old_state, control=default_control
        )
        assert np.all(entity.state == new_state)

    # check that inspection points have been updated
    for target in inspection_sim.targets:
        for point in target.inspection_points.points.values():
            new_point_state, _ = point.dynamics.step(
                step_size=1 / frame_rate,
                state=point.state,
                control=point.control_queue.default_control,
            )
            assert np.all(point.state == new_point_state)


@pytest.mark.parametrize(
    "frame_rate, inspectors, targets",
    [
        (
            1,
            [
                safe_autonomy_simulation.sims.inspection.Inspector(
                    name="inspector",
                    position=np.array([0, 0, 0]),
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
        ),
        (
            1,
            [
                safe_autonomy_simulation.sims.inspection.SixDOFInspector(
                    name="inspector",
                    position=np.array([0, 0, 0]),
                ),
            ],
            [
                safe_autonomy_simulation.sims.inspection.SixDOFTarget(
                    name="target",
                    position=np.array([1, 0, 0]),
                    num_points=10,
                    radius=1,
                    priority_vector=np.array([1, 0, 0]),
                )
            ],
        ),
    ],
)
def test_run_sim(frame_rate, inspectors, targets):
    inspection_sim = safe_autonomy_simulation.sims.inspection.InspectionSimulator(
        frame_rate=frame_rate,
        inspectors=inspectors,
        targets=targets,
        sun=safe_autonomy_simulation.sims.inspection.Sun(),
        binary_ray=True,
    )
    inspection_sim.reset()

    rng = np.random.default_rng()

    prev_inspected = {target.name: 0 for target in inspection_sim.targets}
    for i in range(100):
        for inspector in inspection_sim.inspectors:
            control = rng.uniform(
                inspector.control_queue.control_min,
                inspector.control_queue.control_max,
                size=inspector.control_queue.default_control.shape,
            )
            inspector.add_control(control)
        inspection_sim.step()
        for target in inspection_sim.targets:
            num_inspected = target.inspection_points.get_num_points_inspected()
            assert num_inspected >= prev_inspected[target.name]
            prev_inspected[target.name] = num_inspected
