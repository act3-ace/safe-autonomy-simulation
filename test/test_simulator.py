from typing import Tuple
from numpy import ndarray
import pytest
import safe_autonomy_simulation
import numpy as np


class SimpleDynamics(safe_autonomy_simulation.Dynamics):
    def _step(
        self, step_size: float, state: ndarray, control: ndarray
    ) -> Tuple[ndarray, ndarray]:
        return state + step_size * control, control


TEST_CONTROL_QUEUE = safe_autonomy_simulation.ControlQueue(
    default_control=np.array([1.0])
)


TEST_MATERIAL = safe_autonomy_simulation.Material(
    specular=np.array([0.0, 0.0, 0.0]),
    diffuse=np.array([0.0, 0.0, 0.0]),
    ambient=np.array([0.0, 0.0, 0.0]),
    shininess=0.0,
    reflection=0.0,
)


class SimpleEntity(safe_autonomy_simulation.Entity):
    def __init__(self, name):
        super().__init__(
            name,
            dynamics=SimpleDynamics(),
            control_queue=TEST_CONTROL_QUEUE,
            material=TEST_MATERIAL,
        )
        self.state = 0

    def build_initial_state(self) -> ndarray:
        return np.array([0.0])

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value


@pytest.fixture
def entities():
    return [SimpleEntity(name=i) for i in range(3)]


@pytest.fixture
def simulator(entities):
    return safe_autonomy_simulation.Simulator(frame_rate=1.0, entities=entities)


def test_simulator_init(simulator, entities):
    assert simulator.frame_rate == 1.0
    assert simulator.sim_time == 0.0
    assert len(simulator.entities) == len(entities)
    for entity in entities:
        assert entity in simulator.entities


def test_simulator_reset(simulator):
    for _ in range(10):
        simulator.step()
    simulator.reset()
    assert simulator.sim_time == 0
    for entity in simulator.entities:
        assert entity.state == [0]


def test_simulator_step(simulator):
    for _ in range(10):
        simulator.step()
    assert simulator.sim_time == 10
    for entity in simulator.entities:
        assert entity.state == np.array([10])
