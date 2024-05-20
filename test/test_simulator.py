"""Unit tests for simulator.py"""

import pytest
import numpy as np
from safe_autonomy_simulation.simulator import (
    ContinuousSimulator,
    ControlledContinuousSimulator,
)
from safe_autonomy_simulation.entity import Point


class TestContinuousSimulator:
    @pytest.fixture
    def entity(self):
        return Point(name="test_entity", position=np.zeros(3))
    
    @pytest.fixture
    def sim(self, entity):
        return ContinuousSimulator(
            frame_rate=1, entities={"test_entity": entity}
        )

    def test_init(self, sim, entity):
        assert sim.frame_rate == 1
        assert sim.sim_time == 0
        assert sim.entities == {"test_entity": entity}

    def test_reset(self, sim):
        sim.reset()
        assert sim.sim_time == 0
        assert sim.entities["test_entity"].state == sim.entities["test_entity"].build_initial_state()

    def test_step(self, sim):
        sim.step()
        compare_point = Point(name="test_entity", position=np.zeros(3))
        compare_point.step()
        assert sim.sim_time == 1
        assert sim.entities["test_entity"].state == compare_point.state

    def test_info(self, sim):
        assert sim.info == {"test_entity": sim.entities["test_entity"].state}

    def test_frame_rate_prop(self, sim):
        assert sim.frame_rate == 1

    def test_sim_time_prop(self, sim):
        assert sim.sim_time == 0

    def test_entities_prop(self, sim, entity):
        assert sim.entities == {"test_entity": entity}


class TestControlledContinuousSimulator:
    @pytest.fixture
    def sim(self, entity):
        return ControlledContinuousSimulator(
            frame_rate=1, entities={"test_entity": entity}
        )
    
    def test_add_controls(self, sim):
        control = np.zeros(3)
        sim.add_controls({"test_entity": control})
        assert sim.entities["test_entity"].control_queue.get() == control
