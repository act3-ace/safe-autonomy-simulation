"""Unit tests for simulator.py"""

import pytest
import numpy as np
from safe_autonomy_simulation.spacecraft.point_model import CWHSpacecraft
from safe_autonomy_simulation.docking_simulator import DockingSimulator, DockingSimulatorValidator


# TODO: add test for add_control and check entities in reset/step


@pytest.fixture
def entities():
    e = {
        "entity1": CWHSpacecraft(name="entity1"),
        "entity2": CWHSpacecraft(name="entity2"),
    }
    return e


@pytest.fixture
def sim(entities):

    return DockingSimulator(frame_rate=1, entities=entities)


def test_init(sim, entities):
    assert sim.config.frame_rate == 1
    assert sim._sim_time == 0
    assert sim.config.entities == entities


def test_sim_time_prop(sim):
    assert sim.sim_time == 0


def test_frame_rate_prop(sim):
    assert sim.frame_rate == 1


def test_entities_prop(sim, entities):
    assert sim.entities == entities


def test_get_sim_validator(sim):
    sim_val = sim.get_sim_validator
    assert sim_val, DockingSimulatorValidator


def test_reset(sim):
    sim.reset()
    assert sim.sim_time == 0
    for _, entity in sim.entities.items():
        assert entity


def test_step(sim):
    sim.step()
    assert sim.sim_time == 1


def test_info(sim, entities):
    info_dict = {entity.name: entity.state for _, entity in entities.items()}
    sim_info = sim.info()
    for k, v in sim_info.items():
        assert k in info_dict.keys()
        assert np.array_equal(v, info_dict[k])
