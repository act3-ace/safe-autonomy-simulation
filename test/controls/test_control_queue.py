import pytest
import numpy as np
import safe_autonomy_simulation


TEST_CONTROLS = [
    np.array([0.0]),
    [0.0],
    np.array([0.0, 0.0]),
    [0.0, 0.0],
]
TEST_BOUNDS = [
    (np.array([0.0]), np.array([1.0])),
    ([0.0], [1.0]),
    (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
    ([0.0, 0.0], [1.0, 1.0]),
]
TEST_CONTROL_IDS = [
    "np.array",
    "list",
    "np.array 2D",
    "list 2D",
]


def test_init():
    control_queue = safe_autonomy_simulation.ControlQueue(
        np.array([0.0]), np.array([0.0]), np.array([1.0])
    )
    assert control_queue.empty()
    assert control_queue.default_control == np.array([0.0])
    assert control_queue.control_min == np.array([0.0])
    assert control_queue.control_max == np.array([1.0])


def test_reset():
    control_queue = safe_autonomy_simulation.ControlQueue(
        np.array([0.0]), control_min=np.array([0.0]), control_max=np.array([1.0])
    )
    control_queue.add_control(np.array([1.0]))
    control_queue.reset()
    assert control_queue.empty()


def test_empty():
    control_queue = safe_autonomy_simulation.ControlQueue(
        np.array([0.0]), control_min=np.array([0.0]), control_max=np.array([1.0])
    )
    assert control_queue.empty()


def test_empty_false():
    control_queue = safe_autonomy_simulation.ControlQueue(
        np.array([0.0]), control_min=np.array([0.0]), control_max=np.array([1.0])
    )
    control_queue.add_control(np.array([1.0]))
    assert not control_queue.empty()


def test_next_control_default():
    control_queue = safe_autonomy_simulation.ControlQueue(
        np.array([0.0]), control_min=np.array([0.0]), control_max=np.array([1.0])
    )
    control = control_queue.next_control()
    assert np.allclose(control, np.array([0.0]))


@pytest.mark.parametrize("control, bounds", list(zip(TEST_CONTROLS, TEST_BOUNDS)), ids=TEST_CONTROL_IDS)
def test_next_control(control, bounds):
    control_queue = safe_autonomy_simulation.ControlQueue(
        default_control=bounds[0], control_min=bounds[0], control_max=bounds[1]
    )
    control_queue.add_control(control)
    next_control = control_queue.next_control()
    assert np.allclose(next_control, np.array(control))


@pytest.mark.parametrize("control, bounds", list(zip(TEST_CONTROLS, TEST_BOUNDS)), ids=TEST_CONTROL_IDS)
def test_add_control(control, bounds):
    control_queue = safe_autonomy_simulation.ControlQueue(
        default_control=bounds[0], control_min=bounds[0], control_max=bounds[1]
    )
    control_queue.add_control(control)
    next_control = control_queue.next_control()
    assert np.allclose(next_control, np.array(control))
    assert np.all(next_control <= control_queue.control_max)
    assert np.all(next_control >= control_queue.control_min)
