import pytest
import numpy as np
import safe_autonomy_simulation


TEST_CONTROLS = [np.array([0.0]), [0.0], np.array([0.0])]
TEST_CONTROL_IDS = ["np.array", "list", "np.array"]


def test_init():
    control_queue = safe_autonomy_simulation.controls.NoControl()
    assert control_queue.empty()
    assert np.all(control_queue.default_control == np.empty(0))
    assert np.all(control_queue.control_min == np.empty(0))
    assert np.all(control_queue.control_max == np.empty(0))


def test_reset():
    control_queue = safe_autonomy_simulation.controls.NoControl()
    control_queue.add_control(np.array([1.0]))
    control_queue.reset()
    assert control_queue.empty()


def test_empty_true():
    control_queue = safe_autonomy_simulation.controls.NoControl()
    assert control_queue.empty()


@pytest.mark.parametrize("control", TEST_CONTROLS, ids=TEST_CONTROL_IDS)
def test_empty_always_true(control):
    control_queue = safe_autonomy_simulation.controls.NoControl()
    control_queue.add_control(control)
    assert control_queue.empty()


def test_next_control_default():
    control_queue = safe_autonomy_simulation.controls.NoControl()
    assert np.array_equal(control_queue.next_control(), np.empty(0))


@pytest.mark.parametrize("control", TEST_CONTROLS, ids=TEST_CONTROL_IDS)
def test_next_control(control):
    """Test that next_control() returns the default control vector."""
    control_queue = safe_autonomy_simulation.controls.NoControl()
    control_queue.add_control(control)
    next_control = control_queue.next_control()
    assert np.allclose(next_control, control_queue.default_control)


@pytest.mark.parametrize("control", TEST_CONTROLS, ids=TEST_CONTROL_IDS)
def test_add_control(control):
    """Test that add_control() does not change the control vector."""
    control_queue = safe_autonomy_simulation.controls.NoControl()
    control_queue.add_control(control)
    next_control = control_queue.next_control()
    assert np.allclose(next_control, control_queue.default_control)
    assert np.all(next_control <= control_queue.control_max)
    assert np.all(next_control >= control_queue.control_min)
