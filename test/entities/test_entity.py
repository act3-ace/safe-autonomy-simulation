import numpy as np
import safe_autonomy_simulation


class SimpleDynamics(safe_autonomy_simulation.dynamics.Dynamics):
    def _step(self, step_size, state, control):
        return state + control * step_size, control


class SimpleEntity(safe_autonomy_simulation.Entity):
    def __init__(
        self,
        name,
        dynamics=SimpleDynamics(),
        control_queue=safe_autonomy_simulation.ControlQueue(
            default_control=np.array([1])
        ),
        material=safe_autonomy_simulation.materials.BLACK,
        parent=None,
        children=[],
    ):
        super().__init__(name, dynamics, control_queue, material, parent, children)

    def build_initial_state(self):
        return np.array([0])

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state


def test_init():
    entity = SimpleEntity(name="test_entity")
    assert entity.name == "test_entity"
    assert np.all(entity.state == np.array([0]))
    assert entity.dynamics.__class__.__name__ == "SimpleDynamics"
    assert entity.control_queue.__class__.__name__ == "ControlQueue"
    assert entity.last_control is None
    assert np.all(entity.state_dot == np.zeros_like(entity.state))
    assert len(entity.children) == 0
    assert entity.parent is None
    assert entity.material is safe_autonomy_simulation.materials.BLACK


def test_init_parent():
    parent = SimpleEntity(name="parent")
    entity = SimpleEntity(name="test_entity", parent=parent)
    assert entity.parent == parent
    assert entity in parent.children


def test_init_children():
    child1 = SimpleEntity(name="child1")
    child2 = SimpleEntity(name="child2")
    entity = SimpleEntity(name="test_entity", children={child1, child2})
    assert child1 in entity.children
    assert child2 in entity.children
    assert child1.parent == entity
    assert child2.parent == entity


def test_reset():
    child1 = SimpleEntity(name="child1")
    child2 = SimpleEntity(name="child2")
    entity = SimpleEntity(name="test_entity", children={child1, child2})
    entity.step(step_size=1)
    entity.add_control(np.array([1]))
    entity.reset()
    assert np.all(entity.state == np.array([0]))
    assert entity.last_control is None
    assert np.all(entity.state_dot == np.zeros_like(entity.state))
    assert entity.control_queue.empty()
    for child in entity.children:
        assert np.all(child.state == np.array([0]))
        assert child.last_control is None
        assert np.all(child.state_dot == np.zeros_like(child.state))
        assert child.control_queue.empty()


def test__pre_step():
    child1 = SimpleEntity(name="child1")
    child2 = SimpleEntity(name="child2")
    entity = SimpleEntity(name="test_entity", children={child1, child2})
    entity._pre_step(step_size=1)
    assert np.all(entity.state == np.array([0]))
    assert entity.last_control is None
    assert np.all(entity.state_dot == np.zeros_like(entity.state))
    assert entity.control_queue.empty()
    for child in entity.children:
        assert np.all(child.state == np.array([0]))
        assert child.last_control is None
        assert np.all(child.state_dot == np.zeros_like(child.state))
        assert child.control_queue.empty()


def test__post_step():
    child1 = SimpleEntity(name="child1")
    child2 = SimpleEntity(name="child2")
    entity = SimpleEntity(name="test_entity", children={child1, child2})
    entity._post_step(step_size=1)
    assert np.all(entity.state == np.array([0]))
    assert entity.last_control is None
    assert np.all(entity.state_dot == np.zeros_like(entity.state))
    assert entity.control_queue.empty()
    for child in entity.children:
        assert np.all(child.state == np.array([0]))
        assert child.last_control is None
        assert np.all(child.state_dot == np.zeros_like(child.state))
        assert child.control_queue.empty()


def test_step():
    child1 = SimpleEntity(name="child1")
    child2 = SimpleEntity(name="child2")
    entity = SimpleEntity(name="test_entity", children={child1, child2})
    entity.step(step_size=2)
    assert np.all(entity.state == np.array([2]))
    assert np.all(entity.last_control == np.array([1]))
    assert np.all(entity.state_dot == np.array([1]))
    for child in entity.children:
        assert np.all(child.state == np.array([2]))
        assert np.all(child.last_control == np.array([1]))
        assert np.all(child.state_dot == np.array([1]))


def test_add_control():
    entity = SimpleEntity(name="test_entity")
    entity.add_control(np.array([2]))
    assert np.all(entity.control_queue.next_control() == np.array([2]))


def test__is_descendant():
    parent = SimpleEntity(name="parent")
    child = SimpleEntity(name="child", parent=parent)
    grandchild = SimpleEntity(name="grandchild", parent=child)
    assert parent._is_descendant(parent)
    assert not parent._is_descendant(child)
    assert not parent._is_descendant(grandchild)
    assert child._is_descendant(parent)
    assert child._is_descendant(child)
    assert not child._is_descendant(grandchild)
    assert grandchild._is_descendant(parent)
    assert grandchild._is_descendant(child)
    assert grandchild._is_descendant(grandchild)


def test_add_child():
    parent = SimpleEntity(name="parent")
    child = SimpleEntity(name="child")
    parent.add_child(child)
    assert child in parent.children
    assert child.parent == parent


def test_remove_child():
    parent = SimpleEntity(name="parent")
    child = SimpleEntity(name="child")
    parent.add_child(child)
    parent.remove_child(child)
    assert child not in parent.children
    assert child.parent is None
    assert len(parent.children) == 0
