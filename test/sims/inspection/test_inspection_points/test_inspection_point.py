import pytest
import numpy as np
import safe_autonomy_simulation


@pytest.mark.parametrize(
    "position, inspected, inspector, weight, parent, name",
    [
        (
            np.array([1, 0, 0]),
            False,
            None,
            1,
            safe_autonomy_simulation.entities.Point(
                name="parent", position=np.array([0, 0, 0])
            ),
            "point",
        ),
        (
            np.array([0, 1, 0]),
            True,
            safe_autonomy_simulation.entities.Point(
                name="inspector", position=np.array([0, 0, 0])
            ),
            1,
            safe_autonomy_simulation.entities.Point(
                name="parent", position=np.array([0, 0, 0])
            ),
            "point",
        ),
        (
            np.array([0, 0, 1]),
            False,
            None,
            1,
            safe_autonomy_simulation.entities.Point(
                name="parent", position=np.array([0, 0, 0])
            ),
            "point",
        ),
    ],
)
def test_init(position, inspected, inspector, weight, parent, name):
    inspection_point = safe_autonomy_simulation.sims.inspection.InspectionPoint(
        position=position,
        inspected=inspected,
        inspector=inspector,
        weight=weight,
        parent=parent,
        name=name,
    )
    assert np.all(inspection_point.position == position)
    assert inspection_point.inspected == inspected
    assert inspection_point.inspector == inspector
    assert inspection_point.weight == weight
    assert inspection_point.parent == parent
    assert inspection_point.name == name
    assert np.all(
        inspection_point.state
        == np.concatenate((position, np.zeros(3), [weight], [inspected]))
    )


def test_build_initial_state():
    inspection_point = safe_autonomy_simulation.sims.inspection.InspectionPoint(
        position=np.array([1, 0, 0]),
        inspected=False,
        inspector=None,
        weight=1,
        parent=safe_autonomy_simulation.entities.Point(
            name="parent", position=np.array([0, 0, 0])
        ),
        name="point",
    )
    assert np.all(
        inspection_point.build_initial_state()
        == np.concatenate(
            (
                np.array([1, 0, 0]),
                np.zeros(3),
                np.array([0, 0, 0, 1]),
                np.zeros(3),
                [1],
                [0],
            )
        )
    )


def test_state():
    inspection_point = safe_autonomy_simulation.sims.inspection.InspectionPoint(
        position=np.array([1, 0, 0]),
        inspected=False,
        inspector=None,
        weight=1,
        parent=safe_autonomy_simulation.entities.Point(
            name="parent", position=np.array([0, 0, 0])
        ),
        name="point",
    )
    assert np.all(
        inspection_point.state
        == np.concatenate((np.array([1, 0, 0]), np.zeros(3), [1], [0]))
    )


@pytest.mark.parametrize(
    "new_state, expected",
    [
        (np.array([1, 0, 0, 0, 0, 0, 1, 0]), np.array([1, 0, 0, 0, 0, 0, 1, 0])),
        (np.array([0, 1, 0, 0, 0, 0, 1, 0]), np.array([0, 1, 0, 0, 0, 0, 1, 0])),
        (np.array([0, 0, 1, 0, 0, 0, 1, 0]), np.array([0, 0, 1, 0, 0, 0, 1, 0])),
    ],
)
def test_set_state(new_state, expected):
    inspection_point = safe_autonomy_simulation.sims.inspection.InspectionPoint(
        position=np.array([1, 0, 0]),
        inspected=False,
        inspector=None,
        weight=1,
        parent=safe_autonomy_simulation.entities.Point(
            name="parent", position=np.array([0, 0, 0])
        ),
        name="point",
    )
    inspection_point.state = new_state
    assert np.all(inspection_point.state == expected)


@pytest.mark.parametrize(
    "inspector",
    [
        safe_autonomy_simulation.entities.Point(
            name="inspector", position=np.array([0, 0, 0])
        ),
        None,
    ],
)
def test_inspector(inspector):
    inspection_point = safe_autonomy_simulation.sims.inspection.InspectionPoint(
        position=np.array([1, 0, 0]),
        inspected=False,
        inspector=inspector,
        weight=1,
        parent=safe_autonomy_simulation.entities.Point(
            name="parent", position=np.array([0, 0, 0])
        ),
        name="point",
    )
    assert inspection_point.inspector == inspector


@pytest.mark.parametrize(
    "inspector",
    [
        safe_autonomy_simulation.entities.Point(
            name="inspector", position=np.array([0, 0, 0])
        ),
        None,
    ],
)
def test_set_inspector(inspector):
    inspection_point = safe_autonomy_simulation.sims.inspection.InspectionPoint(
        position=np.array([1, 0, 0]),
        inspected=False,
        inspector=None,
        weight=1,
        parent=safe_autonomy_simulation.entities.Point(
            name="parent", position=np.array([0, 0, 0])
        ),
        name="point",
    )
    inspection_point.inspector = inspector
    assert inspection_point.inspector == inspector


@pytest.mark.parametrize(
    "inspected",
    [
        True,
        False,
    ],
)
def test_inspected(inspected):
    inspection_point = safe_autonomy_simulation.sims.inspection.InspectionPoint(
        position=np.array([1, 0, 0]),
        inspected=inspected,
        inspector=None,
        weight=1,
        parent=safe_autonomy_simulation.entities.Point(
            name="parent", position=np.array([0, 0, 0])
        ),
        name="point",
    )
    assert inspection_point.inspected == inspected


@pytest.mark.parametrize(
    "inspected",
    [
        True,
        False,
    ],
)
def test_set_inspected(inspected):
    inspection_point = safe_autonomy_simulation.sims.inspection.InspectionPoint(
        position=np.array([1, 0, 0]),
        inspected=False,
        inspector=None,
        weight=1,
        parent=safe_autonomy_simulation.entities.Point(
            name="parent", position=np.array([0, 0, 0])
        ),
        name="point",
    )
    inspection_point.inspected = inspected
    assert inspection_point.inspected == inspected


@pytest.mark.parametrize(
    "weight",
    [
        1,
        2,
    ],
)
def test_weight(weight):
    inspection_point = safe_autonomy_simulation.sims.inspection.InspectionPoint(
        position=np.array([1, 0, 0]),
        inspected=False,
        inspector=None,
        weight=weight,
        parent=safe_autonomy_simulation.entities.Point(
            name="parent", position=np.array([0, 0, 0])
        ),
        name="point",
    )
    assert inspection_point.weight == weight


@pytest.mark.parametrize(
    "weight",
    [
        1,
        2,
    ],
)
def test_set_weight(weight):
    inspection_point = safe_autonomy_simulation.sims.inspection.InspectionPoint(
        position=np.array([1, 0, 0]),
        inspected=False,
        inspector=None,
        weight=1,
        parent=safe_autonomy_simulation.entities.Point(
            name="parent", position=np.array([0, 0, 0])
        ),
        name="point",
    )
    inspection_point.weight = weight
    assert inspection_point.weight == weight
