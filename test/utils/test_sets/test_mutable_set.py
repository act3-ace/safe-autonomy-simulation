import pytest
import safe_autonomy_simulation


def test_init_no_args():
    mutable_set = safe_autonomy_simulation.utils.MutableSet()
    assert len(mutable_set) == 0
    assert str(mutable_set) == "MutableSet([])"
    assert repr(mutable_set) == "MutableSet([])"


@pytest.mark.parametrize(
    "elements",
    [
        [1, 2, 3],
        ["a", "b", "c"],
        [1, "a", 2.0],
    ],
)
def test_init_with_args(elements):
    mutable_set = safe_autonomy_simulation.utils.MutableSet(elements)
    assert len(mutable_set) == len(elements)
    assert str(mutable_set) == f"MutableSet({elements})"
    assert repr(mutable_set) == f"MutableSet({elements})"
    for element in elements:
        assert element in mutable_set


def test_add():
    mutable_set = safe_autonomy_simulation.utils.MutableSet()
    mutable_set.add(1)
    assert len(mutable_set) == 1
    assert 1 in mutable_set


def test_remove():
    mutable_set = safe_autonomy_simulation.utils.MutableSet([1, 2, 3])
    mutable_set.remove(2)
    assert len(mutable_set) == 2
    assert 2 not in mutable_set


def test_clear():
    mutable_set = safe_autonomy_simulation.utils.MutableSet([1, 2, 3])
    mutable_set.clear()
    assert len(mutable_set) == 0


def test_pop():
    mutable_set = safe_autonomy_simulation.utils.MutableSet([1, 2, 3])
    element = mutable_set.pop()
    assert len(mutable_set) == 2
    assert element not in mutable_set


def test_difference():
    mutable_set1 = safe_autonomy_simulation.utils.MutableSet([1, 2, 3])
    mutable_set2 = safe_autonomy_simulation.utils.MutableSet([2, 3, 4])
    difference = mutable_set1.difference(mutable_set2)
    assert len(difference) == 1
    assert 1 in difference
    assert 2 not in difference
    assert 3 not in difference
    assert 4 not in difference


def test_difference_update():
    mutable_set1 = safe_autonomy_simulation.utils.MutableSet([1, 2, 3])
    mutable_set2 = safe_autonomy_simulation.utils.MutableSet([2, 3, 4])
    mutable_set1.difference_update(mutable_set2)
    assert len(mutable_set1) == 1
    assert 1 in mutable_set1
    assert 2 not in mutable_set1
    assert 3 not in mutable_set1
    assert 4 not in mutable_set1


def test_intersection():
    mutable_set1 = safe_autonomy_simulation.utils.MutableSet([1, 2, 3])
    mutable_set2 = safe_autonomy_simulation.utils.MutableSet([2, 3, 4])
    intersection = mutable_set1.intersection(mutable_set2)
    assert len(intersection) == 2
    assert 1 not in intersection
    assert 2 in intersection
    assert 3 in intersection
    assert 4 not in intersection


def test_intersection_update():
    mutable_set1 = safe_autonomy_simulation.utils.MutableSet([1, 2, 3])
    mutable_set2 = safe_autonomy_simulation.utils.MutableSet([2, 3, 4])
    mutable_set1.intersection_update(mutable_set2)
    assert len(mutable_set1) == 2
    assert 1 not in mutable_set1
    assert 2 in mutable_set1
    assert 3 in mutable_set1
    assert 4 not in mutable_set1


def test_isdisjoint():
    mutable_set1 = safe_autonomy_simulation.utils.MutableSet([1, 2, 3])
    mutable_set2 = safe_autonomy_simulation.utils.MutableSet([4, 5, 6])
    assert mutable_set1.isdisjoint(mutable_set2)


def test_issubset():
    mutable_set1 = safe_autonomy_simulation.utils.MutableSet([1, 2, 3])
    mutable_set2 = safe_autonomy_simulation.utils.MutableSet([1, 2])
    assert mutable_set2.issubset(mutable_set1)


def test_issuperset():
    mutable_set1 = safe_autonomy_simulation.utils.MutableSet([1, 2, 3])
    mutable_set2 = safe_autonomy_simulation.utils.MutableSet([1, 2])
    assert mutable_set1.issuperset(mutable_set2)


def test_symmetric_difference():
    mutable_set1 = safe_autonomy_simulation.utils.MutableSet([1, 2, 3])
    mutable_set2 = safe_autonomy_simulation.utils.MutableSet([2, 3, 4])
    symmetric_difference = mutable_set1.symmetric_difference(mutable_set2)
    assert len(symmetric_difference) == 2
    assert 1 in symmetric_difference
    assert 2 not in symmetric_difference
    assert 3 not in symmetric_difference
    assert 4 in symmetric_difference


def test_symmetric_difference_update():
    mutable_set1 = safe_autonomy_simulation.utils.MutableSet([1, 2, 3])
    mutable_set2 = safe_autonomy_simulation.utils.MutableSet([2, 3, 4])
    mutable_set1.symmetric_difference_update(mutable_set2)
    assert len(mutable_set1) == 2
    assert 1 in mutable_set1
    assert 2 not in mutable_set1
    assert 3 not in mutable_set1
    assert 4 in mutable_set1


def test_union():
    mutable_set1 = safe_autonomy_simulation.utils.MutableSet([1, 2, 3])
    mutable_set2 = safe_autonomy_simulation.utils.MutableSet([2, 3, 4])
    union = mutable_set1.union(mutable_set2)
    assert len(union) == 4
    assert 1 in union
    assert 2 in union
    assert 3 in union
    assert 4 in union


def test_update():
    mutable_set1 = safe_autonomy_simulation.utils.MutableSet([1, 2, 3])
    mutable_set2 = safe_autonomy_simulation.utils.MutableSet([2, 3, 4])
    mutable_set1.update(mutable_set2)
    assert len(mutable_set1) == 4
    assert 1 in mutable_set1
    assert 2 in mutable_set1
    assert 3 in mutable_set1
    assert 4 in mutable_set1
