import pytest
import safe_autonomy_simulation


def test_init_no_args():
    typed_set = safe_autonomy_simulation.utils.TypedSet(type=int)
    assert len(typed_set) == 0
    assert str(typed_set) == "TypedSet[<class 'int'>]([])"
    assert repr(typed_set) == "TypedSet[<class 'int'>]([])"


def test_init_with_args():
    typed_set = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    assert len(typed_set) == 3
    assert str(typed_set) == "TypedSet[<class 'int'>]([1, 2, 3])"
    assert repr(typed_set) == "TypedSet[<class 'int'>]([1, 2, 3])"


def test_init_with_args_wrong_type():
    with pytest.raises(TypeError):
        safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, "a", 2.0])


def test_add():
    typed_set = safe_autonomy_simulation.utils.TypedSet(type=int)
    typed_set.add(1)
    assert len(typed_set) == 1
    assert 1 in typed_set


def test_add_wrong_type():
    typed_set = safe_autonomy_simulation.utils.TypedSet(type=int)
    with pytest.raises(TypeError):
        typed_set.add("a")


def test_remove():
    typed_set = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set.remove(2)
    assert len(typed_set) == 2
    assert 2 not in typed_set


def test_remove_wrong_type():
    typed_set = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    with pytest.raises(TypeError):
        typed_set.remove("a")


def test_clear():
    typed_set = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set.clear()
    assert len(typed_set) == 0


def test_pop():
    typed_set = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    element = typed_set.pop()
    assert len(typed_set) == 2
    assert element not in typed_set


def test_difference():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[2, 3, 4])
    difference = typed_set1.difference(typed_set2)
    assert type(difference) == type(typed_set1)
    assert len(difference) == 1
    assert 1 in difference
    assert 2 not in difference
    assert 3 not in difference
    assert 4 not in difference


def test_difference_wrong_type():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(
        type=str, elements=["a", "b", "c"]
    )
    with pytest.raises(TypeError):
        typed_set1.difference(typed_set2)


def test_intersection():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[2, 3, 4])
    intersection = typed_set1.intersection(typed_set2)
    assert type(intersection) == type(typed_set1)
    assert len(intersection) == 2
    assert 1 not in intersection
    assert 2 in intersection
    assert 3 in intersection


def test_intersection_wrong_type():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(
        type=str, elements=["a", "b", "c"]
    )
    with pytest.raises(TypeError):
        typed_set1.intersection(typed_set2)


def test_intersection_update():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[2, 3, 4])
    typed_set1.intersection_update(typed_set2)
    assert len(typed_set1) == 2
    assert 1 not in typed_set1
    assert 2 in typed_set1
    assert 3 in typed_set1
    assert 4 not in typed_set1


def test_intersection_update_wrong_type():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(
        type=str, elements=["a", "b", "c"]
    )
    with pytest.raises(TypeError):
        typed_set1.intersection_update(typed_set2)


def test_union():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[2, 3, 4])
    union = typed_set1.union(typed_set2)
    assert type(union) == type(typed_set1)
    assert len(union) == 4
    assert 1 in union
    assert 2 in union
    assert 3 in union
    assert 4 in union


def test_union_wrong_type():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(
        type=str, elements=["a", "b", "c"]
    )
    with pytest.raises(TypeError):
        typed_set1.union(typed_set2)


def test_update():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[2, 3, 4])
    typed_set1.update(typed_set2)
    assert len(typed_set1) == 4
    assert 1 in typed_set1
    assert 2 in typed_set1
    assert 3 in typed_set1
    assert 4 in typed_set1


def test_update_wrong_type():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(
        type=str, elements=["a", "b", "c"]
    )
    with pytest.raises(TypeError):
        typed_set1.update(typed_set2)


def test_difference_update():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[2, 3, 4])
    typed_set1.difference_update(typed_set2)
    assert len(typed_set1) == 1
    assert 1 in typed_set1
    assert 2 not in typed_set1
    assert 3 not in typed_set1
    assert 4 not in typed_set1


def test_difference_update_wrong_type():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(
        type=str, elements=["a", "b", "c"]
    )
    with pytest.raises(TypeError):
        typed_set1.difference_update(typed_set2)


def test_symmetric_difference():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[2, 3, 4])
    symmetric_difference = typed_set1.symmetric_difference(typed_set2)
    assert type(symmetric_difference) == type(typed_set1)
    assert len(symmetric_difference) == 2
    assert 1 in symmetric_difference
    assert 2 not in symmetric_difference
    assert 3 not in symmetric_difference
    assert 4 in symmetric_difference


def test_symmetric_difference_wrong_type():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(
        type=str, elements=["a", "b", "c"]
    )
    with pytest.raises(TypeError):
        typed_set1.symmetric_difference(typed_set2)


def test_symmetric_difference_update():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[2, 3, 4])
    typed_set1.symmetric_difference_update(typed_set2)
    assert len(typed_set1) == 2
    assert 1 in typed_set1
    assert 2 not in typed_set1
    assert 3 not in typed_set1
    assert 4 in typed_set1


def test_symmetric_difference_update_wrong_type():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(
        type=str, elements=["a", "b", "c"]
    )
    with pytest.raises(TypeError):
        typed_set1.symmetric_difference_update(typed_set2)


def test_issubset():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2])
    assert typed_set2.issubset(typed_set1)


def test_issubset_wrong_type():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(type=str, elements=["a", "b"])
    with pytest.raises(TypeError):
        typed_set2.issubset(typed_set1)


def test_issuperset():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2])
    assert typed_set1.issuperset(typed_set2)


def test_issuperset_wrong_type():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(type=str, elements=["a", "b"])
    with pytest.raises(TypeError):
        typed_set1.issuperset(typed_set2)


def test_isdisjoint():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[4, 5, 6])
    assert typed_set1.isdisjoint(typed_set2)


def test_isdisjoint_wrong_type():
    typed_set1 = safe_autonomy_simulation.utils.TypedSet(type=int, elements=[1, 2, 3])
    typed_set2 = safe_autonomy_simulation.utils.TypedSet(
        type=str, elements=["a", "b", "c"]
    )
    with pytest.raises(TypeError):
        typed_set1.isdisjoint(typed_set2)
