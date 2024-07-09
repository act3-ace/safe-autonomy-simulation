import typing
import typing_extensions


T = typing.TypeVar("T")


class MutableSet:
    """An unordered set of mutable objects

    Guarantees that each object is unique in the set.
    Unlike the built-in set, this set allows for mutable
    objects to be elements of the set.
    """

    def __init__(self, elements: list = []):
        self._set: dict = {}
        for element in elements:
            self.add(element)

    def __contains__(self, element) -> bool:
        return id(element) in self._set

    def __len__(self) -> int:
        return len(self._set)

    def __iter__(self):
        return iter(self._set.values())

    def __repr__(self) -> str:
        return f"MutableSet({list(self._set.values())})"

    def __eq__(self, other: typing.Any) -> bool:
        if not isinstance(other, MutableSet):
            return False
        return self._set == other._set

    def __ne__(self, other: typing.Any) -> bool:
        return not self == other

    def __or__(self, other: typing_extensions.Self) -> typing_extensions.Self:
        return self.union(other)

    def __and__(self, other: typing_extensions.Self) -> typing_extensions.Self:
        return self.intersection(other)

    def __sub__(self, other: typing_extensions.Self) -> typing_extensions.Self:
        return self.difference(other)

    def __xor__(self, other: typing_extensions.Self) -> typing_extensions.Self:
        return self.symmetric_difference(other)

    def __ior__(self, other: typing_extensions.Self):
        self.update(other)

    def __iand__(self, other: typing_extensions.Self):
        self.intersection_update(other)

    def __isub__(self, other: typing_extensions.Self):
        self.difference_update(other)

    def __ixor__(self, other: typing_extensions.Self):
        self.symmetric_difference_update(other)

    def __ge__(self, other: typing_extensions.Self) -> bool:
        return self.issuperset(other)

    def __gt__(self, other: typing_extensions.Self) -> bool:
        return self.issuperset(other) and self != other

    def __le__(self, other: typing_extensions.Self) -> bool:
        return self.issubset(other)

    def __lt__(self, other: typing_extensions.Self) -> bool:
        return self.issubset(other) and self != other

    def add(self, element):
        """Add element to set

        Parameters
        ----------
        element
            Element to add to set
        """
        self._set[id(element)] = element

    def remove(self, element):
        """Remove element from set

        Parameters
        ----------
        element : T
            Element to remove from set
        """
        assert element in self, f"Element {element} not found in set"
        del self._set[id(element)]

    def clear(self):
        """Remove all elements from the set"""
        self._set.clear()

    def copy(self) -> typing_extensions.Self:
        """Return a shallow copy of the set

        Returns
        -------
        MutableSet
            Shallow copy of the set
        """
        return self.__class__(elements=list(self._set.values()))

    def pop(self) -> typing.Any:
        """Remove and return an arbitrary element from the set

        Returns
        -------
        Any
            Arbitrary element from the set
        """
        return self._set.popitem()[1]

    def difference(self, other: typing_extensions.Self) -> typing_extensions.Self:
        """Return the difference of two sets as a new set

        Parameters
        ----------
        other : MutableSet
            Set to take the difference with

        Returns
        -------
        MutableSet
            New set with elements in this set but not in the other set
        """
        return self.__class__(
            elements=[element for element in self if element not in other]
        )

    def difference_update(self, other: typing_extensions.Self):
        """Remove all elements of another set from this set

        Parameters
        ----------
        other : MutableSet
            Set of elements to remove from this set
        """
        self._set = {id(element): element for element in self if element not in other}

    def intersection(self, other: typing_extensions.Self) -> typing_extensions.Self:
        """Return the intersection of two sets as a new set

        Parameters
        ----------
        other : MutableSet
            Set to take the intersection with

        Returns
        -------
        MutableSet
            New set with elements common to both sets
        """
        return self.__class__(
            elements=[element for element in self if element in other]
        )

    def intersection_update(self, other: typing_extensions.Self):
        """Remove all elements from this set that are not in another set

        Parameters
        ----------
        other : MutableSet
            Set to take the intersection with
        """
        self._set = {id(element): element for element in self if element in other}

    def isdisjoint(self, other: typing_extensions.Self) -> bool:
        """Return True if the set has no elements in common with another set

        Parameters
        ----------
        other : MutableSet
            Set to compare with

        Returns
        -------
        bool
            True if the sets have no elements in common
        """
        return len(self.intersection(other)) == 0

    def issubset(self, other: typing_extensions.Self) -> bool:
        """Report whether another set contains this set

        Parameters
        ----------
        other : MutableSet
            Set to compare with

        Returns
        -------
        bool
            True if every element in this set is in the other set
        """
        return len(self.difference(other)) == 0

    def issuperset(self, other: typing_extensions.Self) -> bool:
        """Report whether this set contains another set

        Parameters
        ----------
        other : MutableSet
            Set to compare with

        Returns
        -------
        bool
            True if every element in the other set is in this set
        """
        return other.issubset(self)

    def symmetric_difference(
        self, other: typing_extensions.Self
    ) -> typing_extensions.Self:
        """Return the symmetric difference of two sets as a new set

        Parameters
        ----------
        other : MutableSet
            Set to take the symmetric difference with

        Returns
        -------
        MutableSet
            New set with elements in either this set or the other set but not both
        """
        return self.difference(other).union(other.difference(self))

    def symmetric_difference_update(self, other: typing_extensions.Self):
        """Update this set with the symmetric difference of itself and another set

        Parameters
        ----------
        other : MutableSet
            Set to take the symmetric difference with
        """
        self._set = {
            id(element): element
            for element in self.difference(other).union(other.difference(self))
        }

    def union(self, other: typing_extensions.Self) -> typing_extensions.Self:
        """Return the union of two sets as a new set

        Parameters
        ----------
        other : MutableSet
            Set to take the union with

        Returns
        -------
        MutableSet
            New set with all elements from both sets
        """
        return self.__class__(elements=list(self) + list(other))

    def update(self, other: typing_extensions.Self):
        """Update this set with the union of itself and another set

        Parameters
        ----------
        other : MutableSet
            Set to take the union with
        """
        self._set = {id(element): element for element in self.union(other)}


class TypedSet(typing.Generic[T], MutableSet):
    """An unordered set of mutable objects with a specified type

    Guarantees that each object is unique in the set and that
    all elements are of the specified type.
    """

    def __init__(self, type: typing.Type[T], elements: list[T] = []):
        self.type = type
        super().__init__(elements)

    def add(self, element: T):
        """Add element to set

        Parameters
        ----------
        element : T
            Element to add to set
        """
        if not isinstance(element, self.type):
            raise TypeError(f"Element '{element}' is not of type {self.type}")
        super().add(element)

    def remove(self, element: T):
        """Remove element from set

        Parameters
        ----------
        element : T
            Element to remove from set
        """
        if not isinstance(element, self.type):
            raise TypeError(f"Element '{element}' is not of type {self.type}")
        super().remove(element)

    def copy(self) -> typing_extensions.Self:
        """Return a shallow copy of the set

        Returns
        -------
        TypedSet
            Shallow copy of the set
        """
        return self.__class__(type=self.type, elements=list(self._set.values()))

    def pop(self) -> T:
        """Remove and return an arbitrary element from the set

        Returns
        -------
        T
            Arbitrary element from the set
        """
        return super().pop()

    def difference(self, other: typing_extensions.Self) -> typing_extensions.Self:
        """Return the difference of two sets as a new set

        Parameters
        ----------
        other : TypedSet
            Set to take the difference with

        Returns
        -------
        TypedSet
            New set with elements in this set but not in the other set
        """
        if not other.type == self.type:
            raise TypeError(f"Cannot take difference with set of type {other.type}")
        return self.__class__(
            type=self.type,
            elements=[element for element in self if element not in other],
        )

    def intersection(self, other: typing_extensions.Self) -> typing_extensions.Self:
        """Return the intersection of two sets as a new set

        Parameters
        ----------
        other : TypedSet
            Set to take the intersection with

        Returns
        -------
        TypedSet
            New set with elements common to both sets
        """
        if not other.type == self.type:
            raise TypeError(f"Cannot take intersection with set of type {other.type}")
        return self.__class__(
            type=self.type, elements=[element for element in self if element in other]
        )

    def intersection_update(self, other: typing_extensions.Self):
        """Remove all elements from this set that are not in another set

        Parameters
        ----------
        other : TypedSet
            Set to take the intersection with
        """
        if not other.type == self.type:
            raise TypeError(f"Cannot take intersection with set of type {other.type}")
        self._set = {id(element): element for element in self if element in other}

    def union(self, other: typing_extensions.Self) -> typing_extensions.Self:
        """Return the union of two sets as a new set

        Parameters
        ----------
        other : TypedSet
            Set to take the union with

        Returns
        -------
        TypedSet
            New set with all elements from both sets
        """
        if not other.type == self.type:
            raise TypeError(f"Cannot take union with set of type {other.type}")
        return self.__class__(type=self.type, elements=list(self) + list(other))

    def update(self, other: typing_extensions.Self):
        """Update this set with the union of itself and another set

        Parameters
        ----------
        other : TypedSet
            Set to take the union with
        """
        if not other.type == self.type:
            raise TypeError(f"Cannot take union with set of type {other.type}")
        self._set = {id(element): element for element in self.union(other)}

    def difference_update(self, other: typing_extensions.Self):
        """Remove all elements of another set from this set

        Parameters
        ----------
        other : TypedSet
            Set of elements to remove from this set
        """
        if not other.type == self.type:
            raise TypeError(f"Cannot take difference with set of type {other.type}")
        self._set = {id(element): element for element in self if element not in other}

    def symmetric_difference(
        self, other: typing_extensions.Self
    ) -> typing_extensions.Self:
        """Return the symmetric difference of two sets as a new set

        Parameters
        ----------
        other : TypedSet
            Set to take the symmetric difference with

        Returns
        -------
        TypedSet
            New set with elements in either this set or the other set but not both
        """
        if not other.type == self.type:
            raise TypeError(
                f"Cannot take symmetric difference with set of type {other.type}"
            )
        return self.difference(other).union(other.difference(self))

    def symmetric_difference_update(self, other: typing_extensions.Self):
        """Update this set with the symmetric difference of itself and another set

        Parameters
        ----------
        other : TypedSet
            Set to take the symmetric difference with
        """
        if not other.type == self.type:
            raise TypeError(
                f"Cannot take symmetric difference with set of type {other.type}"
            )
        self._set = {
            id(element): element
            for element in self.difference(other).union(other.difference(self))
        }

    def issubset(self, other: typing_extensions.Self) -> bool:
        """Report whether another set contains this set

        Parameters
        ----------
        other : TypedSet
            Set to compare with

        Returns
        -------
        bool
            True if every element in this set is in the other set
        """
        if not other.type == self.type:
            raise TypeError(f"Cannot compare with set of type {other.type}")
        return len(self.difference(other)) == 0

    def issuperset(self, other: typing_extensions.Self) -> bool:
        """Report whether this set contains another set

        Parameters
        ----------
        other : TypedSet
            Set to compare with

        Returns
        -------
        bool
            True if every element in the other set is in this set
        """
        if not other.type == self.type:
            raise TypeError(f"Cannot compare with set of type {other.type}")
        return other.issubset(self)

    def isdisjoint(self, other: typing_extensions.Self) -> bool:
        """Return True if the set has no elements in common with another set

        Parameters
        ----------
        other : MutableSet
            Set to compare with

        Returns
        -------
        bool
            True if the sets have no elements in common
        """
        if not other.type == self.type:
            raise TypeError(f"Cannot compare with set of type {other.type}")
        return len(self.intersection(other)) == 0

    def __or__(self, other: typing_extensions.Self) -> typing_extensions.Self:
        return self.union(other)

    def __and__(self, other: typing_extensions.Self) -> typing_extensions.Self:
        return self.intersection(other)

    def __sub__(self, other: typing_extensions.Self) -> typing_extensions.Self:
        return self.difference(other)

    def __xor__(self, other: typing_extensions.Self) -> typing_extensions.Self:
        return self.symmetric_difference(other)

    def __ior__(self, other: typing_extensions.Self):
        self.update(other)

    def __iand__(self, other: typing_extensions.Self):
        self.intersection_update(other)

    def __isub__(self, other: typing_extensions.Self):
        self.difference_update(other)

    def __ixor__(self, other: typing_extensions.Self):
        self.symmetric_difference_update(other)

    def __ge__(self, other: typing_extensions.Self) -> bool:
        return self.issuperset(other)

    def __gt__(self, other: typing_extensions.Self) -> bool:
        return self.issuperset(other) and self != other

    def __le__(self, other: typing_extensions.Self) -> bool:
        return self.issubset(other)

    def __lt__(self, other: typing_extensions.Self) -> bool:
        return self.issubset(other) and self != other

    def __contains__(self, element: T) -> bool:
        return super().__contains__(element)

    def __repr__(self) -> str:
        return f"TypedSet[{self.type}]({[element for element in self]})"

    def __eq__(self, other: typing.Any) -> bool:
        if not isinstance(other, TypedSet):
            return False
        return self._set == other._set

    def __ne__(self, other: typing.Any) -> bool:
        return not self == other
