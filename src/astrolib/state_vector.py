""" TODO: Module docstring
"""
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Type

from astrolib import Matrix
from astrolib import TimeSpan
from astrolib import Vec3d
from astrolib.base_objects import ElementSetBase
from astrolib.orbit_elements import CartesianElements
from astrolib.orbit_elements import OrbitElementSet


class StateVector():

    @classmethod
    def from_column_matrix(cls, epoch: TimeSpan, elements: Matrix):
        raise NotImplementedError

    def __init__(self, epoch: TimeSpan, components: List[ElementSetBase]):
        self.epoch = epoch
        self._components = components

    def __iter__(self) -> Iterator[ElementSetBase]:
        for component in self._components:
            yield component

    @property
    def num_elements(self) -> int:
        return sum([x.num_elements for x in self._components])

    @property
    def component_types(self) -> Tuple[Type]:
        return tuple(type(x) for x in self._components)

    @property
    def component_indices(self) -> List[Tuple[int, int]]:
        component_indices = []
        start_ind = 0
        for component in self._components:
            end_ind = start_ind + component.num_elements - 1
            component_indices.append((start_ind, end_ind))
            start_ind = end_ind + 1
        return component_indices

    @property
    def rates(self) -> Matrix:
        raise NotImplementedError

    def to_column_matrix(self) -> Matrix:
        return Matrix.from_column_matrices([x.to_column_matrix() for x in self._components])


class OrbitStateVector(StateVector):

    def __init__(self, epoch: TimeSpan, elements: OrbitElementSet):
        super().__init__(epoch, components=[elements])

    @property
    def elements(self) -> OrbitElementSet:
        return self._components[0]

    @elements.setter
    def elements(self, value: OrbitElementSet):
        self._components[0] = value


class CartesianStateVector(OrbitStateVector):

    @classmethod
    def from_column_matrix(cls, epoch: TimeSpan, elements: Matrix) -> 'CartesianStateVector':
        if len(elements) != 6:
            raise ValueError(f"The input matrix is of length {len(elements)} but needs to be of length {CartesianElements.num_elements}")
        return cls(epoch, CartesianElements(Vec3d(*elements[:3].get_col(0)),
                                            Vec3d(*elements[3:].get_col(0))))

    def __init__(self, epoch: TimeSpan = TimeSpan.zero(), elements: CartesianElements = CartesianElements()):
        super().__init__(epoch, elements)

    def __str__(self) -> str:
        return f"Cartesian Orbit State Vector\n\tEpoch: {self.epoch}\n\tElements: {self.elements}"

    @property
    def rates(self) -> Matrix:
        return self.elements.velocity
