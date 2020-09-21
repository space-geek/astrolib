from collections import OrderedDict
from enum import IntEnum
from typing import List

from integrationutils.base_objects import Matrix
from integrationutils.base_objects import TimeSpan


class ElementSetBase():
    """ Class represents a set of generic state vector elements, e.g.
    """

    def __init__(self, elements: List[Matrix]):
        self._elements = Matrix.from_column_matrices(elements)

    @property
    def num_elements(self) -> int:
        return self._elements.num_rows

    def to_column_matrix(self) -> Matrix:
        return self._elements

    def from_column_matrix(self, value: Matrix):
        if not isinstance(value, Matrix) or value.num_cols != 1:
            raise ValueError("Input value must be a column matrix.")
        if value.num_rows != self.num_elements:
            raise ValueError(f"Input column matrix must have {self.num_elements} elements.")
        self._elements = value


class StateVector():

    def __init__(self):
        self.epoch = TimeSpan.undefined()
        self.components: List[ElementSetBase] = []

    @property
    def component_types(self) -> List[ElementSetBase]:
        return [x.__class__ for x in self.components]

    def to_column_matrix(self) -> Matrix:
        return Matrix.from_column_matrices([x.to_column_matrix() for x in self.components])
