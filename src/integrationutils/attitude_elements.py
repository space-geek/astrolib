from typing import List

from integrationutils.element_set_base import ElementSetBase
from integrationutils.base_objects import Matrix
from integrationutils.base_objects import Vec3d


class AttitudeElementSet(ElementSetBase):
    """Class represents a generic set of attitude elements."""

    def __init__(self, elements: List[Matrix]):
        super().__init__(elements)


class Quaternion(AttitudeElementSet):
    """Class represents a quaternion representation of the attitude of an object, with q1, q2, and q3 as the vector components and q4 as the scalar"""

    def __init__(self, x: float, y: float, z: float, w: float):
        super().__init__([Matrix([[x],[y],[z],[w]])])

    @property
    def x(self) -> float:
        return self._elements[0,0]

    @x.setter
    def x(self, value: float):
        self._elements[0,0] = value

    @property
    def y(self) -> float:
        return self._elements[1,0]

    @y.setter
    def y(self, value: float):
        self._elements[1,0] = value

    @property
    def z(self) -> float:
        return self._elements[2,0]

    @z.setter
    def z(self, value: float):
        self._elements[2,0] = value

    @property
    def w(self) -> float:
        return self._elements[3,0]

    @w.setter
    def w(self, value: float):
        self._elements[3,0] = value

    @property
    def vector(self) -> Vec3d:
        return Vec3d(self.x, self.y, self.z)

    @vector.setter
    def vector(self, value: Vec3d):
        self.x = value.x
        self.y = value.y
        self.z = value.z