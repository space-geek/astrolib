from typing import List

from astrolib import Matrix
from astrolib import Vector3
from astrolib.base_objects import ElementSetBase


class AttitudeCoordinates(ElementSetBase):
    """ Class represents a generic set of attitude position coordinates.
    """


class Quaternion(AttitudeCoordinates):
    """Class represents a quaternion representation of the attitude of an object, with q1, q2, and q3 as the vector components and q4 as the scalar"""

    @classmethod
    def identity(cls):
        """Factory method to create a Quaternion representing a zero/identity rotation.

        Returns:
            Quaternion: The Quaternion representing the zero rotation.
        """
        return cls(x=0,y=0,z=0,w=1)

    def __init__(self, x: float, y: float, z: float, w: float):
        super().__init__([Matrix([[x],[y],[z],[w]])])

    def __str__(self) -> str:
        return f'[x = {self.x}, y = {self.y}, z = {self.z}, w = {self.w}]'

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
    def vector(self) -> Vector3:
        return Vector3(self.x, self.y, self.z)

    @vector.setter
    def vector(self, value: Vector3):
        self.x = value.x
        self.y = value.y
        self.z = value.z


class DirectionCosineMatrix(AttitudeCoordinates):

    def __init__(self, dcm: Matrix):
        if not dcm.is_square():
            raise ValueError(f"Input matrix must be square but is instead size {dcm.size}.")
        super().__init__([dcm.to_column_matrix()])

    @property
    def matrix(self) -> Matrix:
        pass


class AttitudeRates(ElementSetBase):

    def __init__(self, rates: Matrix):
        super().__init__(rates)


class AngularVelocity(AttitudeRates):

    def __init__(self, x: float, y: float, z: float):
        super().__init__([Vector3(x,y,z)])


    def __str__(self) -> str:
        return f'[x = {self.x}, y = {self.y}, z = {self.z}]'

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
    def vector(self) -> Vector3:
        return Vector3(self.x, self.y, self.z)

    @vector.setter
    def vector(self, value: Vector3):
        self.x = value.x
        self.y = value.y
        self.z = value.z


class AttitudeElementSet(ElementSetBase):
    """Class represents a generic set of attitude elements."""

    def __init__(self, coordinates: AttitudeCoordinates, rates: AttitudeRates):
        super().__init__([coordinates.to_column_matrix(), rates.to_column_matrix()])
