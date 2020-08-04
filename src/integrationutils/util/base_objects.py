from math import sqrt
from typing import Union

class Vec3d:
    """Class represents a Euclidean vector."""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> str:
        return f'[x = {self.x}, y = {self.y}, z = {self.z}]'

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vec3d):
            return NotImplemented
        return (self.x == other.x and
                self.y == other.y and
                self.z == other.z)

    def __add__(self, other):
        if not isinstance(other, Vec3d):
            return NotImplemented
        return Vec3d(x = self.x + other.x,
                     y = self.y + other.y,
                     z = self.z + other.z)

    def norm(self) -> float:
        """Returns the Euclidean norm of the calling vector."""
        return sqrt(self.x**2 + self.y**2 + self.z**2)

    def norm_2(self) -> float:
        """Returns the square of the Euclidean norm of the calling vector."""
        return self.x**2 + self.y**2 + self.z**2

    def cross(self, other):
        """Returns the cross product of the calling vector with the argument
           vector, computed as C = A x B for C = A.cross(B).
        """
        if not isinstance(other, Vec3d):
            return NotImplemented
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return Vec3d(x,y,z)

    def dot(self, other) -> float:
        """Returns the dot product of the calling vector with the argument
           vector, computed as C = A * B for C = A.dot(B).
        """
        if not isinstance(other, Vec3d):
            return NotImplemented
        return self.x * other.x + self.y * other.y + self.z * other.z


class Matrix3d:
    """Class represents a 3x3 (3 row, 3 column) matrix."""

    @classmethod
    def zero(cls):
        return cls()

    @classmethod
    def identity(cls):
        return cls(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    @classmethod
    def fill(cls, value: float):
        return cls(*([value]*9))

    def __init__(self, M_11: float = 0.0, M_12: float = 0.0, M_13: float = 0.0,
                       M_21: float = 0.0, M_22: float = 0.0, M_23: float = 0.0,
                       M_31: float = 0.0, M_32: float = 0.0, M_33: float = 0.0):
        self.M_11 = M_11
        self.M_12 = M_12
        self.M_13 = M_13
        self.M_21 = M_21
        self.M_22 = M_22
        self.M_23 = M_23
        self.M_31 = M_31
        self.M_32 = M_32
        self.M_33 = M_33

    def __str__(self) -> str:
        return f"[{self.M_11}, {self.M_12}, {self.M_13};\n{self.M_21}, {self.M_22}, {self.M_23};\n{self.M_31}, {self.M_32}, {self.M_33}]"

    def __add__(self, other):
        if not (isinstance(other, Matrix3d) or isinstance(other, float)):
            return NotImplemented
        if isinstance(other, float):
            other = Matrix3d.fill(other)
        pass

    def __radd_(self, other):
        return self.__add__(other)

    def __mult__(self, other):
        if not (isinstance(other, Matrix3d) or isinstance(other, Vec3d)):
            return NotImplemented
        pass

    def __rmult__(self, other):
        if not isinstance(other, Matrix3d):
            return NotImplemented
        pass

