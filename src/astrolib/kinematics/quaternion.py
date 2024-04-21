""" Module contains quaternion-related class and function definitions. 
"""

import math
from typing import Iterator
from typing import Self

from astrolib.matrix import Matrix
from astrolib.vector import Vector3


class Quaternion:
    """Class representing a 4-element quaternion."""

    @classmethod
    def identity(cls) -> Self:
        """TODO: Method docstring"""
        return cls(0, 0, 0, 1)

    def __init__(
        self,
        x: int | float,
        y: int | float,
        z: int | float,
        w: int | float,
    ) -> None:
        self._vector: Vector3 = Vector3(x, y, z)
        self._scalar: int | float = w

    def __repr__(self) -> str:
        return f"Quaternion(x={self.x}, y={self.y}, z={self.z}, w={self.w})"

    def __iter__(self) -> Iterator[float]:
        yield self._vector.x
        yield self._vector.y
        yield self._vector.z
        yield self._scalar

    def __eq__(self, other: Self) -> bool:
        return all(
            getattr(self, x) == getattr(other, x)
            for x in (
                "x",
                "y",
                "z",
                "w",
            )
        )

    def __getitem__(
        self,
        index: int | float | slice,
    ) -> int | float | Matrix:
        match index:
            case int() | float() | slice():
                return self.as_matrix()[index]
            case _:
                raise ValueError(f"Unsupported index type {type(index)} provided.")

    def __sub__(self, other: Self | Matrix) -> Matrix:
        match other:
            case Quaternion():
                return Matrix(
                    [
                        *(self._vector - other.vector),
                        self._scalar - other._scalar,
                    ]
                ).transpose()
            case Matrix():
                return self.as_matrix() - other
            case _:
                pass
        return NotImplemented

    def __rsub__(self, other: Matrix) -> Matrix:
        match other:
            case Matrix():
                return other - self.as_matrix()
            case _:
                pass
        return NotImplemented

    def __mul__(self, other: int | float | Matrix | Self) -> Self | Matrix:
        match other:
            case int() | float():
                return Quaternion(
                    other * self.x,
                    other * self.y,
                    other * self.z,
                    other * self.w,
                )
            case Quaternion():
                return Quaternion(
                    *(
                        self.vector.cross(other.vector)
                        + self.w * other.vector
                        + other.w * self.vector
                    ),
                    self.w * other.w - self.vector.dot(other.vector),
                )
            case Matrix():
                if other.size.num_rows == 1:
                    return self.as_matrix().__mul__(other)
                raise ValueError(
                    "The multiplying matrix must be a row matrix but is instead of size "
                    f"{other.size}. Check dimensionality and try again."
                )
            case _:
                pass
        return NotImplemented

    def __rmul__(self, other: float | int | Matrix) -> Self:
        match other:
            case int() | float():
                return Quaternion(
                    other * self.x,
                    other * self.y,
                    other * self.z,
                    other * self.w,
                )
            case Matrix():
                if other.size.num_cols == 4:
                    return other.__mul__(self.as_matrix())
                raise ValueError(
                    "The multiplying matrix must have 4 columns but is instead of size "
                    f"{other.size}. Check dimensionality and try again."
                )
            case _:
                pass
        return NotImplemented

    def __truediv__(self, other: Self) -> Self:
        match other:
            case Quaternion():
                return Quaternion(
                    *(
                        self.vector.cross(other.vector)
                        + self.w * other.vector
                        - other.w * self.vector
                    ),
                    -self.w * other.w - self.vector.dot(other.vector),
                )
            case _:
                pass
        return NotImplemented

    def __neg__(self) -> Self:
        return Quaternion(*(-x for x in self))

    def __round__(self, ndigits: int | None = None):
        return Quaternion(*(round(x, ndigits=ndigits) for x in self))

    @property
    def x(self) -> int | float:
        """The x component of the vector portion of the quaternion."""
        return self._vector.x

    @x.setter
    def x(self, value: int | float) -> None:
        self._vector.x = value

    @property
    def y(self) -> int | float:
        """The y component of the vector portion of the quaternion."""
        return self._vector.y

    @y.setter
    def y(self, value: int | float) -> None:
        self._vector.y = value

    @property
    def z(self) -> int | float:
        """The z component of the vector portion of the quaternion."""
        return self._vector.z

    @z.setter
    def z(self, value: int | float) -> None:
        self._vector.z = value

    @property
    def w(self) -> int | float:
        """The scalar component of the quaternion."""
        return self._scalar

    @w.setter
    def w(self, value: int | float) -> None:
        """The scalar component of the quaternion."""
        self._scalar = value

    @property
    def vector(self) -> Vector3:
        """The vector components of the quaternion."""
        return self._vector

    @vector.setter
    def vector(self, value: Vector3) -> None:
        """The vector components of the quaternion."""
        self._vector.x = value.x
        self._vector.y = value.y
        self._vector.z = value.z

    def norm(self) -> float:
        """Returns the Euclidean norm of the calling quaternion."""
        return math.sqrt(self.squared_norm())

    def squared_norm(self) -> float:
        """Returns the square of the Euclidean norm of the calling vector."""
        return self.transpose() * self

    def normalize(self) -> Self:
        """Normalizes the calling quaternion in place by its Euclidean norm."""
        m = self.norm()  # TODO Consider if check for 0 magnitude is needed
        self.x /= m
        self.y /= m
        self.z /= m
        self.w /= m

    def normalized(self) -> Self:
        """Returns a copy of the calling quaternion, normalized by its Euclidean norm."""
        m = self.norm()
        return Quaternion(
            self.x / m,
            self.y / m,
            self.z / m,
            self.w / m,
        )

    def as_matrix(self) -> Matrix:
        """Returns a copy of the calling quaternion, expressed as a 4x1 column matrix."""
        return Matrix(
            list([x] for x in self)
        )  # TODO remove list construction when possible

    def transpose(self) -> Matrix:
        """Returns a copy of the calling quaternion, expressed as a 1x4 row matrix."""
        return self.as_matrix().transpose()
