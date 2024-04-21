""" TODO: Module docstring
"""

import math
from typing import Iterator
from typing import Self

from astrolib import Matrix
from astrolib.constants import MACHINE_EPSILON


class Vector3:
    """Class represents a Euclidean vector."""

    # pylint: disable=arguments-differ
    @classmethod
    def zeros(cls) -> Self:
        """TODO: Method docstring"""
        return cls(0, 0, 0)

    # pylint: disable=arguments-differ
    @classmethod
    def ones(cls) -> Self:
        """TODO: Method docstring"""
        return cls(1, 1, 1)

    @classmethod
    def unit_x(cls) -> Self:
        """Instantiates a Vector3 instance containing the X unit vector."""
        return cls(1, 0, 0)

    @classmethod
    def unit_y(cls) -> Self:
        """Instantiates a Vector3 instance containing the Y unit vector."""
        return cls(0, 1, 0)

    @classmethod
    def unit_z(cls) -> Self:
        """Instantiates a Vector3 instance containing the Z unit vector."""
        return cls(0, 0, 1)

    @classmethod
    def from_matrix(cls, mat: Matrix) -> Self:
        """Factory method to construct a Vector3 from a Matrix. The input Matrix must be of size 3x1
            or 1x3 for this operation to be successful.
        Args:
            mat (Matrix): The Matrix from which to construct the Vector3.

        Returns:
            Vector3: The instantiated Vector3 object.

        Raises:
            ValueError: Raised if the input Matrix is not of size 3x1 or 1x3.
        """
        if not isinstance(mat, Matrix):
            raise ValueError(f"Received unsupported input type {type(mat)}.")
        if mat.size not in {(3, 1), (1, 3)}:
            raise ValueError(
                "The multiplying matrix must be a row or column matrix but is instead of size "
                f"{mat.size}. Check dimensionality and try again."
            )
        return cls(*mat)

    def __init__(self, x: int | float, y: int | float, z: int | float):
        # TODO Consider updating to utilize Decimal class for elements
        #      instead of floats to increase numeric precision
        self.x: float = float(x)
        self.y: float = float(y)
        self.z: float = float(z)

    def __str__(self) -> str:
        return f"[{self.x}, {self.y}, {self.z}]"

    def __repr__(self) -> str:
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def __round__(self, ndigits: int | None = None) -> Self:
        return Vector3(
            round(self.x, ndigits),
            round(self.y, ndigits),
            round(self.z, ndigits),
        )

    def __getitem__(self, index: int | slice) -> float | Matrix:
        match index:
            case int():
                return tuple(self)[index]
            case slice():
                return self.as_matrix()[index]
            case _:
                pass
        raise ValueError(f"Unsupported index type {type(index)} provided.")

    def __setitem__(self, index: int | slice, value: int | float) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError(
                "Vector value assignment must be convertible to float, but "
                f"is of unsupported type {type(value)} instead."
            )
        match index:
            case int():
                match index:
                    case 0:
                        self.x = float(value)
                    case 1:
                        self.y = float(value)
                    case 2:
                        self.z = float(value)
                    case _:
                        raise IndexError()
            case slice():
                start = index.start or 0
                stop = index.stop or 2
                step = index.step or 1
                for idx in range(start, stop, step):
                    self[idx] = float(value)
            case _:
                raise ValueError(f"Unsupported index type {type(index)} provided.")

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Self | Matrix | float | int) -> bool:
        match other:
            case int() | float():
                return all(x == other for x in self)
            case Vector3():
                return all(x == y for (x, y) in zip(self, other))
            case Matrix():
                if self.size != other.size:
                    return False
                return all(x == y for (x, y) in zip(self, other))
            case _:
                pass
        return NotImplemented

    def __lt__(self, other: Self | Matrix | float | int) -> bool:
        match other:
            case int() | float():
                return all(x < other for x in self)
            case Vector3():
                return all(x < y for (x, y) in zip(self, other))
            case Matrix():
                if self.size != other.size:
                    return False
                return all(x < y for (x, y) in zip(self, other))
            case _:
                pass
        return NotImplemented

    def __le__(self, other: Self | Matrix | float | int) -> bool:
        match other:
            case int() | float():
                return all(x <= other for x in self)
            case Vector3():
                return all(x <= y for (x, y) in zip(self, other))
            case Matrix():
                if self.size != other.size:
                    return False
                return all(x <= y for (x, y) in zip(self, other))
            case _:
                pass
        return NotImplemented

    def __gt__(self, other: Self | Matrix | float | int) -> bool:
        return not self.__le__(other)

    def __ge__(self, other: Self | Matrix | float | int) -> bool:
        return not self.__lt__(other)

    def __add__(self, other: int | float | Matrix | Self) -> Self:
        match other:
            case int() | float():
                return Vector3(*(x + other for x in self))
            case Vector3():
                return Vector3(*(getattr(self, x) + getattr(other, x) for x in "xyz"))
            case Matrix():
                if other.size != self.size:
                    raise ValueError(
                        f"Matrix must be size {self.size} to be added with Vector3, but is {other.size}."
                    )
                return Vector3(*(self[idx] + other[idx] for idx in range(3)))
            case _:
                pass
        return NotImplemented

    def __radd__(self, other: int | float | Matrix | Self) -> Self:
        return self.__add__(other)

    def __sub__(self, other: int | float | Matrix | Self) -> Self:
        match other:
            case int() | float():
                return Vector3(*(x - other for x in self))
            case Vector3():
                return Vector3(*(getattr(self, x) - getattr(other, x) for x in "xyz"))
            case Matrix():
                if other.size != self.size:
                    raise ValueError(
                        f"Matrix must be size {self.size} to compute difference with Vector3, but is {other.size}."
                    )
                return Vector3(*(self[idx] - other[idx] for idx in range(3)))
            case _:
                pass
        return NotImplemented

    def __rsub__(self, other: float | int | Matrix) -> Self:
        match other:
            case int() | float():
                return Vector3(*(other - x for x in self))
            case Matrix():
                if other.size != self.size:
                    raise ValueError(
                        f"Matrix must be size {self.size} to compute difference with Vector3, but is {other.size}."
                    )
                return Vector3(*(other[idx] - self[idx] for idx in range(3)))
            case _:
                pass
        return NotImplemented

    def __mul__(self, other: int | float | Matrix) -> Self:
        match other:
            case int() | float():
                return Vector3(*(other * x for x in self))
            case Matrix():
                if other.size.num_rows != 1:
                    raise ValueError(
                        "The multiplying matrix must be a row matrix but is instead of size "
                        f"{other.size}. Check dimensionality and try again."
                    )
                return self.as_matrix().__mul__(other)
            case _:
                pass
        return NotImplemented

    def __rmul__(self, other: int | float | Matrix) -> Self:
        match other:
            case int() | float():
                return Vector3(*(other * x for x in self))
            case Matrix():
                if other.size.num_cols != 3:
                    raise ValueError(
                        "The multiplying matrix must have 3 columns but is instead of size "
                        f"{other.size}. Check dimensionality and try again."
                    )
                result: int | float | Matrix = other.__mul__(self.as_matrix())
                if isinstance(result, Matrix):
                    try:
                        return Vector3.from_matrix(result)
                    except ValueError:
                        pass
                return result
            case _:
                pass
        return NotImplemented

    def __abs__(self) -> Self:
        return Vector3(*(abs(x) for x in self))

    def __neg__(self) -> Self:
        return Vector3(*(-1 * x for x in self))

    @property
    def size(self) -> tuple[int, int]:
        """The size of the calling Vector3."""
        return 3, 1

    def norm(self) -> float:
        """Returns the Euclidean norm of the calling vector."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def squared_norm(self) -> float:
        """Returns the square of the Euclidean norm of the calling vector."""
        return self.x**2 + self.y**2 + self.z**2

    def cross(self, other: Self) -> Self:
        """Returns the cross product of the calling vector with the argument
        vector, computed as C = A x B for C = A.cross(B).
        """
        if not isinstance(other, Vector3):
            raise NotImplementedError
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return Vector3(x, y, z)

    def dot(self, other: Self) -> float:
        """Returns the dot product of the calling vector with the argument
        vector, computed as C = A * B for C = A.dot(B).
        """
        if not isinstance(other, Vector3):
            raise NotImplementedError
        return self.x * other.x + self.y * other.y + self.z * other.z

    def vertex_angle(self, other: Self) -> float:
        """Returns the angle between the calling vector and the
            argument vector, measured from the calling vector. If
            either vector is a zero vector an angle of 0.0 radians
            will be returned.

        Args:
            other (Vector3): Vector to which to compute the
                vertex angle.

        Returns:
            float: The angle between the two vectors, expressed
                in radians.
        """
        if not isinstance(other, Vector3):
            raise NotImplementedError
        return math.atan2(self.cross(other).norm(), self.dot(other))

    def normalize(self) -> None:
        """Normalizes the calling vector in place by its Euclidean norm."""
        m = self.norm()
        self.x /= m
        self.y /= m
        self.z /= m

    def normalized(self) -> Self:
        """Returns the calling vector, normalized by its Euclidean norm."""
        m = self.norm()
        return (
            Vector3(self.x / m, self.y / m, self.z / m)
            if abs(m) > MACHINE_EPSILON
            else Vector3.zeros()
        )

    def skew(self) -> Matrix:
        """Returns the skew-symmetric matrix created from the calling vector."""
        return Matrix(
            [
                [0, -self.z, self.y],
                [self.z, 0, -self.x],
                [-self.y, self.x, 0],
            ]
        )

    def as_matrix(self) -> Matrix:
        """Returns a copy of the calling Vector3, expressed as a 3x1 column matrix."""
        return Matrix(list([x] for x in self))

    def transpose(self) -> Matrix:
        """Returns a copy of the calling Vector3, expressed as a 1x3 row matrix."""
        return self.as_matrix().transpose()
