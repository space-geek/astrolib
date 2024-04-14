""" Module contains direction cosine matrix (DCM)-related
class and function definitions. 
"""

import math
from typing import Iterator
from typing import Self

from astrolib.matrix import Matrix
from astrolib.matrix import Vector3


class DirectionCosineMatrix:
    """Class representing a three-dimensional direction
    cosine matrix (DCM).
    """

    @classmethod
    def identity(cls) -> Self:
        """Creates a zero-rotation direction cosine matrix."""
        return cls(Matrix.identity(3))

    @classmethod
    def r_x(cls, angle: float) -> Self:
        """Creates a single-axis rotation matrix around
        the X-axis using the provided angle in radians.
        """
        c_angle = math.cos(angle)
        s_angle = math.sin(angle)
        return cls(
            Matrix(
                [
                    [1, 0, 0],
                    [0, c_angle, s_angle],
                    [0, -s_angle, c_angle],
                ]
            )
        )

    @classmethod
    def r_y(cls, angle: float) -> Self:
        """Creates a single-axis rotation matrix around
        the Y-axis using the provided angle in radians.
        """
        c_angle = math.cos(angle)
        s_angle = math.sin(angle)
        return cls(
            Matrix(
                [
                    [c_angle, 0, -s_angle],
                    [0, 1, 0],
                    [s_angle, 0, c_angle],
                ]
            )
        )

    @classmethod
    def r_z(cls, angle: float) -> Self:
        """Creates a single-axis rotation matrix around
        the Z-axis using the provided angle in radians.
        """
        c_angle = math.cos(angle)
        s_angle = math.sin(angle)
        return cls(
            Matrix(
                [
                    [c_angle, s_angle, 0],
                    [-s_angle, c_angle, 0],
                    [0, 0, 1],
                ]
            )
        )

    def __init__(self, dcm: Matrix) -> None:
        if not isinstance(dcm, Matrix):
            raise ValueError(
                f"Unsupported argument type {type(dcm)} provided for DCM construction."
            )
        if not matrix_is_orthogonal(dcm):
            raise ValueError("Provided matrix must be orthogonal and is not.")
        self._dcm: Matrix = dcm

    def __repr__(self) -> str:
        return (
            f"DirectionCosineMatrix(Matrix([{', '.join(repr(x) for x in self.rows)}]))"
        )

    def __eq__(self, other: Self | Matrix) -> bool:
        match other:
            case DirectionCosineMatrix():
                return self._dcm == other._dcm
            case Matrix():
                return self._dcm == other
            case _:
                pass
        return NotImplemented

    def __getitem__(
        self,
        index: tuple[int | float | slice, int | float | slice],
    ) -> int | float | Matrix:
        match index:
            case (int() | float() | slice(), int() | float() | slice()):
                return self._dcm[index]
            case _:
                raise ValueError(f"Unsupported index type {type(index)} provided.")

    def __sub__(self, other: Self | Matrix) -> Matrix:
        match other:
            case DirectionCosineMatrix():
                return self._dcm - other._dcm
            case Matrix():
                if other.size == (3, 3):
                    result = self._dcm - other
                    return result
                raise ValueError(
                    "The subtracted matrix must be 3x3 square matrix but is instead of size "
                    f"{other.size}. Check dimensionality and try again."
                )
            case _:
                pass
        return NotImplemented

    def __rsub__(self, other: Self | Matrix) -> Matrix:
        return -1.0 * self.__sub__(other)

    def __mul__(self, other: Self | Matrix) -> Self | Matrix:
        match other:
            case DirectionCosineMatrix():
                return DirectionCosineMatrix(self._dcm * other._dcm)
            case Matrix():
                if other.num_rows == 3:
                    result = self._dcm.__mul__(other)
                    try:
                        result = DirectionCosineMatrix(result)
                    except ValueError:
                        pass  # silently ignore
                    return result
                raise ValueError(
                    "The multiplying matrix must have 3 columns but is instead of size "
                    f"{other.size}. Check dimensionality and try again."
                )
            case _:
                pass
        return NotImplemented

    def __rmul__(self, other: Matrix) -> Self:
        match other:
            case Matrix():
                if other.num_cols == 3:
                    result = other * self._dcm
                    try:
                        result = DirectionCosineMatrix(result)
                    except ValueError:
                        pass  # silently ignore
                    return result
                raise ValueError(
                    "The multiplying matrix must have 3 columns but is instead of size "
                    f"{other.size}. Check dimensionality and try again."
                )
            case _:
                pass
        return NotImplemented

    @property
    def rows(self) -> tuple[Vector3, Vector3, Vector3]:
        """Returns a tuple containing the rows of the DCM expressed
        as three-dimensional vectors.
        """
        return (
            Vector3(*self._dcm[0, :]),
            Vector3(*self._dcm[1, :]),
            Vector3(*self._dcm[2, :]),
        )

    @property
    def columns(self) -> tuple[Vector3, Vector3, Vector3]:
        """Returns a tuple containing the columns of the DCM expressed
        as three-dimensional vectors.
        """
        return (
            Vector3(*self._dcm[:, 0]),
            Vector3(*self._dcm[:, 1]),
            Vector3(*self._dcm[:, 2]),
        )

    def as_matrix(self) -> Matrix:
        """Returns a copy of the calling direction cosine matrix, expressed as a 3x3 matrix."""
        return self[:, :]

    def transpose(self) -> Matrix:
        """Returns the transpose of the calling direction cosine matrix, expressed as a 3x3 matrix."""
        return self.as_matrix().transpose()

    @property
    def trace(self) -> float:
        """Returns the trace of the calling direction cosine matrix."""
        return self.as_matrix().trace


def matrix_is_orthogonal(
    c: Matrix,
    /,
    tol: float = 1.0e-6,
) -> bool:
    """Utility function which verifies that the
    provided matrix is orthogonal.

    Args:
        matrix (Matrix): The matrix to evaluate.

    KWArgs:
        tol (float): The tolerance to use when evaluating
            the matrix.

    Returns:
        bool: Boolean flag indicating the provided
            matrix is orthogonal (True) or not (False).
    """
    if not isinstance(c, Matrix):
        raise ValueError(f"Unsupported argument type {type(c)} provided.")

    def orthogonality_conditions() -> Iterator[bool]:
        m, n = c.size
        yield m == n
        yield abs(c.transpose() * c - Matrix.identity(m)) <= tol
        yield abs(c.determinant()) - 1.0 <= tol

    return all(orthogonality_conditions())
