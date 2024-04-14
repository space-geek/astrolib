""" TODO: Module docstring
"""

from copy import copy
from itertools import repeat
import math
from typing import Iterator
from typing import Optional
from typing import Self

from astrolib.constants import MACHINE_EPSILON


# pylint: disable=invalid-name
class Matrix:
    """Class represents a two-dimensional matrix of float values."""

    @classmethod
    def fill(cls, num_rows: int, num_cols: int, fill_value: float) -> Self:
        """TODO: Method docstring"""
        if (num_rows <= 0) or (num_cols <= 0):
            raise ValueError(
                "The number of rows or columns must be positive and greater than 0."
            )
        fill_value = float(fill_value)
        return cls(A=[list(repeat(fill_value, num_cols)) for _ in range(num_rows)])

    @classmethod
    def zeros(cls, num_rows: int, num_cols: int = None) -> Self:
        """TODO: Method docstring"""
        if not num_cols and num_cols != 0.0:
            num_cols = num_rows
        return cls.fill(num_rows, num_cols, 0.0)

    @classmethod
    def ones(cls, num_rows: int, num_cols: int = None) -> Self:
        """TODO: Method docstring"""
        if not num_cols and num_cols != 0.0:
            num_cols = num_rows
        return cls.fill(num_rows, num_cols, 1.0)

    @classmethod
    def identity(cls, dim: int) -> Self:
        """TODO: Method docstring"""
        A = cls.zeros(dim)
        for i in range(dim):
            A[i, i] = 1.0
        return A

    @classmethod
    def empty(cls) -> Self:
        """TODO: Method docstring"""
        return cls()

    @classmethod
    def from_column_matrices(cls, matrices: list[Self]) -> Self:
        """TODO: Method docstring"""
        if not isinstance(matrices, list):
            raise ValueError(
                "Input collection must be a list of matrices to concatenate."
            )
        for A in matrices:
            if not isinstance(A, Matrix):
                raise ValueError(
                    "Input collection must be a list of matrices to concatenate."
                )
            if A.num_cols != 1:
                raise ValueError("Each matrix must be a column matrix to concatenate.")
        return cls([[row] for A in matrices for row in A])

    def __init__(
        self,
        A: Optional[list[int | float] | list[list[int | float]]] = None,
    ) -> None:  # TODO make it so that matrix takes iterables
        """Initialization method for the Matrix class.

        Args:
            A (Optional[list[float] | list[list[float]]]): An object containing
                the intended data to be captured in the resulting Matrix. The
                behavior is as follows based on the content type:
                    None: A null matrix is initialized. Default case.
                    list[int | float]: A row matrix is initialized containing
                        the provided data.
                    list[list[int | float]]: A 2x2 matrix is initialized containing
                        the provided data. Each entry in the inner list is
                        interpreted as a row of the resulting matrix.

        Returns:
            Matrix: The initialized Matrix class instance.
        """
        if A is None:
            A = []
        else:
            if not isinstance(A, list):
                raise ValueError
            if len(A) > 0:
                if not isinstance(A[0], list):
                    if isinstance(A[0], (int, float)):
                        A = [A]
                    else:
                        raise ValueError
            num_cols = len(A[0]) if A else 0
            for row in A:
                if len(row) != num_cols:
                    raise ValueError("Each row must have the same number of columns.")
        self._A: list[list[int | float]] = A

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        def _stringify_row(row: list[str]) -> str:
            return ", ".join([str(x) for x in row])

        match self.size:
            case 1, _:
                data: str = _stringify_row(self._A[0])
            case _, 1:
                data: str = "\n".join(str(row[0]) for row in self._A)
            case _:
                data: str = "\n ".join(_stringify_row(row) for row in self._A)
        return f"[{data}]"

    def __repr__(self) -> str:
        match self.size:
            case 1, _:
                data: str = ", ".join(str(x) for x in self._A[0])
            case _, 1:
                data: str = "; ".join(str(row[0]) for row in self._A)
            case _:
                data: str = "; ".join(", ".join(str(x) for x in row) for row in self)
        return f"[{data}]"

    def __setitem__(
        self,
        indices: tuple[int | slice, int | slice] | int | slice,
        value: float | int | Self,
    ) -> None:
        if isinstance(indices, (int, float, slice)):
            m, n = self.size
            if m == 1:
                indices: tuple[int, int | float | slice] = (0, indices)
            elif n == 1:
                indices: tuple[int | float | slice, int] = (indices, 0)
            else:
                raise ValueError(
                    "Single-index indexing only supported for row or column "
                    f"matrices. Matrix is of size ({m}, {n})."
                )
        match indices:
            case slice(), slice():
                if not isinstance(value, Matrix):  # TODO Support scalar value
                    raise ValueError(
                        "When setting matrix indices using slice notation a Matrix value "
                        "must be used."
                    )
                if value.is_empty:
                    raise ValueError(
                        "When setting matrix indices using slice notation for both the "
                        "row and column indices an empty Matrix value cannot be used."
                    )
                slice_rows_range = range(
                    (indices[0].start or 0),
                    (indices[0].stop or self.num_rows),
                    (indices[0].step or 1),
                )
                slice_cols_range = range(
                    (indices[1].start or 0),
                    (indices[1].stop or self.num_cols),
                    (indices[1].step or 1),
                )
                value_rows_range = range(value.num_rows)
                value_cols_range = range(value.num_cols)
                for i, m in zip(slice_rows_range, value_rows_range):
                    for j, n in zip(slice_cols_range, value_cols_range):
                        self._A[i][j] = value[m, n]
            case slice(), int() | float():
                if not isinstance(value, Matrix):  # TODO Support scalar value
                    raise ValueError(
                        "When setting matrix indices using slice notation a Matrix value "
                        "must be used."
                    )
                slice_range = range(
                    (indices[0].start or 0),
                    (indices[0].stop or self.num_rows),
                    (indices[0].step or 1),
                )
                if value.is_empty:
                    self._A = [
                        [self._A[i][j] for j in range(self.num_cols) if j != indices[1]]
                        for i in slice_range
                    ]
                else:
                    value_range = range(value.num_rows)
                    for i, j in zip(slice_range, value_range):
                        self._A[i][indices[1]] = value[j, 0]
            case int() | float(), slice():
                if not isinstance(value, Matrix):  # TODO Support scalar value
                    raise ValueError(
                        "When setting matrix indices using slice notation a Matrix value "
                        "must be used."
                    )
                slice_range = range(
                    (indices[1].start or 0),
                    (indices[1].stop or self.num_cols),
                    (indices[1].step or 1),
                )
                if value.is_empty:
                    self._A = [
                        [self._A[i][j] for j in slice_range]
                        for i in range(self.num_rows)
                        if i != indices[0]
                    ]
                else:
                    value_range = range(value.num_cols)
                    for i, j in zip(slice_range, value_range):
                        self._A[indices[0]][i] = value[0, j]
            case int() | float(), int() | float():
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        "When setting matrix indices using direct index notation a numeric value must be used."
                    )
                self._A[indices[0]][indices[1]] = value
            case _:
                raise ValueError("Unsupported access indices provided.")

    def __getitem__(
        self,
        indices: tuple[int | float | slice, int | float | slice] | int | float | slice,
    ) -> int | float | Self:
        if isinstance(indices, (int, float, slice)):
            m, n = self.size
            if m == 1:
                indices: tuple[int, int | float | slice] = (0, indices)
            elif n == 1:
                indices: tuple[int | float | slice, int] = (indices, 0)
            else:
                raise ValueError(
                    "Single-index indexing only supported for row or column "
                    f"matrices. Matrix is of size ({m}, {n})."
                )
        match indices:
            case slice(), slice():
                rows_range = range(
                    (indices[0].start or 0),
                    (indices[0].stop or self.num_rows),
                    (indices[0].step or 1),
                )
                cols_range = range(
                    (indices[1].start or 0),
                    (indices[1].stop or self.num_cols),
                    (indices[1].step or 1),
                )
                M: Matrix = Matrix(
                    [[self._A[i][j] for j in cols_range] for i in rows_range]
                )
            case slice(), int() | float():
                M: Matrix = Matrix(
                    [
                        [self._A[i][indices[1]]]
                        for i in range(
                            (indices[0].start or 0),
                            (indices[0].stop or self.num_rows),
                            (indices[0].step or 1),
                        )
                    ]
                )
            case int() | float(), slice():
                M: Matrix = Matrix(
                    [
                        [
                            self._A[indices[0]][i]
                            for i in range(
                                (indices[1].start or 0),
                                (indices[1].stop or self.num_cols),
                                (indices[1].step or 1),
                            )
                        ]
                    ]
                )
            case int() | float(), int() | float():
                M: float = self._A[indices[0]][indices[1]]
            case _:
                raise ValueError("Unsupported access indices provided.")
        return M

    def __iter__(self) -> Iterator[float] | Iterator[list[float]]:
        m, n = self.size
        if m == 1:
            for x in self._A[0]:
                yield x
        elif n == 1:
            for row in self._A:
                yield row[0]
        else:
            for row in self._A:
                yield row

    def __round__(self, ndigits: int | None = None) -> Self:
        A = copy(self._A)  # TODO: Unit tests for this function
        for i, row in enumerate(A):
            for j, value in enumerate(row):
                A[i][j] = round(value, ndigits=ndigits)
        return Matrix(A)

    @property
    def num_rows(self) -> int:
        """TODO: Property docstring"""
        return len(self._A)

    @property
    def num_cols(self) -> int:
        """TODO: Property docstring"""
        return len(self._A[0]) if self._A else 0

    @property
    def size(self) -> tuple[int, int]:
        """TODO: Property docstring"""
        return self.num_rows, self.num_cols

    @property
    def is_empty(self) -> bool:
        """TODO: Property docstring"""
        return not (self.num_rows or self.num_cols)

    def __eq__(self, other: Self | float | int) -> bool:
        match other:
            case int() | float():
                for i in range(self.num_rows):
                    for j in range(self.num_cols):
                        if self[i, j] != other:
                            return False
                return True
            case Matrix():
                if self.size != other.size:
                    return False
                for i in range(self.num_rows):
                    for j in range(self.num_cols):
                        if self[i, j] != other[i, j]:
                            return False
                return True
            case _:
                pass
        return NotImplemented

    def __lt__(self, other: Self | float | int) -> bool:
        match other:
            case int() | float():
                for i in range(self.num_rows):
                    for j in range(self.num_cols):
                        if self[i, j] >= other:
                            return False
                return True
            case Matrix():
                if self.size != other.size:
                    return False
                for i in range(self.num_rows):
                    for j in range(self.num_cols):
                        if self[i, j] >= other[i, j]:
                            return False
                return True
            case _:
                pass
        return NotImplemented

    def __le__(self, other: Self | float | int) -> bool:
        match other:
            case int() | float():
                for i in range(self.num_rows):
                    for j in range(self.num_cols):
                        if self[i, j] > other:
                            return False
                return True
            case Matrix():
                if self.size != other.size:
                    return False
                for i in range(self.num_rows):
                    for j in range(self.num_cols):
                        if self[i, j] > other[i, j]:
                            return False
                return True
            case _:
                pass
        return NotImplemented

    def __gt__(self, other: Self | float | int) -> bool:
        return not self.__le__(other)

    def __ge__(self, other: Self | float | int) -> bool:
        return not self.__lt__(other)

    def __add__(self, other: Self | float | int) -> Self:
        match other:
            case int() | float():
                M = Matrix.zeros(*self.size)
                for i in range(self.num_rows):
                    for j in range(self.num_cols):
                        M[i, j] = self[i, j] + other
                return M
            case Matrix():
                if self.size != other.size:
                    raise ValueError("Matrices must be the same size to be added.")
                M = Matrix.zeros(*self.size)
                for i in range(self.num_rows):
                    for j in range(self.num_cols):
                        M[i, j] = self[i, j] + other[i, j]
                return M
            case _:
                pass
        return NotImplemented

    def __radd__(self, other: Self | float | int) -> Self:
        return self.__add__(other)

    def __sub__(self, other: Self | float | int) -> Self:
        match (other):
            case int() | float():
                M = Matrix.zeros(*self.size)
                for i in range(self.num_rows):
                    for j in range(self.num_cols):
                        M[i, j] = self[i, j] - other
                return M
            case Matrix():
                if self.size != other.size:
                    raise ValueError("Matrices must be the same size to be subtracted.")
                M = Matrix.zeros(*self.size)
                for i in range(self.num_rows):
                    for j in range(self.num_cols):
                        M[i, j] = self[i, j] - other[i, j]
                return M
            case _:
                pass
        return NotImplemented

    def __rsub__(self, other: Self | float | int) -> Self:
        return -1.0 * self.__sub__(other)

    def __mul__(self, other: Self | float | int) -> Self | int | float:
        """Matrix multiplication source:
        The Algorithm Design Manual, Skeina, 3rd Ed.; Section 16.3, p. 472
        """
        if not isinstance(other, (Matrix, float, int)):
            return NotImplemented
        match other:
            case int() | float():
                M = Matrix.zeros(*self.size)
                for i in range(self.num_rows):
                    for j in range(self.num_cols):
                        M[i, j] = other * self[i, j]
            case Matrix():
                x, y = self.size
                yy, z = other.size
                if y != yy:
                    raise ValueError(
                        "Incorrect dimensions for matrix multiplication. Check that the "
                        "number of columns in the first matrix matches the number of "
                        "rows in the second matrix."
                    )
                M = Matrix.zeros(x, z)
                o_t = other.transpose()
                for i in range(x):
                    for j in range(z):
                        M[i, j] = sum(x * y for x, y in zip(self._A[i], o_t._A[j]))
        if M.size == (1, 1):
            M: int | float = M[0]
        return M

    def __rmul__(self, other: int | float) -> Self:
        match other:
            case int() | float():
                return self.__mul__(other)
            case _:
                pass
        return NotImplemented

    def __abs__(self) -> Self:
        M = Matrix.zeros(*self.size)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                M[i, j] = abs(self[i, j])
        return M

    def __neg__(self) -> Self:
        M = Matrix.zeros(*self.size)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                M[i, j] = -self[i, j]
        return M

    def __len__(self) -> int:
        """Returns the length of the calling Matrix, defined as the maximum dimension.

        Returns:
            int: The maximum dimension of the calling Matrix, i.e. max(num_rows, num_cols)
        """
        return int(max(self.size))

    def get_row(self, idx: int) -> Self:
        """TODO: Method docstring"""
        return Matrix([self._A[idx]])

    def get_col(self, idx: int) -> Self:
        """TODO: Method docstring"""
        return Matrix([row[idx] for row in self._A]).transpose()

    def transpose(self) -> Self:
        """Returns the transpose of the calling matrix."""
        M = Matrix.zeros(self.num_cols, self.num_rows)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                M[j, i] = self[i, j]
        return M

    @property
    def diag(self) -> Self:
        """The main diagonal of the calling matrix."""
        return Matrix([[self[idx, idx]] for idx in range(min(self.size))])

    @property
    def trace(self) -> float:
        """The trace of the calling matrix."""
        if not self.is_square:
            raise ValueError(
                "The calling matrix is not square and the trace does not exist."
            )
        return sum(x for x in self.diag)

    def adjoint(self) -> Self:
        """TODO"""

    def determinant(self) -> float:
        """Returns the determinant of the calling matrix."""
        m, n = self.size
        if m != n:
            raise ValueError(
                "The calling matrix is not square and the determinant does not exist."
            )
        if m == 2:
            d = self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0]
        else:
            d = 0.0
            for j in range(self.num_cols):
                A_temp = copy(self[:, :])
                A_temp[0, :] = Matrix.empty()
                A_temp[:, j] = Matrix.empty()
                d += self[0, j] * pow(-1, j) * A_temp.determinant()
        return d

    def inverse(self) -> Self:
        """Returns the inverse of the calling matrix, computed using the cofactor method."""

        def compute_cofactor_matrix(A: Matrix) -> Matrix:
            """Returns the cofactor matrix computed from the input matrix."""
            m, n = A.size
            if m != n:
                raise ValueError(
                    "The input matrix is not square. The cofactor matrix does not "
                    "exist."
                )
            M = Matrix.zeros(*A.size)
            for i in range(A.num_rows):
                for j in range(A.num_cols):
                    A_temp = A[:, :]
                    A_temp[i, :] = Matrix.empty()
                    A_temp[:, j] = Matrix.empty()
                    M[i, j] = pow(-1, i + j) * A_temp.determinant()
            return M

        m, n = self.size
        if m != n:
            raise ValueError(
                "The calling matrix is not square. The matrix inverse does not exist."
            )
        d = self.determinant()
        if not d:
            raise ValueError(
                "The calling matrix is singular. The matrix inverse does not exist."
            )
        return (1 / d) * compute_cofactor_matrix(self).transpose()

    def is_row_matrix(self) -> bool:
        """Returns True if the calling Matrix is a row matrix (i.e. has one row and one or more
            columns), False otherwise.

        Returns:
            bool: Boolean indicator of whether or not the calling matrix is a row matrix.
        """
        return self.num_rows == 1

    def is_column_matrix(self) -> bool:
        """Returns True if the calling Matrix is a column matrix (i.e. has one column and one or
            more rows), False otherwise.

        Returns:
            bool: Boolean indicator of whether or not the calling matrix is a column matrix.
        """
        return self.num_cols == 1

    @property
    def is_square(self) -> bool:
        """Returns True if the calling Matrix is square (i.e. the number of rows equals the
            number of columns), False otherwise.

        Returns:
            bool: Boolean indicator of whether or not the calling matrix is square.
        """
        return self.num_rows == self.num_cols

    def to_column_matrix(self) -> Self:
        """Returns a copy of the calling Matrix expressed as a column matrix, with each row
            stacked in sequence.

        Returns:
            Matrix: A copy of the calling matrix, in column matrix form.
        """
        return Matrix.from_column_matrices([row.transpose() for row in self])


class Vector3(Matrix):
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
    def identity(cls, dim: int) -> Self:
        raise NotImplementedError

    @classmethod
    def fill(cls, num_rows: int, num_cols: int, fill_value: float) -> Self:
        raise NotImplementedError

    @classmethod
    def empty(cls) -> Self:
        raise NotImplementedError

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
    def from_matrix(cls, M: Matrix) -> Self:
        """Factory method to construct a Vector3 from a Matrix. The input Matrix must be of size 3x1
            or 1x3 for this operation to be successful.
        Args:
            M (Matrix): The Matrix from which to construct the Vector3.

        Returns:
            Vector3: The instantiated Vector3 object.

        Raises:
            ValueError: Raised if the input Matrix is not of size 3x1 or 1x3.
        """
        if not isinstance(M, Matrix):
            raise ValueError(f"Received unsupported input type {type(M)}.")
        if M.size not in {(3, 1), (1, 3)}:
            raise ValueError(
                "The multiplying matrix must be a row or column matrix but is instead of size "
                f"{M.size}. Check dimensionality and try again."
            )
        return cls(*M)

    def __init__(self, x: int | float, y: int | float, z: int | float):
        super().__init__([[x], [y], [z]])

    @property
    def x(self) -> float:
        """TODO: Property docstring"""
        return self[0, 0]

    @x.setter
    def x(self, value: float):
        """TODO: Property docstring"""
        self[0, 0] = value

    @property
    def y(self) -> float:
        """TODO: Property docstring"""
        return self[1, 0]

    @y.setter
    def y(self, value: float):
        """TODO: Property docstring"""
        self[1, 0] = value

    @property
    def z(self) -> float:
        """TODO: Property docstring"""
        return self[2, 0]

    @z.setter
    def z(self, value: float):
        """TODO: Property docstring"""
        self[2, 0] = value

    def __str__(self) -> str:
        return f"[x = {self.x}, y = {self.y}, z = {self.z}]"

    def __repr__(self) -> str:
        return f"[{self.x}, {self.y}, {self.z}]"

    def __add__(self, other: Matrix | Self) -> Self:
        return Vector3.from_matrix(super().__add__(other))

    def __sub__(self, other: Matrix | Self) -> Self:
        return Vector3.from_matrix(super().__sub__(other))

    def __rsub__(self, other: Matrix | Self) -> Self:
        return Vector3.from_matrix(super().__rsub__(other))

    def __mul__(self, other: int | float | Matrix) -> Self:
        match other:
            case int() | float():
                return Vector3.from_matrix(super().__mul__(other))
            case Matrix():
                if other.num_rows == 1:
                    return super().__mul__(other)
                raise ValueError(
                    "The multiplying matrix must be a row matrix but is instead of size "
                    f"{other.size}. Check dimensionality and try again."
                )
            case _:
                pass
        return NotImplemented

    def __rmul__(self, other: int | float | Matrix) -> Self:
        match other:
            case int() | float():
                return Vector3.from_matrix(super().__rmul__(other))
            case Matrix():
                if other.num_cols == 3:
                    result: int | float | Matrix = other.__mul__(self)
                    if isinstance(result, Matrix) and result.num_rows == 3:
                        return Vector3.from_matrix(result)
                    return result
                raise ValueError(
                    "The multiplying matrix must have 3 columns but is instead of size "
                    f"{other.size}. Check dimensionality and try again."
                )
            case _:
                pass
        return NotImplemented

    def __abs__(self) -> Self:
        return Vector3.from_matrix(super().__abs__())

    def __neg__(self) -> Self:
        return Vector3.from_matrix(super().__neg__())

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
            return NotImplemented
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return Vector3(x, y, z)

    def dot(self, other: Self) -> float:
        """Returns the dot product of the calling vector with the argument
        vector, computed as C = A * B for C = A.dot(B).
        """
        if not isinstance(other, Vector3):
            return NotImplemented
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
            return NotImplemented
        return math.atan2(self.cross(other).norm(), self.dot(other))

    def normalize(self) -> None:
        """Normalizes the calling vector in place by its Euclidean norm."""
        m = self.norm()
        self[0, 0] /= m
        self[1, 0] /= m
        self[2, 0] /= m

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
