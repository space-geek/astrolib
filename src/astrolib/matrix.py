""" TODO: Module docstring
"""

from copy import copy
from itertools import repeat
from typing import Iterator
from typing import NamedTuple
from typing import Optional
from typing import Self


class TwoDimensionalMatrixSize(NamedTuple):
    """Tuple containing a two-dimensional matrix size."""

    num_rows: int
    num_cols: int


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
                data: str = "\n ".join(str(row[0]) for row in self._A)
            case _:
                data: str = "\n ".join(_stringify_row(row) for row in self._A)
        return f"[{data}]"

    def __repr__(self) -> str:
        data = ", ".join(f"[{', '.join(str(x) for x in row)}]" for row in self._A)
        return f"Matrix([{data}])" if data else "Matrix()"

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
                    (indices[0].stop or self.size.num_rows),
                    (indices[0].step or 1),
                )
                slice_cols_range = range(
                    (indices[1].start or 0),
                    (indices[1].stop or self.size.num_cols),
                    (indices[1].step or 1),
                )
                value_rows_range = range(value.size.num_rows)
                value_cols_range = range(value.size.num_cols)
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
                    (indices[0].stop or self.size.num_rows),
                    (indices[0].step or 1),
                )
                if value.is_empty:
                    self._A = [
                        [
                            self._A[i][j]
                            for j in range(self.size.num_cols)
                            if j != indices[1]
                        ]
                        for i in slice_range
                    ]
                else:
                    value_range = range(value.size.num_rows)
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
                    (indices[1].stop or self.size.num_cols),
                    (indices[1].step or 1),
                )
                if value.is_empty:
                    self._A = [
                        [self._A[i][j] for j in slice_range]
                        for i in range(self.size.num_rows)
                        if i != indices[0]
                    ]
                else:
                    value_range = range(value.size.num_cols)
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
                    (indices[0].stop or self.size.num_rows),
                    (indices[0].step or 1),
                )
                cols_range = range(
                    (indices[1].start or 0),
                    (indices[1].stop or self.size.num_cols),
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
                            (indices[0].stop or self.size.num_rows),
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
                                (indices[1].stop or self.size.num_cols),
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
    def size(self) -> TwoDimensionalMatrixSize:
        """The dimensions of the matrix, expressed as a tuple."""
        return TwoDimensionalMatrixSize(
            len(self._A),
            len(self._A[0]) if self._A else 0,
        )

    @property
    def is_empty(self) -> bool:
        """TODO: Property docstring"""
        return all(x == 0 for x in self.size)

    def __eq__(self, other: Self | float | int) -> bool:
        match other:
            case int() | float():
                for i in range(self.size.num_rows):
                    for j in range(self.size.num_cols):
                        if self[i, j] != other:
                            return False
                return True
            case Matrix():
                if self.size != other.size:
                    return False
                for i in range(self.size.num_rows):
                    for j in range(self.size.num_cols):
                        if self[i, j] != other[i, j]:
                            return False
                return True
            case _:
                pass
        return NotImplemented

    def __lt__(self, other: Self | float | int) -> bool:
        match other:
            case int() | float():
                for i in range(self.size.num_rows):
                    for j in range(self.size.num_cols):
                        if self[i, j] >= other:
                            return False
                return True
            case Matrix():
                if self.size != other.size:
                    return False
                for i in range(self.size.num_rows):
                    for j in range(self.size.num_cols):
                        if self[i, j] >= other[i, j]:
                            return False
                return True
            case _:
                pass
        return NotImplemented

    def __le__(self, other: Self | float | int) -> bool:
        match other:
            case int() | float():
                for i in range(self.size.num_rows):
                    for j in range(self.size.num_cols):
                        if self[i, j] > other:
                            return False
                return True
            case Matrix():
                if self.size != other.size:
                    return False
                for i in range(self.size.num_rows):
                    for j in range(self.size.num_cols):
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
                for i in range(self.size.num_rows):
                    for j in range(self.size.num_cols):
                        M[i, j] = self[i, j] + other
                return M
            case Matrix():
                if self.size != other.size:
                    raise ValueError("Matrices must be the same size to be added.")
                M = Matrix.zeros(*self.size)
                for i in range(self.size.num_rows):
                    for j in range(self.size.num_cols):
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
                for i in range(self.size.num_rows):
                    for j in range(self.size.num_cols):
                        M[i, j] = self[i, j] - other
                return M
            case Matrix():
                if self.size != other.size:
                    raise ValueError("Matrices must be the same size to be subtracted.")
                M = Matrix.zeros(*self.size)
                for i in range(self.size.num_rows):
                    for j in range(self.size.num_cols):
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
                for i in range(self.size.num_rows):
                    for j in range(self.size.num_cols):
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
        for i in range(self.size.num_rows):
            for j in range(self.size.num_cols):
                M[i, j] = abs(self[i, j])
        return M

    def __neg__(self) -> Self:
        M = Matrix.zeros(*self.size)
        for i in range(self.size.num_rows):
            for j in range(self.size.num_cols):
                M[i, j] = -self[i, j]
        return M

    def __len__(self) -> int:
        """Returns the length of the calling Matrix, defined as the maximum dimension.

        Returns:
            int: The maximum dimension of the calling Matrix, i.e. max(num_rows, num_cols)
        """
        return max(self.size)

    def transpose(self) -> Self:
        """Returns the transpose of the calling matrix."""
        transposed_self = Matrix.zeros(self.size.num_cols, self.size.num_rows)
        for i in range(self.size.num_rows):
            for j in range(self.size.num_cols):
                transposed_self[j, i] = self[i, j]
        return transposed_self

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
            for j in range(self.size.num_cols):
                A_temp = copy(self[:, :])
                A_temp[0, :] = Matrix.empty()
                A_temp[:, j] = Matrix.empty()
                d += self[0, j] * pow(-1, j) * A_temp.determinant()
        return d

    def inverse(self) -> Self:
        """Returns the inverse of the calling matrix, computed using the cofactor method."""

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
        return (1 / d) * _compute_cofactor_matrix(self).transpose()

    @property
    def is_row_matrix(self) -> bool:
        """Returns True if the calling Matrix is a row matrix (i.e. has one row and one or more
            columns), False otherwise.

        Returns:
            bool: Boolean indicator of whether or not the calling matrix is a row matrix.
        """
        return self.size.num_rows == 1

    @property
    def is_column_matrix(self) -> bool:
        """Returns True if the calling Matrix is a column matrix (i.e. has one column and one or
            more rows), False otherwise.

        Returns:
            bool: Boolean indicator of whether or not the calling matrix is a column matrix.
        """
        return self.size.num_cols == 1

    @property
    def is_square(self) -> bool:
        """Returns True if the calling Matrix is square (i.e. the number of rows equals the
            number of columns), False otherwise.

        Returns:
            bool: Boolean indicator of whether or not the calling matrix is square.
        """
        return self.size.num_rows == self.size.num_cols


def _compute_cofactor_matrix(A: Matrix) -> Matrix:
    """Returns the cofactor matrix computed from the input matrix."""
    m, n = A.size
    if m != n:
        raise ValueError(
            "The input matrix is not square. The cofactor matrix does not exist."
        )
    M = Matrix.zeros(*A.size)
    for i in range(A.size.num_rows):
        for j in range(A.size.num_cols):
            A_temp = A[:, :]
            A_temp[i, :] = Matrix.empty()
            A_temp[:, j] = Matrix.empty()
            M[i, j] = pow(-1, i + j) * A_temp.determinant()
    return M
