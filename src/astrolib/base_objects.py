""" TODO: Module docstring
"""
from decimal import Decimal
from math import copysign
from math import floor
from math import sqrt
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Union

from astrolib.util.constants import NANOSECONDS_PER_SECOND
from astrolib.util.constants import SECONDS_PER_HOUR
from astrolib.util.constants import SECONDS_PER_MINUTE
from astrolib.util.constants import SECONDS_PER_SOLAR_DAY

_DECIMAL_NANOSECONDS_PER_SECOND = Decimal(str(NANOSECONDS_PER_SECOND))


#pylint: disable=invalid-name
class Matrix:
    """Class represents a two-dimensional matrix of float values."""

    @classmethod
    def fill(cls, num_rows: int, num_cols: int, fill_value: float) -> 'Matrix':
        if (num_rows <= 0) or (num_cols <= 0):
            raise ValueError("The number of rows or columns must be positive and greater than 0.")
        fill_value = float(fill_value)
        return cls(A=[[fill_value for _ in range(int(num_cols))] for _ in range(int(num_rows))])

    @classmethod
    def zeros(cls, num_rows: int, num_cols: int = None) -> 'Matrix':
        if not num_cols and num_cols != 0.0:
            num_cols = num_rows
        return Matrix.fill(num_rows, num_cols, 0.0)

    @classmethod
    def ones(cls, num_rows: int, num_cols: int = None) -> 'Matrix':
        if not num_cols and num_cols != 0.0:
            num_cols = num_rows
        return Matrix.fill(num_rows, num_cols, 1.0)

    @classmethod
    def identity(cls, dim: int) -> 'Matrix':
        A = Matrix.zeros(dim)
        for i in range(dim):
            A[i,i] = 1.0
        return A

    @classmethod
    def empty(cls) -> 'Matrix':
        return Matrix([])

    @classmethod
    def from_column_matrices(cls, matrices) -> 'Matrix':
        if not isinstance(matrices, list):
            raise ValueError('Input collection must be a list of matrices to concatenate.')
        for A in matrices:
            if not isinstance(A, Matrix):
                raise ValueError('Input collection must be a list of matrices to concatenate.')
            if A.num_cols != 1:
                raise ValueError('Each matrix must be a column matrix to concatenate.')
        return Matrix([row for A in matrices for row in A])

    def __init__(self, A: List[List[float]]):
        num_cols = len(A[0]) if A else 0
        for row in A:
            if len(row) != num_cols:
                raise ValueError('Each row must have the same number of columns.')
        self._A = A

    def __str__(self) -> str:
        def stringify_row(row: List):
            return ", ".join([str(x) for x in row])
        mat = "\n ".join(stringify_row(row) for row in self._A)
        return f"[{mat}]"

    def __setitem__(self,
                    indices: Union[Tuple[int, int],
                                   Tuple[int, slice],
                                   Tuple[int, slice],
                                   Tuple[slice, slice],
                                   int,
                                   slice],
                    value: Union[int, 'Matrix']
                    ) -> None:
        if isinstance(indices, (int, slice)):
            if self.is_row_matrix():
                indices = (0, indices)
            elif self.is_column_matrix():
                indices = (indices, 0)
            else:
                raise ValueError("Single-index indexing only supported for row or column matrices.")
        if isinstance(indices[0], slice) and isinstance(indices[1], slice): # expects Matrix
            if not isinstance(value, Matrix):
                raise ValueError("When setting matrix indices using slice notation a Matrix value "
                                 "must be used.")
            if value.is_empty:
                raise ValueError("When setting matrix indices using slice notation for both the "
                                 "row and column indices an empty Matrix value cannot be used.")
            slice_rows_range = range((indices[0].start or 0),
                                     (indices[0].stop or self.num_rows),
                                     (indices[0].step or 1))
            slice_cols_range = range((indices[1].start or 0),
                                     (indices[1].stop or self.num_cols),
                                     (indices[1].step or 1))
            value_rows_range = range(value.num_rows)
            value_cols_range = range(value.num_cols)
            for (i,m) in zip(slice_rows_range, value_rows_range):
                for (j,n) in zip(slice_cols_range, value_cols_range):
                    self._A[i][j] = value[m,n]
        elif isinstance(indices[0], slice): # expects Matrix
            if not isinstance(value, Matrix):
                raise ValueError("When setting matrix indices using slice notation a Matrix value "
                                 "must be used.")
            slice_range = range((indices[0].start or 0),
                                (indices[0].stop or self.num_rows),
                                (indices[0].step or 1))
            if value.is_empty:
                self._A = [[self._A[i][j] for j in range(self.num_cols)
                            if j != indices[1]] for i in slice_range]
            else:
                value_range = range(value.num_rows)
                for (i,j) in zip(slice_range, value_range):
                    self._A[i][indices[1]] = value[j,0]
        elif isinstance(indices[1], slice): # expects Matrix
            if not isinstance(value, Matrix):
                raise ValueError("When setting matrix indices using slice notation a Matrix value "
                                 "must be used.")
            slice_range = range((indices[1].start or 0),
                                (indices[1].stop or self.num_cols),
                                (indices[1].step or 1))
            if value.is_empty:
                self._A = [[self._A[i][j] for j in slice_range] for i in range(self.num_rows)
                            if i != indices[0]]
            else:
                value_range = range(value.num_cols)
                for (i,j) in zip(slice_range, value_range):
                    self._A[indices[0]][i] = value[0,j]
        else: # expects int or float
            if not isinstance(value, (float, int)):
                raise ValueError("When setting matrix indices using direct index notation an int "
                                 "or float value must be used.")
            self._A[indices[0]][indices[1]] = value

    def __getitem__(self,
                    indices: Union[Tuple[int, int],
                                   Tuple[int, slice],
                                   Tuple[int, slice],
                                   Tuple[slice, slice],
                                   int,
                                   slice]
                    ) -> Union[float, 'Matrix']:
        M = None
        if isinstance(indices, (int, slice)):
            if self.is_row_matrix():
                indices = (0, indices)
            elif self.is_column_matrix():
                indices = (indices, 0)
            else:
                raise ValueError("Single-index indexing only supported for row or column matrices.")
        if isinstance(indices[0], slice) and isinstance(indices[1], slice): # returns Matrix
            rows_range = range((indices[0].start or 0),
                               (indices[0].stop or self.num_rows),
                               (indices[0].step or 1))
            cols_range = range((indices[1].start or 0),
                               (indices[1].stop or self.num_cols),
                               (indices[1].step or 1))
            M = Matrix([[self.get_row(i)[j] for j in cols_range] for i in rows_range])
        elif isinstance(indices[0], slice): # returns Matrix
            M = Matrix([[self.get_col(indices[1])[i]]
                         for i in range((indices[0].start or 0),
                                        (indices[0].stop or self.num_rows),
                                        (indices[0].step or 1))])
        elif isinstance(indices[1], slice): # returns Matrix
            M = Matrix([[self.get_row(indices[0])[i]
                         for i in range((indices[1].start or 0),
                                        (indices[1].stop or self.num_cols),
                                        (indices[1].step or 1))]])
        else: # returns float
            M = self._A[indices[0]][indices[1]]
        return M

    def __iter__(self) -> Iterator[List[float]]:
        for row in self._A:
            yield row

    @property
    def num_rows(self) -> int:
        return len(self._A)

    @property
    def num_cols(self) -> int:
        return len(self._A[0]) if self._A else 0

    @property
    def size(self) -> Tuple[int, int]:
        return self.num_rows, self.num_cols

    @property
    def is_empty(self) -> bool:
        return not (bool(self.num_rows) or bool(self.num_cols))

    def __eq__(self, other: Union['Matrix', float, int]) -> bool:
        if not isinstance(other, (Matrix, float, int)):
            return False
        if isinstance(other, (float, int)):
            other = Matrix.fill(*self.size, other)
        if self.size != other.size:
            return False
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self[i,j] != other[i,j]:
                    return False
        return True

    def __lt__(self, other: Union['Matrix', float, int]) -> bool:
        if not isinstance(other, (Matrix, float, int)):
            return False
        if isinstance(other, (float, int)):
            other = Matrix.fill(*self.size, other)
        if self.size != other.size:
            return False
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self[i,j] >= other[i,j]:
                    return False
        return True

    def __le__(self, other: Union['Matrix', float, int]) -> bool:
        if not isinstance(other, (Matrix, float, int)):
            return False
        if isinstance(other, (float, int)):
            other = Matrix.fill(*self.size, other)
        if self.size != other.size:
            return False
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self[i,j] > other[i,j]:
                    return False
        return True

    def __add__(self, other: Union['Matrix', float, int]) -> 'Matrix':
        if not isinstance(other, (Matrix, float, int)):
            return NotImplemented
        if isinstance(other, (float, int)):
            other = Matrix.fill(*self.size, other)
        if self.size != other.size:
            raise ValueError("Matrices must be the same size to be added.")
        M = Matrix.zeros(*self.size)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                M[i,j] = self[i,j] + other[i,j]
        return M

    def __radd__(self, other: Union['Matrix', float, int]) -> 'Matrix':
        return self.__add__(other)

    def __sub__(self, other: Union['Matrix', float, int]) -> 'Matrix':
        return self.__add__(-1.0 * other)

    def __rsub__(self, other: Union['Matrix', float, int]) -> 'Matrix':
        return -1.0 * self.__sub__(other)

    def __mul__(self, other: Union['Matrix', float, int]) -> 'Matrix':
        if not isinstance(other, (Matrix, float, int)):
            return NotImplemented
        if isinstance(other, (float, int)):
            M = Matrix.zeros(*self.size)
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    M[i,j] = other * self[i,j]
        elif isinstance(other, Matrix):
            if other.num_rows != self.num_cols:
                raise ValueError("Incorrect dimensions for matrix multiplication. Check that the "
                                 "number of columns in the first matrix matches the number of "
                                 "rows in the second matrix.")
            M = Matrix.zeros(self.num_rows, other.num_cols)
            # TODO: Evaluate _inner_product() local function performance vs that of nested for loop
            for i in range(self.num_rows):
                for j in range(other.num_cols):
                    M[i,j] = sum([x * y for (x,y) in zip(self.get_row(i), other.get_col(j))])
        return M

    def __rmul__(self, other: Union[float, int]) -> 'Matrix':
        if not isinstance(other, (float, int)):
            return NotImplemented
        return self.__mul__(other)

    def __abs__(self) -> 'Matrix':
        M = Matrix.zeros(*self.size)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                M[i,j] = abs(self[i,j])
        return M

    def __neg__(self) -> 'Matrix':
        M = Matrix.zeros(*self.size)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                M[i,j] = -1 * self[i,j]
        return M

    def __len__(self) -> int:
        """Returns the length of the calling Matrix, defined as the maximum dimension.

        Returns:
            int: The maximum dimension of the calling Matrix, i.e. max(num_rows, num_cols)
        """
        return int(max(self.size))

    def get_row(self, idx: int) -> List[float]:
        return self._A[idx]

    def get_col(self, idx: int) -> List[float]:
        return [row[idx] for row in self._A]

    def transpose(self) -> 'Matrix':
        """Returns the transpose of the calling matrix."""
        M = Matrix.zeros(self.num_cols, self.num_rows)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                M[j,i] = self[i,j]
        return M

    def determinant(self) -> float:
        """Returns the determinant of the calling matrix."""
        m,n = self.size
        if m != n:
            raise ValueError("The calling matrix is not square and the determinant does not exist.")
        if m == 2:
            d = self[0,0] * self[1,1] - self[0,1] * self[1,0]
        else:
            d = 0.0
            for j in range(self.num_cols):
                A_temp = self[:,:]
                A_temp[0,:] = Matrix.empty()
                A_temp[:,j] = Matrix.empty()
                d += (self[0,j] * pow(-1,j) * A_temp.determinant())
        return d

    def inverse(self) -> 'Matrix':
        """ Returns the inverse of the calling matrix, computed using the cofactor method.
        """
        def compute_cofactor_matrix(A: 'Matrix') -> 'Matrix':
            """Returns the cofactor matrix computed from the input matrix."""
            m,n = A.size
            if m != n:
                raise ValueError("The input matrix is not square. The cofactor matrix does not "
                                 "exist.")
            M = Matrix.zeros(*A.size)
            for i in range(A.num_rows):
                for j in range(A.num_cols):
                    A_temp = A[:,:]
                    A_temp[i,:] = Matrix.empty()
                    A_temp[:,j] = Matrix.empty()
                    M[i,j] = pow(-1,i+j) * A_temp.determinant()
            return M
        m,n = self.size
        if m != n:
            raise ValueError("The calling matrix is not square. The matrix inverse does not exist.")
        d = self.determinant()
        if not d:
            raise ValueError("The calling matrix is singular. The matrix inverse does not exist.")
        C = compute_cofactor_matrix(self)
        A_inv = (1 / d) * C.transpose()
        return A_inv

    def is_row_matrix(self) -> bool:
        """ Returns True if the calling Matrix is a row matrix (i.e. has one row and one or more
            columns), False otherwise.

        Returns:
            bool: Boolean indicator of whether or not the calling matrix is a row matrix.
        """
        return self.num_rows == 1

    def is_column_matrix(self) -> bool:
        """ Returns True if the calling Matrix is a column matrix (i.e. has one column and one or
            more rows), False otherwise.

        Returns:
            bool: Boolean indicator of whether or not the calling matrix is a column matrix.
        """
        return self.num_cols == 1

    def is_square(self) -> bool:
        """ Returns True if the calling Matrix is square (i.e. the number of rows equals the
            number of columns), False otherwise.

        Returns:
            bool: Boolean indicator of whether or not the calling matrix is square.
        """
        return self.num_rows == self.num_cols

    def to_column_matrix(self) -> 'Matrix':
        """ Returns a copy of the calling Matrix expressed as a column matrix, with each row
            stacked in order.

        Returns:
            Matrix: A copy of the calling matrix, in column matrix form.
        """
        return Matrix.from_column_matrices([row.transpose() for row in self])


class Vec3d(Matrix):
    """ Class represents a Euclidean vector. """

    #pylint: disable=arguments-differ
    @classmethod
    def zeros(cls) -> 'Vec3d':
        return Vec3d(0,0,0)

    #pylint: disable=arguments-differ
    @classmethod
    def ones(cls) -> 'Vec3d':
        return Vec3d(1,1,1)

    @classmethod
    def identity(cls, dim: int) -> 'Vec3d':
        raise NotImplementedError

    @classmethod
    def fill(cls, num_rows: int, num_cols: int, fill_value: float) -> 'Vec3d':
        raise NotImplementedError

    @classmethod
    def empty(cls) -> 'Vec3d':
        raise NotImplementedError

    @classmethod
    def from_matrix(cls, M: Matrix) -> 'Vec3d':
        """ Factory method to construct a Vec3d from a Matrix. The input Matrix must be of size 3x1
            or 1x3 for this operation to be successful.
        Args:
            M (Matrix): The Matrix from which to construct the Vec3d.

        Raises:
            ValueError: Raised if the input Matrix is not of size 3x1 or 1x3.

        Returns:
            Vec3d: The instantiated Vec3d object.
        """
        if M.size not in {(3,1), (1,3)}:
            raise ValueError("Input matrix must be a row or column matrix of length three.")
        return cls(*(M.get_col(0) if M.size == (3,1) else M.get_row(0)))

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        super().__init__([[x],[y],[z]])

    @property
    def x(self) -> float:
        return self[0,0]

    @x.setter
    def x(self, value: float):
        self[0,0] = value

    @property
    def y(self) -> float:
        return self[1,0]

    @y.setter
    def y(self, value: float):
        self[1,0] = value

    @property
    def z(self) -> float:
        return self[2,0]

    @z.setter
    def z(self, value: float):
        self[2,0] = value

    def __str__(self) -> str:
        return f'[x = {self.x}, y = {self.y}, z = {self.z}]'

    def __add__(self, other: Union[Matrix, 'Vec3d']) -> 'Vec3d':
        return Vec3d.from_matrix(super().__add__(other))

    def __sub__(self, other: Union[Matrix, 'Vec3d']) -> 'Vec3d':
        return Vec3d.from_matrix(super().__sub__(other))

    def __rsub__(self, other: Union[Matrix, 'Vec3d']) -> 'Vec3d':
        return Vec3d.from_matrix(super().__rsub__(other))

    def __abs__(self) -> 'Vec3d':
        return Vec3d.from_matrix(super().__abs__())

    def norm(self) -> float:
        """ Returns the Euclidean norm of the calling vector. """
        return sqrt(self.x**2 + self.y**2 + self.z**2)

    def norm_2(self) -> float:
        """ Returns the square of the Euclidean norm of the calling vector. """
        return self.x**2 + self.y**2 + self.z**2

    def cross(self, other: 'Vec3d') -> 'Vec3d':
        """ Returns the cross product of the calling vector with the argument
            vector, computed as C = A x B for C = A.cross(B).
        """
        if not isinstance(other, Vec3d):
            return NotImplemented
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return Vec3d(x,y,z)

    def dot(self, other: 'Vec3d') -> float:
        """ Returns the dot product of the calling vector with the argument
            vector, computed as C = A * B for C = A.dot(B).
        """
        if not isinstance(other, Vec3d):
            return NotImplemented
        return self.x * other.x + self.y * other.y + self.z * other.z

    def normalize(self) -> 'Vec3d':
        """ Normalizes the calling vector in place by its Euclidean norm. """
        m = self.norm()
        self[0,0] /= m
        self[1,0] /= m
        self[2,0] /= m

    def normalized(self) -> 'Vec3d':
        """ Returns the calling vector, normalized by its Euclidean norm. """
        m = self.norm()
        return Vec3d(self.x/m, self.y/m, self.z/m) if m else Vec3d.zeros()


class TimeSpan:
    """ Class represents a time structure supporting nanosecond precision. """

    @classmethod
    def undefined(cls) -> 'TimeSpan':
        """ Factory method to create an undefined TimeSpan. """
        return cls(None,None)

    @classmethod
    def zero(cls) -> 'TimeSpan':
        """ Factory method to create a zero TimeSpan. """
        return cls(0,0)

    @classmethod
    def from_seconds(cls, seconds: float) -> 'TimeSpan':
        """ Factory method to create a TimeSpan from a number of seconds. """
        return cls(*_decompose_decimal_seconds(seconds))

    @classmethod
    def from_minutes(cls, minutes: float) -> 'TimeSpan':
        """ Factory method to create a TimeSpan from a number of minutes. """
        return cls(*_decompose_decimal_seconds(minutes * SECONDS_PER_MINUTE))

    @classmethod
    def from_hours(cls, minutes: float) -> 'TimeSpan':
        """ Factory method to create a TimeSpan from a number of hours. """
        return cls(*_decompose_decimal_seconds(minutes * SECONDS_PER_HOUR))

    @classmethod
    def from_days(cls, days: float) -> 'TimeSpan':
        """ Factory method to create a TimeSpan from a number of mean solar days. """
        return cls(*_decompose_decimal_seconds(days * SECONDS_PER_SOLAR_DAY))

    def __init__(self, whole_seconds: int, nano_seconds: int):
        def normalize_time(ws: int, ns: int) -> Tuple[int,int]:
            """ Function for normalizing whole vs sub-second digits. """
            ws += (copysign(1,ns) * 1)
            ns -= (copysign(1,ns) * NANOSECONDS_PER_SECOND)
            return ws, ns
        self.whole_seconds = None
        self.nano_seconds = None
        if (whole_seconds is not None) and (nano_seconds is not None):
            while abs(nano_seconds) >= NANOSECONDS_PER_SECOND:
                whole_seconds, nano_seconds = normalize_time(whole_seconds, nano_seconds)
            if copysign(1,whole_seconds) != copysign(1,nano_seconds):
                whole_seconds, nano_seconds = normalize_time(whole_seconds, nano_seconds)
            self.whole_seconds = int(whole_seconds)
            self.nano_seconds = int(nano_seconds)

    def __str__(self):
        return f'[whole_seconds = {self.whole_seconds}, nano_seconds = {self.nano_seconds}]' \
               if self.is_defined() else 'Undefined'

    def __hash__(self) -> int:
        return hash((self.whole_seconds, self.nano_seconds))

    def __eq__(self, other: 'TimeSpan') -> bool:
        if not isinstance(other, TimeSpan):
            return False
        if self.whole_seconds != other.whole_seconds:
            return False
        if self.nano_seconds != other.nano_seconds:
            return False
        return True

    def __lt__(self, other: 'TimeSpan') -> bool:
        if not isinstance(other, TimeSpan):
            return False
        if self.whole_seconds > other.whole_seconds:
            return False
        if self.whole_seconds == other.whole_seconds:
            if self.nano_seconds >= other.nano_seconds:
                return False
        return True

    def __le__(self, other: 'TimeSpan') -> bool:
        if not isinstance(other, TimeSpan):
            return False
        if self.whole_seconds > other.whole_seconds:
            return False
        if self.whole_seconds == other.whole_seconds:
            if self.nano_seconds > other.nano_seconds:
                return False
        return True

    def __add__(self, other: 'TimeSpan') -> 'TimeSpan':
        if not isinstance(other, TimeSpan):
            return NotImplemented
        return TimeSpan(self.whole_seconds + other.whole_seconds,
                        self.nano_seconds + other.nano_seconds)

    def __radd__(self, other: 'TimeSpan') -> 'TimeSpan':
        return self.__add__(other)

    def __sub__(self, other: 'TimeSpan') -> 'TimeSpan':
        if not isinstance(other, TimeSpan):
            return NotImplemented
        return TimeSpan(self.whole_seconds - other.whole_seconds,
                        self.nano_seconds - other.nano_seconds)

    def __rsub__(self, other: 'TimeSpan') -> 'TimeSpan':
        return -1.0 * self.__sub__(other)

    def __mul__(self, other: Union[float, int]) -> 'TimeSpan':
        if not isinstance(other, (float, int)):
            return NotImplemented
        ws, ns = _decompose_decimal_seconds(other * self.whole_seconds)
        return TimeSpan(ws, floor(other * self.nano_seconds) + ns)

    def __rmul__(self, other: Union[float, int]) -> 'TimeSpan':
        if not isinstance(other, (float, int)):
            return NotImplemented
        return self.__mul__(other)

    def __abs__(self) -> 'TimeSpan':
        ws = abs(self.whole_seconds) if self.whole_seconds is not None else None
        ns = abs(self.nano_seconds) if self.nano_seconds is not None else None
        return TimeSpan(ws,ns)

    def __neg__(self) -> 'TimeSpan':
        ws = -1 * self.whole_seconds if self.whole_seconds is not None else None
        ns = -1 * self.nano_seconds if self.nano_seconds is not None else None
        return TimeSpan(ws,ns)

    def is_defined(self) -> bool:
        """ Returns a boolean indicator of whether or not the calling TimeSpan is defined.

        Returns:
            bool: Boolean indicator of whether or not the calling TimeSpan is defined.
        """
        return (self.whole_seconds is not None) and (self.nano_seconds is not None)

    def to_seconds(self) -> float:
        """ Returns the calling TimeSpan's value converted to seconds. This conversion could
            potentially not preserve the calling TimeSpan's precision.
        """
        return self.whole_seconds + (self.nano_seconds / NANOSECONDS_PER_SECOND)

    def to_minutes(self) -> float:
        """ Returns the calling TimeSpan's value converted to minutes. This conversion could
            potentially not preserve the calling TimeSpan's precision.
        """
        return self.to_seconds() / SECONDS_PER_MINUTE

    def to_hours(self) -> float:
        """ Returns the calling TimeSpan's value converted to hours. This conversion could
            potentially not preserve the calling TimeSpan's precision.
        """
        return self.to_seconds() / SECONDS_PER_HOUR

    def to_days(self) -> float:
        """ Returns the calling TimeSpan's value converted to mean solar days. This conversion could
            potentially not preserve the calling TimeSpan's precision.
        """
        return self.to_seconds() / SECONDS_PER_SOLAR_DAY


class ElementSetBase():
    """ Class represents a set of generic state vector elements, e.g. a set of Keplerian orbital
        elements, a set of Cartesian orbital elements, a set of Euler angles and the corresponding
        sequence, etc.
    """

    def __init__(self, elements: List[Matrix]):
        self._elements = Matrix.from_column_matrices(elements)

    @property
    def num_elements(self) -> int:
        return self._elements.num_rows

    def to_column_matrix(self) -> Matrix:
        return self._elements

    def from_column_matrix(self, value: Matrix) -> 'ElementSetBase':
        if not isinstance(value, Matrix) or value.num_cols != 1:
            raise ValueError("Input value must be a column matrix.")
        if value.num_rows != self.num_elements:
            raise ValueError(f"Input column matrix must have {self.num_elements} elements.")
        self._elements = value


def _decompose_decimal_seconds(seconds: float) -> Tuple[int, int]:
    decimal_sec = Decimal(str(seconds))
    return int(decimal_sec), int((decimal_sec % 1) * _DECIMAL_NANOSECONDS_PER_SECOND)
