from math import sqrt
from typing import List
from typing import Tuple
from typing import Union

class Matrix:
    """Class represents a matrix."""

    @classmethod
    def fill(cls, num_rows: int, num_cols: int, fill_value: float):
        if (num_rows <= 0) or (num_cols <= 0):
            raise ValueError("The number of rows or columns must be positive and greater than 0.")
        fill_value = float(fill_value)
        return cls(A=[[fill_value for _ in range(int(num_cols))] for _ in range(int(num_rows))])

    @classmethod
    def zeros(cls, num_rows: int, num_cols: int = None):
        if not num_cols and num_cols != 0.0:
            num_cols = num_rows
        return Matrix.fill(num_rows, num_cols, 0.0)

    @classmethod
    def ones(cls, num_rows: int, num_cols: int = None):
        if not num_cols and num_cols != 0.0:
            num_cols = num_rows
        return Matrix.fill(num_rows, num_cols, 1.0)

    @classmethod
    def identity(cls, dim: int):
        A = Matrix.zeros(dim)
        for i in range(dim):
            A[i,i] = 1.0
        return A

    @classmethod
    def empty(cls):
        return Matrix([])

    def __init__(self, A: List[List]):
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

    def __setitem__(self, indices: Union[Tuple[int,slice],Tuple[int,slice],Tuple[slice,slice]], value):
        if isinstance(indices[0], slice) and isinstance(indices[1], slice): # expects Matrix
            if not isinstance(value, Matrix):
                raise ValueError("When setting matrix indices using slice notation a Matrix value must be used.")
            if value.is_empty:
                raise ValueError("When setting matrix indices using slice notation for both the row and column indices an empty Matrix value cannot be used.")
            slice_rows_range = range((indices[0].start or 0), (indices[0].stop or self.num_rows), (indices[0].step or 1))
            slice_cols_range = range((indices[1].start or 0), (indices[1].stop or self.num_cols), (indices[1].step or 1))
            value_rows_range = range(value.num_rows)
            value_cols_range = range(value.num_cols)
            for (i,m) in zip(slice_rows_range, value_rows_range):
                for (j,n) in zip(slice_cols_range, value_cols_range):
                    self._A[i][j] = value[m,n]
        elif isinstance(indices[0], slice): # expects Matrix
            if not isinstance(value, Matrix):
                raise ValueError("When setting matrix indices using slice notation a Matrix value must be used.")
            slice_range = range((indices[0].start or 0), (indices[0].stop or self.num_rows), (indices[0].step or 1))
            if value.is_empty:
                self._A = [[self._A[i][j] for j in range(self.num_cols) if j != indices[1]] for i in slice_range]
            else:
                value_range = range(value.num_rows)
                for (i,j) in zip(slice_range, value_range):
                    self._A[i][indices[1]] = value[j,0]
        elif isinstance(indices[1], slice): # expects Matrix
            if not isinstance(value, Matrix):
                raise ValueError("When setting matrix indices using slice notation a Matrix value must be used.")
            slice_range = range((indices[1].start or 0), (indices[1].stop or self.num_cols), (indices[1].step or 1))
            if value.is_empty:
                self._A = [[self._A[i][j] for j in slice_range] for i in range(self.num_rows) if i != indices[0]]
            else:
                value_range = range(value.num_cols)
                for (i,j) in zip(slice_range, value_range):
                    self._A[indices[0]][i] = value[0,j]
        else: # expects int or float
            if not (isinstance(value, float) or isinstance(value, int)):
                raise ValueError("When setting matrix indices using direct index notation an int or float value must be used.")
            self._A[indices[0]][indices[1]] = value

    def __getitem__(self, indices: Union[Tuple[int,slice],Tuple[int,slice],Tuple[slice,slice]]):
        M = None
        if isinstance(indices[0], slice) and isinstance(indices[1], slice): # returns Matrix
            rows_range = range((indices[0].start or 0), (indices[0].stop or self.num_rows), (indices[0].step or 1))
            cols_range = range((indices[1].start or 0), (indices[1].stop or self.num_cols), (indices[1].step or 1))
            M = Matrix([[self.get_row(i)[j] for j in cols_range] for i in rows_range])
        elif isinstance(indices[0], slice): # returns Matrix
            M = Matrix([[self.get_col(indices[1])[i]] for i in range((indices[0].start or 0), (indices[0].stop or self.num_rows), (indices[0].step or 1))])
        elif isinstance(indices[1], slice): # returns Matrix
            M = Matrix([[self.get_row(indices[0])[i] for i in range((indices[1].start or 0), (indices[1].stop or self.num_cols), (indices[1].step or 1))]])
        else: # returns float
            M = self._A[indices[0]][indices[1]]
        return M

    @property
    def num_rows(self):
        return len(self._A)

    @property
    def num_cols(self):
        return len(self._A[0]) if self._A else 0

    @property
    def size(self):
        return self.num_rows, self.num_cols

    @property
    def is_empty(self):
        return not (bool(self.num_rows) or bool(self.num_cols))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Matrix):
            return False
        if (self.size != other.size):
            return False
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self[i,j] != other[i,j]:
                    return False
        return True

    def __lt__(self, other) -> bool:
        if not isinstance(other, Matrix):
            return False
        if (self.size != other.size):
            return False
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self[i,j] >= other[i,j]:
                    return False
        return True

    def __le__(self, other) -> bool:
        if not isinstance(other, Matrix):
            return False
        if (self.size != other.size):
            return False
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self[i,j] > other[i,j]:
                    return False
        return True

    def __add__(self, other):
        if not (isinstance(other, Matrix) or \
                isinstance(other, float) or \
                isinstance(other, int)):
            return NotImplemented
        if isinstance(other, float) or isinstance(other, int):
            other = Matrix.fill(*self.size, other)
        if (self.size != other.size):
            raise ValueError("Matrices must be the same size to be added.")
        M = Matrix.zeros(*self.size)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                M[i,j] = self[i,j] + other[i,j]
        return M

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-1.0 * other)

    def __rsub__(self, other):
        return -1.0 * self.__sub__(other)

    def __mul__(self, other):
        if not (isinstance(other, Matrix) or \
                isinstance(other, float) or \
                isinstance(other, int)):
            return NotImplemented
        if isinstance(other, float) or isinstance(other, int):
            M = Matrix.zeros(*self.size)
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    M[i,j] = other * self[i,j]
        elif (isinstance(other, Matrix)):
            if other.num_rows != self.num_cols:
                raise ValueError("Incorrect dimensions for matrix multiplication. Check that the number of columns in the first matrix matches the number of rows in the second matrix.")
            M = Matrix.zeros(self.num_rows, other.num_cols)
            for i in range(self.num_rows):
                for j in range(other.num_cols):
                    M[i,j] = sum([x * y for (x,y) in zip(self.get_row(i), other.get_col(j))])
        return M

    def __rmul__(self, other):
        if not (isinstance(other, float) or isinstance(other, int)):
            return NotImplemented
        return self.__mul__(other)

    def __abs__(self):
        M = Matrix.zeros(*self.size)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                M[i,j] = abs(self[i,j])
        return M

    def get_row(self, idx: int) -> List:
        return self._A[idx]

    def get_col(self, idx: int) -> List:
        return [row[idx] for row in self._A]

    def transpose(self):
        """Returns the transpose of the calling matrix."""
        M = Matrix.zeros(self.num_cols, self.num_rows)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                M[j,i] = self[i,j]
        return M

    def determinant(self):
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

    def inverse(self):
        """Returns the inverse of the calling matrix, computed using
           the cofactor method.
        """
        def compute_cofactor_matrix(A: Matrix):
            """Returns the cofactor matrix for the calling matrix."""
            m,n = A.size
            if m != n:
                raise ValueError("The calling matrix is not square and the cofactor matrix does not exist.")
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
            raise ValueError("The calling matrix is not square and the inverse does not exist.")
        d = self.determinant()
        if not d:
            raise ValueError("The calling matrix is singular and not invertible.")
        C = compute_cofactor_matrix(self)
        A_inv = (1 / d) * C.transpose()
        return A_inv


class Vec3d(Matrix):
    """Class represents a Euclidean vector."""

    @classmethod
    def zeros(cls):
        return Vec3d(0,0,0)

    @classmethod
    def ones(cls):
        return Vec3d(1,1,1)

    @classmethod
    def identity(cls, dim: int):
        raise NotImplementedError()

    @classmethod
    def fill(cls, num_rows: int, num_cols: int, fill_value: float):
        raise NotImplementedError()

    @classmethod
    def from_matrix(cls, M: Matrix):
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

    def __add__(self, other):
        return Vec3d.from_matrix(super().__add__(other))

    def __sub__(self, other):
        return Vec3d.from_matrix(super().__sub__(other))

    def __rsub__(self, other):
        return Vec3d.from_matrix(super().__rsub__(other))

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

    def normalized(self):
        """Returns the calling vector, normalized by its Euclidean norm."""
        norm = self.norm
        return Vec3d(self.x/norm, self.y/norm, self.z/norm)
