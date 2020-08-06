from math import sqrt
from typing import List
from typing import Tuple

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

    def __setitem__(self, indices: Tuple[int,int], value: float):
        self._A[indices[0]][indices[1]] = value

    def __getitem__(self, indices: Tuple[int,int]) -> float:
        return self._A[indices[0]][indices[1]]

    @property
    def num_rows(self):
        return len(self._A)

    @property
    def num_cols(self):
        return len(self._A[0]) if self._A else 0

    @property
    def size(self):
        return self.num_rows, self.num_cols

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
        return NotImplemented

    def __rsub__(self, other):
        return NotImplemented

    def __mult__(self, other):
        return NotImplemented

    def __rmult__(self, other):
        return NotImplemented

    def transpose(self):
        """Returns the transpose of the calling matrix."""
        M = Matrix.zeros(self.num_cols, self.num_rows)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                M[j,i] = self[i,j]
        return M

    def inverse(self):
        """Returns the inverse of the calling matrix, computed using
           the cofactor method.
        """
        raise NotImplementedError


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
        result = super().__add__(other)
        return Vec3d(result[0,0], result[1,0], result[2,0])

    def __radd__(self, other):
        result = super().__radd__(other)
        return Vec3d(result[0,0], result[1,0], result[2,0])

    def __sub__(self, other):
        result = super().__sub__(other)
        return Vec3d(result[0,0], result[1,0], result[2,0])

    def __rsub__(self, other):
        result = super().__rsub__(other)
        return Vec3d(result[0,0], result[1,0], result[2,0])

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
