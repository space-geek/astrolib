""" Module contains unit test definitions for the astrolib.matrix package.
"""

import math
import unittest

from astrolib.matrix import _compute_cofactor_matrix
from astrolib.matrix import Matrix
from astrolib.matrix import Vector3

# pylint: disable=invalid-name
# pylint: disable=line-too-long
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=pointless-statement
# pylint: disable=protected-access


class Test_Matrix(unittest.TestCase):
    def test_constructor(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        B = Matrix([1, 2, 3, 4, 5])
        _ = Matrix([])
        for i in range(A.num_rows):
            for j in range(A.num_cols):
                self.assertTrue(
                    A[i, j] == i * 3 + j + 1,
                    "Matrix construction not done correctly.",
                )
        for i in range(B.num_rows):
            for j in range(B.num_cols):
                self.assertTrue(
                    B[i, j] == i * j + (j + 1),
                    "Matrix construction not done correctly.",
                )
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3, 4, 5]])
        with self.assertRaises(ValueError):
            Matrix(1)
        with self.assertRaises(ValueError):
            Matrix("test")
        with self.assertRaises(ValueError):
            Matrix(["test"])

    def test_get_item(self):
        A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        B = Matrix([[4, 5, 6]])
        C = Matrix([[2], [5], [8]])
        D = Matrix([[4, 6]])
        E = Matrix([[2], [8]])
        F = Matrix([[1, 3], [7, 9]])
        G = Matrix([[2], [5]])
        H = Matrix([[5, 6]])
        self.assertTrue(A[1, 1] == 5, "Matrix indexing not done correctly")
        self.assertTrue(A[1, :] == B, "Matrix indexing not done correctly")
        self.assertTrue(A[:, 1] == C, "Matrix indexing not done correctly")
        self.assertTrue(A[1, 0:3:2] == D, "Matrix indexing not done correctly")
        self.assertTrue(A[0:3:2, 1] == E, "Matrix indexing not done correctly")
        self.assertTrue(A[:, :] == A, "Matrix indexing not done correctly")
        self.assertTrue(A[0:3:2, 0:3:2] == F, "Matrix indexing not done correctly")
        self.assertTrue(C[1] == 5, "Matrix indexing not done correctly.")
        self.assertTrue(C[0:2] == G, "Matrix indexing not done correctly.")
        self.assertTrue(B[1:3] == H, "Matrix indexing not done correctly.")
        with self.assertRaises(ValueError):
            A[0:2]
        with self.assertRaises(ValueError):
            A["abc", "abc"]

    def test_set_item(self):
        A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        B = Matrix([[1, 9, 3], [4, 5, 6], [7, 8, 9]])
        C = Matrix([[1, 3, 3], [4, 6, 6], [7, 9, 9]])
        D = Matrix([[1, 1, 1], [4, 6, 6], [7, 9, 9]])
        E = Matrix([[0, 9, 0], [4, 5, 6], [0, 8, 0]])
        F = Matrix([[1, 9, 1], [1, 5, 1], [1, 8, 1]])
        G = Matrix([[1, 1], [1, 1], [1, 1]])
        H = Matrix([[1, 1], [1, 1]])
        A[0, 1] = 9
        self.assertTrue(A == B, "Matrix assignment not done correctly")
        A[:, 1] = Matrix([[3], [6], [9]])
        self.assertTrue(A == C, "Matrix assignment not done correctly")
        A[0, :] = Matrix.ones(1, 3)
        self.assertTrue(A == D, "Matrix assignment not done correctly")
        A[:, :] = B
        self.assertTrue(A == B, "Matrix assignment not done correctly")
        A[0:3:2, 0:3:2] = Matrix.zeros(2)
        self.assertTrue(A == E, "Matrix assignment not done correctly")
        A[:, 0:3:2] = Matrix.ones(3, 2)
        self.assertTrue(A == F, "Matrix assignment not done correctly")
        with self.assertRaises(ValueError):
            A[:, :] = 1
        with self.assertRaises(ValueError):
            A[1:2, :] = 1
        with self.assertRaises(ValueError):
            A[:, 1:2] = 1
        with self.assertRaises(ValueError):
            A[1, 1] = B
        A[:, 1] = Matrix.empty()
        self.assertTrue(A == G, "Matrix assignment not done correctly")
        A[0, :] = Matrix.empty()
        self.assertTrue(A == H, "Matrix assignment not done correctly")
        with self.assertRaises(ValueError):
            A[:, :] = Matrix.empty()
        mat = Matrix.ones(1, 3)
        mat[1:] = Matrix.zeros(1, 2)
        self.assertTrue(mat == Matrix([1, 0, 0]))
        mat = Matrix.ones(3, 1)
        mat[1:] = Matrix.zeros(2, 1)
        self.assertTrue(mat == Matrix([1, 0, 0]).transpose())
        mat = Matrix.ones(3)
        with self.assertRaises(ValueError):
            mat[1:] = Matrix.zeros(1, 2)
        with self.assertRaises(ValueError):
            mat[1:, 1] = 0.0
        with self.assertRaises(ValueError):
            mat[1, 1:] = 0.0
        with self.assertRaises(ValueError):
            mat["abc", "abc"] = "abc"

    def test_num_rows(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(A.num_rows == 2)
        with self.assertRaises(AttributeError):
            A.num_rows = 1

    def test_num_cols(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(A.num_cols == 3)
        with self.assertRaises(AttributeError):
            A.num_cols = 1

    def test_size(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(A.size == (2, 3))
        with self.assertRaises(AttributeError):
            A.size = 1

    def test_is_empty(self):
        A = Matrix()
        B = Matrix([[1], [2]])
        C = Matrix.empty()
        D = Matrix([])
        self.assertTrue(A.is_empty, "Matrix is empty.")
        self.assertTrue(not B.is_empty, "Matrix is not empty.")
        self.assertTrue(C.is_empty, "Matrix is empty.")
        self.assertTrue(D.is_empty, "Matrix is empty.")

    def test_empty(self):
        A = Matrix.empty()
        self.assertTrue(
            A.size == (0, 0),
            f"Matrix should have size (0, 0), but has size {A.size}.",
        )

    def test_fill(self):
        A = Matrix.fill(2, 2, 1.0)
        B = Matrix([[1, 1], [1, 1]])
        for i in range(2):
            for j in range(2):
                self.assertTrue(
                    A[i, j] == B[i, j], f"Matrix element [{i},{j}] not equal."
                )
        with self.assertRaises(ValueError):
            Matrix.fill(-1, 2, 1)
        with self.assertRaises(ValueError):
            Matrix.fill(0, 2, 1)
        with self.assertRaises(ValueError):
            Matrix.fill(2, -1, 1)
        with self.assertRaises(ValueError):
            Matrix.fill(2, 0, 1)

    def test_zeros(self):
        A = Matrix.zeros(2, 3)
        B = Matrix([[0, 0, 0], [0, 0, 0]])
        C = Matrix.zeros(3)
        D = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        for i in range(2):
            for j in range(3):
                self.assertTrue(
                    A[i, j] == B[i, j], f"Matrix element [{i},{j}] not equal."
                )
        for i in range(3):
            for j in range(3):
                self.assertTrue(
                    C[i, j] == D[i, j], f"Matrix element [{i},{j}] not equal."
                )
        with self.assertRaises(ValueError):
            Matrix.zeros(-1, 2)
        with self.assertRaises(ValueError):
            Matrix.zeros(0, 2)
        with self.assertRaises(ValueError):
            Matrix.zeros(2, -1)
        with self.assertRaises(ValueError):
            Matrix.zeros(2, 0)

    def test_ones(self):
        A = Matrix.ones(2, 3)
        B = Matrix([[1, 1, 1], [1, 1, 1]])
        C = Matrix.ones(3)
        D = Matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        for i in range(2):
            for j in range(3):
                self.assertTrue(
                    A[i, j] == B[i, j], f"Matrix element [{i},{j}] not equal."
                )
        for i in range(3):
            for j in range(3):
                self.assertTrue(
                    C[i, j] == D[i, j], f"Matrix element [{i},{j}] not equal."
                )
        with self.assertRaises(ValueError):
            Matrix.ones(-1, 2)
        with self.assertRaises(ValueError):
            Matrix.ones(0, 2)
        with self.assertRaises(ValueError):
            Matrix.ones(2, -1)
        with self.assertRaises(ValueError):
            Matrix.ones(2, 0)

    def test_identity(self):
        A = Matrix.identity(2)
        B = Matrix([[1, 0], [0, 1]])
        C = Matrix.identity(3)
        D = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for i in range(2):
            for j in range(2):
                self.assertTrue(
                    A[i, j] == B[i, j], f"Matrix element [{i},{j}] not equal."
                )
        for i in range(3):
            for j in range(3):
                self.assertTrue(
                    C[i, j] == D[i, j], f"Matrix element [{i},{j}] not equal."
                )
        with self.assertRaises(ValueError):
            Matrix.identity(-1)
        with self.assertRaises(ValueError):
            Matrix.identity(0)

    def test_equals(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        B = Matrix([[4, 5, 6], [7, 8, 9]])
        C = Matrix([[1, 2, 3], [4, 5, 6]])
        D = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        E = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        F = 1.0
        self.assertTrue(A != B, "Matrices are not equal.")
        self.assertTrue(A == C, "Matrices are equal.")
        self.assertTrue(A != D, "Matrices are not equal.")
        self.assertTrue(A != E, "Matrices are not equal.")
        self.assertTrue(A != F, "Matrices are not equal.")
        # pylint: disable=not-an-iterable
        self.assertTrue(Matrix.ones(*A.size) == F, "Matrices are equal.")
        self.assertTrue(
            Matrix.ones(3) != "abc", "Comparison did not behave as expected."
        )

    def test_comparison(self):
        A = Matrix.ones(3)
        B = 2 * Matrix.ones(3)
        C = Matrix.ones(3)
        D = 3.0
        E = 0.0
        F = 1.0
        self.assertTrue(A < B, "A is less than B")
        self.assertTrue(A <= B, "A is less than or equal to B")
        self.assertTrue(A <= C, "A is less than or equal to C")
        self.assertTrue(A < D, "A is less than D")
        self.assertTrue(A <= D, "A is less than or equal to D")
        self.assertTrue(B > A, "B is greater than A")
        self.assertTrue(B >= A, "B is greater than or equal to A")
        self.assertTrue(C >= A, "C is greater than or equal to A")
        self.assertTrue(D > A, "D is greater than A")
        self.assertTrue(D >= A, "D is greater than or equal to A")
        self.assertTrue(A > E, "A is greater than E")
        self.assertTrue(A >= E, "A is greater than or equal to E")
        self.assertTrue(A >= F, "A is greater than or equal to F")
        self.assertFalse(A < Matrix.zeros(2))
        self.assertFalse(A <= Matrix.zeros(2))
        with self.assertRaises(TypeError):
            A < "foo"
        with self.assertRaises(TypeError):
            A <= "foo"

    def test_neg(self):
        A = Matrix.ones(3, 2)
        B = Matrix.fill(3, 2, -1)
        self.assertTrue(-A == B, "Matrices are equal")

    def test_add(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        B = Matrix([[4, 5, 6], [7, 8, 9]])
        C = Matrix([[5, 7, 9], [11, 13, 15]])
        D = Matrix([[-1, -2, -3], [-4, -5, -6]])
        E = Matrix.ones(1, 1)
        self.assertTrue(A + B == C, "The matrix sum was not computed successfully.")
        self.assertTrue(B + A == C, "The matrix sum was not computed successfully.")
        self.assertTrue(
            A + D == Matrix.zeros(2, 3), "The matrix sum was not computed successfully."
        )
        self.assertTrue(
            D + A == Matrix.zeros(2, 3), "The matrix sum was not computed successfully."
        )
        self.assertTrue(A + 3 == B, "The matrix sum was not computed successfully.")
        self.assertTrue(3 + A == B, "The matrix sum was not computed successfully.")
        with self.assertRaises(TypeError):
            A + "foo"
        with self.assertRaises(TypeError):
            "foo" + A
        with self.assertRaises(ValueError):
            A + E
        with self.assertRaises(ValueError):
            E + A

    def test_subtract(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        B = Matrix([[4, 5, 6], [7, 8, 9]])
        C = Matrix([[-2, -1, 0], [1, 2, 3]])
        D = Matrix([[2, 1, 0], [-1, -2, -3]])
        self.assertTrue(
            A - B == Matrix.fill(2, 3, -3),
            "The matrix difference was not computed successfully.",
        )
        self.assertTrue(
            B - A == Matrix.fill(2, 3, 3),
            "The matrix difference was not computed successfully.",
        )
        self.assertTrue(
            A - 3 == C, "The matrix difference was not computed successfully."
        )
        self.assertTrue(
            3 - A == D, "The matrix difference was not computed successfully."
        )
        with self.assertRaises(TypeError):
            A - "foo"
        with self.assertRaises(TypeError):
            "foo" - A
        with self.assertRaises(ValueError):
            # pylint: disable=expression-not-assigned
            A - Matrix.ones(2)

    def test_mult(self):
        A = Matrix(
            [
                [4, 4, 3, 1, 2],
                [2, 9, 8, 1, 6],
                [3, 6, 8, 6, 5],
                [7, 6, 4, 8, 1],
                [5, 10, 6, 10, 4],
            ]
        )
        B = Matrix(
            [
                [0.1206, 0.2518, 0.9827, 0.9063, 0.0225],
                [0.5895, 0.2904, 0.7302, 0.8797, 0.4253],
                [0.2262, 0.6171, 0.3439, 0.8178, 0.3127],
                [0.3846, 0.2653, 0.5841, 0.2607, 0.1615],
                [0.5830, 0.8244, 0.1078, 0.5944, 0.1788],
            ]
        )
        C = Matrix(
            [
                [5.0696, 5.9343, 8.6829, 11.0466, 3.2483],
                [11.2388, 13.2658, 12.5193, 20.0984, 7.6082],
                [10.9310, 13.1484, 14.1238, 19.0751, 6.9836],
                [8.9460, 8.9203, 17.4160, 17.5733, 5.4307],
                [14.0334, 13.8163, 20.5508, 23.2193, 8.5714],
            ]
        )
        D = Matrix(
            [
                [10.3908, 14.3077, 13.9979, 13.7440, 7.6617],
                [13.4135, 18.8840, 16.0042, 16.5513, 9.1536],
                [10.4585, 16.5556, 13.5137, 12.5758, 7.9429],
                [6.4538, 10.6096, 9.9605, 7.8550, 6.1879],
                [9.3583, 15.7517, 12.6561, 8.5965, 7.9605],
            ]
        )
        E = Matrix(
            [
                [31.1723, 42.9230, 41.9937, 41.2320, 22.9852],
                [40.2406, 56.6520, 48.0126, 49.6538, 27.4608],
                [31.3754, 49.6667, 40.5410, 37.7274, 23.8288],
                [19.3613, 31.8289, 29.8814, 23.5650, 18.5638],
                [28.0750, 47.2552, 37.9684, 25.7895, 23.8815],
            ]
        )
        F = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        G = Matrix([[1, 2], [3, 4], [5, 6]])
        H = Matrix([[22, 28], [49, 64], [76, 100]])
        I = Matrix([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
        tol = 2.0e-3
        self.assertTrue(
            abs(A * B - C) <= tol,
            "The matrix product was not computed successfully.",
        )
        self.assertTrue(
            abs(B * A - D) <= tol,
            "The matrix product was not computed successfully.",
        )
        self.assertTrue(
            abs(3 * D - E) <= tol,
            "The matrix product was not computed successfully.",
        )
        self.assertTrue(
            abs(D * 3 - E) <= tol,
            "The matrix product was not computed successfully.",
        )
        self.assertTrue(
            abs(F * G - H) <= tol,
            "The matrix product was not computed successfully.",
        )

        self.assertTrue(
            isinstance(I.transpose() * I, int),
            "The matrix product was not computed successfully.",
        )
        self.assertTrue(
            abs(I.transpose() * I - 285.0) <= tol,
            "The matrix product was not computed successfully.",
        )
        with self.assertRaises(ValueError):
            G * F
        with self.assertRaises(TypeError):
            "foo" * G

    def test_transpose(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        B = Matrix([[1, 4], [2, 5], [3, 6]])
        self.assertTrue(
            A.transpose() == B, "The matrix transpose was not computed successfully."
        )

    def test_determinant(self):
        A = Matrix([[3, 0, 1], [0, 5, 0], [-1, 1, -1]])
        B = Matrix([[3, 0, 1], [0, 5, 0]])
        self.assertTrue(
            A.determinant() == -10,
            "The matrix determinant was not computed successfully.",
        )
        with self.assertRaises(ValueError):
            B.determinant()
        # TODO Add more test cases

    def test_inverse(self):
        A = Matrix([[3, 0, 1], [0, 5, 0], [-1, 1, -1]])
        B = Matrix([[0.5, -0.1, 0.5], [0, 0.2, 0], [-0.5, 0.3, -1.5]])
        C = Matrix([[3, 0, 1], [0, 5, 0]])
        tol = 1.0e-10
        self.assertTrue(
            A.inverse() - B <= Matrix.fill(A.num_rows, A.num_cols, tol),
            "The matrix inverse was not computed successfully.",
        )
        with self.assertRaises(ValueError):
            C.inverse()
        with self.assertRaises(ValueError):
            Matrix.zeros(3).inverse()
        # TODO Add more test cases

    def test_diag(self) -> None:
        A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        B = Matrix([[1], [5], [9]])
        C = Matrix([[1, 2], [3, 4], [5, 6]])
        D = Matrix([[1], [4]])
        self.assertTrue(A.diag == B)
        self.assertTrue(C.diag == D)

    def test_is_square(self) -> None:
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(A.is_square)
        self.assertTrue(not B.is_square)

    def test_is_row_matrix(self) -> None:
        """Tests for the is_row_matrix method."""
        A = Matrix.ones(1, 3)
        B = Matrix.ones(3, 1)
        C = Matrix.ones(3, 3)
        D = Matrix.ones(1, 1)
        self.assertTrue(A.is_row_matrix)
        self.assertFalse(B.is_row_matrix)
        self.assertFalse(C.is_row_matrix)
        self.assertTrue(D.is_row_matrix)

    def test_is_column_matrix(self) -> None:
        """Tests for the is_column_matrix method."""
        A = Matrix.ones(1, 3)
        B = Matrix.ones(3, 1)
        C = Matrix.ones(3, 3)
        D = Matrix.ones(1, 1)
        self.assertFalse(A.is_column_matrix)
        self.assertTrue(B.is_column_matrix)
        self.assertFalse(C.is_column_matrix)
        self.assertTrue(D.is_column_matrix)

    def test_trace(self) -> None:
        A = Matrix.identity(5)
        B = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        C = Matrix([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(A.trace == 5)
        self.assertTrue(B.trace == 15)
        with self.assertRaises(ValueError):
            C.trace

    @unittest.expectedFailure
    def test_adjoint(self) -> None:
        # TODO add adjoint tests once implementation is complete
        self.assertTrue(False)

    def test_hash(self) -> None:
        """Unit tests for the hash dunder method."""
        mat = Matrix([[1, 2]])
        expected = hash("[1, 2]")
        self.assertTrue(hash(mat) == expected)

    def test_repr(self) -> None:
        """Unit tests for the repr dunder method."""
        self.assertTrue(repr(Matrix()) == "Matrix()")
        self.assertTrue(repr(Matrix.identity(2)) == "Matrix([[1.0, 0.0], [0.0, 1.0]])")
        self.assertTrue(repr(Matrix([1, 2])) == "Matrix([[1, 2]])")
        self.assertTrue(repr(Matrix([[1], [2]])) == "Matrix([[1], [2]])")

    def test_round(self) -> None:
        """Unit tests for the round dunder method."""
        A = Matrix.fill(10, 10, 1.0e-3)
        self.assertTrue(round(A) == A)
        self.assertTrue(round(A, 3) == A)
        # pylint: disable=not-an-iterable
        self.assertTrue(round(A, 2) == Matrix.zeros(*A.size))

    def test_iter(self) -> None:
        """Unit tests for the iter dunder method."""
        self.assertTrue(all(isinstance(x, float) for x in Matrix.ones(2, 1)))
        self.assertTrue(all(isinstance(x, float) for x in Matrix.ones(1, 2)))
        self.assertTrue(all(isinstance(x, list) for x in Matrix.ones(2, 2)))


class Test_Vector3(unittest.TestCase):
    def test_constructor(self):
        A = Vector3(1, 2, 3)
        self.assertIsInstance(A.x, float)
        self.assertIsInstance(A.y, float)
        self.assertIsInstance(A.z, float)
        self.assertTrue(A.x == 1, "The vector was not initialized successfully.")
        self.assertTrue(A.y == 2, "The vector was not initialized successfully.")
        self.assertTrue(A.z == 3, "The vector was not initialized successfully.")

    def test_not_supported_superclass_methods(self) -> None:
        """Tests to verify not supported superclass methods."""
        with self.assertRaises(NotImplementedError):
            Vector3.identity(3)
        with self.assertRaises(NotImplementedError):
            Vector3.fill(1, 2, 3)
        with self.assertRaises(NotImplementedError):
            Vector3.empty()

    def test_ones(self):
        A = Vector3.ones()
        self.assertTrue(A.x == 1, "The vector was not initialized successfully.")
        self.assertTrue(A.y == 1, "The vector was not initialized successfully.")
        self.assertTrue(A.z == 1, "The vector was not initialized successfully.")

    def test_zeros(self):
        A = Vector3.zeros()
        self.assertTrue(
            isinstance(A.x, float), "The vector was not initialized successfully."
        )
        self.assertTrue(
            isinstance(A.y, float), "The vector was not initialized successfully."
        )
        self.assertTrue(
            isinstance(A.z, float), "The vector was not initialized successfully."
        )
        self.assertTrue(A.x == 0, "The vector was not initialized successfully.")
        self.assertTrue(A.y == 0, "The vector was not initialized successfully.")
        self.assertTrue(A.z == 0, "The vector was not initialized successfully.")

    def test_from_matrix(self) -> None:
        """Tests for the from_matrix class method."""
        vec = Vector3.from_matrix(Matrix.ones(3, 1))
        self.assertTrue(vec, Vector3(1, 1, 1))
        vec = Vector3.from_matrix(Matrix.ones(1, 3))
        self.assertTrue(vec, Vector3(1, 1, 1))
        with self.assertRaises(ValueError):
            Vector3.from_matrix("abc")
        with self.assertRaises(ValueError):
            Vector3.from_matrix(Matrix.ones(3))

    def test_add(self):
        A = Vector3(1, 2, 3)
        B = Vector3(4, 5, 6)
        C = Vector3(5, 7, 9)
        self.assertTrue(A + B == C, "The vector sum was not computed successfully.")
        self.assertTrue(B + A == C, "The vector sum was not computed successfully.")
        self.assertTrue(A + 3 == B, "The vector sum was not computed successfully.")
        self.assertTrue(3 + A == B, "The vector sum was not computed successfully.")
        self.assertTrue(A + Matrix.ones(3, 1) == Vector3(2, 3, 4))
        with self.assertRaises(ValueError):
            # pylint: disable=expression-not-assigned
            A + Matrix.ones(3, 3)
        with self.assertRaises(TypeError):
            # pylint: disable=expression-not-assigned
            "foo" + A

    def test_subtract(self):
        A = Vector3(1, 2, 3)
        B = Vector3(4, 5, 6)
        C = Vector3(5, 7, 9)
        self.assertTrue(
            C - B == A, "The vector difference was not computed successfully."
        )
        self.assertTrue(
            C - A == B, "The vector difference was not computed successfully."
        )
        self.assertTrue(
            B - 3 == A, "The vector difference was not computed successfully."
        )
        self.assertTrue(Vector3.unit_x() - Matrix.ones(3, 1) == Vector3(0, -1, -1))
        with self.assertRaises(ValueError):
            # pylint: disable=expression-not-assigned
            A - Matrix.ones(3, 3)
        with self.assertRaises(TypeError):
            # pylint: disable=expression-not-assigned
            "foo" - A

    def test_multiplication(self):
        x = Vector3(1, 2, 3)
        A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = Vector3(14, 32, 50)
        c = Vector3(2, 4, 6)
        self.assertTrue(
            Matrix.identity(3) * x == x,
            "The matrix multiplication with the vector was not computed successfully.",
        )
        self.assertTrue(
            A * x == b,
            "The matrix multiplication with the vector was not computed successfully.",
        )
        self.assertTrue(
            isinstance(A * x, Vector3),
            "The matrix multiplication with the vector was not computed successfully.",
        )
        self.assertTrue(
            2 * x == c,
            "The scalar multiplication with the vector was not computed successfully.",
        )
        self.assertTrue(
            x * 2 == c,
            "The scalar multiplication with the vector was not computed successfully.",
        )
        self.assertTrue(
            isinstance(2 * x, Vector3),
            "The scalar multiplication with the vector was not computed successfully.",
        )
        self.assertTrue(
            b * c.transpose()
            == Matrix(
                [
                    [28, 56, 84],
                    [64, 128, 192],
                    [100, 200, 300],
                ]
            ),
            "The multiplication with the vector was not computed successfully.",
        )
        self.assertTrue(
            isinstance(b * c.transpose(), Matrix),
            "The scalar multiplication with the vector was not computed successfully.",
        )
        with self.assertRaises(ValueError):
            x * A
        with self.assertRaises(TypeError):
            # pylint: disable=expression-not-assigned
            Vector3.unit_x() * "abc"
        self.assertTrue(isinstance(Matrix.ones(1, 3) * Vector3.unit_x(), float))
        self.assertTrue(Matrix.ones(1, 3) * Vector3.unit_x() == 1.0)
        self.assertTrue(isinstance(Matrix.ones(2, 3) * Vector3.unit_x(), Matrix))
        self.assertTrue(Matrix.ones(2, 3) * Vector3.unit_x() == Matrix([[1], [1]]))
        self.assertTrue(isinstance(Matrix.identity(3) * Vector3.unit_x(), Vector3))
        self.assertTrue(Matrix.identity(3) * Vector3.unit_x() == Vector3.unit_x())
        with self.assertRaises(ValueError):
            # pylint: disable=expression-not-assigned
            Matrix.zeros(3, 1) * x
        with self.assertRaises(TypeError):
            # pylint: disable=expression-not-assigned
            "abc" * Vector3.unit_x()

    def test_norm(self):
        A = Vector3(0.5377, 1.8339, -2.2588)
        self.assertTrue(
            A.norm() - 3.0 <= 1.0e-16, "The vector norm was not computed successfully."
        )
        self.assertTrue(
            A.squared_norm() - 9.0 <= 1.0e-16,
            "The vector norm was not computed successfully.",
        )

    def test_cross(self):
        # TODO Update Matrix class to utilize Decimal class for its elements instead of floats to increase numeric precision
        A = Vector3(0.5377, 1.8339, -2.2588)
        B = Vector3(3.0349, 0.7254, -0.0631)
        AxB = Vector3(1.522941669438591, -6.821524812007696, -5.175674650707535)
        BxA = Vector3(-1.522941669438591, 6.821524812007696, 5.175674650707535)
        self.assertTrue(
            abs(A.cross(B) - AxB) <= 1.0e-03 * Vector3.ones(),
            "The cross product was not computed successfully.",
        )
        self.assertTrue(
            abs(B.cross(A) - BxA) <= 1.0e-03 * Vector3.ones(),
            "The cross product was not computed successfully.",
        )
        with self.assertRaises(NotImplementedError):
            A.cross("abc")

    def test_dot(self):
        # TODO Update Matrix class to utilize Decimal class for its elements instead of floats to increase numeric precision
        A = Vector3(0.5377, 1.8339, -2.2588)
        B = Vector3(3.0349, 0.7254, -0.0631)
        dot_product = 3.104517858912047
        self.assertTrue(
            abs(A.dot(B) - dot_product) <= 1.0e-03,
            "The dot product was not computed successfully.",
        )
        self.assertTrue(
            abs(B.dot(A) - dot_product) <= 1.0e-03,
            "The dot product was not computed successfully.",
        )
        with self.assertRaises(NotImplementedError):
            A.dot("abc")

    def test_normalize(self):
        A = Vector3(0.5377, 1.8339, -2.2588)
        A_norm = Vector3(0.1792, 0.6113, -0.7529)
        self.assertTrue(
            abs(A.normalized() - A_norm) <= 1.0e-01 * Vector3.ones(),
            "The normalized form of the vector was not computed successfully.",
        )
        A.normalize()
        self.assertTrue(
            abs(A - A_norm) <= 1.0e-01 * Vector3.ones(),
            "The normalized form of the vector was not computed successfully.",
        )
        self.assertTrue(
            A.norm() == 1.0,
            "The normalized form of the vector was not computed successfully.",
        )
        B = Vector3(-3.0, 0, 0)
        self.assertTrue(
            B.normalized() - Vector3(-1.0, 0, 0) <= 1.0e-6,
            "The vector norm was not computed successfully.",
        )

    def test_neg(self):
        A = Vector3(0.5377, 1.8339, -2.2588)
        self.assertTrue(isinstance(-A, Vector3))
        self.assertTrue(abs(A + -A) <= 1.0e-4)

    def test_vertex_angle(self):
        A = Vector3(1, 0, 0)
        B = Vector3(0, 1, 0)
        C = Vector3(math.cos(math.pi / 6), math.sin(math.pi / 6), 0)
        D = Vector3.zeros()
        self.assertTrue(abs(A.vertex_angle(B) - math.pi / 2) <= 1.0e-6)
        self.assertTrue(abs(A.vertex_angle(C) - math.pi / 6) <= 1.0e-6)
        self.assertTrue(A.vertex_angle(D) == 0.0)
        self.assertTrue(D.vertex_angle(A) == 0.0)
        with self.assertRaises(NotImplementedError):
            A.vertex_angle("abc")

    def test_unit_x(self):
        self.assertTrue(Vector3.unit_x() == Vector3(1, 0, 0))

    def test_unit_y(self):
        self.assertTrue(Vector3.unit_y() == Vector3(0, 1, 0))

    def test_unit_z(self):
        self.assertTrue(Vector3.unit_z() == Vector3(0, 0, 1))

    def test_skew(self):
        A = Vector3(1, 2, 3)
        B = Matrix([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        self.assertTrue(abs(A.skew() - B) <= 1.0e-16)

    def test_str(self) -> None:
        """Tests for the str method."""
        self.assertTrue(str(Vector3(1, 2, 3.0)) == "[1.0, 2.0, 3.0]")

    def test_repr(self) -> None:
        """Tests for the Vector3 repr method."""
        self.assertTrue(repr(Vector3(1, 2, 3.0)) == "Vector3(1.0, 2.0, 3.0)")


class Test_ComputeCofactorMatrix(unittest.TestCase):
    """Class defines unit tests for the
    _compute_cofactor_matrix() function.
    """

    @unittest.expectedFailure
    def test_success(self) -> None:
        """Method defines tests for the function."""
        A = Matrix([[-4, 7], [-11, 9]])
        B = Matrix([[9, 11], [-7, -4]])
        self.assertTrue(_compute_cofactor_matrix(A) == B)

    def test_failure(self) -> None:
        """Method defines tests for the function"""
        with self.assertRaises(ValueError):
            _compute_cofactor_matrix(Matrix.ones(3, 2))
