""" Module contains unit test definitions for the astrolib.matrix package.
"""

import unittest

from astrolib.matrix import _compute_cofactor_matrix
from astrolib.matrix import Matrix

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
        for i in range(A.size.num_rows):
            for j in range(A.size.num_cols):
                self.assertTrue(
                    A[i, j] == i * 3 + j + 1,
                    "Matrix construction not done correctly.",
                )
        for i in range(B.size.num_rows):
            for j in range(B.size.num_cols):
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
        self.assertTrue(A.size.num_rows == 2)
        with self.assertRaises(AttributeError):
            A.size.num_rows = 1

    def test_num_cols(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(A.size.num_cols == 3)
        with self.assertRaises(AttributeError):
            A.size.num_cols = 1

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

    def test_multiplication(self):
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
        with self.assertRaises(TypeError):
            G * "foo"

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
            A.inverse() - B <= Matrix.fill(A.size.num_rows, A.size.num_cols, tol),
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

    def test_str(self) -> None:
        """Tests for the string method."""
        self.assertTrue(str(Matrix.ones(1, 2)) == "[1.0, 1.0]")
        self.assertTrue(str(Matrix.ones(2, 1)) == "[1.0\n 1.0]")
        self.assertTrue(str(Matrix.ones(2, 2)) == "[1.0, 1.0\n 1.0, 1.0]")

    def test_len(self) -> None:
        """Method defines tests for the len dunder method."""
        self.assertTrue(len(Matrix.ones(3, 1)) == 3)
        self.assertTrue(len(Matrix.ones(1, 3)) == 3)
        self.assertTrue(len(Matrix.ones(3, 3)) == 3)


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
