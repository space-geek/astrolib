""" Module contains unit test definitions for the astrolib.matrix package.
"""

import math
import unittest

from astrolib.matrix import Matrix
from astrolib.vector import Vector3

# pylint: disable=invalid-name


class Test_Vector3(unittest.TestCase):
    """Unit tests for the Vector3 class."""

    def test_constructor(self):
        """Unit tests for the class constructor."""
        A = Vector3(1, 2, 3)
        self.assertIsInstance(A.x, float)
        self.assertIsInstance(A.y, float)
        self.assertIsInstance(A.z, float)
        self.assertTrue(A.x == 1, "The vector was not initialized successfully.")
        self.assertTrue(A.y == 2, "The vector was not initialized successfully.")
        self.assertTrue(A.z == 3, "The vector was not initialized successfully.")

    def test_ones(self):
        """Unit tests for the ones method."""
        A = Vector3.ones()
        self.assertTrue(A.x == 1, "The vector was not initialized successfully.")
        self.assertTrue(A.y == 1, "The vector was not initialized successfully.")
        self.assertTrue(A.z == 1, "The vector was not initialized successfully.")

    def test_zeros(self):
        """Unit tests for the zeros method."""
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

    def test_getitem(self) -> None:
        """Tests for vector access by index."""
        v = Vector3(*range(3))
        self.assertTrue(v[0] == 0)
        self.assertTrue(v[1] == 1)
        self.assertTrue(v[2] == 2)
        self.assertTrue(v[0] == v.x)
        self.assertTrue(v[1] == v.y)
        self.assertTrue(v[2] == v.z)
        self.assertIsInstance(v[:], Matrix)
        self.assertTrue(v[1:3] == Matrix([1, 2]).transpose())
        with self.assertRaises(IndexError):
            # pylint: disable=pointless-statement
            v[4]
        with self.assertRaises(ValueError):
            # pylint: disable=pointless-statement
            v["abc"]

    def test_setitem(self) -> None:
        """Tests for vector assignment by index."""
        v = Vector3(*range(3))
        v[0] = 1
        v[1] = 2
        v[2] = 3
        self.assertTrue(v[0] == 1)
        self.assertTrue(v[1] == 2)
        self.assertTrue(v[2] == 3)
        with self.assertRaises(IndexError):
            v[4] = 1
        with self.assertRaises(ValueError):
            v[1:3] = Matrix([4, 5]).transpose()
        v[1:3] = 4
        self.assertTrue(v[1:3] == Matrix([[4], [4]]))
        with self.assertRaises(ValueError):
            # pylint: disable=pointless-statement
            v["abc"] = 0.0

    def test_equals(self) -> None:
        """Tests for the equals dunder method."""
        v = Vector3.ones()
        self.assertTrue(v == 1.0)
        self.assertTrue(v == Vector3.ones())
        self.assertTrue(v == Matrix.ones(3, 1))
        self.assertFalse(v == Matrix.ones(3, 3))
        self.assertFalse(v == "abc")

    def test_less_than(self) -> None:
        """Tests for the less_than dunder method."""
        v = Vector3.ones()
        self.assertFalse(v < 1.0)
        self.assertTrue(v <= 1.0)
        self.assertFalse(v < Vector3.ones())
        self.assertTrue(v <= Vector3.ones())
        self.assertFalse(v < Matrix.ones(3, 1))
        self.assertTrue(v <= Matrix.ones(3, 1))
        self.assertFalse(v < Matrix.ones(3, 3))
        self.assertFalse(v <= Matrix.ones(3, 3))
        with self.assertRaises(TypeError):
            # pylint: disable=pointless-statement
            v < "abc"
        with self.assertRaises(TypeError):
            # pylint: disable=pointless-statement
            v <= "abc"

    def test_add(self):
        """Tests for vector element-wise addition."""
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
            # pylint: disable=pointless-statement
            "foo" + A

    def test_subtract(self):
        """Tests for vector element-wise subtraction."""
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
        self.assertTrue(Matrix.ones(3, 1) - Vector3.unit_x() == Vector3(0, 1, 1))
        self.assertTrue(Vector3.unit_x() - Vector3.ones() == Vector3(0, -1, -1))
        self.assertTrue(1.0 - Vector3.ones() == Vector3.zeros())
        with self.assertRaises(ValueError):
            # pylint: disable=expression-not-assigned
            A - Matrix.ones(3, 3)
        with self.assertRaises(ValueError):
            # pylint: disable=expression-not-assigned
            Matrix.ones(3, 3) - A
        with self.assertRaises(TypeError):
            # pylint: disable=pointless-statement
            "foo" - A
        with self.assertRaises(TypeError):
            # pylint: disable=pointless-statement
            A - "foo"

    def test_multiplication(self):
        """Tests for vector multiplication."""
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
            # pylint: disable=pointless-statement
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
        """Tests for the vector norm method."""
        A = Vector3(0.5377, 1.8339, -2.2588)
        self.assertTrue(
            A.norm() - 3.0 <= 1.0e-16, "The vector norm was not computed successfully."
        )
        self.assertTrue(
            A.squared_norm() - 9.0 <= 1.0e-16,
            "The vector norm was not computed successfully.",
        )

    def test_cross(self):
        """Tests for the vector cross method."""
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
        """Tests for the vector dot method."""
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
        """Tests for the vector normalize method."""
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
        """Tests for negating a vector."""
        A = Vector3(0.5377, 1.8339, -2.2588)
        self.assertTrue(isinstance(-A, Vector3))
        self.assertTrue(abs(A + -A) <= 1.0e-4)

    def test_vertex_angle(self):
        """Tests for the vector vertex_angle method."""
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
        """Tests for the unit_x factory classmethod."""
        self.assertTrue(Vector3.unit_x() == Vector3(1, 0, 0))

    def test_unit_y(self):
        """Tests for the unit_y factory classmethod."""
        self.assertTrue(Vector3.unit_y() == Vector3(0, 1, 0))

    def test_unit_z(self):
        """Tests for the unit_z factory classmethod."""
        self.assertTrue(Vector3.unit_z() == Vector3(0, 0, 1))

    def test_skew(self):
        """Tests for the vector skew method."""
        A = Vector3(1, 2, 3)
        B = Matrix([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        self.assertTrue(abs(A.skew() - B) <= 1.0e-16)

    def test_str(self) -> None:
        """Tests for the str method."""
        self.assertTrue(str(Vector3(1, 2, 3.0)) == "[1.0, 2.0, 3.0]")

    def test_repr(self) -> None:
        """Tests for the Vector3 repr method."""
        self.assertTrue(repr(Vector3(1, 2, 3.0)) == "Vector3(1.0, 2.0, 3.0)")

    def test_hash(self) -> None:
        """Unit tests for the hash dunder method."""
        self.assertTrue(hash(Vector3(1, 2, 3)) == hash("[1.0, 2.0, 3.0]"))

    def test_gt(self) -> None:
        """Unit tests for greater-than conditions."""
        self.assertTrue(Vector3(1, 2, 3) > 0)
        self.assertTrue(Vector3(1, 2, 3) >= 1)

    def test_round(self) -> None:
        """Unit tests for rounding."""
        self.assertTrue(round(Vector3(1.123, 1.123, 1.123)) == 1.0)
        self.assertTrue(round(Vector3(1.123, 1.123, 1.123), 3) == 1.123)
        self.assertTrue(round(Vector3(1.123, 1.123, 1.123), 2) == 1.12)
        self.assertTrue(round(Vector3(1.123, 1.123, 1.123), 1) == 1.1)
        self.assertTrue(round(Vector3(1.123, 1.123, 1.123), 0) == 1.0)
