""" Unit test module for the astrolib.kinematics.dcm module.
"""

import math
import unittest

from astrolib.matrix import Matrix
from astrolib.matrix import Vector3
from astrolib.kinematics.dcm import DirectionCosineMatrix
from astrolib.kinematics.dcm import matrix_is_orthogonal


class TestDirectionCosineMatrix(unittest.TestCase):
    """Test case for the DirectionCosineMatrix class and its methods."""

    def test_matrix_is_orthogonal(self):
        """Tests for the matrix_is_orthogonal function."""
        m1 = Matrix.identity(4)
        m2 = Matrix.ones(2)
        m3 = Matrix.zeros(3, 2)
        m4 = DirectionCosineMatrix.r_x(math.radians(90)).as_matrix()
        m5 = -Matrix.identity(4)

        self.assertTrue(matrix_is_orthogonal(m1))
        self.assertFalse(matrix_is_orthogonal(m2))
        self.assertFalse(matrix_is_orthogonal(m3))
        self.assertTrue(matrix_is_orthogonal(m4))
        self.assertTrue(matrix_is_orthogonal(m5))
        with self.assertRaises(ValueError):
            # pylint: disable=expression-not-assigned
            matrix_is_orthogonal("foo")

    def test_constructor(self):
        """Success tests for the class constructor."""
        expected = Matrix.identity(3)
        actual = DirectionCosineMatrix(expected)
        self.assertTrue(actual == expected)
        actual2 = DirectionCosineMatrix(expected)
        self.assertTrue(actual == actual2)
        with self.assertRaises(ValueError):  # invalid arg
            # pylint: disable=pointless-statement
            DirectionCosineMatrix("foo")
        with self.assertRaises(ValueError):  # wrong size matrix
            # pylint: disable=pointless-statement
            DirectionCosineMatrix(Matrix.ones(4, 3))
        with self.assertRaises(ValueError):  # non-orthogonal matrix
            # pylint: disable=pointless-statement
            DirectionCosineMatrix(Matrix.ones(4))

    def test_getitem(self):
        """Tests for quaternion indexing."""
        machine_precision: float = 1.0e-16
        dcm = DirectionCosineMatrix.r_x(math.radians(90))
        self.assertTrue(dcm[0, 0] - 1.0 <= machine_precision)
        self.assertTrue(dcm[0, 1] - 0.0 <= machine_precision)
        self.assertTrue(dcm[0, 2] - 0.0 <= machine_precision)
        self.assertTrue(dcm[1, 0] - 0.0 <= machine_precision)
        self.assertTrue(dcm[1, 1] - 0.0 <= machine_precision)
        self.assertTrue(dcm[1, 2] - 1.0 <= machine_precision)
        self.assertTrue(dcm[2, 0] - 0.0 <= machine_precision)
        self.assertTrue(dcm[2, 1] + 1.0 <= machine_precision)
        self.assertTrue(dcm[2, 2] - 0.0 <= machine_precision)
        self.assertIsInstance(dcm[:2, :2], Matrix)
        self.assertTrue(dcm[:2, :2].size == (2, 2))
        self.assertTrue(
            dcm[:2, :2] - Matrix([[1.0, 0.0], [0.0, 1.0]]) <= machine_precision
        )
        with self.assertRaises(IndexError):
            # pylint: disable=pointless-statement
            dcm[4, 4]
        with self.assertRaises(ValueError):
            # pylint: disable=pointless-statement
            dcm["foo"]

    def test_identity(self):
        """Tests for the identity classmethod."""
        dcm = DirectionCosineMatrix.identity()
        self.assertTrue(dcm[0, 0] == 1.0)
        self.assertTrue(dcm[0, 1] == 0.0)
        self.assertTrue(dcm[0, 2] == 0.0)
        self.assertTrue(dcm[1, 0] == 0.0)
        self.assertTrue(dcm[1, 1] == 1.0)
        self.assertTrue(dcm[1, 2] == 0.0)
        self.assertTrue(dcm[2, 0] == 0.0)
        self.assertTrue(dcm[2, 1] == 0.0)
        self.assertTrue(dcm[2, 2] == 1.0)

    def test_r_x(self):
        """Tests for the r_x classmethod."""
        machine_precision: float = 1.0e-16
        dcm = DirectionCosineMatrix.r_x(math.radians(90))
        self.assertTrue(dcm[0, 0] - 1.0 <= machine_precision)
        self.assertTrue(dcm[0, 1] - 0.0 <= machine_precision)
        self.assertTrue(dcm[0, 2] - 0.0 <= machine_precision)
        self.assertTrue(dcm[1, 0] - 0.0 <= machine_precision)
        self.assertTrue(dcm[1, 1] - 0.0 <= machine_precision)
        self.assertTrue(dcm[1, 2] - 1.0 <= machine_precision)
        self.assertTrue(dcm[2, 0] - 0.0 <= machine_precision)
        self.assertTrue(dcm[2, 1] + 1.0 <= machine_precision)
        self.assertTrue(dcm[2, 2] - 0.0 <= machine_precision)

    def test_r_y(self):
        """Tests for the r_y classmethod."""
        machine_precision: float = 1.0e-16
        dcm = DirectionCosineMatrix.r_y(math.radians(90))
        self.assertTrue(dcm[0, 0] - 0.0 <= machine_precision)
        self.assertTrue(dcm[0, 1] - 0.0 <= machine_precision)
        self.assertTrue(dcm[0, 2] + 1.0 <= machine_precision)
        self.assertTrue(dcm[1, 0] - 0.0 <= machine_precision)
        self.assertTrue(dcm[1, 1] - 1.0 <= machine_precision)
        self.assertTrue(dcm[1, 2] - 0.0 <= machine_precision)
        self.assertTrue(dcm[2, 0] - 1.0 <= machine_precision)
        self.assertTrue(dcm[2, 1] - 0.0 <= machine_precision)
        self.assertTrue(dcm[2, 2] - 0.0 <= machine_precision)

    def test_r_z(self):
        """Tests for the r_z classmethod."""
        machine_precision: float = 1.0e-16
        dcm = DirectionCosineMatrix.r_z(math.radians(90))
        self.assertTrue(dcm[0, 0] - 0.0 <= machine_precision)
        self.assertTrue(dcm[0, 1] - 1.0 <= machine_precision)
        self.assertTrue(dcm[0, 2] - 0.0 <= machine_precision)
        self.assertTrue(dcm[1, 0] + 1.0 <= machine_precision)
        self.assertTrue(dcm[1, 1] - 0.0 <= machine_precision)
        self.assertTrue(dcm[1, 2] - 0.0 <= machine_precision)
        self.assertTrue(dcm[2, 0] - 0.0 <= machine_precision)
        self.assertTrue(dcm[2, 1] - 0.0 <= machine_precision)
        self.assertTrue(dcm[2, 2] - 1.0 <= machine_precision)

    def test_rows(self):
        """Tests for the rows property."""
        dcm = DirectionCosineMatrix.identity()
        self.assertTrue(len(dcm.rows) == 3)
        self.assertTrue(all((isinstance(x, Vector3) for x in dcm.rows)))
        self.assertTrue(dcm.rows[0] == Vector3.unit_x())
        self.assertTrue(dcm.rows[1] == Vector3.unit_y())
        self.assertTrue(dcm.rows[2] == Vector3.unit_z())

    def test_columns(self):
        """Tests for the columns property."""
        dcm = DirectionCosineMatrix.identity()
        self.assertTrue(len(dcm.columns) == 3)
        self.assertTrue(all((isinstance(x, Vector3) for x in dcm.rows)))
        self.assertTrue(dcm.columns[0] == Vector3.unit_x())
        self.assertTrue(dcm.columns[1] == Vector3.unit_y())
        self.assertTrue(dcm.columns[2] == Vector3.unit_z())

    def test_as_matrix(self):
        """Tests for the as_matrix method."""
        dcm = DirectionCosineMatrix.identity()
        self.assertTrue(isinstance(dcm.as_matrix(), Matrix))
        self.assertTrue(dcm.as_matrix().size == (3, 3))
        self.assertTrue(dcm.as_matrix() == Matrix.identity(3))

    def test_transpose(self):
        """Tests for the transpose method."""
        dcm = DirectionCosineMatrix.r_y(math.radians(90))
        self.assertTrue(isinstance(dcm.transpose(), Matrix))
        self.assertTrue(dcm.transpose().size == (3, 3))
        self.assertTrue(
            dcm.transpose() == DirectionCosineMatrix.r_y(math.radians(-90)).as_matrix()
        )

    def test_repr(self):
        """Tests for DCM repr."""
        dcm = DirectionCosineMatrix.identity()
        self.assertTrue(
            repr(dcm)
            == "DirectionCosineMatrix(Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))"
        )

    def test_equals(self):
        """Tests for DCM equality comparison."""
        dcm = DirectionCosineMatrix.identity()
        self.assertTrue(dcm == DirectionCosineMatrix.identity())
        self.assertTrue(dcm == Matrix.identity(3))
        self.assertFalse(dcm == "foo")

    def test_subtraction(self):
        """Tests for DCM component-wise subtraction"""
        dcm = DirectionCosineMatrix.identity()
        self.assertIsInstance(dcm - dcm, Matrix)
        self.assertIsInstance(dcm - Matrix.identity(3), Matrix)
        self.assertTrue((dcm - dcm).size == (3, 3))
        self.assertTrue(dcm - dcm == Matrix.zeros(3))
        self.assertTrue(dcm - Matrix.identity(3) == Matrix.zeros(3))
        self.assertTrue(Matrix.identity(3) - dcm == Matrix.zeros(3))
        with self.assertRaises(ValueError):
            # pylint: disable=expression-not-assigned
            dcm - Matrix.ones(4)
        with self.assertRaises(TypeError):
            # pylint: disable=pointless-statement
            dcm - "abc"

    def test_multiplication(self):
        """Tests for DCM multiplication."""
        tol: float = 1.2e-16
        dcm1 = DirectionCosineMatrix.identity()
        dcm2 = DirectionCosineMatrix.r_x(math.radians(30))
        self.assertIsInstance(dcm1 * dcm2, DirectionCosineMatrix)
        self.assertIsInstance(dcm1 * Matrix.identity(3), DirectionCosineMatrix)
        self.assertIsInstance(dcm1 * Matrix.ones(3), Matrix)
        self.assertIsInstance(Matrix.identity(3) * dcm1, DirectionCosineMatrix)
        self.assertIsInstance(Matrix.ones(4, 3) * dcm1, Matrix)
        self.assertTrue(dcm1 * dcm2 == dcm2)
        self.assertTrue(dcm2 * dcm1 == dcm2)
        self.assertTrue(dcm2 * Matrix.identity(3) == dcm2)
        self.assertTrue(Matrix.identity(3) * dcm2 == dcm2)
        print(abs(dcm2 * dcm2 - DirectionCosineMatrix.r_x(math.radians(60))))
        self.assertTrue(
            abs(dcm2 * dcm2 - DirectionCosineMatrix.r_x(math.radians(60))) <= tol
        )
        self.assertTrue(dcm2 * dcm2.transpose() == DirectionCosineMatrix.identity())
        with self.assertRaises(ValueError):
            # pylint: disable=expression-not-assigned
            dcm1 * Matrix.ones(4)
        with self.assertRaises(TypeError):
            # pylint: disable=pointless-statement
            dcm1 * "abc"
        with self.assertRaises(ValueError):
            # pylint: disable=expression-not-assigned
            Matrix.ones(4) * dcm1
        with self.assertRaises(TypeError):
            # pylint: disable=pointless-statement
            "abc" * dcm1

    def test_trace(self) -> None:
        """Tests for the trace property."""
        A = DirectionCosineMatrix.identity()
        B = DirectionCosineMatrix.r_x(math.radians(90))
        self.assertTrue(A.trace == 3)
        self.assertTrue(B.trace == 1)
