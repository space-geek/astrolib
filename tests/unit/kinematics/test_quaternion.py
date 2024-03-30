""" Unit test module for the astrolib.kinematics.quaternion module.
"""

import unittest

from astrolib.matrix import Matrix
from astrolib.matrix import Vector3
from astrolib.kinematics.quaternion import Quaternion


class TestQuaternion(unittest.TestCase):
    """Test case for the Quaternion class and its methods."""

    def test_constructor(self):
        """Test for the class constructor."""
        expected = Matrix(list(range(0, 4))).transpose()
        actual = Quaternion(*range(4))
        self.assertTrue(actual[:] == expected)

    def test_identity(self):
        """Test for the identity class method."""
        expected = Matrix([0, 0, 0, 1]).transpose()
        self.assertTrue(Quaternion.identity()[:] == expected)

    def test_properties(self):
        """Test for the various read/write properties
        of the class.
        """
        x, y, z, w = range(4)
        quat = Quaternion(x, y, z, w)
        self.assertTrue(quat.x == x)
        self.assertTrue(quat.y == y)
        self.assertTrue(quat.z == z)
        self.assertTrue(quat.w == w)
        self.assertTrue(quat.vector == Vector3(x, y, z))
        quat.x = 4
        self.assertTrue(quat.x == 4)
        quat.y = 5
        self.assertTrue(quat.y == 5)
        quat.z = 6
        self.assertTrue(quat.z == 6)
        quat.w = 7
        self.assertTrue(quat.w == 7)
        quat.vector = Vector3(*(-1 * x for x in range(3)))
        print(quat)
        self.assertTrue(quat.x == x)
        self.assertTrue(quat.y == -y)
        self.assertTrue(quat.z == -z)
        self.assertTrue(quat.w == 7)
        self.assertTrue(quat.vector == Vector3(x, -y, -z))

    def test_getitem(self):
        """Tests for quaternion indexing."""
        quat = Quaternion(*range(4))
        self.assertTrue(quat[0] == quat.x)
        self.assertTrue(quat[1] == quat.y)
        self.assertTrue(quat[2] == quat.z)
        self.assertTrue(quat[3] == quat.w)
        self.assertIsInstance(quat[:], Matrix)
        self.assertTrue(quat[:].size == (4, 1))
        self.assertTrue(quat[:] == Matrix([*range(4)]).transpose())
        self.assertIsInstance(quat[:2], Matrix)
        self.assertTrue(quat[:2].size == (2, 1))
        self.assertTrue(quat[:2] == Matrix([*range(2)]).transpose())
        with self.assertRaises(IndexError):
            # pylint: disable=pointless-statement
            quat[4]
        with self.assertRaises(ValueError):
            # pylint: disable=pointless-statement
            quat["foo"]

    def test_norm(self):
        """Tests for the norm method."""
        self.assertTrue(Quaternion(0, 0, 0, 0).norm() == 0.0)
        self.assertTrue(Quaternion.identity().norm() == 1.0)
        self.assertAlmostEqual(Quaternion(1, 2, 3, 4).norm(), 5.477225575)

    def test_normalize(self):
        """Tests for the normalize method."""
        quat = Quaternion(0, 0, 0, 10)
        quat.normalize()
        self.assertAlmostEqual(quat, Quaternion.identity())

        quat = Quaternion(2, 2, 2, 2)
        quat.normalize()
        self.assertAlmostEqual(quat, Quaternion(0.5, 0.5, 0.5, 0.5))

        quat = Quaternion(2, 0, -2, 0)
        quat.normalize()
        self.assertAlmostEqual(quat, Quaternion(0.7071068, 0, -0.7071068, 0))

    def test_normalized(self):
        """Tests for the normalized method."""
        quat = Quaternion(0, 0, 0, 10)
        self.assertAlmostEqual(quat.normalized(), Quaternion.identity())

        quat = Quaternion(2, 2, 2, 2)
        self.assertAlmostEqual(quat.normalized(), Quaternion(0.5, 0.5, 0.5, 0.5))

        quat = Quaternion(2, 0, -2, 0)
        self.assertAlmostEqual(
            quat.normalized(), Quaternion(0.7071068, 0, -0.7071068, 0)
        )

    def test_neg(self):
        """Tests for quaternion negation."""
        quat = Quaternion(*range(4))
        self.assertTrue(-quat == Quaternion(0, -1, -2, -3))

    def test_subtraction(self):
        """Tests for quaternion component-wise subtraction"""
        q1 = Quaternion(*range(4))
        q2 = Quaternion(*reversed(range(4)))
        self.assertIsInstance(q1 - q2, Matrix)
        self.assertTrue((q1 - q2).size == (4, 1))
        self.assertTrue(q1 - q2 == Matrix([-3, -1, 1, 3]).transpose())

    def test_multiplication(self):
        """Tests for quaternion multiplication."""
        q1 = Quaternion(*range(4))
        self.assertIsInstance(10 * q1, Quaternion)
        self.assertTrue(3 * q1 == Quaternion(0, 3, 6, 9))
        self.assertTrue(3.0 * q1 == Quaternion(0, 3, 6, 9))
        self.assertTrue(-3.0 * q1 == Quaternion(0, -3, -6, -9))
        self.assertTrue(q1 * 3 == Quaternion(0, 3, 6, 9))
        self.assertTrue(q1 * 3.0 == Quaternion(0, 3, 6, 9))
        self.assertTrue(q1 * (-3.0) == Quaternion(0, -3, -6, -9))

        self.assertTrue(q1.transpose() * q1 == 14.0)
        self.assertIsInstance(Matrix.identity(4) * q1, Matrix)
        self.assertIsInstance(q1 * Matrix.ones(1, 4), Matrix)
        with self.assertRaises(ValueError):
            # Invalid dimensionality: (4,1) * (4, 4)
            # pylint: disable=expression-not-assigned
            q1 * Matrix.identity(4)
        # TODO more matrix mult

        # TODO quaternion mult

    def test_division(self):
        """Tests for quaternion division."""
        q1 = Quaternion(*range(4)).normalized()
        q2 = Quaternion(*reversed(range(4))).normalized()
        print(q1 / q2)
        # self.assertFalse(True)
        # TODO quaternion division test case

    def test_as_matrix(self):
        """Tests for the as_matrix method."""
        quat = Quaternion(*range(4))
        self.assertTrue(isinstance(quat.as_matrix(), Matrix))
        self.assertTrue(quat.as_matrix().size == (4, 1))
        self.assertTrue(quat.as_matrix() == Matrix([0, 1, 2, 3]).transpose())

    def test_transpose(self):
        """Tests for the transpose method."""
        quat = Quaternion(*range(4))
        self.assertTrue(isinstance(quat.transpose(), Matrix))
        self.assertTrue(quat.transpose().size == (1, 4))
        self.assertTrue(quat.transpose() == Matrix([0, 1, 2, 3]))
