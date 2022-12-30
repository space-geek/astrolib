""" Unit test module for the astrolib.integration.euler module.
"""
import unittest

from astrolib.base_objects import Matrix
from astrolib.base_objects import TimeSpan
from astrolib.integration.euler import integrate


class Test_Euler(unittest.TestCase):
    def test_simple(self):
        t_0 = TimeSpan.zero()
        X_0 = Matrix.ones(2, 1)
        h = TimeSpan.from_seconds(1)
        t_f, X_f, h_actual = integrate(
            t_0,
            X_0,
            h,
            lambda t, X: Matrix.fill(*X.size, 1.0),
        )
        self.assertTrue(t_f == h, "Integration did not work.")
        self.assertTrue(X_f == Matrix.fill(*X_0.size, 2.0), "Integration did not work.")
        self.assertTrue(h_actual == h, "Integration did not work.")

    # TODO: Add more extensive Euler integration tests
