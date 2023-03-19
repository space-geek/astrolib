""" Unit test module for the astrolib.integration.euler module.
"""
import unittest

from astrolib.base_objects import Matrix
from astrolib.base_objects import TimeSpan
from astrolib.integration.euler import integrate


class Test_Euler(unittest.TestCase):
    def test_simple(self):
        t_0 = 0.0
        x_0 = Matrix.ones(2, 1)
        h = 1.0
        results = integrate(
            t_0,
            x_0,
            h,
            lambda t, x: Matrix.fill(*x.size, 1.0),
        )
        self.assertTrue(results.epoch == h, "Integration did not work.")
        self.assertTrue(
            results.state == Matrix.fill(*x_0.size, 2.0),
            "Integration did not work.",
        )
        self.assertTrue(results.total_step_seconds == h, "Integration did not work.")

    # TODO: Add more extensive Euler integration tests
