import unittest

from astrolib.base_objects import Matrix
from astrolib.interpolation.lagrange import interpolate as interpolate_lagrange


class Test_Lagrange(unittest.TestCase):
    def test_1(self):
        x_0 = 8.4
        xvals = Matrix([[8.1, 8.3, 8.6, 8.7]])
        yvals = Matrix([[16.94410, 17.56492, 18.50515, 18.82091]])
        self.assertTrue(
            interpolate_lagrange(xvals[0, 1:3], yvals[0, 1:3], x_0)
            == 17.878329999999998,
            "Lagrange interpolation failed.",
        )
        self.assertTrue(
            interpolate_lagrange(
                xvals[0, 1:3].transpose(), yvals[0, 1:3].transpose(), x_0
            )
            == 17.878329999999998,
            "Lagrange interpolation failed.",
        )
        self.assertTrue(
            interpolate_lagrange(xvals[0, 1:3].transpose(), yvals[0, 1:3], x_0)
            == 17.878329999999998,
            "Lagrange interpolation failed.",
        )
        self.assertTrue(
            interpolate_lagrange(xvals[0, 1:3], yvals[0, 1:3].transpose(), x_0)
            == 17.878329999999998,
            "Lagrange interpolation failed.",
        )
        self.assertTrue(
            interpolate_lagrange(xvals[0, 1:4], yvals[0, 1:4], x_0)
            == 17.877155000000002,
            "Lagrange interpolation failed.",
        )
        self.assertTrue(
            interpolate_lagrange(xvals, yvals, x_0) == 17.877142500000001,
            "Lagrange interpolation failed.",
        )
        with self.assertRaises(ValueError):
            interpolate_lagrange(Matrix.ones(3, 2), Matrix.ones(3, 2), 0.0)
        with self.assertRaises(ValueError):
            interpolate_lagrange(Matrix.ones(3, 1), Matrix.ones(4, 1), 0.0)

    def test_2(self):
        # Example 2 from Numerical Analysis, Burden & Faires 10th Edition, page 108
        x_0 = 3.0
        xvals = Matrix([[2, 2.75, 4]])
        yvals = Matrix([[1 / x for x in xvals.get_row(0)]])
        self.assertTrue(
            interpolate_lagrange(xvals, yvals, x_0) == 0.3295454545454546,
            "Lagrange interpolation failed.",
        )


if __name__ == "__main__":
    unittest.main()
