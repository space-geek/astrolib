""" Unit test module for the astrolib.integration.rk4 module.
"""

import math
from typing import Callable
from typing import List
from typing import Tuple
import unittest

from astrolib.matrix import Matrix
from astrolib.integration.rk4 import integrate


class Test_RK4(unittest.TestCase):

    tolerance: float = 1.0e-7

    def test_simple(self):
        t_0 = 0.0
        X_0 = Matrix.ones(2, 1)
        h = 1.0
        results = integrate(
            t_0,
            X_0,
            h,
            lambda t, X: Matrix.fill(*X.size, 1.0),
        )
        self.assertTrue(results.epoch == h, "Integration did not work.")
        self.assertTrue(
            results.state == Matrix.fill(*X_0.size, 2.0), "Integration did not work."
        )
        self.assertTrue(results.total_step_seconds == h, "Integration did not work.")

    def test_burden_faires_5_4_13a(self):
        self._integrate_and_validate(
            truth_values=[
                (0.5, 0.2969975),
                (1.0, 3.3143118),
            ],
            t_0=0.0,
            x_0=0.0,
            step_size=0.5,
            x_dyn=lambda t, y: t * math.exp(3 * t) - 2 * y,
            tol=Test_RK4.tolerance,
        )

    def test_burden_faires_5_4_13b(self):
        self._integrate_and_validate(
            truth_values=[
                (2.5, 1.8333234),
                (3.0, 2.4999712),
            ],
            t_0=2.0,
            x_0=1.0,
            step_size=0.5,
            x_dyn=lambda t, y: 1 + math.pow(t - y, 2),
            tol=Test_RK4.tolerance,
        )

    def test_burden_faires_5_4_13c(self):
        self._integrate_and_validate(
            truth_values=[
                (1.25, 2.7789095),
                (1.50, 3.6081647),
                (1.75, 4.4792846),
                (2.00, 5.3862426),
            ],
            t_0=1.0,
            x_0=2.0,
            step_size=0.25,
            x_dyn=lambda t, y: 1 + (y / t),
            tol=Test_RK4.tolerance,
        )

    def test_burden_faires_5_4_13d(self):
        self._integrate_and_validate(
            truth_values=[
                (0.25, 1.3291650),
                (0.50, 1.7305336),
                (0.75, 2.0415436),
                (1.00, 2.1180636),
            ],
            t_0=0.0,
            x_0=1.0,
            step_size=0.25,
            x_dyn=lambda t, y: math.cos(2 * t) + math.sin(3 * t),
            tol=Test_RK4.tolerance,
        )

    def test_burden_faires_5_4_15a(self):
        self._integrate_and_validate(
            truth_values=[
                (1.2, 1.0149520),
                (1.5, 1.0672620),
                (1.7, 1.1106547),
                (2.0, 1.1812319),
            ],
            t_0=1.0,
            x_0=1.0,
            step_size=0.1,
            x_dyn=lambda t, y: y / t - math.pow(y / t, 2),
            tol=Test_RK4.tolerance,
        )

    def test_burden_faires_5_4_15b(self):
        self._integrate_and_validate(
            truth_values=[
                (1.4, 0.4896842),
                (2.0, 1.6612651),
                (2.4, 2.8764941),
                (3.0, 5.8738386),
            ],
            t_0=1.0,
            x_0=0.0,
            step_size=0.2,
            x_dyn=lambda t, y: 1 + y / t + math.pow(y / t, 2),
            tol=Test_RK4.tolerance,
        )

    def test_burden_faires_5_4_15c(self):
        self._integrate_and_validate(
            truth_values=[
                (0.4, -1.6200576),
                (1.0, -1.2384307),
                (1.4, -1.1146769),
                (2.0, -1.0359922),
            ],
            t_0=0.0,
            x_0=-2.0,
            step_size=0.2,
            x_dyn=lambda t, y: -(y + 1) * (y + 3),
            tol=Test_RK4.tolerance,
        )

    def test_burden_faires_5_4_15d(self):
        self._integrate_and_validate(
            truth_values=[
                (0.2, 0.1627655),
                (0.5, 0.2774767),
                (0.7, 0.5001579),
                (1.0, 1.0023207),
            ],
            t_0=0.0,
            x_0=1 / 3,
            step_size=0.1,
            x_dyn=lambda t, y: -5 * y + 5 * math.pow(t, 2) + 2 * t,
            tol=Test_RK4.tolerance,
        )

    def test_burden_faires_5_4_13c_matrix(self):
        self._integrate_and_validate(
            truth_values=[
                (1.25, Matrix([[2.7789095], [2.7789095]])),
                (1.50, Matrix([[3.6081647], [3.6081647]])),
                (1.75, Matrix([[4.4792846], [4.4792846]])),
                (2.00, Matrix([[5.3862426], [5.3862426]])),
            ],
            t_0=1.0,
            x_0=Matrix([[2.0], [2.0]]),
            step_size=0.25,
            x_dyn=lambda t, y: Matrix([[1 + (y[0, 0] / t)], [1 + (y[1, 0] / t)]]),
            tol=Test_RK4.tolerance,
        )

    # TODO Add more matrix test case(s)

    def _integrate_and_validate(
        self,
        truth_values: List[Tuple[float, float | Matrix]],
        t_0: float,
        x_0: float | Matrix,
        step_size: float,
        x_dyn: Callable[[float, float | Matrix], float | Matrix],
        tol: float,
    ):
        for step_num in range(len(truth_values)):
            results = integrate(t_0 + step_num * step_size, x_0, step_size, x_dyn)
            y_truth = next((x for t, x in truth_values if t == results.epoch), None)
            t_truth = t_0 + (step_num + 1) * step_size
            self.assertTrue(
                abs(results.epoch - t_truth) <= tol,
                f"{abs(results.epoch - t_truth)} is not less than {tol}",
            )
            self.assertTrue(results.total_step_seconds == step_size)
            if y_truth is not None:
                self.assertTrue(
                    abs(results.state - y_truth) <= tol,
                    f"{abs(results.state - y_truth)} is not less than {tol}",
                )
            x_0 = results.state
