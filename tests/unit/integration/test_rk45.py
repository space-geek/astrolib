""" Unit test module for the astrolib.integration.rk4 module.
"""

import math
from typing import Callable
from typing import List
from typing import NamedTuple
import unittest

from astrolib.matrix import Matrix
from astrolib.integration.rk45 import integrate
from astrolib.integration.rk45 import _integrate_single_step


class TruthDataset(NamedTuple):
    i: int
    t_i: float
    w_i: float | Matrix
    h_i: float


class Test_RK45(unittest.TestCase):

    tolerance: float = 1.0e-6

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

    def test_burden_faires_ch5_ex1(self):
        self._integrate_and_validate(
            truth_values=[
                TruthDataset(1, 0.2500000, 0.9204886, 0.2500000),
                TruthDataset(2, 0.4865522, 1.3964910, 0.2365522),
                TruthDataset(3, 0.7293332, 1.9537488, 0.2427810),
                TruthDataset(4, 0.9793332, 2.5864260, 0.2500000),
                TruthDataset(5, 1.2293332, 3.2604605, 0.2500000),
                TruthDataset(6, 1.4793332, 3.9520955, 0.2500000),
                TruthDataset(7, 1.7293332, 4.6308268, 0.2500000),
                TruthDataset(8, 1.9793332, 5.2574861, 0.2500000),
                TruthDataset(9, 2.0000000, 5.3054896, 0.0206668),
            ],
            t_0=0.0,
            x_0=0.5,
            x_dyn=lambda t, y: y - math.pow(t, 2) + 1,
            h_max=0.25,
            h_min=0.01,
            rel_tol=1.0e-5,
            tol=Test_RK45.tolerance,
        )

    def test_burden_faires_5_5_1a(self):
        self._integrate_and_validate(
            truth_values=[
                TruthDataset(1, 0.2093900, 0.0298184, 0.2093900),
                TruthDataset(3, 0.5610469, 0.4016438, 0.1777496),
                TruthDataset(5, 0.8387744, 1.5894061, 0.1280905),
                TruthDataset(7, 1.0000000, 3.2190497, 0.0486737),
            ],
            t_0=0.0,
            x_0=0.0,
            x_dyn=lambda t, y: t * math.exp(3 * t) - 2 * y,
            h_max=0.25,
            h_min=0.045,
            rel_tol=1.0e-4,
            tol=Test_RK45.tolerance,
        )

    def test_burden_faires_5_5_1b(self):
        self._integrate_and_validate(
            truth_values=[
                TruthDataset(1, 2.2500000, 1.4499988, 0.2500000),
                TruthDataset(2, 2.5000000, 1.8333332, 0.2500000),
                TruthDataset(3, 2.7500000, 2.1785718, 0.2500000),
                TruthDataset(4, 3.0000000, 2.5000005, 0.2500000),
            ],
            t_0=2.0,
            x_0=1.0,
            x_dyn=lambda t, y: 1 + math.pow(t - y, 2),
            h_max=0.25,
            h_min=0.05,
            rel_tol=1.0e-4,
            tol=Test_RK45.tolerance,
        )

    def test_burden_faires_5_5_1c(self):
        self._integrate_and_validate(
            truth_values=[
                TruthDataset(1, 1.2500000, 2.7789299, 0.2500000),
                TruthDataset(2, 1.5000000, 3.6081985, 0.2500000),
                TruthDataset(3, 1.7500000, 4.4793288, 0.2500000),
                TruthDataset(4, 2.0000000, 5.3862958, 0.2500000),
            ],
            t_0=1.0,
            x_0=2.0,
            x_dyn=lambda t, y: 1 + (y / t),
            h_max=0.25,
            h_min=0.05,
            rel_tol=1.0e-4,
            tol=Test_RK45.tolerance,
        )

    def test_burden_faires_5_5_1d(self):
        self._integrate_and_validate(
            truth_values=[
                TruthDataset(1, 0.2500000, 1.3291478, 0.2500000),
                TruthDataset(2, 0.5000000, 1.7304857, 0.2500000),
                TruthDataset(3, 0.7500000, 2.0414669, 0.2500000),
                TruthDataset(4, 1.0000000, 2.1179750, 0.2500000),
            ],
            t_0=0.0,
            x_0=1.0,
            x_dyn=lambda t, y: math.cos(2 * t) + math.sin(3 * t),
            h_max=0.25,
            h_min=0.05,
            rel_tol=1.0e-4,
            tol=Test_RK45.tolerance,
        )

    def test_burden_faires_5_5_3a(self):
        self._integrate_and_validate(
            truth_values=[
                TruthDataset(1, 1.1101946, 1.0051237, 0.1101946),
                TruthDataset(5, 1.7470584, 1.1213948, 0.2180472),
                TruthDataset(7, 2.3994350, 1.2795396, 0.3707934),
                TruthDataset(11, 4.0000000, 1.6762393, 0.1014853),
            ],
            t_0=1.0,
            x_0=1.0,
            x_dyn=lambda t, y: (y / t) - math.pow(y / t, 2),
            h_max=0.5,
            h_min=0.05,
            rel_tol=1.0e-6,
            tol=Test_RK45.tolerance,
        )

    def test_burden_faires_5_5_3b(self):
        self._integrate_and_validate(
            truth_values=[
                TruthDataset(4, 1.5482238, 0.7234123, 0.1256486),
                TruthDataset(7, 1.8847226, 1.3851234, 0.1073571),
                TruthDataset(10, 2.1846024, 2.1673514, 0.0965027),
                TruthDataset(16, 2.6972462, 4.1297939, 0.0778628),
                TruthDataset(21, 3.0000000, 5.8741059, 0.0195070),
            ],
            t_0=1.0,
            x_0=0.0,
            x_dyn=lambda t, y: 1 + (y / t) + math.pow(y / t, 2),
            h_max=0.5,
            h_min=0.01,
            rel_tol=1.0e-6,
            tol=Test_RK45.tolerance,
        )

    def test_burden_faires_5_5_3c(self):
        self._integrate_and_validate(
            truth_values=[
                TruthDataset(1, 0.1633541, -1.8380836, 0.1633541),
                TruthDataset(5, 0.7585763, -1.3597623, 0.1266248),
                TruthDataset(9, 1.1930325, -1.1684827, 0.1048224),
                TruthDataset(13, 1.6229351, -1.0749509, 0.1107510),
                TruthDataset(17, 2.1074733, -1.0291158, 0.1288897),
                TruthDataset(23, 3.0000000, -1.0049450, 0.1264618),
            ],
            t_0=0.0,
            x_0=-2.0,
            x_dyn=lambda t, y: -(y + 1) * (y + 3),
            h_max=0.5,
            h_min=0.05,
            rel_tol=1.0e-6,
            tol=Test_RK45.tolerance,
        )

    def test_burden_faires_5_5_3d(self):
        self._integrate_and_validate(
            truth_values=[
                TruthDataset(1, 0.3986051, 0.3108201, 0.3986051),
                TruthDataset(3, 0.9703970, 0.2221189, 0.2866710),
                TruthDataset(5, 1.5672905, 0.1133085, 0.3042087),
                TruthDataset(8, 2.0000000, 0.0543454, 0.0902302),
            ],
            t_0=0.0,
            x_0=(1.0 / 3.0),
            x_dyn=lambda t, y: (t + 2 * math.pow(t, 3)) * math.pow(y, 3) - t * y,
            h_max=0.5,
            h_min=0.05,
            rel_tol=1.0e-6,
            tol=Test_RK45.tolerance,
        )

    def test_burden_faires_5_5_1d_multistep(self):
        truth_values = [
            TruthDataset(1, 0.2500000, 1.3291478, 0.2500000),
            TruthDataset(2, 0.5000000, 1.7304857, 0.2500000),
            TruthDataset(3, 0.7500000, 2.0414669, 0.2500000),
            TruthDataset(4, 1.0000000, 2.1179750, 0.2500000),
        ]
        t_0 = 0.0
        x_0 = 1.0
        x_dyn = lambda t, y: math.cos(2 * t) + math.sin(3 * t)
        h_max = 0.25
        h_min = 0.001
        rel_tol = 1.0e-8
        tol = 1e-5  # Increased because of adjusted rel_tol from truth dataset
        for step_num in range(len(truth_values)):
            results = integrate(
                t_0 + step_num * h_max,
                x_0,
                h_max,
                x_dyn,
                rel_tol=rel_tol,
                min_step_size=h_min,
            )
            y_truth = next(
                (x.w_i for x in truth_values if x.t_i == results.epoch),
                None,
            )
            t_truth = t_0 + (step_num + 1) * h_max
            self.assertTrue(
                abs(results.epoch - t_truth) <= tol,
                f"Epoch: {abs(results.epoch - t_truth)} is not less than {tol}",
            )
            self.assertTrue(results.total_step_seconds == h_max)
            if y_truth is not None:
                self.assertTrue(
                    abs(results.state - y_truth) <= tol,
                    f"State: {abs(results.state - y_truth)} is not less than {tol}",
                )
            x_0 = results.state

    def _integrate_and_validate(
        self,
        truth_values: List[TruthDataset],
        t_0: float,
        x_0: float | Matrix,
        x_dyn: Callable[[float, float | Matrix], float | Matrix],
        h_max: float,
        h_min: float,
        rel_tol: float,
        tol: float,
    ):
        cur_step: int = 0
        step_size = h_max
        for truth_data in truth_values:
            while cur_step < truth_data.i:
                results = _integrate_single_step(
                    t_0,
                    x_0,
                    min(
                        step_size,
                        truth_data.t_i - t_0,
                    ),
                    h_max,
                    h_min,
                    rel_tol,
                    x_dyn,
                )
                t_0 = results.epoch
                x_0 = results.state
                step_size = results.projected_step_seconds
                cur_step += 1
                print("------------------------------")
                print(truth_data)
                print(cur_step, results)
            self.assertTrue(
                abs(results.epoch - truth_data.t_i) <= tol,
                f"Epoch: {abs(results.epoch - truth_data.t_i)} is not less than {tol}",
            )
            self.assertTrue(
                abs(results.total_step_seconds - truth_data.h_i) <= tol,
                f"Step size: {abs(results.total_step_seconds - truth_data.h_i)} is not less than {tol}",
            )
            self.assertTrue(
                abs(results.state - truth_data.w_i) <= tol,
                f"State: {abs(results.state - truth_data.w_i)} is not less than {tol}",
            )
