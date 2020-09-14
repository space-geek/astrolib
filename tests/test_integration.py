import unittest

from integrationutils.base_objects import Matrix
from integrationutils.integration.euler import integrate as euler_integrate
from integrationutils.integration.rk4 import integrate as rk4_integrate
from integrationutils.integration.rk45 import integrate as rk45_integrate
from integrationutils.time_objects import TimeSpan


class Test_Euler(unittest.TestCase):

    def test_simple(self):
        t_0 = TimeSpan.zero()
        X_0 = Matrix.ones(2,1)
        h = TimeSpan.from_seconds(1)
        t_f, X_f, h_actual = euler_integrate(t_0, X_0, h, _simple_dynamics_func)
        self.assertTrue(t_f == h, "Integration did not work.")
        self.assertTrue(X_f == Matrix.fill(*X_0.size, 2.0), "Integration did not work.")
        self.assertTrue(h_actual == h, "Integration did not work.")

    # TODO: Add more extensive Euler integration tests

class Test_RK4(unittest.TestCase):

    def test_simple(self):
        t_0 = TimeSpan.zero()
        X_0 = Matrix.ones(2,1)
        h = TimeSpan.from_seconds(1)
        t_f, X_f, h_actual = rk4_integrate(t_0, X_0, h, _simple_dynamics_func)
        self.assertTrue(t_f == h, "Integration did not work.")
        self.assertTrue(X_f == Matrix.fill(*X_0.size, 2.0), "Integration did not work.")
        self.assertTrue(h_actual == h, "Integration did not work.")

    # TODO: Add more extensive RK4 integration tests

class Test_RK45(unittest.TestCase):

    def test_simple(self):
        t_0 = TimeSpan.zero()
        X_0 = Matrix.ones(2,1)
        h = TimeSpan.from_seconds(1)
        t_f, X_f, h_actual = rk45_integrate(t_0, X_0, h, _simple_dynamics_func)
        self.assertTrue(t_f == h, "Integration did not work.")
        self.assertTrue(X_f == Matrix.fill(*X_0.size, 2.0), "Integration did not work.")
        self.assertTrue(h_actual == h, "Integration did not work.")

    # TODO: Add more extensive RK45 integration tests

def _simple_dynamics_func(t: TimeSpan, X: Matrix) -> Matrix:
    return Matrix.fill(*X.size, 1.0)

if __name__ == '__main__':
    unittest.main()
