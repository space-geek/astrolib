""" TODO: Module docstring
"""
from typing import Callable
from typing import Tuple

from astrolib import Matrix
from astrolib import TimeSpan


def integrate(t_0: TimeSpan,
              x_0: Matrix,
              h: TimeSpan,
              dynamics_func: Callable[[TimeSpan, Matrix], Matrix]
              ) -> Tuple[TimeSpan, Matrix, TimeSpan]:
    """ Function utilizes a 4th-order Runge-Kutta integration scheme to integrate
        the input state vector from the initial to the final epoch.

    Arguments:
        t_0 (TimeSpan)                                      Epoch of the initial state.
        x_0 (Matrix)                                        Initial state vector, in column matrix form.
        h (TimeSpan)                                        Step size for integration step to take.
        dynamics_func ([Callable[TimeSpan, Matrix],Matrix]) State vector dynamics function.

    Returns:
        Tuple[TimeSpan, Matrix, TimeSpan]                   Tuple of the propagated state vector epoch, the
                                                            propagated state vector, and the step size taken.
    """

    # Compute the RK coefficients:
    k_1 = h.to_seconds() * dynamics_func(t_0, x_0)
    k_2 = h.to_seconds() * dynamics_func(t_0 + (1.0/2.0) * h, x_0 + (1.0/2.0) * k_1)
    k_3 = h.to_seconds() * dynamics_func(t_0 + (1.0/2.0) * h, x_0 + (1.0/2.0) * k_2)
    k_4 = h.to_seconds() * dynamics_func(t_0 + (    1.0) * h, x_0 + (    1.0) * k_3)

    # Build up the new state vector:
    t_n = t_0 + h
    x_n = x_0 + (1.0/6.0) * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)

    return t_n, x_n, h
