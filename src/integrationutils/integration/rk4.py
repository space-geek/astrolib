from typing import Callable
from typing import Tuple

from integrationutils.base_objects import Matrix
from integrationutils.base_objects import TimeSpan


def integrate(t_0: TimeSpan, X_0: Matrix, h: TimeSpan, dynamics_func: Callable[[TimeSpan, Matrix],Matrix]) -> Tuple[TimeSpan, Matrix, TimeSpan]:
    """ Function utilizes a 4th-order Runge-Kutta integration scheme to integrate
        the input state vector from the initial to the final epoch.

    Arguments:
        t_0 (TimeSpan)                                      Epoch of the initial state.
        X_0 (Matrix)                                        Initial state vector, in column matrix form.
        h (TimeSpan)                                        Step size for integration step to take.
        dynamics_func ([Callable[TimeSpan, Matrix],Matrix]) State vector dynamics function.

    Returns:
        Tuple[TimeSpan, Matrix, TimeSpan]                   Tuple of the propagated state vector epoch, the
                                                            propagated state vector, and the step size taken.
    """

    # Compute the RK coefficients:
    k1 = h.to_seconds() * dynamics_func(t_0, X_0)
    k2 = h.to_seconds() * dynamics_func(t_0 + (1.0/2.0) * h, X_0 + (1.0/2.0) * k1)
    k3 = h.to_seconds() * dynamics_func(t_0 + (1.0/2.0) * h, X_0 + (1.0/2.0) * k2)
    k4 = h.to_seconds() * dynamics_func(t_0 + (    1.0) * h, X_0 + (    1.0) * k3)

    # Build up the new state vector:
    t_n = t_0 + h
    X_n = X_0 + (1.0/6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return t_n, X_n, h
