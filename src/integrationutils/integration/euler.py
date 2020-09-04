from typing import Callable
from typing import Tuple

from integrationutils.util.base_objects import Matrix
from integrationutils.util.time_objects import TimeSpan


def integrate(t_0: TimeSpan, X_0: Matrix, h: TimeSpan, dynamics_func: Callable[[TimeSpan, Matrix],Matrix]) -> Tuple[TimeSpan, Matrix, TimeSpan]:
    """ Function utilizes a first-order/Euler integration scheme to integrate
        the input state from the initial to the final epoch.

    Arguments:
        t_0 (TimeSpan)                                      Epoch of the initial state.
        X_0 (Matrix)                                        Initial state vector, in column matrix form.
        h (TimeSpan)                                        Step size for integration step to take.
        dynamics_func ([Callable[TimeSpan, Matrix],Matrix]) State vector dynamics function.

    Returns:
        Tuple[TimeSpan, Matrix, TimeSpan]                   Tuple of the propagated state vector epoch, the
                                                            propagated state vector, and the step size taken.
    """
    return t_0 + h, X_0 + h * dynamics_func(t_0, X_0), h