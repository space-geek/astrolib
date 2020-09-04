from typing import Callable
from typing import Tuple

from integrationutils.util.base_objects import Matrix
from integrationutils.util.constants import MINIMUM_STEP_SIZE_IN_SECONDS
from integrationutils.integration.errors import MinimumStepSizeExceededError
from integrationutils.util.time_objects import TimeSpan

_DEFAULT_RELATIVE_ERROR_TOLERANCE = 1.0e-12
_MINIMUM_STEP_SIZE = TimeSpan.from_seconds(MINIMUM_STEP_SIZE_IN_SECONDS)
_MAXIMUM_STEP_SIZE_SCALE_FACTOR = 4.0
_MINIMUM_STEP_SIZE_SCALE_FACTOR = 0.1


def integrate(t_0: TimeSpan, X_0: Matrix, h_max: TimeSpan, dynamics_func: Callable[[TimeSpan, Matrix],Matrix],
              rel_tol: float = _DEFAULT_RELATIVE_ERROR_TOLERANCE, h_min: TimeSpan = _MINIMUM_STEP_SIZE) -> Tuple[TimeSpan, Matrix, TimeSpan]:
    """ Function utilizes a Runge-Kutta-Fehlberg integration scheme, utilizing a Runge-Kutta method with local
        trucation error of order five to estimate the local error in a Runge-Kutta method of order four, to,
        integrate the input state vector from the initial to the final epoch. The algorithm is given by
        Algorithm 5.3 in Numerical Analysis, Burden & Faires, 10th Ed, p. 297.

    Args:
        t_0 (TimeSpan)                                      Epoch of the initial state.
        X_0 (Matrix)                                        Initial state vector, in column matrix form.
        h (TimeSpan)                                        Maximum step size for integration step to take.
        dynamics_func ([Callable[TimeSpan, Matrix],Matrix]) State vector dynamics function.

    KWArgs:
        rel_tol (float)                                     Relative error tolerance for variable step
                                                            calculation
        h_min (TimeSpan)                                    Minimum step size for integration step to take.

    Returns:
        Tuple[TimeSpan, Matrix, TimeSpan]                   Tuple of the propagated state vector epoch, the
                                                            propagated state vector, and the step size taken.
    """

    # Initialize the step to attempt using the maximum step size:
    h = h_max

    # Perform an integration step:
    while True:

        # Compute the RK coefficients:
        k1 = h.to_seconds() * dynamics_func(t_0,                                X_0)
        k2 = h.to_seconds() * dynamics_func(t_0 + (  1.0/4.0) * h.to_seconds(), X_0 + (1.0/4.0) * k1)
        k3 = h.to_seconds() * dynamics_func(t_0 + (  3.0/8.0) * h.to_seconds(), X_0 + (     3.0/32.0) * k1 + (     9.0/32.0) * k2)
        k4 = h.to_seconds() * dynamics_func(t_0 + (12.0/13.0) * h.to_seconds(), X_0 + (1932.0/2197.0) * k1 - (7200.0/2197.0) * k2 + (7296.0/2197.0) * k3)
        k5 = h.to_seconds() * dynamics_func(t_0 + (      1.0) * h.to_seconds(), X_0 + (  439.0/216.0) * k1 - (          8.0) * k2 + ( 3680.0/513.0) * k3 - ( 845.0/4104.0) * k4)
        k6 = h.to_seconds() * dynamics_func(t_0 + (  1.0/2.0) * h.to_seconds(), X_0 - (     8.0/27.0) * k1 + (          2.0) * k2 - (3544.0/2565.0) * k3 + (1859.0/4104.0) * k4 - (11.0/40.0) * k5)

        # Compute the error term:
        R = (1.0 / h.to_seconds()) * abs((1.0/360.0) * k1 - (128.0/4275.0) * k3 - (2197.0/75240.0) * k4 + (1.0/50.0) * k5 + (2.0/55.0) * k6)

        # Check against the relative error tolerance:
        if R <= rel_tol:

            # Compute the new state vector using the 4th-order Runge-Kutta
            # approximation:
            t_n = t_0 + h
            X_n = X_0 + (25.0/216.0) * k1 + ( 1408.0/2565.0) * k3 + (  2197.0/4104.0) * k4 - ( 1.0/5.0) * k5

            # Break out of the processing loop:
            break

        else:

            # Scale the step size, bounding by the extrema limits:
            delta = 0.84 * pow(rel_tol / R, 0.25)
            if delta < _MINIMUM_STEP_SIZE_SCALE_FACTOR:
                delta = _MINIMUM_STEP_SIZE_SCALE_FACTOR
            elif delta > _MAXIMUM_STEP_SIZE_SCALE_FACTOR:
                delta = _MAXIMUM_STEP_SIZE_SCALE_FACTOR
            h *= delta
            if h > h_max:
                h = h_max
            elif h < h_min:
                raise MinimumStepSizeExceededError(h.to_seconds(), h_min.to_seconds)

    return t_n, X_n, h