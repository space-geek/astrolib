""" TODO: Module docstring
"""

from typing import Callable

from astrolib.matrix import Matrix
from astrolib.integration import IntegratorResults


def integrate(
    t_0: float,
    x_0: float | Matrix,
    step_size: float,
    dynamics_func: Callable[[float, float | Matrix], float | Matrix],
) -> IntegratorResults:
    """Function utilizes a 4th-order Runge-Kutta integration scheme to integrate
        the input state vector from the initial to the final epoch.

    Arguments:
        t_0 (float): Epoch of the initial state.
        x_0 (float | Matrix): Initial state vector, in column matrix form.
        h (float): Step size for integration step to take.
        dynamics_func (Callable[float, float | Matrix], float | Matrix]): State vector dynamics function.

    Returns:
        IntegratorResults: Tuple containing the integration results.
    """

    # Cache the actual step size:
    h = step_size

    # Compute the RK coefficients:
    k_1 = h * dynamics_func(t_0, x_0)
    k_2 = h * dynamics_func(t_0 + (1.0 / 2.0) * h, x_0 + (1.0 / 2.0) * k_1)
    k_3 = h * dynamics_func(t_0 + (1.0 / 2.0) * h, x_0 + (1.0 / 2.0) * k_2)
    k_4 = h * dynamics_func(t_0 + (1.0) * h, x_0 + (1.0) * k_3)

    # Build up and return the new state vector:
    return IntegratorResults(
        epoch=t_0 + h,
        state=x_0 + (1.0 / 6.0) * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4),
        total_step_seconds=h,
        intermediate_step_seconds=[
            h,
        ],
    )
