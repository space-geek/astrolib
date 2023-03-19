""" TODO: Module docstring
"""
from typing import Callable

from astrolib.base_objects import Matrix
from astrolib.integration import IntegratorResults


def integrate(
    t_0: float,
    x_0: float | Matrix,
    step_size: float,
    dynamics_func: Callable[[float, float | Matrix], float | Matrix],
) -> IntegratorResults:
    """Function utilizes a first-order/Euler integration scheme to integrate
    the input state from the initial to the final epoch.

    Arguments:
        t_0 (float): Epoch of the initial state.
        x_0 (float | Matrix): Initial state vector, in column matrix form.
        step_size (float): Step size for integration step to take.
        dynamics_func (Callable[[float, float | Matrix], float | Matrix]): State vector dynamics function.

    Returns:
        IntegratorResults: Tuple containing the integration results.
    """
    return IntegratorResults(
        epoch=t_0 + step_size,
        state=x_0 + step_size * dynamics_func(t_0, x_0),
        total_step_seconds=step_size,
        intermediate_step_seconds=[
            step_size,
        ],
    )
