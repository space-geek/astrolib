""" TODO: Module docstring
"""
from typing import Callable
from typing import List

from astrolib.base_objects import Matrix
from astrolib.constants import MACHINE_EPSILON
from astrolib.integration.errors import MinimumStepSizeExceededError
from astrolib.integration import IntegratorResults

# NOTE: Set to 1.0e-3 to be consistent with Matlab https://www.mathworks.com/help/matlab/ref/odeset.html
_DEFAULT_RELATIVE_ERROR_TOLERANCE: float = 1.0e-3
_MINIMUM_STEP_SIZE_IN_SECONDS: float = 1.0e-9
_MAXIMUM_STEP_SIZE_SCALE_FACTOR: float = 4.0
_MINIMUM_STEP_SIZE_SCALE_FACTOR: float = 0.1


def integrate(
    t_0: float,
    x_0: float | Matrix,
    max_step_size: float,
    dynamics_func: Callable[[float, float | Matrix], float | Matrix],
    rel_tol: float = _DEFAULT_RELATIVE_ERROR_TOLERANCE,
    min_step_size: float = _MINIMUM_STEP_SIZE_IN_SECONDS,
) -> IntegratorResults:
    """Function utilizes a Runge-Kutta-Fehlberg integration scheme, utilizing a Runge-Kutta method with local
        trucation error of order five to estimate the local error in a Runge-Kutta method of order four, to,
        integrate the input state vector from the initial to the final epoch. The algorithm is given by
        Algorithm 5.3 in Numerical Analysis, Burden & Faires, 10th Ed, p. 297.

    Args:
        t_0 (float): Epoch of the initial state.
        x_0 (float | Matrix): Initial state vector, in column matrix form.
        max_step_size (float): Maximum step size for integration step to take.
        dynamics_func (Callable[[float, float | Matrix], float | Matrix]): State vector dynamics function.

    Keyword Args:
        rel_tol (float) Relative error tolerance for variable step calculation
        min_step_size (float): Minimum step size for integration step to take.

    Returns:
        IntegratorResults: Tuple containing the integration results.

    Raises:
        MinimumStepSizeExceededError: Raised if the variable step calculation attempts
            apply a step size below the input minimum step
            size.
    """
    # Initialize the end epoch:
    t_f: float = t_0 + max_step_size

    # Initialize the looping variables:
    t_n: float = t_0
    x_n: float | Matrix = x_0
    intermediate_step_sizes: List[float] = []
    projected_step_size: float = max_step_size

    # Integrate to the end epoch:
    while t_n < t_f:

        # Initialize the step to attempt using the maximum step size:
        h: float = min(t_f - t_n, projected_step_size)

        # Integrate for a single step:
        results = _integrate_single_step(
            t_n,
            x_n,
            h,
            max_step_size,
            min_step_size,
            rel_tol,
            dynamics_func,
        )

        # Update looping variables and store outputs:
        t_n = results.epoch
        x_n = results.state
        intermediate_step_sizes.extend(results.intermediate_step_seconds)
        projected_step_size = results.projected_step_seconds

    return IntegratorResults(
        epoch=t_n,
        state=x_n,
        total_step_seconds=t_f - t_0,
        intermediate_step_seconds=intermediate_step_sizes,
        projected_step_seconds=projected_step_size,
    )


def _integrate_single_step(
    t_0: float,
    x_0: float | Matrix,
    step_size: float,
    max_step_size: float,
    min_step_size: float,
    rel_tol: float,
    dynamics_func: Callable[[float, float | Matrix], float | Matrix],
) -> IntegratorResults:
    """Function utilizes a Runge-Kutta-Fehlberg integration scheme, utilizing a Runge-Kutta method with local
        trucation error of order five to estimate the local error in a Runge-Kutta method of order four, to
        integrate the input state vector for one integration step meeting the error tolerance. The algorithm
        is given by Algorithm 5.3 in Numerical Analysis, Burden & Faires, 10th Ed, p. 297.

    Args:
        t_0 (float): Epoch of the initial state.
        x_0 (float | Matrix): Initial state vector, in column matrix form.
        step_size (float): The initial integration step size to take, in seconds.
        max_step_size (float): Maximum step size for integration step to take, in seconds.
        min_step_size (float): Minimum step size for integration step to take, in seconds.
        rel_tol (float) Relative error tolerance for variable step calculation.
        dynamics_func (Callable[[float, float | Matrix], float | Matrix]): State vector dynamics function.

    Returns:
        IntegratorResults: Tuple containing the integration results.

    Raises:
        MinimumStepSizeExceededError: Raised if the variable step calculation attempts
            apply a step size below the input minimum step
            size.
    """
    # Check the provided step size against the maximum/minimum values:
    if step_size < min_step_size:
        raise MinimumStepSizeExceededError(step_size, min_step_size)
    if step_size > max_step_size:
        step_size = max_step_size

    # Perform an integration step:
    h: float = step_size
    while True:

        # Compute the RK coefficients:
        k_1: float | Matrix = h * dynamics_func(
            t_0,
            x_0,
        )
        k_2: float | Matrix = h * dynamics_func(
            t_0 + (1.0 / 4.0) * h,
            x_0 + (1.0 / 4.0) * k_1,
        )
        k_3: float | Matrix = h * dynamics_func(
            t_0 + (3.0 / 8.0) * h,
            x_0 + (3.0 / 32.0) * k_1 + (9.0 / 32.0) * k_2,
        )
        k_4: float | Matrix = h * dynamics_func(
            t_0 + (12.0 / 13.0) * h,
            x_0
            + (1932.0 / 2197.0) * k_1
            - (7200.0 / 2197.0) * k_2
            + (7296.0 / 2197.0) * k_3,
        )
        k_5: float | Matrix = h * dynamics_func(
            t_0 + (1.0) * h,
            x_0
            + (439.0 / 216.0) * k_1
            - (8.0) * k_2
            + (3680.0 / 513.0) * k_3
            - (845.0 / 4104.0) * k_4,
        )
        k_6: float | Matrix = h * dynamics_func(
            t_0 + (1.0 / 2.0) * h,
            x_0
            - (8.0 / 27.0) * k_1
            + (2.0) * k_2
            - (3544.0 / 2565.0) * k_3
            + (1859.0 / 4104.0) * k_4
            - (11.0 / 40.0) * k_5,
        )

        # Compute the error term:
        inner_term: float | Matrix = (
            (1.0 / 360.0) * k_1
            - (128.0 / 4275.0) * k_3
            - (2197.0 / 75240.0) * k_4
            + (1.0 / 50.0) * k_5
            + (2.0 / 55.0) * k_6
        )
        if isinstance(inner_term, Matrix):  # TODO evaluate this statement
            relative_error: float = (1.0 / h) * max(
                abs(inner_term)
            )  # inner_term.norm()
        else:
            relative_error: float = (1.0 / h) * abs(inner_term)

        # Compute the step size scaling term:
        q_scale: float = 0.84 * pow(
            rel_tol / relative_error if abs(relative_error) > MACHINE_EPSILON else 1.0,
            0.25,
        )
        if q_scale <= _MINIMUM_STEP_SIZE_SCALE_FACTOR:
            q_scale = _MINIMUM_STEP_SIZE_SCALE_FACTOR
        elif q_scale > _MAXIMUM_STEP_SIZE_SCALE_FACTOR:
            q_scale = _MAXIMUM_STEP_SIZE_SCALE_FACTOR

        # Check against the relative error tolerance:
        if relative_error <= rel_tol:

            # Compute the new state vector using the 4th-order Runge-Kutta
            # approximation:
            t_n: float = t_0 + h
            x_n: float | Matrix = (
                x_0
                + (25.0 / 216.0) * k_1
                + (1408.0 / 2565.0) * k_3
                + (2197.0 / 4104.0) * k_4
                - (1.0 / 5.0) * k_5
            )

            # Compute the projected step size for the next integration
            # step:
            projected_step_size: float = q_scale * h
            if projected_step_size > max_step_size:
                projected_step_size = max_step_size

            # Break out of the processing loop:
            break

        # Scale the step size, bounding by the extrema limits:
        h *= q_scale
        if h > max_step_size:
            h = max_step_size
        if h < min_step_size:
            raise MinimumStepSizeExceededError(h, min_step_size)

    return IntegratorResults(
        epoch=t_n,
        state=x_n,
        total_step_seconds=h,
        intermediate_step_seconds=[
            h,
        ],
        projected_step_seconds=projected_step_size,
    )
