""" TODO: Module docstring
"""
from typing import Callable
from typing import List
from typing import Tuple

from astrolib import Matrix
from astrolib.util.constants import MAX_ZERO_THRESHOLD_VALUE


def interpolate(x_vals: Matrix, y_vals: Matrix, x_q: Matrix) -> Matrix:

    # Cache the number of x- and y-values provided:
    n = len(x_vals)
    m = len(y_vals)

    # Compute the cubic spline interpolants and interpolated values based
    # on the number of y-values provided:
    if m == n: # Natural spline
        p, p_x, coeffs, coeffs_raw = _build_natural_spline_poly(x_vals, y_vals)
    elif m == (n + 2): # Clamped spline
        p, p_x, coeffs, coeffs_raw = _build_clamped_spline_poly(x_vals, y_vals)
    else:
        raise ValueError("Invalid input data provided to cubic spline interpolation function. " \
                         "X- and Y-vectors must be the same size or the Y-vector must have exactly " \
                         "two more elements than the X-vector if clamped behavior is desired.")

    # Approximate the requested values using the cubic spline interpolant:
    return p(x_q)


def _build_natural_spline_poly(x_vals: Matrix, y_vals: Matrix) -> Tuple[Callable[[Matrix], Matrix], List[Callable[[Matrix], Matrix]], Matrix, Matrix]:

    # Cache the number of x-values provided:
    n = len(x_vals)

    # Check that the number of y-values provided is valid:
    if len(y_vals) != n:
        raise ValueError('Invalid input data provided to clamped cubic spline interpolation function. X- and Y-vectors must be the same size.')

    # Initialize vectors for the calculation:
    a = y_vals[1:]
    b = Matrix.zeros(1,n-1)
    c = Matrix.zeros(*b.size)
    d = Matrix.zeros(*b.size)
    alpha_vec = Matrix.zeros(1,n)
    l_vec  = Matrix.zeros(*alpha_vec.size)
    mu_vec = Matrix.zeros(*alpha_vec.size)
    z_vec = Matrix.zeros(*alpha_vec.size)

    # Compute the h-values:
    h_vec = Matrix.zeros(1, n-1)
    for i in range(0, n-1):
        h_vec[i] = x_vals[i+1] - x_vals[i]

    # Compute the alpha values:
    for i in range(1, n-1):
        if i == n-2:
            import pdb;pdb.set_trace()
        alpha_vec[i] = (3 / h_vec[i]) * (a[i+1] - a[i]) - (3 / h_vec[i-1]) * (a[i] - a[i-1])

    # Compute the other parameters needed for computation of the
    # coefficients:
    l_vec[1] = 1.0
    mu_vec[1] = 0.0
    z_vec[1] = 0.0
    for i in range(1, n-1):
        l_vec[i] = 2 * (x_vals[i+1] - x_vals[i-1]) - h_vec[i-1] * mu_vec[i-1]
        mu_vec[i] = h_vec[i] / l_vec[i]
        z_vec[i] = (alpha_vec[i] - h_vec[i-1] * z_vec[i-1]) / l_vec[i]
    l_vec[-1] = 1.0
    z_vec[-1] = 0.0

    # Compute the remaining coefficients:
    c_jp1 = 0.0
    for j in range((n-1), 0, -1):
        c[j] = z_vec[j] - mu_vec[j] * c_jp1
        b[j] = ((a[j+1] - a[j]) / h_vec[j]) - ((h_vec[j] * (c_jp1 + 2 * c[j])) / 3)
        d[j] = (c_jp1 - c[j]) / (3 * h_vec[j])
        c_jp1 = c[j]

    # Build up and return the output objects:
    return _make_output_objects(x_vals, a, b, c, d)


def _build_clamped_spline_poly(x_vals: Matrix, y_vals: Matrix) -> Tuple[Callable[[Matrix], Matrix], List[Callable[[Matrix], Matrix]], Matrix, Matrix]:
    #TODO Port guts of clamped spline function to solve for coefficients
    return _make_output_objects(x_vals, a, b, c, d)


def _evaluate_piecewise_polynomial(x_vals: Matrix, Px: List[Callable[[Matrix], Matrix]], xq: Matrix) -> Matrix:
    n = len(xq)
    yq = Matrix.zeros(*xq.size)
    import pdb; pdb.set_trace()
    for i, x in range(n):
        j = next(x for x, v in enumerate(x_vals) if xq[i] >= x)
        if j > n-1:
            j = n-1
        P = Px[j]
        yq[i] = P(xq[i])
    return yq


def _make_output_objects(x_vals: Matrix, a: Matrix, b: Matrix, c: Matrix, d: Matrix) -> Tuple[Callable[[Matrix], Matrix], List[Callable[[Matrix], Matrix]], Matrix, Matrix]:

    # Cache the number of x-values provided:
    n = len(x_vals)

    # Build up the output objects:
    """ NOTE: This output is of the form:
                  P       == Callable to evaluate piecewise polynomials
                  Px      == Cell array of piecewise polynomials
                  coeffs  == Matrix of simplified polynomial term coefficients
                  coeffs_raw  == Matrix of polynomial term coefficients a, b, c, & d
              where
                  Px{j} = @(x) coeffs(j,1)*x^3 + coeffs(j,2)*x^2 + coeffs(j,3)*x + coeffs(j,4)
              and
                  Px{j} = @(x) coeffs_raw(j,1)*(x - x_vals[j])^3 + coeffs_raw(j,2)*(x - x_vals[j])^2 + coeffs_raw(j,3)*(x - x_vals[j]) + coeffs_raw(j,4)
    """
    coeffs = Matrix.zeros(n-1, 4)
    coeffs_raw = Matrix.zeros(*coeffs.size)
    p_x = []
    for j in range(n-1):
        coeffs[j, 0] = d[j]
        coeffs[j, 1] = -3 * d[j] * x_vals[j] + c[j]
        coeffs[j, 2] = 3 * d[j] * x_vals[j]^2 - 2 * c[j] * x_vals[j] + b[j]
        coeffs[j, 3] = -d[j] * x_vals[j]^3 + c[j] * x_vals[j]^2 - b[j] * x_vals[j] + a[j]
        for k in range(4):
            if abs(coeffs[j, k]) < MAX_ZERO_THRESHOLD_VALUE:
                coeffs[j, k] = 0.0
        p_x.append(lambda x: coeffs[j, 0] * pow(x, 3) + coeffs[j, 1] * pow(x, 2) + coeffs[j, 2] * x + coeffs[j, 3])
        coeffs_raw[j, 0] = d[j]
        coeffs_raw[j, 1] = c[j]
        coeffs_raw[j, 2] = b[j]
        coeffs_raw[j, 3] = a[j]
    p = lambda x: _evaluate_piecewise_polynomial(x_vals, p_x, x)
    return p, p_x, coeffs, coeffs_raw
