from typing import List

from astrolib.base_objects import Matrix


def interpolate(x_vals: Matrix, y_vals: Matrix, x_0: float) -> float:
    if not x_vals.is_row_matrix():
        x_vals = x_vals.transpose()
    if not x_vals.is_row_matrix():
        raise ValueError("Invalid input data provided to Lagrange polynomial fit function. X-data must be a row or column vector.")
    if not y_vals.is_row_matrix():
        y_vals = y_vals.transpose()
    if not y_vals.is_row_matrix():
        raise ValueError("Invalid input data provided to Lagrange polynomial fit function. Y-data must be a row or column vector.")
    if x_vals.size != y_vals.size:
        raise ValueError("Invalid input data provided to Lagrange polynomial fit function. X- and Y-vectors must be the same size.")
    y_0 = 0.0
    for i in range(0, x_vals.num_cols):
        L = lambda x: 1.0
        for j in range(0, y_vals.num_cols):
            L = lambda x: ((x - x_vals[0,j]) / (x_vals[0,i] - x_vals[0,j])) * L(x)
        y_0 = y_0 + y_vals[0,i] * L(x_0)
    return y_0
