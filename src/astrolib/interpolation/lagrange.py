from typing import List

from astrolib.base_objects import Matrix


def interpolate(x_vals: Matrix, y_vals: Matrix, x_0: float) -> float:
    if x_vals.is_column_matrix():
        x_vals = x_vals.transpose()
    if y_vals.is_column_matrix():
        y_vals = y_vals.transpose()
    if not (x_vals.is_row_matrix() or y_vals.is_row_matrix()) or (x_vals.size != y_vals.size):
        raise ValueError("Invalid input data provided to Lagrange polynomial fit function. X- and Y-data must be row or column vectors and the same length.")
    y_0 = 0.0
    for i in range(0, x_vals.num_cols):
        L = 1.0
        for j in range(0, y_vals.num_cols):
            if j != i:
                L = ((x_0 - x_vals[0,j]) / (x_vals[0,i] - x_vals[0,j])) * L
        y_0 = y_0 + y_vals[0,i] * L
    return y_0
