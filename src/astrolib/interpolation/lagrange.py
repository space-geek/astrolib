from typing import List

from astrolib.base_objects import Matrix


def interpolate(x_vals: Matrix, y_vals: Matrix, x_0: float) -> float:
    def validate_input_data(data: Matrix) -> bool:
        if not data.is_row_matrix():
            data.transpose()
        return data.is_row_matrix()
    if not validate_input_data(x_vals):
        raise ValueError("Invalid input data provided to Lagrange polynomial fit function. X-data must be a row or column vector.")
    if not validate_input_data(y_vals):
        raise ValueError("Invalid input data provided to Lagrange polynomial fit function. Y-data must be a row or column vector.")
    if x_vals.size != y_vals.size:
        raise ValueError("Invalid input data provided to Lagrange polynomial fit function. X- and Y-vectors must be the same size.")
    y_0 = 0.0
    for i in range(0, x_vals.num_cols):
        L = 1.0
        for j in range(0, y_vals.num_cols):
            if j != i:
                L = ((x_0 - x_vals[0,j]) / (x_vals[0,i] - x_vals[0,j])) * L
        y_0 = y_0 + y_vals[0,i] * L
    return y_0
