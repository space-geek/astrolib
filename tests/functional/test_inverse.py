""" TODO: Module docstring
"""
import math

from astrolib import Matrix



def linspace(min_val, max_val, num_steps):
    if num_steps == 1:
        yield min_val
    step_size = (max_val - min_val) / num_steps
    for i in range(num_steps):
        yield min_val + i * step_size

def poly_fit(order: int, x_vals: Matrix, y_vals: Matrix) -> Matrix:
    a_matrix = Matrix([[pow(row, order - idx) for idx in range(order + 1)] \
        for row in x_vals.get_col(0)])
    return a_matrix.pseudo_inverse() * y_vals

def linear_solver_gauss_seidel(A: Matrix, b: Matrix, X_0, omega: float, tol: float, max_iter: int) -> Matrix:
    def _resid(x: Matrix) -> Matrix:
        return b - A * x
    def _norm(x: Matrix) -> float:
        val = 0.0
        for i, _ in enumerate(x):
            val += pow(x[i], 2)
        return math.sqrt(val)
    x_k = X_0
    num_iter = 0
    while _norm(_resid(x_k) > tol):
        D = Matrix.diagonal(A.get_diagonal())
        L = -Matrix.lower_triangle(A, -1)
        U = -Matrix.upper_triangle(A, 1)
        # X_k = (D - omega * L) \ (((1 - omega) * D + omega * U) * x_k + omega * b) #TODO implement gaussian elimination to replace "\"
        num_iter += 1
        if num_iter > max_iter:
            raise Exception("Failed to converge.")
    return x_k

if __name__ == "__main__":
    order = 7
    x_vals = list(linspace(0, 2 * math.pi, 10))
    y_vals = [math.cos(x) for x in x_vals]

    coeffs = poly_fit(order, Matrix([x_vals]).transpose(), Matrix([y_vals]).transpose())

    print(" + ".join(f"{c}x^{order - i}" for i, c in enumerate(coeffs)))
