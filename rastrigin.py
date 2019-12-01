import numpy as np


def _dfn(x_i, a=10):
    return 2 * np.pi * a * np.sin(2 * np.pi * x_i) + 2 * x_i


def _d2fn(x_i, a=10):
    return 2 + 4 * np.pi ** 2 * a * np.cos(2 * np.pi * x_i)


def _fn_term(x_i, a=10):
    return x_i ** 2 - a * np.cos(2 * np.pi * x_i)


def fn2d(x, y, a=10):
    print()
    return 2 * a + x**2 - a * np.cos(2 * np.pi * x) + y**2 - a * np.cos(2 * np.pi * y)


def fn(x, a=10):
    return a * x.size + np.apply_along_axis(_fn_term, 0, x, a).sum()


def hessian(x, a=10):
    n: int = x.size
    h_matrix = np.zeros((n, n))
    row, col = np.diag_indices_from(h_matrix)
    h_matrix[row, col] = np.apply_along_axis(_d2fn, 0, x, a)

    return h_matrix


def gradient(x, a=10):
    return np.apply_along_axis(_dfn, 0, x, a)
