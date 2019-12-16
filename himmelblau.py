import numpy as np


# Частная производная по X
def _dfn_x(x, y):
    return x*(4 * x**2 + 4*y - 42) + 2 * y**2 - 14


# Частная производная по Y
def _dfn_y(x, y):
    return 2 * x**2 + y * (4 * x + 4 * y ** 2 - 26) - 22


# Функция для отрисовки
def fn2d(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


# Функция для расчетов, используется в методах
def fn(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def gradient(x):
    return np.array([
        _dfn_x(x[0], x[1]),
        _dfn_y(x[0], x[1])
    ])
