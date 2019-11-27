from math import cos, sin, pi


def fn(x_input, a=10):
    """
    Вычислить функцию Растригина от вектора
    :param x_input: входной вектор
    :param a: коэффициент фукнции
    :return: значение функция Растригина в заданной входным вектором точке
    """
    n = len(x_input)
    result = a * n

    for x_i in x_input:
        result += x_i ** 2 - a * cos(2 * pi * x_i)

    return result


def private_derivative(x_i, a=10):
    return 2 * x_i + 2 * pi * a * sin(2 * pi * x_i)


def gradient(x_input, negative=True, a=10):
    grad = []

    for x_i in x_input:
        p_deriv = private_derivative(x_i, a)

        grad.append(-p_deriv if negative else p_deriv)

    return grad
