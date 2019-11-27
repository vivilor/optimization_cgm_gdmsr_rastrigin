import rastrigin
from math import fabs


def square_norm(x_input):
    result = 0

    for x_i in x_input:
        result += x_i ** 2

    return result


def scalar_prod(x_a, x_b):
    result = 0

    for i in range(len(x_a)):
        result += x_a[i] * x_b[i]

    return result


def vector_multiply(x_vec, a):
    result_vec = []

    for x_i in x_vec:
        result_vec.append(x_i * a)

    return result_vec


def vector_sum(x_a, x_b):
    result_vec = []

    for i in range(len(x_a)):
        result_vec.append(x_a[i] + x_b[i])

    return result_vec


def vector_diff(x_a, x_b):
    result_vec = []

    for i in range(len(x_a)):
        result_vec.append(x_a[i] - x_b[i])

    return result_vec


def fletcher_reeves_coeff(grad, grad_prev):
    return \
        square_norm(grad) /\
        square_norm(grad_prev)


def polak_ribiere(grad, grad_prev):
    return \
        scalar_prod(grad, vector_diff(grad, grad_prev)) /\
        square_norm(grad_prev)


def search_direction(grad, grad_prev, dir_prev):
    if not grad_prev or not dir_prev:
        return grad
    else:
        return vector_sum(grad, vector_multiply(dir_prev, polak_ribiere(grad, grad_prev)))


def cgm(fn, dfn, x_start, eps):
    """
    Найти минимум функции `fn` методом сопряженных градиентов
    с начальным приближением `x_start` и точностью `eps`
    :param fn:
    :param dfn:
    :param x_start:
    :param eps:
    :return:
    """
    # Размерность пространства
    dim = len(x_start)

    # Номер шага
    k = 0

    is_not_finished = True
    grad = None
    grad_prev = None
    dir = None
    dir_prev = None
    x_current = x_start.copy()

    f = .0
    f_prev = .0

    while is_not_finished:
        print('Step #{0}'.format(k))
        if k > 0:
            grad_prev = grad.copy()

        grad = dfn(x_current)

        print('Current grad: {0}'.format(grad))

        dir = search_direction(grad, grad_prev, dir_prev)

        print('Current direction: {0}'.format(dir))

        x_current = vector_sum(x_current, vector_multiply(dir, 1e-5))

        print('Current X: {0}'.format(x_current))

        if dir:
            dir_prev = dir.copy()

        f_prev = f

        f = fn(x_current)

        if k > 0 and fabs(f - f_prev) < eps:
            print('{0} - {1} = {2}\n'.format(f, f_prev, f - f_prev))
            return x_current

        k += 1

