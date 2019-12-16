import numpy as np
from scipy.optimize import minimize_scalar


def _lambda_coeff(fn, x_start, ort):
    """
    Функция для минимизации
    f(a) = F(X - aG)
    f(a) - функция для минимизации
    F(X - aG) - исследуемая функция (Химмельблау)
    X - точка, от которой идет направление
    G - вектор градиента функции, который задает направление
    а - искомый коэффициент

    :param x_start: точка, от которой идет направление
    :param fn: исследуемая функция (Растригин)
    :return: значение коэффициента, при котором функция принимает минимум
    """

    fn_to_minimize = lambda l: fn(x_start + l * ort)

    # Результат оптимизации по направлению
    result = minimize_scalar(fn_to_minimize)

    # Значение коэффициента при котором функция принимает минимум
    lambda_min = result.x

    return lambda_min


def _update_trace(trace, x_current, fn_val):
    trace['x'] = np.append(trace['x'], x_current[0])
    trace['y'] = np.append(trace['y'], x_current[1])
    trace['f'] = np.append(trace['f'], fn_val)


def _ort(idx, n):
    ort_vector = [0 if i is not idx else 1 for i in range(n)]

    return np.array(ort_vector)


def _print_step_index(k):
    print('# Step #{0} ###########################'.format(k))


def _print_x_and_f_values(x_current, f_current):
    print('x\t\t{0}'.format(x_current))
    print('F(x)\t\t{0}'.format(f_current))


def _print_fn_values_diff(fn_values_diff):
    print('F_k-1(x) - F_k(x) = ', fn_values_diff)


def cdm(fn, x_start, eps=1e-6):
    k = 0

    dim = x_start.size

    # TODO: сделать так везде
    trace = {
        'x': np.empty(0),
        'y': np.empty(0),
        'f': np.empty(0)
    }

    x_prev = None
    f_prev = None

    x_current = x_start.copy()
    f_current = fn(x_current)

    _print_x_and_f_values(x_current, f_current)
    _update_trace(trace, x_current, f_current)

    while True:
        _print_step_index(k)

        x_1 = None

        x_prev = x_current.copy()
        f_prev = f_current

        for i in range(dim):
            e_i = _ort(i, dim)

            lambda_i = _lambda_coeff(fn, x_current, e_i)

            x_current = x_current + lambda_i * e_i

            _update_trace(trace, x_current, fn(x_current))

            if i is 0:
                x_1 = x_current.copy()

        e_1 = _ort(0, dim)

        lambda_tilde = _lambda_coeff(fn, x_current, e_1)

        x_n_1 = x_current + lambda_tilde * e_1

        _update_trace(trace, x_n_1, fn(x_n_1))

        p = x_n_1 - x_1

        lambda_final = _lambda_coeff(fn, x_1, p)

        x_current = x_1 + lambda_final * p

        f_current = fn(x_current)

        f_diff = np.abs(f_current - f_prev)

        _print_fn_values_diff(f_diff)
        _print_x_and_f_values(x_current, f_current)
        _update_trace(trace, x_current, f_current)

        if f_diff < eps:
            return trace['x'], trace['y'], trace['f']

        k += 1



