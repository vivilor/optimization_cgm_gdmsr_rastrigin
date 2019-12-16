import rastrigin
import numpy as np
from scipy.optimize import minimize_scalar


def _store_points(x_current, fn_val, x_points, y_points, fn_points):
    return \
        np.append(x_points, x_current[0]),\
        np.append(y_points, x_current[1]),\
        np.append(fn_points, fn_val)


def _print_x_and_fn_values(x_current, fn_val):
    print('x\t\t{0}'.format(x_current))
    print('F(x)\t\t{0}'.format(fn_val))


def _print_fn_values_diff(fn_values_diff):
    print('F_k-1(x) - F_k(x) = ', fn_values_diff)


def _lambda_coeff(f_prev, f_cur, grad_prev_norm):
    return 2 * (f_prev - f_cur) / grad_prev_norm


def gdm_sr(fn, grad_fn, x_start, eps, lambda_start=-0.5, k_max=-1):
    # Номер шага
    k = 0

    # Хранилище точек
    x_points = np.empty(0)
    y_points = np.empty(0)
    fn_points = np.empty(0)

    x_current = None

    fn_val = None
    prev_fn_val = None

    gradient_norm = None
    lambda_coeff = None

    should_check_step_index = True

    if k_max < 0:
        should_check_step_index = False

    while True:
        print('### Step #{0} ###'.format(k))

        if k is 0:
            lambda_coeff = lambda_start
            x_current = np.copy(x_start)
            fn_val = fn(x_current)

            _print_x_and_fn_values(x_current, fn_val)

            x_points, y_points, fn_points = \
                _store_points(x_current, fn_val, x_points, y_points, fn_points)

        gradient_norm = np.linalg.norm(grad_fn(x_current))

        prev_fn_val = fn_val

        x_current = x_current - lambda_coeff * gradient_norm

        fn_val = fn(x_current)

        fn_values_diff = np.abs(prev_fn_val - fn_val)

        _print_fn_values_diff(fn_values_diff)
        _print_x_and_fn_values(x_current, fn_val)

        x_points, y_points, fn_points =\
            _store_points(x_current, fn_val, x_points, y_points, fn_points)

        if fn_values_diff < eps:
            return x_points, y_points, fn_points

        if should_check_step_index and k is k_max:
            return x_points, y_points, fn_points

        lambda_coeff = _lambda_coeff(prev_fn_val, fn_val, gradient_norm)

        k += 1

