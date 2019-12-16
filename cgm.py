import numpy as np
from math import isnan
from scipy.optimize import minimize_scalar


def _alpha_coeff_lin(x_current, fn, gradient, a=10):
    alpha_min = minimize_scalar(
        lambda alpha: fn(x_current - alpha * gradient, a)
    ).x

    return alpha_min


def _alpha_coeff(gradient, direction, hessian) -> float:
    nom = np.matmul(gradient.T, direction)
    den = np.matmul(np.matmul(direction.T, hessian), direction)
    alpha = -float(nom / den)

    return alpha


def _beta_coeff(gradient, direction, hessian) -> float:
    nom = np.matmul(np.matmul(gradient.T, hessian), direction)
    den = np.matmul(np.matmul(direction.T, hessian), direction)
    beta = float(nom / den)

    return beta


def _store_points(x_current, fn_val, x_points, y_points, fn_points):
    return \
        np.append(x_points, x_current[0]),\
        np.append(y_points, x_current[1]),\
        np.append(fn_points, fn_val)


def _print_step_index(k):
    print('###########################')
    print('# Step #{0}'.format(k))
    print('###########################')


def _print_and_x_fn_values(x_current, fn_val):
    print('x\t\t{0}'.format(x_current))
    print('F(x)\t\t{0}'.format(fn_val))


def _print_gradient_and_direction(k, gradient, direction):
    print('grad F(x)\t{0}'.format(gradient))
    print('d_{0}\t{1}'.format(k, direction))


def _print_fn_values_diff(fn_values_diff):
    print('F_k-1(x) - F_k(x) = ', fn_values_diff)


def cgm(fn, grad_fn, hessian_fn, x_start, eps, k_max=-1, a=10):
    # Размерность пространства
    dim = x_start.size

    # Номер шага
    k = 0

    # Хранилище точек
    x_points = np.empty(0)
    y_points = np.empty(0)
    fn_points = np.empty(0)

    x_current = None

    fn_val = None
    prev_fn_val = None

    hessian = None
    gradient = None
    direction = None

    should_check_step_index = True

    if k_max < 0:
        should_check_step_index = False

    while True:
        _print_step_index(k)

        if k is 0:
            x_current = np.copy(x_start)
            fn_val = fn(x_current, a)
            _print_and_x_fn_values(x_current, fn_val)
            x_points, y_points, fn_points =\
                _store_points(x_current, fn_val, x_points, y_points, fn_points)

        gradient = grad_fn(x_current, a)
        # gradient /= np.linalg.norm(gradient)

        if k % 2:
            beta = _beta_coeff(gradient, direction, hessian)

            if isnan(beta):
                print('Beta is nan')
                return x_points, y_points, fn_points

            hessian = hessian_fn(x_current, a)

            direction = -gradient - beta * direction
        else:
            hessian = hessian_fn(x_current, a)
            print('>> update direction')
            direction = -gradient

        # direction /= np.linalg.norm(direction)

        _print_gradient_and_direction(k, gradient, direction)

        # alpha = _alpha_coeff(gradient, direction, hessian)
        alpha = _alpha_coeff_lin(x_current, fn, gradient, a)

        if isnan(alpha):
            print('Alpha is nan')
            return x_points, y_points, fn_points

        x_current = x_current + alpha * direction

        prev_fn_val = fn_val

        fn_val = fn(x_current, a)

        fn_values_diff = np.abs(prev_fn_val - fn_val)

        _print_fn_values_diff(fn_values_diff)
        _print_and_x_fn_values(x_current, fn_val)

        x_points, y_points, fn_points =\
            _store_points(x_current, fn_val, x_points, y_points, fn_points)

        if fn_values_diff < eps:
            return x_points, y_points, fn_points

        if should_check_step_index and k is k_max:
            return x_points, y_points, fn_points

        k += 1
