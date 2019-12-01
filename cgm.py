import numpy as np


def _alpha_coeff(gradient, direction, hessian) -> float:
    nom = np.dot(gradient, direction)
    den = np.matmul(np.matmul(direction.T, hessian), direction)
    alpha = -float(nom / den)

    return alpha


def _beta_coeff(gradient, direction, hessian) -> float:
    nom = np.matmul(np.matmul(gradient.T, hessian), direction)
    den = np.matmul(np.matmul(direction.T, hessian), direction)
    beta = float(nom / den)

    return beta


def cgm(fn, grad_fn, hessian_fn, x_start, eps, k_max=-1):
    # Размерность пространства
    dim = x_start.size

    # Номер шага
    k = 1
    x_points = np.empty(0)
    y_points = np.empty(0)
    fn_points = np.empty(0)

    is_not_finished = True
    should_check_step_index = True

    x_current = np.copy(x_start)
    gradient = grad_fn(x_current)
    direction = -gradient

    fn_val = fn(x_current)

    print('Step #{0}'.format(1))
    print('x\t{0}'.format(x_current))
    print('F(x)\t{0}'.format(fn_val))

    prev_fn_val = fn_val + eps * 2.

    x_points = np.append(x_points, x_current[0])
    y_points = np.append(y_points, x_current[1])
    fn_points = np.append(fn_points, fn_val)

    if k_max < 0:
        should_check_step_index = False

    while is_not_finished:
        print('Step #{0}'.format(k))

        hessian = hessian_fn(x_current)

        alpha = _alpha_coeff(gradient, direction, hessian)

        if alpha is np.nan:
            print('Alpha is nan')
            return x_points, y_points, fn_points

        x_current = x_current + alpha * direction

        prev_fn_val = fn_val

        fn_val = fn(x_current)

        print('x\t{0}'.format(x_current))
        print('F(x)\t{0}'.format(fn_val))

        x_points = np.append(x_points, x_current[0])
        y_points = np.append(y_points, x_current[1])
        fn_points = np.append(fn_points, fn_val)

        fn_values_diff = np.abs(prev_fn_val - fn_val)
        print('F_k-1(x) - F_k(x) = ', fn_values_diff)

        if fn_values_diff < eps:
            return x_points, y_points, fn_points

        if should_check_step_index and k is k_max:
            return x_points, y_points, fn_points

        gradient = grad_fn(x_current)

        beta = _beta_coeff(gradient, direction, hessian)

        if beta is np.nan:
            print('Beta is nan')
            return x_points, y_points, fn_points

        direction = -gradient + beta * direction
        k += 1
