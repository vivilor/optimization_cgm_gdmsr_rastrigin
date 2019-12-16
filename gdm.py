import rastrigin
import numpy as np
from scipy.optimize import minimize_scalar


def _store_points(x_current, fn_val, x_points, y_points, fn_points):
    return \
        np.append(x_points, x_current[0]),\
        np.append(y_points, x_current[1]),\
        np.append(fn_points, fn_val)


def _print_step_index(k):
    print('# Step #{0} ###########################'.format(k))


def _print_and_x_fn_values(x_current, fn_val):
    print('x\t\t{0}'.format(x_current))
    print('F(x)\t\t{0}'.format(fn_val))


def _print_fn_values_diff(fn_values_diff):
    print('F_k-1(x) - F_k(x) = ', fn_values_diff)


def _alpha_coeff(x_current, fn, gradient, a=10):
    """
    Функция для минимизации
    f(a) = F(X - aG)
    f(a) - функция для минимизации
    F(X - aG) - исследуемая функция (Растригин)
    X - точка, от которой идет направление
    G - вектор градиента функции, который задает направление
    а - искомый коэффициент

    :param x_current: точка, от которой идет направление
    :param fn: исследуемая функция (Химмельблау)
    :param gradient: вектор градиента функции, который задает направление
    :param a: искомый коэффициент
    :return: значение коэффициента, при котором функция принимает минимум
    """
    fn_to_minimize = lambda alpha: fn(x_current - alpha * gradient)

    # Результат оптимизации по направлению
    result = minimize_scalar(fn_to_minimize)

    # Значение коэффициента при котором функция принимает минимум
    alpha_min = result.x

    return alpha_min


def gdm(fn, grad_fn, x_start, eps, k_max=-1):
    # Номер шага
    k = 0

    # Хранилище точек
    x_points = np.empty(0)
    y_points = np.empty(0)
    fn_points = np.empty(0)

    x_current = None

    fn_val = None
    prev_fn_val = None

    gradient = None

    should_check_step_index = True

    if k_max < 0:
        should_check_step_index = False

    while True:
        print('### Step #{0} ###'.format(k))

        if k is 0:
            x_current = np.copy(x_start)
            fn_val = fn(x_current)

            _print_and_x_fn_values(x_current, fn_val)

            x_points, y_points, fn_points = \
                _store_points(x_current, fn_val, x_points, y_points, fn_points)

        gradient = np.linalg.norm(grad_fn(x_current))

        alpha = _alpha_coeff(x_current, fn, gradient)

        prev_fn_val = fn_val

        x_current = x_current - alpha * gradient

        fn_val = fn(x_current)

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

