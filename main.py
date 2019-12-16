import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import rastrigin as target_f
from gdm import gdm
from cdm_bigor import cdm

should_plot_contour = True
no_plot = False

if __name__ == '__main__':
    x_start = np.array([
        -0.9,
        0.42
    ])

    # Метод сопряженных направлений
    x1, y1, f1 = cdm(
        target_f.fn,
        x_start,
        1e-10
    )

    # Метод градиентного спуска
    x2, y2, f2 = gdm(
        target_f.fn,
        target_f.gradient,
        x_start,
        1e-10,
        100
    )

    if no_plot:
        exit(0)

    x_list = np.linspace(-1, 1, 128)
    y_list = np.linspace(-1, 1, 128)

    X, Y = np.meshgrid(x_list, y_list)
    Z = target_f.fn2d(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if should_plot_contour:
        ax.contour3D(
            X,
            Y,
            Z,
            20,  # количестов линий уровня
            alpha=0.3,  # прозрачность
            cmap=plt.get_cmap('plasma')  # цветовая схема
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f')
        ax.view_init(100, 150) # начальное положение камеры
    else:
        ax.plot_surface(X, Y, Z, alpha=0.3, cmap=plt.get_cmap('plasma'))

    plt.plot(x1[0:2], y1[0:2], f1[0:2], 'mo-')
    plt.plot(x1[1:], y1[1:], f1[1:], 'ro-')

    plt.plot(x2[0:2], y2[0:2], f2[0:2], 'co-')
    plt.plot(x2[1:], y2[1:], f2[1:], 'go-')

    plt.show()