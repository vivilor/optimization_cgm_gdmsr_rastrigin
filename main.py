import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import rastrigin
from cgm import cgm

should_plot_contour = False
no_plot = True

if __name__ == '__main__':
    x_start = np.array([
        -0.02000000,
        -0.020000000
    ])

    print(x_start, x_start.size)

    x, y, f = cgm(
        rastrigin.fn,
        rastrigin.gradient,
        rastrigin.hessian,
        x_start,
        1e-10
    )

    print(x)
    print(y)
    print(f)

    if no_plot:
        exit(0)

    x_list = np.linspace(-2.5, 2.5, 512)
    y_list = np.linspace(-2.5, 2.5, 512)

    X, Y = np.meshgrid(x_list, y_list)
    Z = rastrigin.fn2d(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if should_plot_contour:
        ax.contour3D(X, Y, Z, 50)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(100, 150)
    else:
        ax.plot_wireframe(X, Y, Z, color='b', alpha=0.3)
    plt.plot(x, y, f, 'ro-')
    plt.show()