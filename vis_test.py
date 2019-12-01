import numpy as np
import rastrigin

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#
# x_list = np.linspace(-5.12, 5.12, 512)
# y_list = np.linspace(-5.12, 5.12, 512)
#
# X, Y = np.meshgrid(x_list, y_list)
# Z = rastrigin.fn2d(X, Y)
# a = '3d'

# if a is '3d':
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.contour3D(X, Y, Z, 50)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.view_init(100, 150)
# else:
#     fig, ax = plt.subplots(1, 1)
#     cp = ax.contourf(X, Y, Z)
#
#     # Add a color bar to a plot
#     fig.colorbar(cp)
#
#     ax.set_title('Rastrigin (Contour)')
#
#     # ax.set_xlabel('x (cm)')
#     # ax.set_ylabel('y')
# plt.show()


x = np.empty(0)
a = np.array([.8, -.8])
x = np.append(x, a[0])
print(rastrigin.fn(np.array([0.8, -0.8])))
print(rastrigin.fn2d(0.8, -0.8))
print(rastrigin._fn_term(0.8) + rastrigin._fn_term(-0.8))
print(np.apply_along_axis(rastrigin._fn_term, 0, a, 10).sum())
print(x)

