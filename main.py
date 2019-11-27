import rastrigin
from cgm import cgm

if __name__ == '__main__':
    x_start = [
        1.5,
        1.5
    ]

    # print(rastrigin.fn(x_start))

    cgm(rastrigin.fn, rastrigin.gradient, x_start, 1e-6)