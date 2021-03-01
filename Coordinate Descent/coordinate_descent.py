import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from autograd import elementwise_grad as egrad
from autograd import jacobian
import autograd.numpy as np


def f(x):
    """
    Your function to compute

    Parameters
    ----------
    x_1: float
    x_2: float

    Returns
    -------
    out: float

    """
    x_1, x_2 = x
    out = 2.9 * x_1 ** 2 + 0.8 * x_1 * x_2
    out += 3.3 * x_2 ** 2 - 1.5 * x_1 + 3.1 * x_2
    return out


def stop_condition(grad, eps):
    return np.linalg.norm(grad) <= eps


if __name__ == "__main__":
    n = 2
    epsilon = 0.1
    x = np.array([5., 8.])
    iteration = 0
    verbose = 1
    grad = egrad(f)
    H_f = jacobian(grad)
    color = 'black'

    if n == 2:  # if dimension = 2 we could plot function
        x_line = np.arange(-10, 10, 0.05)
        y_line = np.arange(-10, 10, 0.05)
        x_grid, y_grid = np.meshgrid(x_line, y_line, sparse=True)
        function_values = f([x_grid, y_grid])

        plt.pcolormesh(x_line, y_line, function_values, cmap=cm.get_cmap('inferno_r'), alpha=0.8)

    while not stop_condition(grad(x), epsilon) and iteration < 4:
        iteration += 1

        if verbose == 1:
            print('-' * 40)
            print(f'Iteration: {iteration}')

        for i in range(n):
            print(f'Fixing coordinates exept: {i}')

            e = np.zeros(n)
            e[i] = 1

            if verbose == 1:
                print(f'Basis vector: {e}')

            step =  (grad(x) @ grad(x)) / ((H_f(x) @ grad(x)) @ grad(x))

            if verbose == 1:
                print(f'Step: {step:.4f}')

            x_new  = x - step * grad(x) * e

            if n == 2:
                plt.plot([x[0], x_new[0]], [x[1], x_new[1]], 'o-', color=color, lw=0.1, markersize=1)

            x = x_new

            if verbose == 1:
                print(f'Function value: {f(x)}')


plt.scatter(x[0],x[1])
plt.show()
