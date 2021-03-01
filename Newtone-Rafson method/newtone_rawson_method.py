import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from autograd import elementwise_grad as egrad
from autograd import jacobian
import autograd.numpy as np
from numpy.linalg.linalg import norm
from numpy.linalg.linalg import eigvals
from numpy.linalg.linalg import inv
from numpy import eye


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
    return norm(grad) <= eps


def is_pos_def(x):
    return np.all(eigvals(x) > 0)


if __name__ == "__main__":
    n = 2
    iteration = 0
    epsilon = 0.1
    verbose = 1
    color = 'black'
    H_f = jacobian(egrad(f))
    grad = egrad(f)
    descent_p = np.zeros(n)
    betta = 0
    x = np.array([8., 8.], dtype=np.float32)

    if n == 2:  # if dimension = 2 we could plot function
        x_line = np.arange(-10, 10, 0.01)
        y_line = np.arange(-10, 10, 0.01)
        x_grid, y_grid = np.meshgrid(x_line, y_line, sparse=True)
        function_values = f([x_grid, y_grid])

        plt.pcolormesh(x_line, y_line, function_values, cmap=cm.get_cmap('inferno_r'), alpha=0.8)

    if verbose == 1:
        print('Starting...')
        print('Starting point x_0:', x)

    while True:
        iteration += 1

        if verbose == 1:
            print('-' * 40)
            print(f'Iteration: {iteration}')
        # calculate f value at point
        x_value = f(x)

        if verbose == 1:
            print(f'X value: {x_value}')

        if verbose == 1:
            print(f'Gradient: {grad(x)}')

        # check stop condition
        if stop_condition(grad(x), epsilon):
            break

        if is_pos_def(H_f(x)):
            H_f_inv = inv(H_f(x))
        else:
            H_f_inv = eye(n)

        p_k = - H_f_inv @ grad(x)

        step = (grad(x) @ p_k) / ((H_f(x) @ p_k) @ p_k)

        if verbose == 1:
            print(f'Step: {step}')

        x_new = x + step * H_f_inv @ grad(x)

        print(x, x_new)
        if n == 2:
            plt.plot([x[0], x_new[0]], [x[1], x_new[1]], 'o-', color=color, lw=0.1, markersize=1)
        x = x_new

        if verbose == 1:
            print(f'X new coordinates: {x}')
            print('End of iteration')
            print('-' * 40)

plt.scatter(x[0], x[1])
plt.show()
