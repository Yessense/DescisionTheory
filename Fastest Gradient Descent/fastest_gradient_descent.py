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


# scipy.optimize.approx_fprime(np.array([1,1]), f, 1e-5)

def stop_condition(grad, eps):
    return np.linalg.norm(grad) <= eps


if __name__ == "__main__":
    n = 2
    iteration = 0
    epsilon = 0.1
    verbose = 1
    color = 'black'
    H_f = jacobian(egrad(f))

    if n == 2:  # if dimension = 2 we could plot function
        x_line = np.arange(-10, 10, 0.05)
        y_line = np.arange(-10, 10, 0.05)
        x_grid, y_grid = np.meshgrid(x_line, y_line, sparse=True)
        function_values = f([x_grid, y_grid])

        plt.pcolormesh(x_line, y_line, function_values, cmap=cm.get_cmap('inferno_r'), alpha=0.8)

    x = np.array([-8, -8], dtype=np.float32)
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

        # get grad coords
        grad = scipy.optimize.approx_fprime(x, f, 1e-5)


        if verbose == 1:
            print(f'Gradient: {grad}')

        # check stop condition
        if stop_condition(grad, epsilon):
            break

        step =  (grad @ grad) / ((H_f(x) @ grad ) @ grad)
        if verbose == 1:
            print(f'Step: {step}')

        x_new = x - step * grad
        x_new_value = f(x_new)

        while x_new_value >= x_value:
            step /= 2
            x_new = x - step * grad
            x_new_value = f(x_new)

        if n == 2:
            plt.plot([x[0], x_new[0]],[x[1], x_new[1]], 'o-',color=color, lw=0.1, markersize=1)
        x = x_new

        if verbose == 1:
            print(f'X new coordinates: {x}')
            print('End of iteration')
            print('-' * 40)

plt.scatter(x[0],x[1])
plt.show()
