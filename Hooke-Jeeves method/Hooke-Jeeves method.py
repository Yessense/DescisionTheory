import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def stop_condition(steps, eps):
    return np.all(steps < eps)


if __name__ == "__main__":
    n = 2
    step = 0.2
    d = 2
    epsilon = 0.0001
    iteration = 0
    x = np.array([8., 8.])
    steps = np.array([step] * n)
    m = 2
    verbose = 1
    color = 'black'

    if n == 2:  # if dimension = 2 we could plot function
        x_line = np.arange(-10, 10, 0.01)
        y_line = np.arange(-10, 10, 0.01)
        x_grid, y_grid = np.meshgrid(x_line, y_line, sparse=True)
        function_values = f([x_grid, y_grid])

        plt.pcolormesh(x_line, y_line, function_values, cmap=cm.get_cmap('inferno_r'), alpha=0.8)

    if verbose == 1:
        print('Starting...')

    while not stop_condition(steps, epsilon) and iteration < 50:
        x_old = x + 0
        x_value = f(x)
        iteration += 1
        plt.scatter(x[0], x[1])


        if verbose == 1:
            print(f'Iteration: {iteration}')
            print('-' * 40)
            print(f'Function value at x: {x_value:0.4f}')
            print(f'Steps: {steps}')

        for i in range(n):
            x_check = x + 0
            x_check[i] += steps[i]
            if verbose == 1:
                print(f'Checking {i} coordinate')
                print('- - ' * 15)
                print(f'Vector + {steps[i]} at {i} dim = {x_check}')
                print(f'Function value at {x_check} = {f(x_check):0.4f}')

            if f(x_check) < x_value:
                print(f'X checked value: {f(x_check):0.4f} < {x_value:0.4f}')
                x = x_check
                continue
            else:
                x_check[i] -= 2 * steps[i]
                print(f'Vector + {steps[i]} at {i} dim = {x_check}')

            if f(x_check) < x_value:
                print(f'X checked value: {f(x_check):0.4f} < {x_value:0.4f}')
                x = x_check

            print('----<')

        if verbose == 1:
            print(f'X: {x}')

        if np.allclose(x_old, x):
            if verbose == 1:
                print(f'Reducing steps: {steps} -> {steps / 2}')
            steps = steps / 2
        else:
            x_p = x + m * (x - x_old)

            if verbose == 1:
                print(f'Pattern search')
                print(f'X(p): {np.round(x_p,3)}')
                print(f'F(X(p)): {f(x_p):.4f}')
                print(f'F(X(p)) = {f(x_p):.4f} vs F(x) = {f(x):0.4f}')

            if f(x_p) < f(x):
                x = x_p

plt.show()
