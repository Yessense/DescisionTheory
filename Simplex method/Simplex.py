import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def f(x_1, x_2):
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
    out = 2.9 * x_1 ** 2 + 0.8 * x_1 * x_2
    out += 3.3 * x_2 ** 2 - 1.5 * x_1 + 3.1 * x_2
    return out


def delta(n: float, length: float, i_equals_j):
    """
    Function allows find all vertices

    Parameters
    ----------
    n: int
        dimension
    length: float
        length of simplex edge
    i_equals_j: bool
        formula is changed when i equals j

    Returns
    -------
    out: float

    """
    out = (n + 1) ** 0.5 - 1

    if i_equals_j:
        out += n

    out /= n * 2 ** 0.5
    out *= length
    return out


def create_start_vectors(start_point):
    """
    Function creates started simplex points

    Parameters
    ----------
    start_point: ndarray
        1d - array length of n

    Returns
    -------
    out: ndarray
        Simplex matrix size(n, n+1)
    """

    n = len(start_point)

    S = np.zeros((n, n + 1))

    S[:] = start_point[:, np.newaxis]

    d1 = delta(n, length, True)
    d2 = delta(n, length, False)

    for i in range(n):
        for j in range(1, n + 1):
            if i + 1 == j:
                S[i, j] += d1
            else:
                S[i, j] += d2
    return S


def find_max_value(f, S_arr, inverse=False, verbose=0):
    """
    Function find max/min element and index

    Parameters
    ----------
    f: func
        function applied to vector elements
    S_arr: ndarray
        coordinates of vertices
    inverse: bool
        False if we finding max element, True otherwise
    verbose: int

    Returns
    -------
    out: tuple
        max element, index of max element
    """
    values = [f(*X) for X in S_arr.T]

    if inverse:
        max_elem = min(values)
    else:
        max_elem = max(values)

    index_of_max = values.index(max_elem)

    if verbose == 1:
        print('Values: ', values)
        print('Max element: ', max_elem)
        print('Index of max: ', index_of_max)

    return max_elem, index_of_max


def find_center_of_gravity(index_to_remove: int, S_arr, verbose=0):
    """
    Function finds simplex center of gravity with/without index to remove column

    Parameters
    ----------
    index_to_remove: int
        index to remove for reflection
    S_arr: ndarray
        coordinates of vertices
    verbose: int

    Returns
    -------
    out: ndarray
        vector to center gravity point
    """
    to_take = np.ones(n + 1).astype(bool)

    if index_to_remove != -1:
        to_take[index_to_remove] = False

    center_of_gravity = np.average(S_arr[:, to_take], axis=1)

    if verbose == 1:
        print('Columns where to find center of gravity', to_take)
        print('Array without this index')
        print(S[:, to_take])
        print('Center of gravity:', center_of_gravity)
    return center_of_gravity


def find_reflection(x, center_of_gravity):
    """
    Find reflection of vector x across center of gravity

    Parameters
    ----------
    x: ndarray
        1-d vector
    center_of_gravity: ndarray
        1-d vector to point

    Returns
    -------
    out: ndarray
        1-d vector to reflected point

    """
    return 2 * center_of_gravity - x


def stop_condition(S_arr, verbose=0):
    """
    Checking for stop algorithm condition

    Parameters
    ----------
    S_arr: ndarray
        coordinates of vertices
    verbose: int

    Returns
    -------
    out: bool
        if abs of difference is less < epsilon, then we stop
    """
    center_of_gravity = find_center_of_gravity(-1, S_arr=S)

    if verbose == 1:
        print('Center of gravity: ', center_of_gravity)

    absolute = S_arr - center_of_gravity[:, np.newaxis]

    if verbose == 1:
        print('Difference:')
        print(absolute)

    absolute = absolute ** 2

    if verbose == 1:
        print('Squared:')
        print(absolute)

    absolute = np.sum(absolute, axis=0)

    if verbose == 1:
        print('Sum:')
        print(absolute)
    if verbose == 2:
        print('Absolute difference for all values:', np.round(absolute, 3))
    return np.all(absolute < epsilon)


def reduction(index_of_min, S_arr, verbose=0):
    """
    Shrinks the sides 2 times

    Parameters
    ----------
    index_of_min: int
        index of vector with minimum value
    S_arr: ndarray
        coordinates of vertices
    verbose:  int

    Returns
    -------
    out: None

    """
    to_take = np.ones(n + 1).astype(bool)
    to_take[index_of_min] = False

    x_r = S_arr[:, index_of_min]

    if verbose == 1:
        print('To take:', to_take)
        print('X_r', x_r)

    S_arr[:, to_take] = x_r[:, np.newaxis] + 0.5 * (S_arr[:, to_take] - x_r[:, np.newaxis])


if __name__ == '__main__':
    verbose = 1 # verbose parameter (could be 0, 1, 2) 0 - zero verbose, 1 - standard, 2 - special
    n = 2 # dimension
    length = 8. # length of simplex edge
    epsilon = 0.1 # precision

    start_vertex = np.array([1, 1]) # start vector (should be n - dimensioned)

    print('Start vertex:', start_vertex)
    S = create_start_vectors(start_point=start_vertex)

    if n == 2: # if dimension = 2 we could plot function
        x_line = np.arange(-10, 10, 0.05)
        y_line = np.arange(-10, 10, 0.05)
        x_grid, y_grid = np.meshgrid(x_line, y_line, sparse=True)
        function_values = f(x_grid, y_grid)

        plt.pcolormesh(x_line, y_line, function_values, cmap=cm.get_cmap('inferno_r'), alpha=0.8)

    iterations = 0
    color = 'black'
    while not stop_condition(S):
        iterations += 1
        max_elem, index_of_max = find_max_value(f, S_arr=S, verbose=0)

        horisontal = S[0]
        vertical = S[1]

        if n == 2:
            plt.plot(horisontal[[0, 1]], vertical[[0, 1]], color=color)
            plt.plot(horisontal[[1, 2]], vertical[[1, 2]], color=color)
            plt.plot(horisontal[[0, 2]], vertical[[0, 2]], color=color)

        if verbose == 1:
            print('----------------------')
            print('Iteration:', iterations)
            print('Max element: ', max_elem)

        center_of_gravity = find_center_of_gravity(index_of_max, S_arr=S, verbose=0)
        x_new = find_reflection(S[:, index_of_max], center_of_gravity)
        x_new_value = f(*x_new)

        if verbose == 1:
            print('Center of gravity:', center_of_gravity)
            print('X new:', x_new)
            print('X new value:', x_new_value)

        if x_new_value < max_elem:
            S[:, index_of_max] = x_new
            print(f'X new {x_new_value:0.3f} < max_element {max_elem:0.3f} ')
            print('Next iteration')
        else:
            minimum, index_of_min = find_max_value(f, S_arr=S, inverse=True)
            reduction(index_of_min, S_arr=S, verbose=0)
            print('Performing reduction')
        print('----------------------')

    if n == 2:
        plt.show()

    print('Stop condition reached!\n')
    stop_condition(verbose=2, S_arr=S)

    min_elem, index_of_min = find_max_value(f, S_arr=S, inverse=True)
    print('\nIterations completed:', iterations)
    print('Lowest function value: ', min_elem)
