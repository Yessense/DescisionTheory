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


def create_start_vectors(start_point, length):
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


def find_n_max_values(f, S_arr, n=1, inverse=False, verbose=0):
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
    print(values)
    if inverse:
        index = np.argsort(values)[:n]
    else:
        index = np.argsort(values)[-n:]

    elements = []
    for idx in index:
        elements.append(values[idx])

    if verbose == 1:
        print('Values: ', values)
        print('Max element: ', index)
        print('Index of max: ', elements)

    return elements, index


def find_center_of_gravity(index_to_remove: int, S_arr, n=1, verbose=0):
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
        print(S_arr[:, to_take])
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


def reduction(index_of_min, S_arr, n, verbose=0):
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


def stop_condition(S_arr, epsilon, n, verbose=0):
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
    center_of_gravity = find_center_of_gravity(-1, S_arr=S_arr, n=n)

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

    absolute = np.sum(absolute)
    sigma = absolute / len(S_arr)
    sigma = np.sqrt(sigma)

    if verbose == 1:
        print('Sum:')
        print(absolute)
    if verbose == 2:
        print('Absolute difference for all values:', np.round(absolute, 3))
    return sigma < epsilon


def change_size(center_of_gravity, vector, coeff):
    return center_of_gravity + coeff * (vector - center_of_gravity)


if __name__ == "__main__":
    verbose = 1
    n = 2
    length = 8
    epsilon = 0.1

    start_vertex = np.array([16, 16])  # start vector (should be n - dimensioned)

    print('Start vertex:', start_vertex)
    S = create_start_vectors(start_point=start_vertex, length=length)

    if n == 2:  # if dimension = 2 we could plot function
        x_line = np.arange(-10, 10, 0.05)
        y_line = np.arange(-10, 10, 0.05)
        x_grid, y_grid = np.meshgrid(x_line, y_line, sparse=True)
        function_values = f(x_grid, y_grid)

        plt.pcolormesh(x_line, y_line, function_values, cmap=cm.get_cmap('inferno_r'), alpha=0.8)

    iterations = 0
    color = 'black'
    betta = 2.
    gamma = 0.5

    while not stop_condition(S, epsilon, n=n):
        iterations += 1

        print(S)
        max_elems, indexes = find_n_max_values(f, S, 2)
        # Step 3
        # find 3 points 2 max, 1 lowest
        # ---------------------------------------


        second_max_elem, max_elem = max_elems
        second_max_elem_index, max_elem_index= indexes

        low_elems, indexes = find_n_max_values(f, S, 1, True)
        low_elem, low_elem_index = low_elems[0], indexes[0]

        print(max_elem_index, second_max_elem_index, low_elem_index)
        # ---------------------------------------

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


        # Step 4
        # find center of gravity
        # ---------------------------------------
        center_of_gravity = find_center_of_gravity(max_elem_index, S, n=2)
        # ---------------------------------------

        # Step 5
        # find reflection
        # ---------------------------------------
        x_new = find_reflection(S[:, max_elem_index], center_of_gravity)
        x_new_value = f(*x_new)
        # ---------------------------------------

        if verbose == 1:
            print('Center of gravity:', center_of_gravity)
            print('X new:', x_new)
            print('X new value:', x_new_value)

        # if x_new_value < low_elem:
        #     x_new_streched = change_size(center_of_gravity, x_new, betta)
        #     x_new_streched_value = f(*x_new_streched)
        #
        #     if x_new_streched_value < low_elem:
        #         S[:, max_elem_index] = x_new_streched
        #     else:
        #         S[:, max_elem_index] = x_new
        #     continue
        # elif (low_elem < x_new_value) and (x_new_value < second_max_elem):
        #     S[:, max_elem_index] = x_new
        #     continue
        # elif (second_max_elem < x_new_value) and (x_new_value < max_elem_index):
        #     S[:, max_elem_index] = x_new
        # else:
        #     pass
        #
        # x_s = change_size(center_of_gravity, S[:, max_elem_index], gamma)
        # x_s_value = f(*x_s)
        #
        # if x_s_value < f(*S[:, max_elem_index]):
        #     S[:, max_elem_index] = x_s
        # else:
        #     min_el, index_of_min_el = find_n_max_values(f, S, 1, True)
        #     min_el, index_of_min_el = min_el[0], index_of_min_el[0]
        #
        #     reduction(index_of_min_el, S, n)

        # Step 6
        # ---------------------------------------
        if x_new_value < max_elem:
            S[:, max_elem_index] = x_new

            # Step 7
            # ---------------------------------------
            if x_new_value < low_elem:
                # If ok, perform stretching
                x_new_streched = change_size(center_of_gravity, x_new, betta)
                x_new_streched_value = f(*x_new_streched)

                # Step 8
                # ---------------------------------------
                if x_new_streched_value < x_new_value:
                    # if ok, set x_new_stretched as x_new and go again --> 12
                    S[:, max_elem_index] = x_new_streched
                continue


        # Step 9
        # ---------------------------------------
        if second_max_elem < x_new_value and x_new_value < max_elem:
            x_new_reduced = change_size(center_of_gravity, x_new, gamma)
            x_new_reduced_value = f(*x_new_reduced)
            # Step 10
            # ---------------------------------------
            if x_new_reduced_value < x_new_value:
                # if ok, set x_new_reduced as x_new and go again --> 12
                S[:, max_elem_index] = x_new_reduced
                continue

        # Step 11
        # ---------------------------------------

        min_el, index_of_min_el = find_n_max_values(f, S, 1, True)
        min_el, index_of_min_el = min_el[0], index_of_min_el[0]

        reduction(index_of_min_el, S, n)



    if n == 2:
        plt.show()

    print('Stop condition reached!\n')
    stop_condition(verbose=2, S_arr=S, n=2, epsilon=epsilon)

    min_elem, index_of_min = find_n_max_values(f, S_arr=S, inverse=True)
    min_elem = min_elem[0]
    print('\nIterations completed:', iterations)
    print('Lowest function value: ', round(min_elem, 3))



