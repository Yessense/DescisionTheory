import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

n = 2
length = 8.
epsilon = 0.1

start_vertex = np.array([1, 1])
print('Start vertex:', start_vertex)


def f(x_1, x_2):
    out = 2.9 * x_1 ** 2 + 0.8 * x_1 * x_2
    out += 3.3 * x_2 ** 2 - 1.5 * x_1 + 3.1 * x_2
    return out


def delta(n: float, length: float, i_equals_j):
    out = (n + 1) ** 0.5 - 1

    if i_equals_j:
        out += n

    out /= n * 2 ** 0.5
    out *= length
    return out


S = np.zeros((n, n + 1))

S[:] = start_vertex[:, np.newaxis]

d1 = delta(n, length, True)
d2 = delta(n, length, False)

for i in range(n):
    for j in range(1, n + 1):
        if i + 1 == j:
            S[i, j] += d1
        else:
            S[i, j] += d2


# print('Coordinates:')
# print(S)


def find_max_value(f, inverse=False, verbose=0):
    values = [f(*X) for X in S.T]

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


def find_center_of_gravity(index_to_remove: int, verbose=0):
    to_take = np.ones(n + 1).astype(bool)

    if index_to_remove != -1:
        to_take[index_to_remove] = False

    center_of_gravity = np.average(S[:, to_take], axis=1)

    if verbose == 1:
        print('Columns where to find center of gravity', to_take)
        print('Array without this index')
        print(S[:, to_take])
        print('Center of gravity:', center_of_gravity)
    return center_of_gravity


def find_reflection(x, center_of_gravity):
    return 2 * center_of_gravity - x


def stop_condition(verbose=0):
    center_of_gravity = find_center_of_gravity(-1)

    if verbose == 1:
        print('Center of gravity: ', center_of_gravity)

    absolute = S - center_of_gravity[:, np.newaxis]

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


def reduction(index_of_min, verbose=0):
    to_take = np.ones(n + 1).astype(bool)
    to_take[index_of_min] = False

    x_r = S[:, index_of_min]

    if verbose == 1:
        print('To take:', to_take)
        print('X_r', x_r)

    S[:, to_take] = x_r[:, np.newaxis] + 0.5 * (S[:, to_take] - x_r[:, np.newaxis])


x_line = np.arange(-10, 10, 0.05)
y_line = np.arange(-10, 10, 0.05)
x_grid, y_grid = np.meshgrid(x_line, y_line, sparse=True)
function_values = f(x_grid, y_grid)

plt.pcolormesh(x_line, y_line, function_values, cmap=cm.get_cmap('inferno_r'), alpha=0.8)

iterations = 0
verbose = 1
while not stop_condition():
    iterations += 1
    max_elem, index_of_max = find_max_value(f, verbose=1)

    horisontal = S[0]
    vertical = S[1]

    plt.plot(horisontal[[0,1]], vertical[[0,1]], color='black')
    plt.plot(horisontal[[1,2]], vertical[[1,2]], color='black')
    plt.plot(horisontal[[0,2]], vertical[[0,2]], color='black')

    if verbose == 1:
        print('----------------------')
        print('Iteration:', iterations)
        print('Max element: ', max_elem)

    center_of_gravity = find_center_of_gravity(index_of_max, verbose=1)
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
        minimum, index_of_min = find_max_value(f, True)
        reduction(index_of_min, verbose=0)
        print('Performing reduction')
    print('----------------------')

plt.show()

print('Stop condition reached!\n')
stop_condition(verbose=2)

min_elem, index_of_min = find_max_value(f, inverse=True)
print('\nIterations completed:', iterations)
print('Lowest function value: ', min_elem)

if __name__ == '__main__':
    # gg = np.arange(plt)
    pass
