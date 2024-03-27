import numpy as np


# f2_bounds = [-100, 100]
# rastrigin_bounds = [-5.12, 5.12]
# sphere_bounds = [-100, 100]
# griewank_bounds = [-600, 600]

def sphere(x):
    return sum([i ** 2 for i in x])


def rastrigin_function(x):
    n = len(x)
    sum_values = 0
    for i in range(n):
        sum_values += x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]) + 10
    return sum_values


def f2_function(x):
    sum_values = 0
    for i in range(len(x)):
        sum_values += ((x[i] - i) ** 2)
    return sum_values


def griewank_function(x):
    n = len(x)
    sum_term = np.sum(np.square(x) / 4000)
    product_term = np.prod(np.cos(np.array(x) / np.sqrt(range(1, n + 1))))
    return 1 + sum_term - product_term
