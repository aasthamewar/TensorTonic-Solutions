import numpy as np

def euclidean_distance(x, y):

    if len(x) != len(y):
        raise ValueError

    sum_sq = 0

    for i in range(len(x)):
        diff = x[i] - y[i]
        sum_sq += diff ** 2

    return np.sqrt(sum_sq)