import numpy as np

def manhattan_distance(x, y):
    z = 0
    for i in range(len(x)):
        z += np.abs(x[i] - y[i])
    return float(z)