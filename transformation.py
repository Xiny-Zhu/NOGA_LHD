import numpy as np


def _x_b(ndarrayX, b):
    X_b = (ndarrayX + b) % ndarrayX.shape[0]
    return X_b


def williamsT(Xb):
    N = Xb.shape[0]
    n = Xb.shape[1]
    W = np.zeros((N, n))
    for i in range(0, N):
        for j in range(0, n):
            if Xb[i, j] < N / 2:
                W[i, j] = 2 * Xb[i, j]
            else:
                W[i, j] = 2 * (N - Xb[i, j]) - 1
    return W
