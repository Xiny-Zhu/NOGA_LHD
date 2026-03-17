import numpy as np


def coef(Z):
    N = Z.shape[0]
    n = Z.shape[1]
    rou_r = []
    R = np.zeros((n, n))
    for j in range(0, n):
        for i in range(j, n):
            rou_1 = np.dot(Z[:, j], Z[:, i])
            rou_2 = 12 * rou_1 / (N * (N * N - 1)) - 3 * (N - 1) / (N + 1)
            rou_r.append(rou_2)
    abs_rou_r = np.abs(rou_r)
    for i in range(0, n):
        for j in range(0, n):
            if i < j:
                R[i, j] = abs_rou_r[int(i * n - i * (i - 1) / 2 + (j - i))]
            elif i > j:
                R[i, j] = R[j, i]
            else:
                R[i, j] = 0
    return R
