import numpy as np


def _distance(C):
    N = C.shape[0]
    A = []
    for i in range(0, N):
        for j in range(i + 1, N):
            A = np.r_[A, abs(C[i, :] - C[j, :])]
    A = A.reshape((int(N * (N - 1) / 2), C.shape[1]))
    return A


def min_distance(C):
    A = _distance(C)
    sum_A = np.sum(A, axis=1)
    min_distance = np.min(sum_A)
    return min_distance


def fai_p(C, p=50):
    A = _distance(C)
    sum_A = np.sum(A, axis=1)

    unique_elements, counts_elements = np.unique(sum_A, return_counts=True)
    count_A = np.column_stack((unique_elements, counts_elements))
    fai_p = 0
    for i in range(count_A.shape[0]):
        if count_A[i, 0] > 0:
            value = (count_A[i, 0] ** -p) * count_A[i, 1]
            fai_p += value

    return (fai_p) ** (1 / p)


def _distance2(C):
    """
    Return all pairwise L2 distances between rows of C.
    """
    C = np.asarray(C)
    N = C.shape[0]
    A = []
    for i in range(N):
        for j in range(i + 1, N):
            d = C[i, :] - C[j, :]
            A = np.r_[A, np.linalg.norm(d, ord=2)]
    return A


def min_distance2(C):
    """
    Return the minimum pairwise L2 distance between rows of C.
    """
    A = _distance2(C)
    return float(np.min(A)) if A.size > 0 else 0.0
