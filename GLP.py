def _gcd(m, n):
    while n > 0:
        m, n = n, m % n
    return m


def _prime(run_times):
    H = []
    for i in range(run_times):
        if _gcd(run_times, i) == 1:
            H.append(i)
    return H


def get_GLP(run_times):
    X = []
    for i in range(1, run_times + 1):
        row = []
        for j in _prime(run_times):
            row.append((i * j) % run_times)
        X.append(row)
    return X
