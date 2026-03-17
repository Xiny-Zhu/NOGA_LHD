import numpy as np
from transformation import williamsT as wt
from transformation import _x_b as xb


def candidate(X, N, p):
    if N % 2 == 0:
        C = wt(xb(X, 0))
        for b in range(1, int(N / 2)):
            Z = wt(xb(X, b))
            C = np.hstack((C, Z))
    elif ((N - 1) / 2) % 2 == 0:
        b_star = int((N - 1) / 4)
        Z = wt(xb(X, b_star))
        C = Z[:, : int(p / 2)]
        for b in range(0, b_star):
            Z = wt(xb(X, b))
            C = np.hstack((C, Z))
        for b in range(int((N - 1) / 2) + 1, N - 1):
            Z = wt(xb(X, b))
            C = np.hstack((C, Z))
    else:
        b_star = (3 * N - 1) / 4
        Z = wt(xb(X, b_star))
        C = Z[:, : int(p / 2)]
        for b in range(0, int(((N - 1) / 2 + 1) / 2)):
            Z = wt(xb(X, b))
            C = np.hstack((C, Z))
        for b in range(int((N - 1) / 2) + 1, N):
            Z = wt(xb(X, b))
            C = np.hstack((C, Z))
    return C


def is_valid_column(column_index, combination, ROU, self_constraint):
    for index in combination:
        if ROU[index, column_index] >= self_constraint:
            return False
    return True


def population_individual_xiu(
    columns,
    ROU1,
    k,
    POP_SIZE,
    self_constraint,
    *,
    max_try_per_individual=2000,
    relax_step=0.02,
    max_constraint=1.0,
    base_mode="random",  
    rng=None,
):
    """
    Construct an initial GA population of size POP_SIZE.
    Each individual is a list of length k (subset of column indices) that satisfies
    the correlation constraint: max corr between any pair in the subset <= threshold.

    Parameters
    ----------
    columns : list[int]
        Candidate column indices (typically 0..p-1).
    ROU1 : np.ndarray
        Precomputed absolute correlation matrix (or something equivalent used in is_valid_column).
    k : int
        Design size (chromosome length), i.e., number of columns in each individual.
    POP_SIZE : int
        GA population size, i.e., number of individuals to generate.
    self_constraint : float
        Initial correlation threshold θ (max allowed correlation).
    max_try_per_individual : int
        Max attempts before relaxing θ.
    relax_step : float
        Additive relaxation step Δθ when not enough feasible individuals can be found.
    max_constraint : float
        Upper bound of θ.
    base_mode : str
        "random": base column chosen randomly each attempt
        "cycle": base column cycles through columns deterministically
    rng : np.random.Generator or None
        Random number generator; if None, use np.random.default_rng().

    Returns
    -------
    population : list[list[int]]
        List of individuals, each a list of k column indices.
        If strict constraints make it impossible, θ will be relaxed gradually up to max_constraint.
    """
    if rng is None:
        rng = np.random.default_rng()

    columns = list(columns)
    p = len(columns)
    if k > p:
        raise ValueError(f"k={k} cannot exceed number of candidate columns p={p}")

    population = []
    population_set = set()  # for uniqueness (store sorted tuple)

    theta = float(self_constraint)
    cycle_ptr = 0

    def pick_base():
        nonlocal cycle_ptr
        if base_mode == "cycle":
            b = columns[cycle_ptr % p]
            cycle_ptr += 1
            return b
        # default random
        return int(rng.choice(columns))

    # Helper: try to build one feasible individual under current theta
    def build_one(theta_cur):
        # attempt multiple times, each time with a different random base and scan order
        for _ in range(max_try_per_individual):
            base = pick_base()
            S = [base]
            used = set(S)

            # random scan order over remaining candidates
            remaining = [c for c in columns if c not in used]
            rng.shuffle(remaining)

            for cand in remaining:
                if len(S) >= k:
                    break
                if is_valid_column(cand, S, ROU1, theta_cur):
                    S.append(cand)
                    used.add(cand)

            if len(S) == k:
                # canonical form for dedup
                key = tuple(sorted(S))
                if key not in population_set:
                    return S, key
        return None, None

    # Main loop: fill population, relaxing theta if needed
    while len(population) < POP_SIZE:
        indiv, key = build_one(theta)
        if indiv is not None:
            population.append(indiv)
            population_set.add(key)
            continue

        # Could not find a new feasible individual under current theta => relax
        if theta >= max_constraint - 1e-12:
            # already fully relaxed; cannot generate more unique individuals
            break
        theta = min(theta + relax_step, max_constraint)

    return population
