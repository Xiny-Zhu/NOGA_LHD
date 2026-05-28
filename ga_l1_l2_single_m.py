# -*- coding: utf-8 -*-
# Integrated L1/L2 distance version: choose distance_type="L1" or "L2" when calling.

import time
import random
import numpy as np
from GLP import get_GLP
from GLP import _prime
from transformation import _x_b as xb
from transformation import williamsT as wt
from candidate import candidate
from candidate import candidate_ROU
from candidate import is_valid_column
from coef import coef
from distance import min_distance
from distance import min_distance2



# ============================================================
# 0. Random number utilities
# ============================================================

def _create_rng(seed=None):
    """
    Create an independent random number generator for the current run.

    np.random.seed and random.seed are also set for compatibility with
    legacy functions that may use np.random or random.

    The main GA random operations in this file use rng.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        return np.random.default_rng(seed)
    else:
        return np.random.default_rng()


# ============================================================
# 1. Initial population generation
# ============================================================

def population_individual_xiu(
    columns,
    ROU1,
    k,
    POP_SIZE,
    self_constraint,
    *,
    max_try_per_individual=2000,
    relax_step=0.02,#0.05
    max_constraint=1.0,
    base_mode="random",
    rng=None,
):
    """
    Construct an initial GA population of size POP_SIZE.

    Each individual is a list of length k, i.e., a subset of column indices.
    It satisfies the correlation constraint:
        max correlation between any pair in the subset <= theta.

    Parameters
    ----------
    columns : list[int]
        Candidate column indices.
    ROU1 : np.ndarray
        Precomputed absolute correlation matrix.
    k : int
        Chromosome length.
    POP_SIZE : int
        Population size.
    self_constraint : float
        Initial correlation threshold.
    max_try_per_individual : int
        Max attempts before relaxing theta.
    relax_step : float
        Additive relaxation step.
    max_constraint : float
        Upper bound of theta.
    base_mode : str
        "random" or "cycle".
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    population : list[list[int]]
    """
    if rng is None:
        # Default branch for calls without an explicit rng.
        # In normal use, rng is passed from run_ga_for_N.
        rng = np.random.default_rng()

    columns = list(columns)
    p = len(columns)

    if k > p:
        raise ValueError(f"k={k} cannot exceed number of candidate columns p={p}")

    if base_mode not in ["random", "cycle"]:
        raise ValueError("base_mode must be 'random' or 'cycle'.")

    population = []
    population_set = set()

    theta = float(self_constraint)
    cycle_ptr = 0

    def pick_base():
        nonlocal cycle_ptr

        if base_mode == "cycle":
            b = columns[cycle_ptr % p]
            cycle_ptr += 1
            return int(b)

        return int(rng.choice(columns))

    def build_one(theta_cur):
        for _ in range(max_try_per_individual):
            base = pick_base()
            S = [base]
            used = set(S)

            remaining = [c for c in columns if c not in used]
            rng.shuffle(remaining)

            for cand in remaining:
                if len(S) >= k:
                    break

                if is_valid_column(cand, S, ROU1, theta_cur):
                    S.append(int(cand))
                    used.add(int(cand))

            if len(S) == k:
                key = tuple(sorted(S))
                if key not in population_set:
                    return S, key

        return None, None

    while len(population) < POP_SIZE:
        indiv, key = build_one(theta)

        if indiv is not None:
            population.append(indiv)
            population_set.add(key)
            continue

        if theta >= max_constraint - 1e-12:
            break

        theta = min(theta + relax_step, max_constraint)

    return population


# ============================================================
# 2. Distance criterion utilities: L1 / L2
# ============================================================

def _normalize_distance_type(distance_type):
    """
    Normalize distance_type to "L1" or "L2".
    """
    if isinstance(distance_type, str):
        dt = distance_type.strip().upper()
        if dt in ["L1", "1", "MANHATTAN"]:
            return "L1"
        if dt in ["L2", "2", "EUCLIDEAN"]:
            return "L2"

    if distance_type == 1:
        return "L1"
    if distance_type == 2:
        return "L2"

    raise ValueError("distance_type must be 'L1' or 'L2'.")


def _compute_min_distance(LHD, distance_type="L1"):
    """
    Compute the minimum inter-point distance under the selected criterion.

    distance_type="L1": use min_distance(LHD).
    distance_type="L2": use min_distance2(LHD).
    """
    dt = _normalize_distance_type(distance_type)

    if dt == "L1":
        return min_distance(LHD)

    return min_distance2(LHD)


def _distance_usl(N, k, distance_type="L1"):
    """
    Upper specification limit for distance desirability.

    For L1 distance:
        USL_d = (N + 1) * k // 3

    For L2 distance:
        USL_d = ((N + 1) * k * N // 6) ** 0.5
    """
    dt = _normalize_distance_type(distance_type)

    if dt == "L1":
        return (N + 1) * k // 3

    return ((N + 1) * k * N // 6) ** 0.5


# ============================================================
# 3. Desirability and fitness
# ============================================================

def desirability_larger_better(x, LSL, USL, alpha=1):
    """
    Larger-is-better desirability.
    Used for minimum distance.
    """
    if x <= LSL:
        return 0.0
    elif x >= USL:
        return 1.0
    else:
        return ((x - LSL) / (USL - LSL)) ** alpha


def desirability_smaller_better(x, LSL, USL, alpha=1):
    """
    Smaller-is-better desirability.
    Used for maximum correlation.
    """
    if x <= LSL:
        return 1.0
    elif x >= USL:
        return 0.0
    else:
        return ((USL - x) / (USL - LSL)) ** alpha


def fitness_desirability(
    population,
    C,
    LSL_d,
    USL_d,
    LSL_rho,
    USL_rho,
    w_d,
    w_rho,
    alpha=1,
    distance_type="L1",
):
    """
    Compute desirability-based fitness for each individual.
    """
    fitness_list = []

    for individual in population:
        individual = np.array(individual, dtype=int)
        LHD = C[:, individual]

        d_min_value = _compute_min_distance(LHD, distance_type=distance_type)
        rho_max_value = np.max(coef(LHD))

        d1 = desirability_larger_better(d_min_value, LSL_d, USL_d, alpha)
        d2 = desirability_smaller_better(rho_max_value, LSL_rho, USL_rho, alpha)

        if w_d + w_rho == 0:
            raise ValueError("w_d + w_rho must be positive.")

        D = ((d1 ** w_d) * (d2 ** w_rho)) ** (1 / (w_d + w_rho))

        fitness_list.append(D)

    return np.array(fitness_list, dtype=float)


# ============================================================
# 3. Other individual generation utilities with controlled rng
# ============================================================

def generate_individual(columns, k, population_size, max_attempts=1000, rng=None):
    """
    Randomly generate multiple unique individuals.
    """
    if rng is None:
        rng = np.random.default_rng()

    unique_individuals = set()
    attempts = 0

    columns = list(columns)

    while len(unique_individuals) < population_size and attempts < max_attempts:
        individual = rng.permutation(columns)[:k]
        unique_individuals.add(tuple(int(x) for x in individual))
        attempts += 1

    if len(unique_individuals) < population_size:
        raise ValueError("Unable to generate a sufficient number of unique solutions.")

    initial_population = np.array(list(unique_individuals), dtype=int)
    return initial_population


# ============================================================
# 4. GA operators: selection, crossover, and mutation
# ============================================================

def roulette_wheel_selection(
    population,
    fitness_values,
    C,
    rou_constraint,
    POP_SIZE,
    rng,
):
    """
    Roulette wheel selection with correlation constraint.
    """
    fitness_values = np.array(fitness_values, dtype=float)

    if len(population) == 0:
        raise ValueError("Population is empty in roulette_wheel_selection.")

    ROU = constraint(population, C)
    ROU = np.array(ROU, dtype=float)

    valid_mask = ROU <= rou_constraint

    # Construct selection probabilities
    total_fitness = np.sum(fitness_values)

    if total_fitness <= 0 or not np.isfinite(total_fitness):
        selection_probs = np.ones(len(population), dtype=float) / len(population)
    else:
        selection_probs = fitness_values / total_fitness

    # Set the probabilities of constraint-violating individuals to zero
    selection_probs[~valid_mask] = 0.0

    prob_sum = np.sum(selection_probs)

    # If all probabilities are zero, select uniformly among feasible individuals.
    # If no individual is feasible, select uniformly from the whole population.
    if prob_sum <= 0 or not np.isfinite(prob_sum):
        if np.any(valid_mask):
            selection_probs = valid_mask.astype(float)
            selection_probs = selection_probs / np.sum(selection_probs)
        else:
            selection_probs = np.ones(len(population), dtype=float) / len(population)
    else:
        selection_probs = selection_probs / prob_sum

    selected_indices = rng.choice(
        len(population),
        size=POP_SIZE,
        replace=True,
        p=selection_probs,
    )

    selected_population = [
        np.array(population[int(i)], dtype=int).copy()
        for i in selected_indices
    ]

    return selected_population


def crossover(selected_population, rng):
    """
    Order-based crossover.
    """
    offspring_population = []

    if len(selected_population) == 0:
        return offspring_population

    size = len(selected_population[0])

    for pair_idx in range(0, len(selected_population) - 1, 2):
        individual1 = list(selected_population[pair_idx])
        individual2 = list(selected_population[pair_idx + 1])

        cx_point1 = int(rng.integers(size))
        cx_point2 = int(rng.integers(size - 1))

        if cx_point2 >= cx_point1:
            cx_point2 += 1
        else:
            cx_point1, cx_point2 = cx_point2, cx_point1

        offspring1 = [-1] * size
        offspring2 = [-1] * size

        offspring1[cx_point1:cx_point2 + 1] = individual1[cx_point1:cx_point2 + 1]
        offspring2[cx_point1:cx_point2 + 1] = individual2[cx_point1:cx_point2 + 1]

        idx1 = 0
        idx2 = 0

        for j in range(size):
            if j < cx_point1 or j > cx_point2:
                while individual2[idx1] in offspring1:
                    idx1 += 1
                offspring1[j] = individual2[idx1]
                idx1 += 1

                while individual1[idx2] in offspring2:
                    idx2 += 1
                offspring2[j] = individual1[idx2]
                idx2 += 1

        offspring_population.append(offspring1)
        offspring_population.append(offspring2)

    # If the population size is odd, copy the last individual directly
    if len(selected_population) % 2 == 1:
        offspring_population.append(list(selected_population[-1]).copy())

    return offspring_population


def mutate_population(selected_population, C, mutation_rate, rng):
    """
    Mutation operator.
    """
    new_population = []

    for individual in selected_population:
        individual = list(individual)

        if rng.random() < mutation_rate:
            mutated_individual = individual.copy()

            index_to_mutate = int(rng.integers(len(individual)))
            existing_values = set(mutated_individual)

            while True:
                random_value = int(rng.integers(C.shape[1]))

                if random_value not in existing_values:
                    mutated_individual[index_to_mutate] = random_value
                    break

            new_population.append(mutated_individual)
        else:
            new_population.append(individual.copy())

    return np.array(new_population, dtype=int)


# ============================================================
# 5. Single GA run for a fixed N and m
# ============================================================

def run_ga_for_N(
    N,
    m,
    POP_SIZE=50,
    GENERATIONS=200,
    w_d=0.5,
    w_rho=0.5,
    self_constraint=0.3,
    distance_type="L1",
    seed=None,
    verbose=True,
    time_limit_sec=300.0,
):
    """
    Run GA for a fixed N and m.
    """
    distance_type = _normalize_distance_type(distance_type)
    rng = _create_rng(seed)

    t_start = time.perf_counter()

    k = int(m)

    if k <= 0:
        raise ValueError("m must be a positive integer.")

    X = np.array(get_GLP(N))
    n = X.shape[1]

    C = np.array(candidate(X, N, n))
    ROU1 = np.around(coef(C), 3)

    columns = list(range(C.shape[1]))

    LSL_d = 0
    USL_d = _distance_usl(N, k, distance_type=distance_type)
    LSL_rho = 0
    USL_rho = 1.0

    populations = population_individual_xiu(
        columns,
        ROU1,
        k,
        POP_SIZE,
        self_constraint,
        rng=rng,
    )

    if len(populations) == 0:
        elapsed_sec = time.perf_counter() - t_start
        return {
            "N": int(N),
            "m": int(k),
            "distance_type": distance_type,
            "best_individual": None,
            "maximin_distance": None,
            "max_rho": None,
            "fitness": None,
            "hit_time_limit": False,
            "elapsed_sec": float(elapsed_sec),
            "gens_done": 0,
        }

    global_best_fitness = -np.inf
    global_best_individual = None

    gens_done = 0
    hit_time_limit = False

    for gen in range(GENERATIONS):

        if (time.perf_counter() - t_start) >= time_limit_sec:
            hit_time_limit = True
            break

        fitness_values = fitness_desirability(
            populations,
            C,
            LSL_d,
            USL_d,
            LSL_rho,
            USL_rho,
            w_d,
            w_rho,
            alpha=1,
            distance_type=distance_type,
        )

        current_best_idx = int(np.argmax(fitness_values))
        current_best_fitness = float(fitness_values[current_best_idx])
        current_best_individual = np.array(
            populations[current_best_idx],
            dtype=int,
        ).copy()

        if current_best_fitness > global_best_fitness:
            global_best_fitness = current_best_fitness
            global_best_individual = current_best_individual.copy()

        while True:
            if (time.perf_counter() - t_start) >= time_limit_sec:
                hit_time_limit = True
                break

            selected_population = roulette_wheel_selection(
                populations,
                fitness_values,
                C,
                self_constraint,
                POP_SIZE,
                rng,
            )

            crossed_population = crossover(selected_population, rng)

            next_population = mutate_population(
                crossed_population,
                C,
                mutation_rate=0.8,
                rng=rng,
            )

            next_fitness_values = fitness_desirability(
                next_population,
                C,
                LSL_d,
                USL_d,
                LSL_rho,
                USL_rho,
                w_d,
                w_rho,
                alpha=1,
                distance_type=distance_type,
            )

            if np.max(next_fitness_values) >= current_best_fitness:
                populations = next_population
                fitness_values = next_fitness_values
                break

        gens_done = gen + 1

        if verbose:
            elapsed = time.perf_counter() - t_start
            print(
                f"[N={N}] Gen {gen + 1}, "
                f"best fitness = {global_best_fitness:.6g}, "
                f"elapsed={elapsed:.2f}s"
            )

        if hit_time_limit:
            break

    elapsed_sec = time.perf_counter() - t_start

    if global_best_individual is None:
        return {
            "N": int(N),
            "m": int(k),
            "distance_type": distance_type,
            "best_individual": None,
            "maximin_distance": None,
            "max_rho": None,
            "fitness": None,
            "hit_time_limit": bool(hit_time_limit),
            "elapsed_sec": float(elapsed_sec),
            "gens_done": int(gens_done),
        }

    LHD = C[:, global_best_individual]
    maximin_distance = _compute_min_distance(LHD, distance_type=distance_type)
    max_rho = np.max(coef(LHD))

    return {
        "N": int(N),
        "m": int(k),
        "distance_type": distance_type,
        "best_individual": global_best_individual.copy(),
        "maximin_distance": float(maximin_distance),
        "max_rho": float(max_rho),
        "fitness": float(global_best_fitness),
        "hit_time_limit": bool(hit_time_limit),
        "elapsed_sec": float(elapsed_sec),
        "gens_done": int(gens_done),
    }
