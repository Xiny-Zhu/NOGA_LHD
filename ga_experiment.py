import time
import random
import numpy as np
import pandas as pd

from GLP import get_GLP, _prime
from candidate import candidate, population_individual_xiu
from coef import coef
from distance import min_distance, min_distance2


def desirability_larger_better(x, LSL, USL, alpha=1):
    if x <= LSL:
        return 0.0
    elif x >= USL:
        return 1.0
    else:
        return ((x - LSL) / (USL - LSL)) ** alpha


def desirability_smaller_better(x, LSL, USL, alpha=1):
    if x <= LSL:
        return 1.0
    elif x >= USL:
        return 0.0
    else:
        return ((USL - x) / (USL - LSL)) ** alpha


def fitness_desirability(population, C, LSL_d, USL_d, LSL_rho, USL_rho, w_d, w_rho, alpha=1):
    fitness_list = []

    for individual in population:
        LHD = C[:, individual]

        d_min_value = min_distance(LHD)
        rho_max_value = np.max(coef(LHD))

        d1 = desirability_larger_better(d_min_value, LSL_d, USL_d, alpha)
        d2 = desirability_smaller_better(rho_max_value, LSL_rho, USL_rho, alpha)

        D = ((d1 ** w_d) * (d2 ** w_rho)) ** (1 / (w_d + w_rho))
        fitness_list.append(D)

    return np.array(fitness_list)


def generate_individual(columns, k, population_size, max_attempts=1000):
    unique_individuals = set()
    attempts = 0

    while len(unique_individuals) < population_size and attempts < max_attempts:
        individual = np.random.permutation(columns)[:k]
        unique_individuals.add(tuple(individual))
        attempts += 1

    if len(unique_individuals) < population_size:
        raise ValueError("Unable to generate a sufficient number of unique solutions.")

    Initial_population = np.array(list(unique_individuals), dtype=int)
    return Initial_population


def constraint(population, C):
    ROU = [np.max(coef(C[:, individual])) for individual in population]
    ROU = np.array(ROU)
    return ROU


def roulette_wheel_selection(population, fitness_values, C, rou_constraint, POP_SIZE):
    fitness_values = np.array(fitness_values)
    total_fitness = np.sum(fitness_values)
    selection_probs = fitness_values / total_fitness

    ROU = constraint(population, C)
    valid_indices = np.where(ROU > rou_constraint)
    selection_probs[valid_indices] = 0

    selection_probs = selection_probs / np.sum(selection_probs)
    selected_indices = np.random.choice(len(population), size=POP_SIZE, p=selection_probs)
    selected_population = [population[i] for i in selected_indices]

    return selected_population


def crossover(selected_population):
    offspring_population = []
    for i in range(0, len(selected_population) - 1, 2):
        individual1 = selected_population[i]
        individual2 = selected_population[i + 1]

        size = len(selected_population[0])
        cx_point1 = np.random.randint(size)
        cx_point2 = np.random.randint(size - 1)
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
        for i in range(size):
            if i < cx_point1 or i > cx_point2:
                while individual2[idx1] in offspring1:
                    idx1 += 1
                offspring1[i] = individual2[idx1]
                idx1 += 1

                while individual1[idx2] in offspring2:
                    idx2 += 1
                offspring2[i] = individual1[idx2]
                idx2 += 1

        offspring_population.append(offspring1)
        offspring_population.append(offspring2)

    return offspring_population


def mutate_population(selected_population, C, mutation_rate):
    mutate_population = []
    for individual in selected_population:
        if np.random.rand() < mutation_rate:
            mutated_individual = individual.copy()
            index_to_mutate = np.random.randint(len(individual))
            existing_values = set(mutated_individual)

            while True:
                random_value = np.random.randint(C.shape[1])
                if random_value not in existing_values:
                    mutated_individual[index_to_mutate] = random_value
                    break

            mutate_population.append(mutated_individual)
        else:
            mutate_population.append(individual)

    return np.array(mutate_population)


def run_ga_for_N(
    N,
    k,
    POP_SIZE=50,
    GENERATIONS=200,
    w_d=0.5,
    w_rho=0.5,
    self_constraint=0.3,
    seed=None,
    verbose=True,
    time_limit_sec=300.0,
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    t_start = time.perf_counter()

    
    X = np.array(get_GLP(N))
    n = X.shape[1]

    C = np.array(candidate(X, N, n))
    ROU1 = np.around(coef(C), 3)

    columns = list(range(C.shape[1]))

    LSL_d = 0
    USL_d = (N + 1) * k // 3
    LSL_rho = 0.05
    USL_rho = 1.0

    populations = population_individual_xiu(columns, ROU1, k, POP_SIZE, self_constraint)

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
        )

        current_best_fitness = float(np.max(fitness_values))
        current_best_individual = populations[int(np.argmax(fitness_values))]

        if current_best_fitness > global_best_fitness:
            global_best_fitness = current_best_fitness
            global_best_individual = current_best_individual

        while True:
            if (time.perf_counter() - t_start) >= time_limit_sec:
                hit_time_limit = True
                break

            selected_population = roulette_wheel_selection(
                populations, fitness_values, C, self_constraint, POP_SIZE
            )

            crossed_population = crossover(selected_population)
            next_population = mutate_population(crossed_population, C, mutation_rate=0.8)

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
                f"best fitness = {global_best_fitness:.6g}, elapsed={elapsed:.2f}s"
            )

        if hit_time_limit:
            break

    elapsed_sec = time.perf_counter() - t_start

    if global_best_individual is None:
        return {
            "N": N,
            "best_individual": None,
            "maximin_distance": None,
            "maximin_distance_l2": None,
            "max_rho": None,
            "fitness": None,
            "generations_done": gens_done,
            "hit_time_limit": hit_time_limit,
            "elapsed_sec": elapsed_sec,
        }

    LHD = C[:, global_best_individual]
    Maximin_distance = min_distance(LHD)
    Maximin_distance2 = min_distance2(LHD)
    Max_ROU = np.max(coef(LHD))

    return {
        "N": N,
        "best_individual": global_best_individual,
        "maximin_distance": float(Maximin_distance),
        "maximin_distance_l2": float(Maximin_distance2),
        "max_rho": float(Max_ROU),
        "fitness": float(global_best_fitness),
        "generations_done": gens_done,
        "hit_time_limit": hit_time_limit,
        "elapsed_sec": elapsed_sec,
    }


def batch_run_to_csv(
    N_min=7,
    N_max=30,
    csv_name="GA_results_mean.csv",
    POP_SIZE=50,
    GENERATIONS=200,
    w_d=0.5,
    w_rho=0.5,
    self_constraint=0.3,
    time_limit_sec=300.0,
    seed=42,
):
    records = []

    for N in range(N_min, N_max + 1):
        print(f"========== Running N={N} ==========")
        result = run_ga_for_N(
            N=N,
            k=len(_prime(N)),
            POP_SIZE=POP_SIZE,
            GENERATIONS=GENERATIONS,
            w_d=w_d,
            w_rho=w_rho,
            self_constraint=self_constraint,
            seed=seed,
            verbose=True,
            time_limit_sec=time_limit_sec,
        )
        records.append(result)

    df = pd.DataFrame(records)
    df.to_csv(csv_name, index=False)
    print(f"\nResults saved to {csv_name}")
    return df
