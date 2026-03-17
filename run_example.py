from ga_experiment import run_ga_for_N


if __name__ == "__main__":
    result = run_ga_for_N(
        N=18,
        k=6,
        POP_SIZE=50,
        GENERATIONS=50,
        self_constraint=0.3,
        seed=42,
        verbose=True,
        time_limit_sec=60.0,
    )
    print(result)
