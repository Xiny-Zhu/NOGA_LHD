from ga_l1_l2_single_m import run_ga_for_N

res = run_ga_for_N(
    N=18,
    m=6,
    POP_SIZE=50,
    GENERATIONS=200,
    w_d=0.5,
    w_rho=0.5,
    self_constraint=0.1,
    distance_type="L2",
    seed=18123,
    verbose=True,
    time_limit_sec=300.0,
)

print(res)
