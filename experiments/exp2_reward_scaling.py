from common import run

for alg in ["DSAC_V1", "DSAC_V2"]:
    for rs in [0.01, 0.1, 1.0, 10.0]:
        run("dsac-t-experiments", f"exp2_rs_{alg}", f"{alg}_rs{rs}_s42",
            "gym_halfcheetah", algorithm=alg, seed=42,
            max_iteration=100_000, eval_interval=2500, reward_scale=rs)
