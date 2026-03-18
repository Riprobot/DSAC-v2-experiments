from common import run

ENVS = {
    "gym_pendulum":    dict(max_iteration=8_000,   eval_interval=200,  buffer_warm_size=1000, log_save_interval=100),
    "gym_hopper":      dict(max_iteration=100_000, eval_interval=2500, gamma=0.999),
    "gym_halfcheetah": dict(max_iteration=100_000, eval_interval=2500),
}

for env_id in ENVS:
    for alg in ["DSAC_V1", "DSAC_V2"]:
        run("dsac-t-experiments", f"exp1_{env_id}", f"{alg}_{env_id}_s42",
            env_id, algorithm=alg, seed=42, **ENVS[env_id])
