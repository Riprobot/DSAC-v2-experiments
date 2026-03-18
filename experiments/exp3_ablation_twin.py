from common import run
from dsac_v2_ablations import DSAC_V2_SingleQ

cfg = dict(max_iteration=100_000, eval_interval=2500, gamma=0.999)

run("dsac-t-experiments", "exp3_twin_ablation", "DSAC_V2_twin_gym_hopper_s42",
    "gym_hopper", seed=42, **cfg)

run("dsac-t-experiments", "exp3_twin_ablation", "DSAC_V2_single_gym_hopper_s42",
    "gym_hopper", alg_cls=DSAC_V2_SingleQ, seed=42, **cfg)
