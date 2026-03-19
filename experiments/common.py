import sys
from pathlib import Path

import wandb
from torch.utils.tensorboard import SummaryWriter

DSAC_ROOT = Path(__file__).resolve().parent.parent / "DSAC-v2"
for _p in [str(DSAC_ROOT), str(DSAC_ROOT / "utils"), str(DSAC_ROOT / "env_gym")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.initialization import create_env, create_alg, create_buffer
from training.off_sampler import create_sampler
from training.evaluator import create_evaluator
from training.trainer import create_trainer
from utils.init_args import init_args


class WandbWriter(SummaryWriter):
    """SummaryWriter that also sends every scalar to wandb.
    Skips tags where step != iteration (replay-sample/time axes).
    Renames tags to match old wandb format."""
    _SKIP = {"Evaluation/2.", "Evaluation/3.", "Evaluation/4."}
    _RENAME = {"Evaluation/1. TAR-RL iter": "eval/mean_return"}

    def add_scalar(self, tag, value, global_step=None, **kw):
        super().add_scalar(tag, value, global_step, **kw)
        if not any(tag.startswith(s) for s in self._SKIP):
            wtag = self._RENAME.get(tag, tag.replace("/", "_").replace("-", "_"))
            wandb.log({wtag: value}, step=global_step)


def build_args(env_id, algorithm="DSAC_V2", seed=42,
               max_iteration=100_000, **overrides):
    args = dict(
        env_id=env_id, algorithm=algorithm, seed=seed,
        enable_cuda=True, trainer="off_serial_trainer", action_type="continu",
        value_func_name="ActionValueDistri", value_func_type="MLP",
        value_hidden_sizes=[256, 256, 256], value_hidden_activation="gelu",
        value_output_activation="linear", value_min_log_std=-8, value_max_log_std=8,
        policy_func_name="StochaPolicy", policy_func_type="MLP",
        policy_act_distribution="TanhGaussDistribution",
        policy_hidden_sizes=[256, 256, 256], policy_hidden_activation="gelu",
        policy_output_activation="linear", policy_min_log_std=-20, policy_max_log_std=0.5,
        value_learning_rate=1e-4, policy_learning_rate=1e-4, alpha_learning_rate=3e-4,
        sample_batch_size=20, buffer_warm_size=10_000, buffer_max_size=1_000_000,
        buffer_name="replay_buffer", replay_batch_size=256,
        max_iteration=max_iteration, eval_interval=2500, log_save_interval=500,
        apprfunc_save_interval=50_000, num_eval_episode=10, sample_interval=1,
        gamma=0.99, tau=0.005, auto_alpha=True, alpha=0.2, delay_update=2,
        TD_bound=20.0, bound=True,
        reward_scale=1.0, reward_shift=None, max_episode_steps=None,
        save_folder=None, ini_network_dir=None, is_render=False, noise_params=None,
    )
    args.update(overrides)
    return args


def run(project, group, name, env_id, algorithm="DSAC_V2",
        alg_cls=None, seed=42, max_iteration=100_000, **overrides):
    args = build_args(env_id, algorithm, seed, max_iteration, **overrides)
    env = create_env(**args)
    args = init_args(env, **args)

    alg = alg_cls(**args) if alg_cls else create_alg(**args)
    sampler = create_sampler(**args)
    buffer = create_buffer(**args)
    evaluator = create_evaluator(**args)

    safe = {k: v for k, v in args.items()
            if isinstance(v, (int, float, str, bool, type(None)))}
    wandb.init(project=project, group=group, name=name,
               config=safe, reinit=True)

    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)
    trainer.writer.close()
    trainer.writer = WandbWriter(log_dir=args["save_folder"], flush_secs=20)
    trainer.train()
    wandb.finish()
