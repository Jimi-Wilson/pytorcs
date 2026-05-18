"""
train_ppo.py  –  Core PPO training infrastructure for TORCS (Docker Native)
"""

import argparse
import atexit
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from callbacks import TerminalLoggingCallback, LapTimeCallback, BestLapSaveCallback

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from torcs_env import TorcsEnv
from utils import get_timestamped_run_dir

DEFAULT_PPO_KWARGS: dict = {
    "policy":        "MlpPolicy",
    "device": "cpu",
    "learning_rate": 3e-4,
    "n_steps":       2048,
    "batch_size":    256,
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_range":    0.2,
    "ent_coef":      0.005,
    "vf_coef":       0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": [256, 256], "activation_fn": nn.Tanh},
    "verbose": 1,
}

DEFAULT_ENV_KWARGS: dict = {
    "sensor_features": ["speedX", "angle", "trackPos", "track"],
    "truncate_limit":  10_000,
    "frame_skip": 1,
}

@dataclass
class RunConfig:
    reward_fn:       Callable
    run_name:        str            = "default_run"
    total_timesteps: int            = 1_000_000
    ppo_overrides:   dict           = field(default_factory=dict)
    env_overrides:   dict           = field(default_factory=dict)
    bc_kwargs:       Optional[dict] = None
    num_envs:        int            = 1
    base_port:       int            = 3001
    callbacks:       Optional[List] = None


def build_default_callbacks(run_dir: Path, run_name: str) -> list:
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000, save_path=str(run_dir / "checkpoints"), name_prefix=run_name
    )
    return [
        checkpoint_cb,
        LapTimeCallback(verbose=1),
        BestLapSaveCallback(save_dir=run_dir / "best_models", verbose=1),
        TerminalLoggingCallback()
    ]


def make_env(port: int, cfg: RunConfig, env_kwargs: dict, monitor_dir: Path):
    from wrappers import FrameSkipWrapper
    def _init():
        skip = env_kwargs.get("frame_skip", 1)
        torcs_kwargs = {k: v for k, v in env_kwargs.items() if k != "frame_skip"}
        env = TorcsEnv(**torcs_kwargs, reward_function=cfg.reward_fn, port=port)
        return Monitor(FrameSkipWrapper(env, skip=skip) if skip > 1 else env, str(monitor_dir))
    return _init


def train(cfg: RunConfig, exp_dir: Path, resume_path: Optional[str] = None) -> None:
    env_kwargs = {**DEFAULT_ENV_KWARGS, **cfg.env_overrides}

    if resume_path:
        checkpoint = Path(resume_path).resolve()
        run_dir = checkpoint.parents[1]
    else:
        run_dir = get_timestamped_run_dir(exp_dir, cfg.run_name)

    for d in ["checkpoints", "best_models", "tensorboard", "monitor"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    print(f"\nTORCS PPO Run: {run_dir.name} | Envs: {cfg.num_envs}")

    sys.path.insert(0, str(_HERE / "docker"))
    import orchestrate
    orchestrate.launch(num_envs=cfg.num_envs, base_port=cfg.base_port)
    atexit.register(orchestrate.stop, num_envs=cfg.num_envs, base_port=cfg.base_port)

    env_fns = [make_env(cfg.base_port + i, cfg, env_kwargs, run_dir / "monitor") for i in range(cfg.num_envs)]
    vec_env = SubprocVecEnv(env_fns) if cfg.num_envs > 1 else DummyVecEnv(env_fns)

    ppo_kwargs = {**DEFAULT_PPO_KWARGS, **cfg.ppo_overrides, "tensorboard_log": str(run_dir / "tensorboard")}

    if resume_path:
        print(f"Resuming target checkpoint: {checkpoint.name}")
        load_kwargs = {k: v for k, v in ppo_kwargs.items() if k not in ("policy", "policy_kwargs")}
        model = PPO.load(str(checkpoint), env=vec_env, **load_kwargs)
    else:
        model = PPO(env=vec_env, **ppo_kwargs)
        if cfg.bc_kwargs:
            from bc import pretrain_with_bc
            model = pretrain_with_bc(model, cfg, exp_dir, run_dir, env_kwargs)

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=cfg.callbacks if cfg.callbacks is not None else build_default_callbacks(run_dir, cfg.run_name),
        reset_num_timesteps=not resume_path,
        tb_log_name=cfg.run_name,
    )
    model.save(str(run_dir / f"{cfg.run_name}_final"))


def load_config(config_path: str) -> tuple[RunConfig, Path]:
    config_path = Path(config_path).resolve()
    if str(config_path.parent) not in sys.path:
        sys.path.insert(0, str(config_path.parent))
    sys.modules.setdefault("train_ppo", sys.modules["__main__"])

    spec = importlib.util.spec_from_file_location("experiment_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cfg, config_path.parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on TORCS")
    parser.add_argument("--config", required=True, help="Path to experiment config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Explicit path to a .zip checkpoint to resume training")
    args = parser.parse_args()

    cfg, exp_dir = load_config(args.config)
    train(cfg, exp_dir, resume_path=args.resume)
