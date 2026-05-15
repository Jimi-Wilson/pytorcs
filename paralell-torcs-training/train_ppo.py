"""
train_ppo.py  –  Modular PPO training infrastructure for TORCS
==============================================================

Usage
-----
  # New run
  python train_ppo.py --config experiments/corkscrew_v1/config-1.py

  # Resume from latest checkpoint
  python train_ppo.py --config experiments/corkscrew_v1/config-1.py --resume

  # Evaluate best saved model
  python train_ppo.py --config experiments/corkscrew_v1/config-1.py --eval [--episodes N]

How configs work
----------------
Each experiment lives in its own folder under experiments/.
The config-1.py in that folder creates a single `cfg = RunConfig(...)` instance.
Only override what you want to change — everything else falls back to the defaults
defined here (DEFAULT_PPO_KWARGS, DEFAULT_ENV_KWARGS).

Docker / multi-env
------------------
Set  use_docker=True  and  num_envs=N  in RunConfig.
train_ppo.py will call orchestrate.launch() at startup and orchestrate.stop()
on exit (even on crash, via atexit). Each env gets its own container and port.
"""

import atexit
import datetime
import importlib.util
import math
import os
import sys
import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch._C import Stream

# Adjust path so we can import gym_torcs modules regardless of cwd
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from torcs_env import TorcsEnv
from utils import kill_torcs_instance


# ─────────────────────────────────────────────────────────────────────────────
#  Defaults  (override via RunConfig.ppo_overrides / env_overrides)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PPO_KWARGS: dict = {
    "policy":        "MlpPolicy",
    "device":        "cpu",       # MlpPolicy has poor GPU utilisation
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
    "policy_kwargs": {
        "net_arch":      [256, 256],
        "activation_fn": nn.Tanh,
    },
    "verbose": 1,
}

DEFAULT_ENV_KWARGS: dict = {
    "sensor_features": ["speedX", "angle", "trackPos", "track"],
    "truncate_limit":  10_000,
    "frame_skip":      1,          # 1 = no skipping; set >1 to repeat actions
}


# ─────────────────────────────────────────────────────────────────────────────
#  RunConfig  –  the contract between train_ppo.py and each experiment config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    """
    Defines a training experiment. Create one instance called `cfg` in your
    experiments/<name>/config-1.py file.

    Required
    --------
    reward_fn : callable(previous_sensors, current_sensors) -> float

    Optional (all have sensible defaults)
    ------
    total_timesteps : int
        Total PPO environment steps.
    ppo_overrides : dict
        Keys/values to merge on top of DEFAULT_PPO_KWARGS.
    env_overrides : dict
        Keys/values to merge on top of DEFAULT_ENV_KWARGS.
    bc_kwargs : dict or None
        Behavioural-cloning settings. Set to None to skip BC entirely.
        Expected keys:
            teacher_policy      : callable(obs) -> action
            collection_episodes : int
            only_completed_laps : bool
            epochs              : int
            batch_size          : int
            lr                  : float
    num_envs : int
        Number of parallel TORCS environments. >1 requires use_docker=True.
    use_docker : bool
        True  = docker path (orchestrate.launch / kill_torcs_on for resets)
        False = local TORCS path (kill_torcs_instance for resets)
    base_port : int
        First UDP port. Docker envs use base_port+0 … base_port+(num_envs-1).
    callbacks : list or None
        If set, replaces the default callback list entirely.
    """

    reward_fn:       Callable
    run_name:        str            = "default_run"
    total_timesteps: int            = 1_000_000
    ppo_overrides:   dict           = field(default_factory=dict)
    env_overrides:   dict           = field(default_factory=dict)
    bc_kwargs:       Optional[dict] = None
    num_envs:        int            = 1
    use_docker:      bool           = False
    base_port:       int            = 3001
    callbacks:       Optional[List] = None
    resume_path:     str            = None


# ─────────────────────────────────────────────────────────────────────────────
#  Callbacks
# ─────────────────────────────────────────────────────────────────────────────

class LapTimeCallback(BaseCallback):
    """Logs every completed lap time to TensorBoard and stdout."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.best_lap = float("inf")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            lt = info.get("lap_time", 0)
            if lt and lt > 0:
                self.logger.record("custom/lap_time", lt)
                if lt < self.best_lap:
                    self.best_lap = lt
                    self.logger.record("custom/best_lap_time", lt)
                    print(f"\n  🏁 New best lap: {lt:.2f}s  (step {self.num_timesteps})")
        return True


class BestLapSaveCallback(BaseCallback):
    """
    Saves the model whenever a new best lap time is recorded.

    Multi-env safe: inspects info dicts from all envs each step without
    maintaining a shared episode buffer (which would mix trajectories
    across envs and corrupt the data).
    """

    def __init__(self, save_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.best_lap = float("inf")

    def _on_training_start(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        for done, info in zip(
            self.locals.get("dones", [False]),
            self.locals.get("infos", [{}]),
        ):
            if not done:
                continue

            lt = info.get("lap_time", 0)
            if lt and lt > 0 and lt < self.best_lap:
                self.best_lap = lt
                model_path = self.save_dir / f"best_lap_{lt:.2f}s_step{self.num_timesteps}"
                self.model.save(str(model_path))
                print(f"\n  🏆 New best lap: {lt:.2f}s → {model_path.name}.zip")

        return True


def build_default_callbacks(exp_dir: Path, run_name: str) -> list:
    """Returns the standard callback list, all writing into exp_dir."""
    checkpoint_dir = exp_dir / f"{run_name}/checkpoints"
    best_models_dir = exp_dir / f"{run_name}/best_models"

    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = str(checkpoint_dir),
        name_prefix = run_name,
    )
    lap_time_cb = LapTimeCallback(verbose=1)
    lap_save_cb = BestLapSaveCallback(save_dir=best_models_dir, verbose=1)

    return [checkpoint_cb, lap_time_cb, lap_save_cb]


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint / model helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(exp_dir: Path) -> Optional[Path]:
    """Returns the most recently modified .zip in exp_dir/checkpoints/, or None."""
    checkpoints = list((exp_dir / "checkpoints").glob("*.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def find_best_model(exp_dir: Path) -> Optional[Path]:
    """
    Returns the best-lap model from exp_dir/best_models/.
    Prefers the file with the lowest lap time embedded in the filename
    (pattern: best_lap_<time>s_step<N>.zip).
    Falls back to the most recently modified .zip if no time can be parsed.
    """
    candidates = list((exp_dir / "best_models").glob("*.zip"))
    if not candidates:
        return None

    def _lap_time(p: Path) -> float:
        try:
            # filename: best_lap_111.10s_step676119.zip
            return float(p.stem.split("best_lap_")[1].split("s_step")[0])
        except (IndexError, ValueError):
            return float("inf")

    return min(candidates, key=_lap_time)


# ─────────────────────────────────────────────────────────────────────────────
#  Behavioural Cloning
# ─────────────────────────────────────────────────────────────────────────────

def collect_bc_data(
    cfg: RunConfig,
    exp_dir: Path,
    env_kwargs: dict,
    reset_fn: Callable,
) -> Path:
    """
    Rolls out cfg.bc_kwargs['teacher_policy'] for N episodes and saves
    transitions to exp_dir/expert_data.npz. Returns the path.
    """
    bc = cfg.bc_kwargs
    teacher      = bc["teacher_policy"]
    episodes     = bc.get("collection_episodes", 30)
    only_done    = bc.get("only_completed_laps", True)

    print(f"\n{'='*60}")
    print(f"  BC data collection — {episodes} episodes")
    print(f"{'='*60}\n")

    # Build env through make_env so frame skipping (and any future wrappers)
    # are applied consistently with RL training.
    env = make_env(
        port        = cfg.base_port,
        cfg         = cfg,
        env_kwargs  = env_kwargs,
        reset_fn    = reset_fn,
        monitor_dir = exp_dir / "monitor",
    )()

    all_obs, all_actions = [], []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = truncated = False
        ep_obs, ep_actions = [], []

        while not (done or truncated):
            action = teacher(obs)
            ep_obs.append(obs.copy())
            ep_actions.append(np.array(action, dtype=np.float32))
            obs, _, done, truncated, info = env.step(action)

        completed = info.get("lap_time", 0) and info["lap_time"] > 0
        status    = "✓" if completed else "✗"
        print(f"  Episode {ep+1:3d}/{episodes}  {status}  dist={info.get('dist_raced',0):.0f}m")

        if not only_done or completed:
            all_obs.extend(ep_obs)
            all_actions.extend(ep_actions)

    expert_path = exp_dir / "expert_data.npz"
    np.savez_compressed(
        expert_path,
        observations = np.array(all_obs,     dtype=np.float32),
        actions      = np.array(all_actions, dtype=np.float32),
    )
    print(f"\n  Saved {len(all_obs):,} transitions → {expert_path}\n")
    return expert_path


def pretrain_bc(model: PPO, expert_path: Path, bc_kwargs: dict) -> PPO:
    import torch
    import torch.nn as nn
    """Warm-starts the PPO actor with behavioural cloning on expert transitions."""
    epochs     = bc_kwargs.get("epochs", 20)
    batch_size = bc_kwargs.get("batch_size", 256)
    lr         = bc_kwargs.get("lr", 1e-3)

    print(f"\n{'='*60}")
    print(f"  Behavioural Cloning pre-training from {expert_path.name}")
    print(f"{'='*60}")

    data       = np.load(expert_path)
    obs_t      = torch.tensor(data["observations"].astype(np.float32))
    actions_t  = torch.tensor(data["actions"].astype(np.float32))
    print(f"  Expert transitions : {len(obs_t):,}")
    print(f"  obs shape          : {obs_t.shape}")
    print(f"  actions shape      : {actions_t.shape}\n")

    dataset = torch.utils.data.TensorDataset(obs_t, actions_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Only train the actor (policy net + action head), not the value head
    actor_params = (
        list(model.policy.mlp_extractor.policy_net.parameters()) +
        list(model.policy.action_net.parameters())
    )
    optimizer = torch.optim.Adam(actor_params, lr=lr)
    loss_fn   = nn.MSELoss()

    model.policy.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(model.device)
            act_batch = act_batch.to(model.device)

            features     = model.policy.extract_features(obs_batch)
            latent_pi, _ = model.policy.mlp_extractor(features)
            mean_actions = model.policy.action_net(latent_pi)

            loss = loss_fn(mean_actions, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  BC Epoch [{epoch+1:02d}/{epochs}]  loss={total_loss / len(loader):.5f}")

    print("\n  BC pre-training complete.\n")
    model.policy.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_reset_fn(use_docker: bool) -> Callable:
    """Returns the appropriate TORCS kill function for local vs docker."""
    if use_docker:
        # Import here so docker dependency is only required when actually used
        sys.path.insert(0, str(_HERE / "docker"))
        from orchestrate import kill_torcs_on
        return kill_torcs_on
    return kill_torcs_instance


def make_env(port: int, cfg: RunConfig, env_kwargs: dict, reset_fn: Callable, monitor_dir: Path):
    """Returns a thunk (no-arg callable) that constructs a monitored TorcsEnv.

    frame_skip is consumed here and never forwarded to TorcsEnv.
    The FrameSkipWrapper sits between TorcsEnv and Monitor so that Monitor
    records metrics at the agent-step rate, not the raw tick rate.
    """
    from wrappers import FrameSkipWrapper

    def _init():
        skip = env_kwargs.get("frame_skip", 1)
        torcs_kwargs = {k: v for k, v in env_kwargs.items() if k != "frame_skip"}
        env = TorcsEnv(
            **torcs_kwargs,
            reward_function = cfg.reward_fn,
            reset_fn        = reset_fn,
            port            = port,
        )
        if skip > 1:
            env = FrameSkipWrapper(env, skip=skip)
        return Monitor(env, str(monitor_dir))
    return _init


# ─────────────────────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: RunConfig, exp_dir: Path, resume: bool = False) -> None:
    run_name   = cfg.run_name
    env_kwargs = {**DEFAULT_ENV_KWARGS, **cfg.env_overrides}
    reset_fn   = _make_reset_fn(cfg.use_docker)

    # ── Create output directories ──────────────────────────────────────────
    for d in ["checkpoints", "best_models", "tensorboard", "monitor"]:
        (exp_dir / d).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TORCS PPO — {run_name}")
    print(f"  Envs: {cfg.num_envs}  |  Docker: {cfg.use_docker}")
    print(f"{'='*60}\n")

    # ── Docker lifecycle ───────────────────────────────────────────────────
    if cfg.use_docker:
        # Import orchestrate from docker/ folder inside gym_torcs/
        docker_dir = _HERE / "docker"
        sys.path.insert(0, str(docker_dir))
        import orchestrate

        print(f"  Launching {cfg.num_envs} TORCS container(s) (ports {cfg.base_port}–{cfg.base_port + cfg.num_envs - 1})…")
        orchestrate.launch(num_envs=cfg.num_envs, base_port=cfg.base_port)
        atexit.register(orchestrate.stop, num_envs=cfg.num_envs, base_port=cfg.base_port)
        print("  Containers up.\n")

    # ── Build vectorised environment ───────────────────────────────────────
    env_fns = [
        make_env(
            port        = cfg.base_port + i,
            cfg         = cfg,
            env_kwargs  = env_kwargs,
            reset_fn    = reset_fn,
            monitor_dir = exp_dir / "monitor",
        )
        for i in range(cfg.num_envs)
    ]

    if cfg.num_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
        print(f"  Using SubprocVecEnv with {cfg.num_envs} workers.\n")
    else:
        vec_env = DummyVecEnv(env_fns)
        print("  Using DummyVecEnv (single env).\n")

    # ── Build / load model ─────────────────────────────────────────────────
    ppo_kwargs = {
        **DEFAULT_PPO_KWARGS,
        **cfg.ppo_overrides,
        "tensorboard_log": str(exp_dir / "tensorboard"),
    }

    if resume:
        if cfg.resume_path is not None:
            print(f"  Resuming from {cfg.resume_path}\n")
            load_kwargs = {k: v for k, v in ppo_kwargs.items() if k not in ("policy", "policy_kwargs")}
            model = PPO.load(str(cfg.resume_path), env=vec_env, **load_kwargs)

        else:

            checkpoint = find_latest_checkpoint(exp_dir)
            if checkpoint is None:
                raise FileNotFoundError(
                    f"--resume specified but no checkpoint found in {exp_dir / 'checkpoints'}"
                )
            print(f"  Resuming from {checkpoint.name}\n")
            # Don't pass policy/policy_kwargs when loading — they're frozen in the zip
            load_kwargs = {k: v for k, v in ppo_kwargs.items() if k not in ("policy", "policy_kwargs")}
            model = PPO.load(str(checkpoint), env=vec_env, **load_kwargs)
    else:
        model = PPO(env=vec_env, **ppo_kwargs)

    # ── Behavioural Cloning warm-start (skipped on resume) ────────────────
    if cfg.bc_kwargs and not resume:
        expert_path = exp_dir / "expert_data.npz"
        if not expert_path.exists():
            expert_path = collect_bc_data(cfg, exp_dir, env_kwargs, reset_fn)
        else:
            print(f"  Found existing expert data at {expert_path.name} — skipping collection.\n")
        model = pretrain_bc(model, expert_path, cfg.bc_kwargs)
    elif not resume:
        print("  [INFO] bc_kwargs is None — skipping BC pre-training.\n")

    # ── Callbacks ──────────────────────────────────────────────────────────
    callbacks = cfg.callbacks if cfg.callbacks is not None else build_default_callbacks(exp_dir, run_name)

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"  Starting PPO for {cfg.total_timesteps:,} steps")
    print(f"  Best models  → {exp_dir / 'best_models'}")
    print(f"  Checkpoints  → {exp_dir / 'checkpoints'}")
    print(f"  TensorBoard  : tensorboard --logdir {exp_dir / 'tensorboard'}\n")

    start_time = time.time()
    model.learn(
        total_timesteps     = cfg.total_timesteps,
        callback            = callbacks,
        reset_num_timesteps = not resume,
        tb_log_name         = run_name,
    )

    final_path = exp_dir / f"{run_name}_final"
    model.save(str(final_path))
    elapsed_seconds = time.time() - start_time
    formatted_time = str(datetime.timedelta(seconds=int(elapsed_seconds)))
    print(f"\n  Training complete in {formatted_time}. Final model → {final_path}.zip")


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(cfg: RunConfig, exp_dir: Path, episodes: int = 5) -> None:
    env_kwargs = {**DEFAULT_ENV_KWARGS, **cfg.env_overrides}
    reset_fn   = _make_reset_fn(cfg.use_docker)

    best_model = find_best_model(exp_dir)
    if best_model is None:
        raise FileNotFoundError(f"No model found in {exp_dir / 'best_models'}")

    print(f"\n  Evaluating {best_model.name} for {episodes} episodes\n")

    # Use make_env so frame skipping is applied identically to training.
    from wrappers import FrameSkipWrapper
    skip = env_kwargs.get("frame_skip", 1)
    torcs_kwargs = {k: v for k, v in env_kwargs.items() if k != "frame_skip"}
    env = TorcsEnv(**torcs_kwargs, reward_function=cfg.reward_fn, reset_fn=reset_fn, port=cfg.base_port)
    if skip > 1:
        env = FrameSkipWrapper(env, skip=skip)
    model = PPO.load(str(best_model))

    lap_times = []
    for ep in range(episodes):
        obs, _       = env.reset()
        done = truncated = False
        total_reward = 0.0
        step         = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step         += 1

        lt = info.get("lap_time", 0)
        if lt:
            lap_times.append(lt)

        print(
            f"  Episode {ep+1}/{episodes} | steps={step} | "
            f"reward={total_reward:.1f} | dist={info.get('dist_raced',0):.0f}m | "
            f"lap={'✓ ' + f'{lt:.2f}s' if lt else '✗'}"
        )

    if lap_times:
        print(f"\n  Best lap  : {min(lap_times):.2f}s")
        print(f"  Mean lap  : {np.mean(lap_times):.2f}s")
    else:
        print("\n  No completed laps recorded.")


# ─────────────────────────────────────────────────────────────────────────────
#  Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> tuple[RunConfig, Path]:
    """
    Dynamically imports the given config-1.py and returns (cfg, exp_dir).
    exp_dir is the folder containing the config file.
    The config must define a module-level `cfg = RunConfig(...)`.
    Also adds the config's directory to sys.path so the config can do
    `from reward import ...` without any path fiddling.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    exp_dir = config_path.parent

    # Add experiment dir to path so `from reward import ...` works in config
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    # Also add gym_torcs dir so `from train_ppo import RunConfig` works
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))

    # When train_ppo.py runs as __main__ its RunConfig lives in sys.modules["__main__"].
    # The config's `from train_ppo import RunConfig` would otherwise import a *second*
    # copy of train_ppo, producing a different RunConfig class and failing isinstance.
    # Aliasing __main__ → train_ppo makes both sides share the same class object.
    sys.modules.setdefault("train_ppo", sys.modules["__main__"])

    spec   = importlib.util.spec_from_file_location("experiment_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "cfg"):
        raise AttributeError(
            f"{config_path} must define a module-level `cfg = RunConfig(...)` instance."
        )

    cfg = module.cfg
    if not isinstance(cfg, RunConfig):
        raise TypeError(f"`cfg` in {config_path} must be a RunConfig instance, got {type(cfg)}")

    return cfg, exp_dir


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / evaluate a PPO agent on TORCS")
    parser.add_argument("--config",   required=True,          help="Path to experiment config-1.py")
    parser.add_argument("--resume",   action="store_true",    help="Resume from latest checkpoint")
    parser.add_argument("--eval",     action="store_true",    help="Evaluate best model instead of training")
    parser.add_argument("--episodes", type=int, default=5,    help="Number of eval episodes (default 5)")
    args = parser.parse_args()

    cfg, exp_dir = load_config(args.config)

    if args.eval:
        evaluate(cfg, exp_dir, episodes=args.episodes)
    else:
        train(cfg, exp_dir, resume=args.resume)
