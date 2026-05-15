"""
train_ppo.py  –  PPO agent for TORCS Corkscrew (Stable-Baselines3)
====================================================================
Workflow
--------
1.  (Optional) Collect expert data with pid_racer.py first:
        python pid_racer.py --episodes 30 --output expert_data.npz --only_completed

2.  Train the PPO agent (BC warm-start is automatic if expert_data.npz exists):
        python train_ppo.py

3.  Evaluate the best saved checkpoint:
        python train_ppo.py --eval --model logs/best_model/best_model.zip

Key design choices
------------------
* SB3 PPO with MlpPolicy – proven stable for continuous control
* Behavioural-Cloning pre-training if expert_data.npz is present
* NO separate EvalCallback — TORCS is a single stateful process; running a second
  env mid-training corrupts the UDP connection and causes training instability.
  Instead, LapSaveCallback tracks best laps inline from training rollouts and
  saves model weights whenever a new best lap time is recorded.
* CheckpointCallback keeps periodic snapshots every 50k steps
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_torcs.torcs_env import TorcsEnv

# ─────────────────────────────────────────────
#  Paths & hyper-parameters
# ─────────────────────────────────────────────
LOG_DIR          = Path("logs")
BEST_MODEL_DIR   = LOG_DIR / "best_model"
CHECKPOINT_DIR   = LOG_DIR / "checkpoints"
TENSORBOARD_DIR  = LOG_DIR / "tensorboard"
COMPLETED_DIR    = LOG_DIR / "completed_runs"
EXPERT_DATA_PATH = Path("expert_data.npz")

# PPO hyper-parameters (tuned for TORCS continuous control)
PPO_KWARGS = dict(
    policy            = "MlpPolicy",
    learning_rate     = 3e-4,
    n_steps           = 2048,       # steps collected per update
    batch_size        = 256,
    n_epochs          = 10,
    gamma             = 0.99,
    gae_lambda        = 0.95,
    clip_range        = 0.2,
    ent_coef          = 0.005,      # small entropy bonus to avoid premature convergence
    vf_coef           = 0.5,
    max_grad_norm     = 0.5,
    policy_kwargs     = dict(
        net_arch       = [256, 256],   # two hidden layers
        activation_fn  = nn.Tanh,
    ),
    tensorboard_log   = str(TENSORBOARD_DIR),
    verbose           = 1,
)

# Behavioural-Cloning pre-training settings
BC_EPOCHS      = 20        # epochs over the expert dataset
BC_BATCH_SIZE  = 256
BC_LR          = 1e-3

# Total PPO environment steps
TOTAL_TIMESTEPS = 1_000_000

# Sensor features (must match what pid_racer collected)
SENSOR_FEATURES = ["speedX", "angle", "trackPos", "track"]


# ─────────────────────────────────────────────
#  Corkscrew-specific reward function
# ─────────────────────────────────────────────
import math

def corkscrew_reward(previous, current) -> float:
    """
    Reward shaping tuned for the Corkscrew track:
    - Maximise forward progress projected onto track direction
    - Soft corridor penalty (±0.6) — tighter than default to hug the racing line
    - Speed bonus for maintaining high speed
    - Lateral slip penalty
    - Large lap-completion bonus scaled by inverse lap-time (fastest = highest reward)
    - Off-track penalty with partial distance credit
    """
    angle      = current.angle
    track_pos  = current.trackPos
    speed_x    = current.speedX
    speed_y    = current.speedY
    dist_raced = current.distRaced

    # --- Progress ---
    progress = ((dist_raced - previous.distRaced) / 10.0) * math.cos(angle)

    # --- Track-position penalty: tighter corridor than default ---
    abs_pos     = abs(track_pos)
    pos_penalty = 0.8 * ((abs_pos - 0.6) ** 2) if abs_pos > 0.6 else 0.0

    # --- Speed bonus: reward maintaining high speed (scaled to 0–0.3) ---
    speed_bonus = 0.001 * max(0.0, speed_x - 80.0)

    # --- Lateral slip penalty ---
    lateral_penalty = 0.005 * abs(speed_y / 300.0)

    # --- Angle penalty: penalise large misalignment ---
    angle_penalty = 0.3 * (angle ** 2)

    terminal_bonus = 0.0
    lap_done       = (
            current.curLapTime > 0 and
            current.distRaced   > 10.0 and
            current.lastLapTime > 0.0 and
            current.curLapTime  < 0.5
    )
    off_track = abs(track_pos) > 1.2

    if lap_done:
        lap_time       = current.lastLapTime
        terminal_bonus = 5000.0 + (10_000.0 / max(lap_time, 1.0))
        print(f"  ✓ LAP COMPLETE  time={lap_time:.2f}s  bonus={terminal_bonus:.1f}")
    elif off_track:
        partial_credit = min(dist_raced / 1000.0, 20.0)
        terminal_bonus = -150.0 + partial_credit
        print(f"  ✗ OFF TRACK  dist={dist_raced:.0f}m")

    return float(
        progress
        + speed_bonus
        - pos_penalty
        - lateral_penalty
        - angle_penalty
        + terminal_bonus
    )


# ─────────────────────────────────────────────
#  Behavioural Cloning pre-training
# ─────────────────────────────────────────────

def pretrain_bc(model: PPO, expert_path: Path) -> PPO:
    """Warm-start the PPO actor with behavioural cloning on expert transitions."""
    print(f"\n{'='*60}")
    print(f"  Behavioural Cloning pre-training from {expert_path}")
    print(f"{'='*60}")

    data        = np.load(expert_path)
    obs_np      = data["observations"].astype(np.float32)
    actions_np  = data["actions"].astype(np.float32)

    print(f"  Expert transitions : {len(obs_np):,}")
    print(f"  obs shape          : {obs_np.shape}")
    print(f"  actions shape      : {actions_np.shape}\n")

    obs_t     = torch.tensor(obs_np)
    actions_t = torch.tensor(actions_np)
    dataset   = torch.utils.data.TensorDataset(obs_t, actions_t)
    loader    = torch.utils.data.DataLoader(
        dataset, batch_size=BC_BATCH_SIZE, shuffle=True
    )

    # Only train the actor (policy network), not the value head
    actor_params = list(model.policy.mlp_extractor.policy_net.parameters()) + \
                   list(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(actor_params, lr=BC_LR)
    loss_fn   = nn.MSELoss()

    model.policy.train()
    for epoch in range(BC_EPOCHS):
        total_loss = 0.0
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(model.device)
            act_batch = act_batch.to(model.device)

            # Forward pass through actor
            features      = model.policy.extract_features(obs_batch)
            latent_pi, _  = model.policy.mlp_extractor(features)
            mean_actions  = model.policy.action_net(latent_pi)

            loss = loss_fn(mean_actions, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  BC Epoch [{epoch+1:02d}/{BC_EPOCHS}]  loss={avg_loss:.5f}")

    print("\n  BC pre-training complete.\n")
    model.policy.eval()
    return model


# ─────────────────────────────────────────────
#  Lap-time tracking callback
# ─────────────────────────────────────────────

class LapTimeCallback(BaseCallback):
    """Logs best lap time to tensorboard and stdout."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_lap = float("inf")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            lt = info.get("lap_time", 0)
            if lt and lt > 0:
                self.logger.record("custom/lap_time", lt)
                if lt < self.best_lap:
                    self.best_lap = lt
                    self.logger.record("custom/best_lap_time", lt)
                    print(f"\n  🏁 New best lap: {lt:.2f}s  (step {self.num_timesteps})")
        return True


class LapSaveCallback(BaseCallback):
    """
    Buffers every transition during an episode.
    When a lap completes, saves obs/actions/rewards to a .npz file in
    logs/completed_runs/ — one file per completed lap, named by lap time.
    Also saves the model weights whenever a new best lap is recorded.
    """

    def __init__(self, save_dir: Path, verbose=0):
        super().__init__(verbose)
        self.save_dir    = save_dir
        self.best_lap    = float("inf")
        self._ep_obs     = []
        self._ep_actions = []
        self._ep_rewards = []

    def _on_training_start(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        # Buffer current transition (VecEnv returns batched arrays)
        self._ep_obs.append(self.locals["obs_tensor"].cpu().numpy().copy())
        self._ep_actions.append(self.locals["actions"].copy())
        self._ep_rewards.append(self.locals["rewards"].copy())

        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])

        for done, info in zip(dones, infos):
            if not done:
                continue

            lt = info.get("lap_time", 0)
            if lt and lt > 0:
                # Completed lap — save the full episode
                obs_arr     = np.concatenate(self._ep_obs,     axis=0).astype(np.float32)
                actions_arr = np.concatenate(self._ep_actions, axis=0).astype(np.float32)
                rewards_arr = np.concatenate(self._ep_rewards, axis=0).astype(np.float32)

                fname = self.save_dir / f"lap_{lt:.2f}s_step{self.num_timesteps}.npz"
                np.savez_compressed(
                    fname,
                    observations = obs_arr,
                    actions      = actions_arr,
                    rewards      = rewards_arr,
                    lap_time     = np.array([lt], dtype=np.float32),
                )
                print(f"\n  💾 Saved completed lap ({lt:.2f}s) → {fname.name}")

                # Save model weights if new best lap
                if lt < self.best_lap:
                    self.best_lap = lt
                    model_path = self.save_dir / f"best_lap_{lt:.2f}s_step{self.num_timesteps}"
                    self.model.save(str(model_path))
                    print(f"  🏆 New best lap model saved → {model_path.name}.zip")

            # Always clear buffer at episode boundary
            self._ep_obs.clear()
            self._ep_actions.clear()
            self._ep_rewards.clear()

        return True


# ─────────────────────────────────────────────
#  Environment factory
# ─────────────────────────────────────────────

def make_env(port: int = 3001):
    def _init():
        env = TorcsEnv(
            sensor_features  = SENSOR_FEATURES,
            reward_function  = corkscrew_reward,
            truncate_limit   = 10_000,
            port             = port,
        )
        env = Monitor(env, str(LOG_DIR / "monitor"))
        return env
    return _init


# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────

def train(resume_path: str | None = None):
    for d in [LOG_DIR, BEST_MODEL_DIR, CHECKPOINT_DIR, TENSORBOARD_DIR, COMPLETED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  TORCS Corkscrew — PPO Training")
    print("="*60)

    # Single env only — TORCS is a stateful single process, two envs will
    # fight over the same UDP port and corrupt each other's connections.
    vec_env = DummyVecEnv([make_env(port=3001)])

    if resume_path:
        print(f"\n  Resuming from {resume_path}")
        model = PPO.load(resume_path, env=vec_env, **{
            k: v for k, v in PPO_KWARGS.items()
            if k not in ("policy", "policy_kwargs")
        })
    else:
        model = PPO(env=vec_env, **PPO_KWARGS)

    # Behavioural-cloning warm start
    if EXPERT_DATA_PATH.exists() and not resume_path:
        model = pretrain_bc(model, EXPERT_DATA_PATH)
    elif not EXPERT_DATA_PATH.exists():
        print(
            "\n  [INFO] No expert_data.npz found — skipping BC pre-training.\n"
            "         Run pid_racer.py first for faster convergence.\n"
        )

    # Callbacks — no EvalCallback (would launch a second TORCS, breaking training)
    checkpoint_callback = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = str(CHECKPOINT_DIR),
        name_prefix = "ppo_corkscrew",
    )
    lap_time_callback = LapTimeCallback(verbose=1)
    lap_save_callback = LapSaveCallback(save_dir=COMPLETED_DIR, verbose=1)

    print(f"\n  Starting PPO training for {TOTAL_TIMESTEPS:,} steps")
    print(f"  Best lap models  → {COMPLETED_DIR}")
    print(f"  Checkpoints      → {CHECKPOINT_DIR}")
    print(f"  Tensorboard: tensorboard --logdir {TENSORBOARD_DIR}\n")

    model.learn(
        total_timesteps     = TOTAL_TIMESTEPS,
        callback            = [checkpoint_callback, lap_time_callback, lap_save_callback],
        reset_num_timesteps = not bool(resume_path),
        tb_log_name         = "ppo_corkscrew",
    )

    final_path = str(LOG_DIR / "ppo_corkscrew_final")
    model.save(final_path)
    print(f"\n  Training complete. Final model saved to {final_path}.zip")
    print(f"  Best lap model in {COMPLETED_DIR}")


# ─────────────────────────────────────────────
#  Evaluation / deployment
# ─────────────────────────────────────────────

def evaluate(model_path: str, episodes: int = 5):
    print(f"\n  Evaluating {model_path} for {episodes} episodes\n")

    env   = TorcsEnv(
        sensor_features = SENSOR_FEATURES,
        reward_function = corkscrew_reward,
        truncate_limit  = 10_000,
        port            = 3001,
    )
    model = PPO.load(model_path)

    lap_times = []
    for ep in range(episodes):
        obs, _       = env.reset()
        done         = False
        truncated    = False
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


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / evaluate PPO on TORCS Corkscrew")
    parser.add_argument("--eval",     action="store_true",  help="Run evaluation instead of training")
    parser.add_argument("--model",    type=str, default=None, help="Path to .zip model (for --eval or resume)")
    parser.add_argument("--episodes", type=int, default=5,   help="Eval episodes (default 5)")
    parser.add_argument("--resume",   action="store_true",   help="Resume training from --model checkpoint")
    args = parser.parse_args()

    if args.eval:
        if not args.model:
            args.model = str(BEST_MODEL_DIR / "best_model.zip")
        evaluate(args.model, episodes=args.episodes)
    elif args.resume:
        if not args.model:
            raise ValueError("--resume requires --model <path>")
        train(resume_path=args.model)
    else:
        train()