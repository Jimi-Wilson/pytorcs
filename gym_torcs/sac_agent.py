import argparse
import os
import sys
from pathlib import Path
from time import sleep

import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from torcs_env import TorcsEnv
from rewards import corkscrew_reward
from utils import SaveBestLapCallback, create_run_folder

sys.path.insert(0, str(Path(__file__).parent.parent / "docker"))
import orchestrate

SENSOR_FEATURES = ["speedX", "angle", "trackPos", "track"]


def make_env(rank: int, base_port: int):
    def init():
        port = base_port + rank
        env = TorcsEnv(
            sensor_features=SENSOR_FEATURES,
            truncate_limit=10_000,
            port=port,
            reward_function=corkscrew_reward,
        )
        return env
    return init


def collect_bc_data(model_path, num_episodes, base_port: int):
    env = TorcsEnv(
        sensor_features=SENSOR_FEATURES,
        truncate_limit=10_000,
        port=base_port,
        reward_function=corkscrew_reward,
    )

    orchestrate.launch(num_envs=1, base_port=base_port)

    model = PPO.load(model_path, env=env, device="cpu")

    bc_data = []

    for episode in range(num_episodes):
        obs, _ = env.reset(headless=False)
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            bc_data.append((obs, next_obs, action, reward, terminated, truncated, info))

            obs = next_obs
            done = terminated or truncated

        print(f"Collected episode {episode + 1}/{num_episodes} — {len(bc_data)} steps total")

    env.close()
    return bc_data


def fill_replay_buffer(model, bc_data):
    n = model.n_envs
    print(f"Filling replay buffer with {len(bc_data)} BC steps (n_envs={n})...")
    for obs, next_obs, action, reward, terminated, truncated, info in bc_data:
        model.replay_buffer.add(
            obs=np.stack([obs] * n),
            next_obs=np.stack([next_obs] * n),
            action=np.stack([action] * n),
            reward=np.full(n, reward, dtype=np.float32),
            done=np.full(n, float(terminated), dtype=np.float32),
            infos=[info] * n,
        )
    print(f"Replay buffer filled: {model.replay_buffer.size()} steps.")


def train(num_envs: int, base_port: int, resume_path=None, clone_model_path=None, clone_episodes=None):
    run_dir = create_run_folder()

    train_env = SubprocVecEnv([make_env(i, base_port) for i in range(num_envs)])

    if resume_path:
        print(f"Resuming training from {resume_path}...")
        model = SAC.load(resume_path, env=train_env)
    else:
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            tensorboard_log=run_dir,
        )

    if clone_model_path and clone_episodes:
        print(f"Collecting {clone_episodes} BC episodes from {clone_model_path}...")
        bc_data = collect_bc_data(clone_model_path, clone_episodes, base_port)
        fill_replay_buffer(model, bc_data)

    orchestrate.launch(num_envs=num_envs, base_port=base_port)

    try:
        print("Starting training...")
        callback = SaveBestLapCallback(save_dir=run_dir)
        model.learn(total_timesteps=1_000_000, callback=callback)
        model.save(os.path.join(run_dir, "final_model"))
        print(f"Training complete. Model saved to {run_dir}")
    finally:
        print("[orchestrate] Shutting down TORCS containers...")
        orchestrate.stop(num_envs=num_envs, base_port=base_port)


def evaluate(visual, model_path):
    env = TorcsEnv()

    model = SAC.load(model_path, env=env)

    obs, info = env.reset(headless=not visual)

    print(f"Starting evaluation of model {model_path}... (Visual mode: {visual})")

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    lap_time = info.get("lap_time", "No lap time recorded (e.g., crashed or truncated)")
    dist_raced = info.get("dist_raced", "No distance recorded")

    print("Evaluation complete!")
    print(f"Lap time: {lap_time}")
    print(f"Distance raced: {dist_raced}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Run non-visual evaluation")
    parser.add_argument("--vis-eval", action="store_true", help="Run visual evaluation")
    parser.add_argument("--model", type=str, default=None, help="Path to .zip model (for evaluation or resume)")
    parser.add_argument("--resume", action="store_true", default=None, help="Resume training from model checkpoint")
    parser.add_argument("--clone-model", type=str, default=None, help="Path to .zip model for behavioural cloning")
    parser.add_argument("--clone-episodes", type=int, default=None, help="Number of episodes to collect for BC")
    parser.add_argument("--num-envs", type=int, default=int(os.getenv("NUM_ENVS", 2)), help="Number of parallel TORCS environments (default: NUM_ENVS from .env or 2)")
    parser.add_argument("--base-port", type=int, default=int(os.getenv("BASE_PORT", 3001)), help="First UDP port to use (default: BASE_PORT from .env or 3001)")

    args = parser.parse_args()

    if args.eval or args.vis_eval:
        if not args.model:
            print("No model specified for evaluation")
            exit(1)

        evaluate(args.vis_eval, args.model)

    elif args.resume:
        if not args.model:
            print("No model specified for resuming")
            exit(1)

        train(num_envs=args.num_envs, base_port=args.base_port, resume_path=args.model)
    else:
        if args.clone_model and not args.clone_episodes:
            print("No number of episodes specified for BC")
            exit(1)
        elif args.clone_episodes and not args.clone_model:
            print("No model specified for BC")
            exit(1)

        train(
            num_envs=args.num_envs,
            base_port=args.base_port,
            clone_model_path=args.clone_model,
            clone_episodes=args.clone_episodes,
        )
