import numpy as np
from pathlib import Path
import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm


def _run_visual_evaluation(env: gym.Env, model: BaseAlgorithm) -> None:
    """Continuous evaluation loop without resets for visual inspection."""
    print("\n[Evaluate] Starting up visual evaluation. Press Ctrl+C to stop.")
    print("[Evaluate] The simulation will run continuously and will never trigger an environment reset.\n")

    obs, _ = env.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            status = info.get("status_flag", "ACTIVE")
            print(
                f"\rSpeed: {info.get('speed_x', 0.0):.1f} km/h | "
                f"Dist Raced: {info.get('dist_raced', 0.0):.0f}m | "
                f"Track Pos: {info.get('track_pos', 0.0):.2f} | "
                f"Status: {status:<12}",
                end="",
                flush=True
            )

    except KeyboardInterrupt:
        print("\n\n[Evaluate] Visual window execution halted. Tearing down container architecture...")


def _run_episodic_evaluation(env: gym.Env, model: BaseAlgorithm, episodes: int) -> None:
    """Fixed number of deterministic episodes to collect lap statistics."""
    print(f"\n[Evaluate] Starting deterministic evaluation for {episodes} episodes...\n")
    lap_times = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = truncated = False
        total_reward = 0.0
        step = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

        lap_time = info.get("lap_time", 0)
        if lap_time and lap_time > 0:
            lap_times.append(lap_time)

        status = f"✓ {lap_time:.2f}s" if lap_time else "✗"
        print(
            f"  Episode {ep + 1:02d}/{episodes:02d} | steps={step:<5d} | "
            f"reward={total_reward:<8.1f} | dist={info.get('dist_raced', 0):.0f}m | "
            f"lap={status}"
        )

    print(f"\n{'=' * 50}\n  Evaluation Summary\n{'=' * 50}")
    if lap_times:
        print(f"  Completed Laps : {len(lap_times)} / {episodes}")
        print(f"  Best Lap Time  : {min(lap_times):.2f}s")
        print(f"  Mean Lap Time  : {np.mean(lap_times):.2f}s")
    else:
        print("  No completed laps recorded during this evaluation.")
    print(f"{'=' * 50}\n")


def evaluate_algorithm(
        algo_class: type[BaseAlgorithm],
        env: gym.Env,
        model_path: Path,
        episodes: int = 5,
        visual: bool = False
) -> None:
    """Universal evaluation entry point for any Stable Baselines 3 algorithm."""

    print(f"[Evaluate] Loading target model: {model_path.name} as {algo_class.__name__}")

    # Load the model directly from the class defined in the config (PPO, SAC, etc.)
    model = algo_class.load(str(model_path), device="cpu")

    if visual:
        _run_visual_evaluation(env, model)
    else:
        _run_episodic_evaluation(env, model, episodes)