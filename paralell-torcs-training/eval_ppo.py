import argparse
import atexit
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from torcs_env import TorcsEnv
from train_ppo import load_config, DEFAULT_ENV_KWARGS, RunConfig


def evaluate(cfg: RunConfig, model_path: Path, episodes: int, visual: bool = False) -> None:
    env_kwargs = {**DEFAULT_ENV_KWARGS, **cfg.env_overrides}
    sys.path.insert(0, str(_HERE / "docker"))
    import orchestrate

    print(f"\n[INFO] Launching TORCS container on port {cfg.base_port}...")
    orchestrate.launch(num_envs=1, base_port=cfg.base_port, visual=visual)
    atexit.register(orchestrate.stop, num_envs=1, base_port=cfg.base_port, visual=visual)

    from wrappers import FrameSkipWrapper
    skip = env_kwargs.get("frame_skip", 1)
    torcs_kwargs = {k: v for k, v in env_kwargs.items() if k != "frame_skip"}

    env = TorcsEnv(**torcs_kwargs, reward_function=cfg.reward_fn, port=cfg.base_port, skip_reset_kill=visual)
    if skip > 1:
        env = FrameSkipWrapper(env, skip=skip)

    print(f"[INFO] Loading target model: {model_path.name}")
    model = PPO.load(str(model_path), device="cpu")

    if visual:
        print("\n[LAUNCH] Entering infinite visual loop. Press Ctrl+C to terminate.")
        print("The simulation will run continuously and will NEVER trigger an environment reset.\n")
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

                if obs is None or (np.all(obs == 0) and env.unwrapped.client.sock is None):
                    print("\n[WARN] Lost connection to the TORCS socket pipeline.")
                    break
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("\n\n[EXIT] Visual window execution halted. Tearing down container architecture...")

    else:
        print(f"\nStarting deterministic evaluation for {episodes} episodes...\n")
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

            lt = info.get("lap_time", 0)
            if lt and lt > 0:
                lap_times.append(lt)

            status = f"✓ {lt:.2f}s" if lt else "✗"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate or visually run a trained PPO agent on TORCS")
    parser.add_argument("--config", required=True, help="Path to experiment config file")
    parser.add_argument("--model", required=True, help="Explicit path to the .zip model checkpoint")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes (ignored if --visual is used)")
    parser.add_argument("--visual", action="store_true",
                        help="Run the agent in an infinite loop without container or environment resets")
    args = parser.parse_args()

    cfg, _ = load_config(args.config)
    evaluate(cfg, Path(args.model).resolve(), args.episodes, visual=args.visual)
