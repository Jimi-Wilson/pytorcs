import argparse
import atexit
import sys

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from pytorcs.trainer import train_algorithm
from pytorcs.utils import load_config, get_timestamped_run_dir, get_orchestrator, make_env


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="Path to the config file (e.g., experiments/PPO-1/config-1.py)"
                        )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Explicit path to a .zip checkpoint to resume training"
    )

    args = parser.parse_args()

    print(f"[Train] Loading configuration file: {args.config}")
    try:
        config, exp_dir = load_config(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        sys.exit(1)

    run_dir = get_timestamped_run_dir(exp_dir, config.run_name)
    for d in ["checkpoints", "best_models", "monitor"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    if config.num_envs > 32:
        print(f"[Train] Warning: Number of environments capped at 32 (requested: {config.num_envs}).")
        config.num_envs = 32

    print(f"[Train] Launching {config.num_envs} TORCS container(s)...")
    orchestrate = get_orchestrator()
    orchestrate.launch(num_envs=config.num_envs, base_port=config.base_port)
    atexit.register(orchestrate.stop, num_envs=config.num_envs, base_port=config.base_port)

    env_kwargs = config.env_kwargs.copy()
    env_fns = [
        make_env(config.base_port + i, config, env_kwargs, run_dir / "monitor")
        for i in range(config.num_envs)
    ]

    vec_env = SubprocVecEnv(env_fns) if config.num_envs > 1 else DummyVecEnv(env_fns)

    print(f"\n[Train] Starting run: {run_dir.name}")
    print(f"[Train] Algorithm: {config.algorithm_class.__name__}")

    train_algorithm(
        algorithm_class=config.algorithm_class,
        env=vec_env,
        config=config,
        run_dir=run_dir,
        resume_path=args.resume
    )

    print("\n[Train] Training complete. Cleaning up containers...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[Train] Keyboard interrupt by user. Exiting...")
        sys.exit(0)