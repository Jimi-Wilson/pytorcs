import argparse
import sys
import atexit
from pathlib import Path

from pytorcs.utils import load_config, get_orchestrator
from pytorcs.torcs_env import TorcsEnv
from pytorcs.wrappers import FrameSkipWrapper

from pytorcs.evaluator import evaluate_algorithm


def main():
    parser = argparse.ArgumentParser(description="Universal Evaluation Script for SB3 on TORCS")
    parser.add_argument("--config", required=True, help="Path to experiment config file")
    parser.add_argument("--model", required=True, help="Explicit path to the .zip model checkpoint")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes (ignored if --visual is used)")
    parser.add_argument("--visual", action="store_true",
                        help="Run the agent in an infinite loop without container or environment resets")
    args = parser.parse_args()

    print(f"[Evaluate] Loading configuration file: {args.config}")
    try:
        config, _ = load_config(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        sys.exit(1)

    orchestrate = get_orchestrator()
    orchestrate.launch(num_envs=1, base_port=config.base_port, visual=args.visual)
    atexit.register(orchestrate.stop, num_envs=1, base_port=config.base_port, visual=args.visual)

    env_kwargs = config.env_kwargs.copy()
    skip = env_kwargs.pop("frame_skip", 1)

    env = TorcsEnv(
        **env_kwargs,
        reward_function=config.reward_function,
        port=config.base_port,
        skip_reset_kill=args.visual
    )

    if skip > 1:
        env = FrameSkipWrapper(env, skip=skip)

    print(f"\n[Evaluate] Algorithm: {config.algorithm_class.__name__}")

    evaluate_algorithm(
        algo_class=config.algorithm_class,
        env=env,
        model_path=Path(args.model).resolve(),
        episodes=args.episodes,
        visual=args.visual
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Evaluate] Keyboard interrupt by user. Exiting...")
        sys.exit(0)