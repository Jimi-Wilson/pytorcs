"""
SACPID training entry-point for the TORCS Corkscrew hotlap.

Usage:
    conda activate torcs_env
    # (Start TORCS manually first, or use autostart.sh so port 3101 is open)
    python train_sacpid.py --stage 1
"""

import argparse
import omnisafe

# Importing the wrapper module triggers @env_register, which makes
# 'TorcsSafe-v0' visible to omnisafe.Agent().
import torcs_omnisafe_wrapper  # noqa: F401


def train(stage: int) -> None:
    print(f"--- Initialising OmniSafe SACPID Training (Stage {stage}) ---")

    # Curriculum definitions from the architecture document
    if stage == 1:
        # Stage 1: basic controls on a flat track. Tight track limits.
        cost_limit  = 5.0
        track_limit = 1.00
    elif stage == 2:
        # Stage 2: geometric racing line. Allow 2-wheels-off.
        cost_limit  = 10.0
        track_limit = 1.05
    else:
        # Stage 3: exploit physics. High risk tolerance.
        cost_limit  = 15.0
        track_limit = 1.10

    custom_cfgs = {
        'train_cfgs': {
            'device': 'cpu',        # change to 'cuda:0' if GPU is available
            'total_steps': 1_000_000,
            'vector_env_nums': 1,   # TORCS UDP supports 1 client at a time
            'parallel': 1,
        },
        'algo_cfgs': {
            'cost_limit': cost_limit,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 50,
        },
    }

    env_id = 'TorcsSafe-v0'

    # Pass track_limit through the config so OmniSafe forwards it to TorcsSafeEnv.__init__
    custom_cfgs['env_cfgs'] = {'track_limit': track_limit}

    # Instantiate the agent.
    # OmniSafe discovers TorcsSafe-v0 through the @env_register decorator
    # that fires when we imported torcs_omnisafe_wrapper above.
    agent = omnisafe.Agent(
        'SACPID',
        env_id,
        custom_cfgs=custom_cfgs,
    )

    print(f"Cost budget (d): {cost_limit}  |  Track limit: {track_limit}")
    print("NOTE: Make sure TORCS is running so UDP port 3101 is open.")

    agent.learn()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SACPID TORCS trainer')
    parser.add_argument('--stage', type=int, default=1,
                        choices=[1, 2, 3],
                        help='Curriculum training stage')
    args = parser.parse_args()
    train(args.stage)
