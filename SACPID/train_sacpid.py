"""
SACPID training entry-point for the TORCS Corkscrew (Laguna Seca) hotlap.

Usage:
    conda activate torcs_env
    # Start TORCS first (manually or via ./autostart.sh), then:
    python train_sacpid.py --stage 1

    # Spot instances: use persistent log dir and frequent saves; resume after interrupt:
    python train_sacpid.py --stage 1 --log-dir /data/sacpid-runs --save-freq 10
    python train_sacpid.py --stage 1 --resume-from /data/sacpid-runs/run-xxx

Weights & Biases: set WANDB_API_KEY or run `wandb login` to log runs to the web.
"""

import argparse
import os
import signal
import torch
import omnisafe

# Importing the wrapper module triggers @env_register, which makes
# 'TorcsSafe-v0' visible to omnisafe.Agent().
import torcs_omnisafe_wrapper  # noqa: F401

# Set by SIGTERM handler (e.g. AWS Spot interrupt). Not yet read by the training loop;
# use --save-freq and --log-dir so checkpoints are saved regularly; resume with --resume-from.
_spot_interrupt_requested = False


def _handle_spot_interrupt(signum, frame):
    global _spot_interrupt_requested
    _spot_interrupt_requested = True
    print("\n[Spot interrupt] SIGTERM received; will save and exit when possible.")


def train(
    stage: int,
    *,
    resume_from: str | None = None,
    log_dir: str | None = None,
    save_freq: int = 10,
) -> None:
    print(f"--- Initialising OmniSafe SACPID Training (Stage {stage}) ---")

    # Curriculum definitions from the architecture document.
    # Target track: Corkscrew (Laguna Seca) only; Stage 1 may use a flat track (longstr) if available.
    if stage == 1:
        # Stage 1: basic controls on a flat track. Tight track limits.
        cost_limit  = 5.0
        track_limit = 1.00
    elif stage == 2:
        # Stage 2: geometric racing line on Corkscrew. Allow 2-wheels-off.
        cost_limit  = 10.0
        track_limit = 1.05
    else:
        # Stage 3: exploit physics on Corkscrew. High risk tolerance.
        cost_limit  = 15.0
        track_limit = 1.10

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    custom_cfgs = {
        'train_cfgs': {
            'device': device,
            'total_steps': 1_000_000,
            'vector_env_nums': 1,   # TORCS UDP supports 1 client at a time
            'parallel': 1,
        },
        'algo_cfgs': {
            'cost_limit': cost_limit,
        },
        'logger_cfgs': {
            'use_wandb': True,
            'wandb_project': 'sacpid-torcs-corkscrew',
            'save_model_freq': save_freq,
        },
    }
    if log_dir is not None:
        custom_cfgs['logger_cfgs']['log_dir'] = os.path.abspath(log_dir)
    if resume_from is not None:
        custom_cfgs['logger_cfgs']['resume'] = resume_from
        custom_cfgs['logger_cfgs']['resume_from'] = resume_from

    env_id = 'TorcsSafe-v0'

    # Pass env options so the wrapper can apply stage-dependent reward (e.g. centerline penalty in Stage 1)
    custom_cfgs['env_cfgs'] = {
        'track_limit': track_limit,
        'stage': stage,
        'reward_scale': 0.01,
        'centerline_penalty': 1.0,
    }

    # Instantiate the agent.
    # OmniSafe discovers TorcsSafe-v0 through the @env_register decorator
    # that fires when we imported torcs_omnisafe_wrapper above.
    agent = omnisafe.Agent(
        'SACPID',
        env_id,
        custom_cfgs=custom_cfgs,
    )

    print(f"Cost budget (d): {cost_limit}  |  Track limit: {track_limit}")
    print("NOTE: Make sure TORCS is running with the correct track so UDP port 3101 is open.")
    if custom_cfgs['logger_cfgs']['use_wandb']:
        print("Logging to Weights & Biases (project: sacpid-torcs-corkscrew).")
    if log_dir:
        print(f"Checkpoints/logs: {custom_cfgs['logger_cfgs']['log_dir']}")
    if resume_from:
        print(f"Resuming from: {resume_from}")

    # Handle Spot interrupt (AWS sends SIGTERM ~2 min before reclaim)
    signal.signal(signal.SIGTERM, _handle_spot_interrupt)

    agent.learn()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SACPID TORCS trainer (Spot-friendly: use --log-dir and --save-freq)')
    parser.add_argument('--stage', type=int, default=1,
                        choices=[1, 2, 3],
                        help='Curriculum training stage')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to existing run directory to resume (e.g. after Spot interrupt)')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for checkpoints and logs (use EBS path on Spot so new instance can resume)')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save model every N epochs (default 10 for Spot; lower = less work lost on interrupt)')
    args = parser.parse_args()
    train(args.stage, resume_from=args.resume_from, log_dir=args.log_dir, save_freq=args.save_freq)
