import math
from typing import Callable
from train_ppo import RunConfig
from reward_functions import corkscrew_reward_1

def exp_decay_schedule(initial_value: float, min_lr: float = 1e-6, decay_rate: float = 5.0) -> Callable[[float], float]:
    """
    Exponential learning rate decay with a safety floor.
    """
    def func(progress_remaining: float) -> float:
        decay_factor = math.exp(-decay_rate * (1.0 - progress_remaining))
        return min_lr + (initial_value - min_lr) * decay_factor
    return func

cfg = RunConfig(
    run_name="phasase",
    reward_fn       = corkscrew_reward_1,
    total_timesteps = 2_000_000, # A shorter run is usually enough for fine-tuning

    ppo_overrides = {
        "n_steps":       2048,
        "batch_size":    2048,
        "gamma":         0.999,
        "ent_coef":      0.001,  # Almost zero randomness. Pure precision.
        "learning_rate": exp_decay_schedule(1e-4, min_lr=1e-6),
        "clip_range":    0.1,
    },

    env_overrides = {
        "sensor_features": ["speedX", "speedY", "angle", "trackPos", "track", "rpm", "gear"],
        "truncate_limit":  10_000,
        "frame_skip":      1,
    },

    num_envs   = 16,
    base_port  = 3001,
)