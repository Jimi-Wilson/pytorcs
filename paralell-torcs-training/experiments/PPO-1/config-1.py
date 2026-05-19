from train_ppo import RunConfig
from reward_functions import corkscrew_reward_1


cfg = RunConfig(
    run_name="learning-the-basics",
    reward_fn       = corkscrew_reward_1,
    total_timesteps = 1_000_000,

    ppo_overrides = {
        "n_steps":       256,
        "batch_size":    1024,
    },

    env_overrides = {
        "sensor_features": ["speedX", "speedY", "angle", "trackPos", "track", "rpm", "gear"],
        "truncate_limit":  10_000,
        "frame_skip":      1,
    },

    num_envs   = 16,
    base_port  = 3001,
)
