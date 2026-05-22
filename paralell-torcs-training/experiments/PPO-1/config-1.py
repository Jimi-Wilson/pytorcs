from pytorcs.configs import SACRunConfig
from reward_functions import corkscrew_reward_1

config = SACRunConfig(
    run_name="Big-boy",
    reward_function= corkscrew_reward_1,
    total_timesteps = 10_000_000,

    sac_kwargs = {
        "policy":       "MlpPolicy",
        "buffer_size":  50_000,
        "batch_size":   1024,
        "train_freq":     8,
        "gradient_steps": 8,
    },

    env_kwargs = {
        "sensor_features": ["speedX", "speedY", "angle", "trackPos", "track", "rpm", "gear"],
        "truncate_limit":  10_000,
        "frame_skip":      4,
    },

    num_envs   = 8,
    base_port  = 3001,
)