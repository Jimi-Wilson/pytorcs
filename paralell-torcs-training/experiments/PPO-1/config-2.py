from train_ppo import RunConfig
from reward_functions import corkscrew_reward_1

cfg = RunConfig(
    run_name="new-gears",
    reward_fn       = corkscrew_reward_1,
    total_timesteps = 5_000_000,

    ppo_overrides = {
        "n_steps":       1024,   # Expanded horizon for high-speed cornering
        "batch_size":    2048,
        "gamma":         0.999,  # Force the agent to care about corner exits, not just entries
        "ent_coef":      0.01,   # Bump entropy slightly to encourage exploring the new grip limits
        #"learning_rate": 1e-4,   # Lower learning rate to protect the driving skills it just learned
    },

    env_overrides = {
        "sensor_features": ["speedX", "speedY", "angle", "trackPos", "track", "rpm", "gear"],
        "truncate_limit":  10_000,
        "frame_skip":      1,
    },

    num_envs   = 16,
    base_port  = 3001,
)