from pytorcs.configs import PPORunConfig
import example_reward_function


"""
   This is an example of a config file for a PPO run.
   Main difference between different algorithms is the `algorithm_name`_kwargs which are specific to each algorithm.
   To find out the algorithm specific kwargs, please refer to the stable-baselines3 documentation. (https://stable-baselines3.readthedocs.io/en/master/index.html)
"""
config = PPORunConfig(
    run_name = "example_experiment",
    reward_function = example_reward_function.reward_function,
    total_timesteps = 1_000_000,

    env_kwargs={
        "sensor_features": ["speedX", "speedY", "angle", "trackPos", "track", "rpm", "gear"],
        "truncate_limit": 10_000,
        "frame_skip": 1,
    },

    num_envs = 1,
    base_port = 3001,
    callbacks = None,

    ppo_kwargs = {
        "policy": "MlpPolicy",
        "device": "cpu"
    }


)

"""
This is an example of a config file for a SAC run. See the SAC folder in examples for the model and reward functions. For test running.
"""

from pytorcs.configs import SACRunConfig

config = SACRunConfig(
    run_name="Big-boy",
    # reward_function= corkscrew_reward_1,
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