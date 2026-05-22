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