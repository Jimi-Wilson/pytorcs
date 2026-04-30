from stable_baselines3 import PPO
from gym_torcs.torcs_env import TorcsEnv

# This is an example of how to train a model using stable-baselines3
# stable-baselines3 is the easiest way to train a model as it works directly with gym
# You can do other ways like custom NN with pytorch/tensorflow but stable-baselines3 is just easier
# DOCS: https://stable-baselines3.readthedocs.io/en/master/

env = TorcsEnv()

model = PPO("MlpPolicy", env, n_steps=4096, verbose=1, max_grad_norm=0.5)

model.learn(total_timesteps=100000)

model.save("model")