from stable_baselines3 import PPO
from gym_torcs.torcs_env import TorcsEnv

"""
This is a barebones example of how to use a trained model.
Just load the model and run it.
"""

env = TorcsEnv()

model = PPO.load("model", env=env)

obs, info = env.reset(options={"headless": False})

while True:
    action, _states = model.predict(obs, deterministic=True)
    _, _, _, _, _ = env.step(action)
