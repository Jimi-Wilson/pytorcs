"""
Quick smoke test: connect to TORCS on port 3101 and drive with random actions.
No training, no OmniSafe — just verifies the wrapper and TORCS comms work.

Usage:
    1. Launch TORCS:  torcs -nofuel -nodamage -nolaptime &
    2. Set up a Quick Race with 'scr_server' driver, click New Race.
    3. Run:  python test_drive.py
"""

import numpy as np
from gym_torcs import TorcsEnv

def main():
    print("Connecting to TORCS on port 3101...")
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    obs, info = env.reset()

    print("Connected! Driving with random actions for 200 steps...")
    for step in range(200):
        # Random steering [-1, 1] and random throttle [-1, 1]
        action = np.array([
            np.random.uniform(-0.3, 0.3),   # gentle random steering
            np.random.uniform(0.2, 0.8),     # mostly forward throttle
        ])

        obs, reward, done, truncated, info = env.step(action)

        if step % 20 == 0:
            speedX = env.client.S.d.get('speedX', 0)
            trackPos = env.client.S.d.get('trackPos', 0)
            print(f"  Step {step:03d} | Speed: {speedX:6.1f} km/h | TrackPos: {trackPos:+.2f} | Reward: {reward:+.2f}")

        if done:
            print(f"  Episode ended at step {step}. Resetting...")
            obs, info = env.reset()

    print("Test complete! The car drove for 200 steps.")
    env.end()

if __name__ == '__main__':
    main()
