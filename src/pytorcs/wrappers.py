import gymnasium as gym


class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        if skip < 1:
            raise ValueError(f"FrameSkipWrapper: skip must be >= 1, got {skip}")
        self._skip = skip

    @property
    def skip(self) -> int:
        return self._skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info