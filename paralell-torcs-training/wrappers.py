import gymnasium as gym


class FrameSkipWrapper(gym.Wrapper):
    """
    Repeats the agent's action for `skip` consecutive TORCS physics ticks.

    Behaviour
    ---------
    - Rewards  : summed across all skipped ticks.
    - Observation : returned from the *last* tick only (observation space
      is therefore identical to the wrapped env — no architecture changes
      needed).
    - Early exit : if a terminal (done) or truncated signal arrives during
      the skip loop the episode ends immediately and that signal is
      propagated, so the agent never steps past a terminal state.

    Parameters
    ----------
    env  : the environment to wrap (must be a TorcsEnv or compatible gym.Env)
    skip : number of physics ticks to repeat each action (default 4).
           skip=1 is a no-op (identical to the unwrapped env).
    """

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

    # reset() is inherited from gym.Wrapper and delegates to self.env.reset()
    # — no override needed.
