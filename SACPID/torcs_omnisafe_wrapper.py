from __future__ import annotations

"""
OmniSafe CMDP wrapper for TORCS (vtorcs-RL-color).
Registers a custom environment with OmniSafe's env_register so that
omnisafe.Agent('SACPID', 'TorcsSafe-v0') works out of the box.

Implements 33 engineered features for a Corkscrew hotlap and returns
a per-step binary cost for the SACPID Lagrangian constraint.
"""

import torch
import numpy as np
from gymnasium import spaces

from omnisafe.envs.core import CMDP, env_register
from gym_torcs import TorcsEnv


@env_register
class TorcsSafeEnv(CMDP):
    """TORCS environment wrapped as an OmniSafe CMDP."""

    # OmniSafe discovers environments by this class variable.
    _support_envs = ['TorcsSafe-v0']

    # OmniSafe wrappers we do NOT need (we handle resets/time limits ourselves)
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False

    _action_space: spaces.Box
    _observation_space: spaces.Box

    def __init__(self, env_id: str, **kwargs) -> None:
        super().__init__(env_id)
        self._num_envs = 1  # required by OmniSafe adapter

        self.track_limit: float = kwargs.get('track_limit', 1.05)

        # Base TORCS environment (vision=False, throttle=True for full control)
        self._env = TorcsEnv(vision=False, throttle=True, gear_change=False)

        # Action: [steering, accel]  both in [-1, 1]
        self._action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # Observation: 33 engineered features
        self._num_obs = 33
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._num_obs,), dtype=np.float32,
        )

        self._alpha = 0.2  # EMA smoothing factor for G-force features

        self._reset_state_tracking()

    # ------------------------------------------------------------------
    # Required CMDP interface
    # ------------------------------------------------------------------

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    def set_seed(self, seed: int) -> None:
        pass  # TORCS doesn't support seeded runs

    def close(self) -> None:
        self._env.end()

    def render(self):
        pass  # TORCS renders its own window

    def save(self):
        return {}

    # ------------------------------------------------------------------
    # step / reset
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        """
        OmniSafe CMDP.step must return:
          (obs, reward, cost, terminated, truncated, info)
        all as torch.Tensors (scalars for reward/cost/terminated/truncated).
        """
        # Convert torch action → numpy for TORCS
        np_action = action.detach().cpu().numpy()

        # Step the base environment
        # gym_torcs returns: (obs_namedtuple, reward, done_bool, truncated_bool, info_dict)
        raw_obs, reward, done, _truncated, _info = self._env.step(np_action)

        # Build engineered state vector
        obs_np = self._engineer_features(raw_obs)

        # ---- CMDP cost calculation ----
        cost = 0.0
        try:
            track_pos = self._env.client.S.d['trackPos']
            angle     = self._env.client.S.d['angle']
            speedX    = self._env.client.S.d['speedX']
        except Exception:
            track_pos = 0.0
            angle     = 0.0
            speedX    = float(raw_obs.speedX) * self._env.default_speed

        # Constraint 1/2: track-limit violation
        if abs(track_pos) > self.track_limit:
            cost = 1.0

        # Constraint 3: kinematic failure (backwards / reversing)
        if abs(angle) > (np.pi / 2.0) or speedX < -1.0:
            cost = 1.0
            done = True

        # Collision / damage proxy (gym_torcs returns reward == -1 on damage)
        if reward == -1:
            cost = 1.0

        terminated = bool(done)
        truncated  = False

        obs_np = np.nan_to_num(obs_np)

        return (
            torch.as_tensor(obs_np, dtype=torch.float32),
            torch.as_tensor(reward, dtype=torch.float32),
            torch.as_tensor(cost, dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.bool),
            torch.as_tensor(truncated, dtype=torch.bool),
            {},
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        OmniSafe CMDP.reset must return: (obs_tensor, info_dict)
        """
        self._reset_state_tracking()

        # gym_torcs.reset returns (obs_namedtuple, {})
        raw_obs, _ = self._env.reset(relaunch=True)

        obs_np = self._engineer_features(raw_obs)

        return torch.as_tensor(obs_np, dtype=torch.float32), {}

    # ------------------------------------------------------------------
    # Feature engineering  (33 features)
    # ------------------------------------------------------------------

    def _reset_state_tracking(self) -> None:
        """Reset historical trackers for derivative features (G-force etc.)."""
        self.prev_speedX = 0.0
        self.prev_speedY = 0.0
        self.prev_speedZ = 0.0
        self.ema_accelX  = 0.0
        self.ema_accelY  = 0.0
        self.ema_accelZ  = 0.0

    def _engineer_features(self, raw_obs) -> np.ndarray:
        """
        Turn a gym_torcs namedtuple into a flat 33-dim numpy array.

        Layout (33):
          [0-3]   speedX, speedY, speedZ, trackPos    (4 base kinematics)
          [4-5]   angle, rpm                           (2 car state)
          [6-13]  slip_angle, tire_slip, accelX,       (8 derived features)
                  accelY, accelZ, curvature,
                  panic_ttb, placeholder
          [14-32] track sensors (19 lidar rays)
        """
        norm_speedX = float(raw_obs.speedX)
        norm_speedY = float(raw_obs.speedY)
        norm_speedZ = float(raw_obs.speedZ)

        # Track-edge lidar (19 values, already /200 in gym_torcs)
        track_sensors = np.asarray(raw_obs.track, dtype=np.float32)

        # ---- trackPos & angle from the raw client dict ----
        try:
            track_pos = float(self._env.client.S.d['trackPos'])
        except Exception:
            track_pos = 0.0

        try:
            angle = float(self._env.client.S.d['angle']) / np.pi
        except Exception:
            angle = 0.0

        # ---- 1. Chassis slip angle ----
        slip_angle = float(np.arctan2(norm_speedY, norm_speedX + 1e-8)) / (np.pi / 2.0)

        # ---- 2. Longitudinal tire slip (rear wheels) ----
        avg_rear_spin = (raw_obs.wheelSpinVel[2] + raw_obs.wheelSpinVel[3]) / 2.0 / 100.0
        tire_slip_delta = float(np.clip(avg_rear_spin - norm_speedX, -1.0, 1.0))

        # ---- 3. EMA G-forces (X, Y) ----
        raw_ax = norm_speedX - self.prev_speedX
        self.ema_accelX = (1.0 - self._alpha) * self.ema_accelX + self._alpha * raw_ax
        self.prev_speedX = norm_speedX

        raw_ay = norm_speedY - self.prev_speedY
        self.ema_accelY = (1.0 - self._alpha) * self.ema_accelY + self._alpha * raw_ay
        self.prev_speedY = norm_speedY

        norm_accelX = float(np.clip(self.ema_accelX / 10.0, -1.0, 1.0))
        norm_accelY = float(np.clip(self.ema_accelY / 10.0, -1.0, 1.0))

        # ---- 4. EMA vertical G-force (suspension unload) ----
        raw_az = norm_speedZ - self.prev_speedZ
        self.ema_accelZ = (1.0 - self._alpha) * self.ema_accelZ + self._alpha * raw_az
        self.prev_speedZ = norm_speedZ

        norm_accelZ = float(np.clip(self.ema_accelZ / 5.0, -1.0, 1.0))

        # ---- 5. Track curvature ----
        # track_sensors are ALREADY normalised (/200) in gym_torcs, so
        # the delta is already in a sensible [-1, 1] range.  Do NOT divide
        # by 200 again (that was a bug in the previous version).
        curve_delta = float(track_sensors[18] - track_sensors[0])

        # ---- 6. Time-to-boundary panic ----
        dist_to_edge = 1.0 - abs(track_pos)
        moving_outward = (track_pos * norm_speedY) > 0
        if moving_outward and abs(norm_speedY) > 0.05:
            ttb = dist_to_edge / (abs(norm_speedY) + 1e-6)
            panic_ttb = float(np.clip(1.0 / ttb, 0.0, 1.0))
        else:
            panic_ttb = 0.0

        # ---- Assemble 33-dim vector ----
        rpm_norm = float(raw_obs.rpm) / 10000.0

        state = np.array([
            # Base kinematics (4)
            norm_speedX, norm_speedY, norm_speedZ, track_pos,
            # Car state (2)
            angle, rpm_norm,
            # Derived features (8)
            slip_angle, tire_slip_delta,
            norm_accelX, norm_accelY, norm_accelZ,
            curve_delta, panic_ttb,
            0.0,   # placeholder (expand later, e.g. lap-time delta)
        ], dtype=np.float32)

        # Append 19 lidar rays
        state = np.concatenate([state, track_sensors])

        return state
