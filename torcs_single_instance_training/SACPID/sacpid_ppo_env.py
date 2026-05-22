from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import sacpid_env_presets
import torcs_omnisafe_wrapper as tow


class SacpidPpoEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium adapter that reuses SACPID reward/cost/termination semantics."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        stage: int,
        port: int = 3001,
        policy_action_dim: int = 3,
        track_limit: float | None = None,
        brake_sector: tuple[float, float, float] | None = None,
        csv_log_dir: str | None = None,
        env_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = sacpid_env_presets.sacpid_env_config_dict(
            int(stage),
            device="cpu",
            port=int(port),
            track_limit=track_limit,
        )
        cfg["policy_action_dim"] = int(policy_action_dim)
        cfg["auto_reset_on_done"] = False
        if brake_sector is not None:
            start_m, end_m, cap_kmh = brake_sector
            cfg["brake_sector_start_m"] = float(start_m)
            cfg["brake_sector_end_m"] = float(end_m)
            cfg["brake_sector_speed_cap_kmh"] = float(cap_kmh)
            if float(cfg.get("brake_sector_overspeed_penalty_scale", 0.0)) == 0.0:
                cfg["brake_sector_overspeed_penalty_scale"] = 0.04
            if float(cfg.get("brake_sector_brake_bonus_scale", 0.0)) == 0.0:
                cfg["brake_sector_brake_bonus_scale"] = 0.10
        if env_overrides:
            cfg.update(env_overrides)

        self._cfg = cfg
        self._slot = tow._TorcsSimSlot(port=int(port), cfg=cfg, label="ppo")
        if csv_log_dir:
            tow.set_sacpid_csv_log_dir(csv_log_dir)

        # Action layout: [steer, accel] for dim=2, [steer, accel, brake] for dim=3.
        # Must match policy_action_dim so PPO sizes its output head correctly.
        _act_low  = np.array([-1.0, 0.0, 0.0], dtype=np.float32)[:int(policy_action_dim)]
        _act_high = np.array([ 1.0, 1.0, 1.0], dtype=np.float32)[:int(policy_action_dim)]
        self.action_space = spaces.Box(low=_act_low, high=_act_high, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._slot._num_obs,),
            dtype=np.float32,
        )
        self._last_tele: dict[str, Any] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        obs = self._slot.reset(relaunch=False)
        self._last_tele = {}
        return np.asarray(obs, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, rew, cost, terminated, truncated, base_info, tele = self._slot.step(action)
        self._last_tele = tele

        info: dict[str, Any] = dict(base_info)
        info["cost"] = float(cost)
        info["termination_reason"] = str(base_info.get("termination_reason", "unknown"))
        info["sacpid_episode_length"] = int(tele.get("wrapper_episode_step", 0))
        info["sacpid_max_abs_track_pos"] = float(self._slot._wandb_ep_max_abs_track_pos)
        info["sacpid_dist_raced_m"] = float(tele.get("dist_raced_m", 0.0))
        info["sacpid_dist_from_start_m"] = float(tele.get("dist_from_start_m", 0.0))
        info["speedX_kmh"] = float(tele.get("speedX_kmh", 0.0))
        info["obs_dist_along"] = float(tele.get("obs_dist_along", 0.0))
        info["raw_gym_reward"] = float(tele.get("raw_gym_reward", 0.0))
        info["raw_reward_is_neg_one"] = bool(tele.get("raw_reward_minus_one_seen", False))
        info["action_repeat"] = int(tele.get("action_repeat", 1))
        info["action_repeat_steps_executed"] = int(tele.get("action_repeat_steps_executed", 1))
        if bool(terminated) or bool(truncated):
            print(
                "[PPO env done] reason=%s len=%d cost=%.3f raw_reward=%.3f repeat=%d/%d speedX_kmh=%.2f trackPos_max=%.3f"
                % (
                    str(info.get("termination_reason", "unknown")),
                    int(info.get("sacpid_episode_length", 0)),
                    float(info.get("cost", 0.0)),
                    float(info.get("raw_gym_reward", 0.0)),
                    int(info.get("action_repeat_steps_executed", 1)),
                    int(info.get("action_repeat", 1)),
                    float(info.get("speedX_kmh", 0.0)),
                    float(info.get("sacpid_max_abs_track_pos", 0.0)),
                )
            )

        return (
            np.asarray(obs, dtype=np.float32),
            float(rew),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self) -> None:
        self._slot.close()

