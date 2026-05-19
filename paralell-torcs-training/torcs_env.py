import copy
import math
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType

from torcs_client import Sensor

from torcs_client import TorcsClient

_HERE = Path(__file__).parent.resolve()
if str(_HERE / "docker") not in sys.path:
    sys.path.insert(0, str(_HERE / "docker"))
from orchestrate import kill_torcs_on


class TorcsEnv(gym.Env):
    SENSOR_CONFIG = {
        "angle": (1, 3.14159),
        "curLapTime": (1, 150.0),
        "damage": (1, 10000.0),
        "distFromStart": (1, 10000.0),
        "distRaced": (1, 10000.0),
        "focus": (5, 200.0),
        "fuel": (1, 100.0),
        "gear": (1, 6.0),
        "lastLapTime": (1, 150.0),
        "opponents": (36, 200.0),
        "racePos": (1, 1.0),
        "rpm": (1, 18000.0),
        "speedX": (1, 360.0),
        "speedY": (1, 300.0),
        "speedZ": (1, 300.0),
        "track": (19, 200.0),
        "trackPos": (1, 2.0),
        "wheelSpinVel": (4, 100.0),
        "z": (1, 1.0)
    }

    def __init__(self, reward_function: Callable[[Sensor, Sensor, np.ndarray, Optional[np.ndarray]], float], sensor_features=None, truncate_limit=5000, port: int = 3001,
                 skip_reset_kill: bool = False):
        super().__init__()

        self.client = None
        self.port = port
        self.time_step = 0
        self.truncate_limit = truncate_limit
        self.reward_function = reward_function
        self.sensor_features = sensor_features or ["speedX", "angle", "trackPos", "track"]
        self.previous_action = None

        # Action space: [Steering (-1 to 1), Acceleration/Brake (-1 to 1)]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        total_dims = sum(self.SENSOR_CONFIG[feat][0] for feat in self.sensor_features)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dims,), dtype=np.float32
        )

        # Shift streaks configuration

        self._auto_gear_up_rpm = (11000, 12500, 13500, 14500, 16000, 999_999)
        self._auto_gear_down_rpm = (0, 4000, 6500, 9000, 11500, 13500)

        self._auto_gear_confirm_steps = 7
        self._gear_up_streak = 0
        self._gear_dn_streak = 0

        self.skip_reset_kill = skip_reset_kill


    def step(self, action: ActType):

        # Standardize the raw action
        raw_steer = float(action[0])

        # Apply cubic non-linear mapping
        self.client.actions.steering = raw_steer ** 3

        pedal = float(action[1])

        if pedal > 0:
            self.client.actions.accel = pedal ** 2
            raw_brake = 0.0
        else:
            self.client.actions.accel = 0.0
            raw_brake = abs(pedal)

        self.client.actions.gear = self.change_gear(int(self.client.sensors.gear), float(self.client.sensors.rpm))
        self.client.actions.brake = self.filter_abs(self.client.sensors.speedX, self.client.sensors.wheelSpinVel,
                                                    raw_brake)

        # Startup launch override
        if self.time_step < 100:
            self.client.actions.accel = 1.0
            self.client.actions.steering = 0.0
            self.client.actions.brake = 0.0
            self.client.actions.gear = 1

        prev_sensors = copy.deepcopy(self.client.sensors)
        self.client.send_action()
        raw_sensors = self.client.get_sensors()

        if raw_sensors is None:
            return np.zeros(self.observation_space.shape), 0.0, True, False, {}

        self.client.sensors.update(raw_sensors)
        observation = self.process_sensors()
        reward = self.reward_function(
            prev_sensors,
            self.client.sensors,
            action,
            self.previous_action
        )

        if not np.isfinite(reward):
            reward = 0.0

        truncated = self.time_step >= self.truncate_limit
        anti_stall = False

        velocity = math.sqrt((self.client.sensors.speedX ** 2) + (self.client.sensors.speedY ** 2) + (self.client.sensors.speedZ ** 2))

        if self.time_step > 150 and velocity < 5.0:
            truncated = True
            reward -= 120.0
            anti_stall = True

        info = {
            "lap_time": self.client.sensors.lastLapTime,
            "dist_raced": self.client.sensors.distRaced,
            "speed_x": self.client.sensors.speedX,
            "track_pos": self.client.sensors.trackPos,
            "status_flag": "LAP_COMPLETE" if self.is_lap_complete() else
            "OFF_TRACK" if self.is_off_track() else
            "ANTI_STALL" if anti_stall else
            "TRUNCATED" if truncated else "ACTIVE"
        }

        self.previous_action = copy.deepcopy(action)

        self.time_step += 1
        return observation, reward, self.is_terminal(), truncated, info

    def reset(self, seed: int=None, options: dict[str, Any]=None) -> np.ndarray:
        super().reset(seed=seed)
        self.time_step = 0
        self._gear_up_streak = 0
        self._gear_dn_streak = 0

        if self.client is not None and self.client.sock is not None:
            self.client.sock.close()

        if not self.skip_reset_kill:
            kill_torcs_on(self.port)
        else:
            print("\n[INFO] Visual Mode: Bypassing container process reset. Connecting to active window...")
        self.client = TorcsClient(self.port)

        raw_obs = self.client.get_sensors()
        if raw_obs:
            self.client.sensors.update(raw_obs)

        return self.process_sensors(), {}

    def is_terminal(self) -> bool:
        return self.is_lap_complete() or self.is_off_track()

    def is_lap_complete(self) -> bool:
        lap_started = self.client.sensors.curLapTime > 1.0 or self.client.sensors.distRaced > 10.0
        return bool(lap_started and self.client.sensors.lastLapTime > 0.0 and self.client.sensors.curLapTime < 0.5)

    def is_off_track(self) -> bool:
        return abs(self.client.sensors.trackPos) > 1.2

    def process_sensors(self) -> np.ndarray:
        sensor_values = []
        for feature in self.sensor_features:
            raw_value = getattr(self.client.sensors, feature)
            length, scale = self.SENSOR_CONFIG[feature]
            array = np.atleast_1d(raw_value).astype(np.float32)

            if array.size != length:
                raise ValueError(f"Sensor '{feature}' got {array.size} elements, expected {length}.")
            sensor_values.append(array / scale)

        obs_array = np.concatenate(sensor_values)
        return np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=-1.0)


    # RPM auto-shifter (debounced): short-shift ups for high-gear bias; downshift only when heavily lugged.
    def change_gear(self, current_gear: int, rpm: float) -> int:
        n = self._auto_gear_confirm_steps
        if current_gear < 1:
            self._gear_up_streak = self._gear_dn_streak = 0
            return 1

        new_gear = current_gear
        want_up = current_gear < 6 and rpm >= float(self._auto_gear_up_rpm[current_gear - 1])
        want_dn = current_gear > 1 and rpm <= float(self._auto_gear_down_rpm[current_gear - 1])

        if want_up and not want_dn:
            self._gear_dn_streak = 0
            self._gear_up_streak += 1
            if self._gear_up_streak >= n:
                new_gear = current_gear + 1
        elif want_dn and not want_up:
            self._gear_up_streak = 0
            self._gear_dn_streak += 1
            if self._gear_dn_streak >= n:
                new_gear = current_gear - 1
        else:
            self._gear_up_streak = self._gear_dn_streak = 0

        if new_gear != current_gear:
            self._gear_up_streak = self._gear_dn_streak = 0
        return new_gear

    # Translated abs code from: https://computerscience.missouristate.edu/SAIL/_Files/Simulated-Car-Racing-Championship-Competition-Software-Manual.pdf
    @staticmethod
    def filter_abs(speed_x: float, wheel_spin_vel: list, brake: float) -> float:
        wheel_radius = [0.3306, 0.3306, 0.3276, 0.3276]
        abs_slip, abs_range, abs_min_speed = 2.0, 3.0, 3.0
        speed_ms = speed_x / 3.6

        if speed_ms < abs_min_speed:
            return brake

        wheel_speed = sum(wheel_spin_vel[i] * wheel_radius[i] for i in range(4))
        slip = speed_ms - (wheel_speed / 4.0)

        if slip > abs_slip:
            brake = brake - (slip - abs_slip) / abs_range

        return max(0.0, brake)