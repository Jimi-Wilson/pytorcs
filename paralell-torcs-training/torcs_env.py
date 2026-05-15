import copy
import math
import time

import gymnasium as gym
import numpy as np

from torcs_client import TorcsClient
from utils import kill_torcs_instance

class TorcsEnv(gym.Env):
    # Defining features, their size and scaling factors
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
        "rpm": (1, 10000.0),
        "speedX": (1, 300.0),
        "speedY": (1, 300.0),
        "speedZ": (1, 300.0),
        "track": (19, 200.0),
        "trackPos": (1, 2.0),
        "wheelSpinVel": (4, 100.0),
        "z": (1, 1.0)
    }

    def __init__(self, sensor_features=None, reward_function=None, reset_fn=None, truncate_limit=5000, port: int = 3001):
        super().__init__()

        self.client = None
        self.port = port
        self.time_step = 0
        self.initial_reset = True
        self.truncate_limit = truncate_limit
        self.reward_function = reward_function or self.calculate_reward
        self.reset_fn = reset_fn or (lambda p: kill_torcs_instance(p))

        # Setting default features if none are provided
        if sensor_features is None:
            self.sensor_features = ["speedX", "angle", "trackPos", "track"]
        else:
            self.sensor_features = sensor_features

        # --- ACTION SPACE ---
        # The agent controls 3 things: [Steering, Acceleration]
        # Steering: -1.0 (Right) to 1.0 (Left)
        # Accel:     -1.0 (Break) to 1.0 (Full Gas)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        total_dims = sum(self.SENSOR_CONFIG[feat][0] for feat in self.sensor_features)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dims,),
            dtype=np.float32
        )

        # Debounced auto-shifter: early upshift (high-gear bias), very late downshift; gentle confirmation.
        self._auto_gear_up_rpm = (4300, 4500, 4700, 4900, 5100, 999_999)
        self._auto_gear_down_rpm = (0, 950, 1050, 1150, 1250, 1350)
        self._auto_gear_confirm_steps = 7
        self._gear_up_streak = 0
        self._gear_dn_streak = 0

    def step(self, action):
        # Updating actions with new ones
        self.client.actions.steering = float(action[0]) * 0.5
        pedal = float(action[1])


        if pedal > 0:
            self.client.actions.accel = pedal
            raw_brake = 0.0
        else:
            self.client.actions.accel = 0.0
            raw_brake = abs(pedal)

        # Changing gears and applying ABS if needed
        self.client.actions.gear = self.change_gear(
            int(self.client.sensors.gear), float(self.client.sensors.rpm)
        )

        self.client.actions.brake = self.filter_abs(
            self.client.sensors.speedX,
            self.client.sensors.wheelSpinVel,
            raw_brake
        )

        if self.time_step < 30:
            self.client.actions.accel = 1.0
            self.client.actions.steering = 0.0
            self.client.actions.brake = 0.0


        # Sending actions to torcs
        prev_sensors = copy.deepcopy(self.client.sensors)
        self.client.send_action()
        raw_sensors = self.client.get_sensors()

        if raw_sensors is None:
            return np.zeros(self.observation_space.shape), 0.0, True, False, {}

        self.client.sensors.update(raw_sensors)

        # Processing sensors, calculating reward, and checking termination
        observation = self.process_sensors()
        reward = self.reward_function(prev_sensors, self.client.sensors)

        # Sanitise reward to prevent NaN/Inf propagation
        if not np.isfinite(reward):
            reward = 0.0

        truncated = self.time_step >= self.truncate_limit

        info = {
            "lap_time": self.client.sensors.lastLapTime,
            "dist_raced": self.client.sensors.distRaced,
            "speed_x": self.client.sensors.speedX,
            "track_pos": self.client.sensors.trackPos,
        }

        if truncated: print(f"  ⏱ Truncated  dist={info["dist_raced"]:.0f}m")
        if self.is_lap_complete(): print(f"  ✓ LAP COMPLETE  time={info["lap_time"]:.2f}s")
        if self.is_off_track(): print(f"  ✗ OFF TRACK  dist={info["dist_raced"]:.0f}m")


        # Anti-Stall code
        if self.time_step > 150 and self.client.sensors.speedX < 5.0:
            print(" ⏱ Anti-Stall activated")
            truncated = True
            reward -= 120.0

        self.time_step += 1
        return observation, reward, self.is_terminal(), truncated, info

    def reset(self, relaunch=False, seed=None, options=None, headless=True):
        super().reset(seed=seed)
        self.time_step = 0
        self._gear_up_streak = 0
        self._gear_dn_streak = 0

        if self.client is not None and self.client.sock is not None:
            self.client.sock.close()

        self.reset_fn(self.port)
        self.client = TorcsClient(self.port)
        self.initial_reset = False

        raw_obs = self.client.get_sensors()
        if raw_obs:
            self.client.sensors.update(raw_obs)
        observation = self.process_sensors()

        return observation, {}

    # Generated with IBM Granite
    def calculate_reward(self, previous, current) -> float:
        angle = current.angle
        track_pos = current.trackPos
        speed_y = current.speedY
        dist_raced = current.distRaced

        # Progress: distance covered this step, projected onto track direction
        progress = ((dist_raced - previous.distRaced) / 10) * math.cos(angle)

        # Smooth quadratic penalty outside a ±0.7 corridor — allows apex lines
        abs_pos = abs(track_pos)
        pos_penalty = 0.5 * ((abs_pos - 0.7) ** 2) if abs_pos > 0.7 else 0

        # Lateral slip penalty — discourages oscillation and oversteer
        lateral_penalty = 0.005 * abs(speed_y / 300)

        # Terminal bonuses
        terminal_bonus = 0
        if self.is_lap_complete():
            lap_time = current.lastLapTime
            terminal_bonus = 5000 + (6000 / lap_time)
            print(f"  ✓ LAP COMPLETE  time={lap_time:.2f}s  bonus={terminal_bonus:.1f}")
        elif self.is_off_track():
            partial_credit = min(dist_raced / 1000, 20)
            terminal_bonus = -100 + partial_credit
            print(f"  ✗ OFF TRACK  dist={dist_raced:.0f}m")

        return float(progress - pos_penalty - lateral_penalty + terminal_bonus)


    def is_terminal(self):
        return self.is_lap_complete() or self.is_off_track()

    def is_lap_complete(self) -> bool:
        lap_started = self.client.sensors.curLapTime > 1.0 or self.client.sensors.distRaced > 10.0
        return bool(lap_started and self.client.sensors.lastLapTime > 0.0 and self.client.sensors.curLapTime < 0.5)

    def is_off_track(self):
        return abs(self.client.sensors.trackPos) > 1.2

    def process_sensors(self) -> np.ndarray:
        """
        Converts raw sensor values into a normalised numpy array.
        """
        sensor_values = []

        for feature in self.sensor_features:
            raw_value = getattr(self.client.sensors, feature)
            length, scale = self.SENSOR_CONFIG[feature]

            array = np.atleast_1d(raw_value).astype(np.float32)

            if array.size != length:
                raise ValueError(
                    f"Sensor '{feature}' returned {array.size} values, "
                    f"expected {length}. Check SENSOR_CONFIG."
                )

            sensor_values.append(array / scale)

        obs_array = np.concatenate(sensor_values)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs_array


    # RPM auto-shifter (debounced): short-shift ups for high-gear bias; downshift only when heavily lugged.
    def change_gear(self, current_gear: int, rpm: float) -> int:
        n = self._auto_gear_confirm_steps
        up = self._auto_gear_up_rpm
        dn = self._auto_gear_down_rpm

        if current_gear < 1:
            self._gear_up_streak = self._gear_dn_streak = 0
            return 1

        new_gear = current_gear
        want_up = current_gear < 6 and rpm >= float(up[current_gear - 1])
        want_dn = current_gear > 1 and rpm <= float(dn[current_gear - 1])

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
            self._gear_up_streak = 0
            self._gear_dn_streak = 0

        if new_gear != current_gear:
            self._gear_up_streak = 0
            self._gear_dn_streak = 0
        return new_gear

    # Translated abs code from: https://computerscience.missouristate.edu/SAIL/_Files/Simulated-Car-Racing-Championship-Competition-Software-Manual.pdf
    @staticmethod
    def filter_abs(speed_x: float, wheel_spin_vel: list, brake: float) -> float:
        wheel_radius = [0.3306, 0.3306, 0.3276, 0.3276]
        abs_slip = 2.0
        abs_range = 3.0
        abs_min_speed = 3.0

        # Convert car speed from km/h to m/s
        speed_ms = speed_x / 3.6

        # When speed is lower than min speed for ABS, do nothing
        if speed_ms < abs_min_speed:
            return brake

        # Compute the speed of wheels in m/s
        wheel_speed = 0.0
        for i in range(4):
            wheel_speed += wheel_spin_vel[i] * wheel_radius[i]

        # Slip is the difference between actual speed of car and average speed of wheels
        slip = speed_ms - (wheel_speed / 4.0)

        # When slip is too high, apply ABS by reducing the braking force
        if slip > abs_slip:
            brake = brake - (slip - abs_slip) / abs_range

        # Ensure brake does not go below 0
        return max(0.0, brake)
