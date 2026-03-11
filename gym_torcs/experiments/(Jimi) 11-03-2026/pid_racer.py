import numpy as np
from gym_torcs.torcs_env import TorcsEnv

PI = 3.14159265359

# ================= PARAMETERS =================
TARGET_SPEED             = 160    # Target speed in km/h
STEER_GAIN               = 25     # Reduced from 30 — was over-steering on straights
CENTERING_GAIN           = 0.3    # Slightly increased for tighter line-keeping
BRAKE_THRESHOLD          = 0.35   # Slightly lower — brake a touch earlier on big angles
ENABLE_TRACTION_CONTROL  = True

SAFE_GENTLE_CORNER_SPEED = 120    # Lowered — corkscrew gentle corners are still fast
SAFE_SHARP_CORNER_SPEED  = 65     # Back to 65 — corkscrew hairpins need it
TARGET_STRAIGHT_SPEED    = 180    # Slightly lower ceiling — less speed to scrub off
CORNER_READING           = 3.0    # Raised — detect corners earlier from side sensors
SLOW_DOWN_DISTANCE       = 80     # Raised from 60 — start braking earlier
STRAIGHT_DISTANCE        = 80     # Unchanged
BRAKING_INTENSITY        = 0.6    # Raised — more committed braking
STEERING_EFFECT          = 1.2    # Unchanged

# Action noise — adds small Gaussian perturbations to expert actions each step
# This creates variation between episodes for better NN generalisation
STEER_NOISE  = 0.03   # Std dev of steering noise  (keep small — steer is sensitive)
ACCEL_NOISE  = 0.04   # Std dev of accel noise
BRAKE_NOISE  = 0.02   # Std dev of brake noise (tiny — wrong braking = off track)
# ==============================================


# Exponential moving average to smooth the corner bias signal across steps
# Alpha=0.15 means each new reading contributes 15% — higher = more reactive
_bias_ema = 0.0
_BIAS_ALPHA = 0.15

def unpack_observation(obs: np.ndarray) -> dict:
    """Reverse TorcsEnv normalisation -> physical units."""
    S = {}
    S["speedX"]       = float(obs[0]) * 300.0
    S["angle"]        = float(obs[1]) * PI
    S["trackPos"]     = float(obs[2]) * 2.0
    S["track"]        = [float(v) * 200.0 for v in obs[3:22]]
    S["wheelSpinVel"] = None
    return S


# ================= HELPERS =================

def get_min_sensor_data(S):
    return min(min(S["track"][:9]), min(S["track"][10:]))


def is_corner(S, min_reading):
    return min_reading < CORNER_READING or S["track"][9] < S["speedX"] * 0.65


def is_straight(current_speed, forward_length):
    return current_speed >= (TARGET_SPEED - 5) and forward_length > STRAIGHT_DISTANCE


def hold_acceleration(S, safe_speed):
    return is_corner(S, get_min_sensor_data(S)) and S["speedX"] > safe_speed


def slow_down(S):
    # Widen the forward cone to sensors 6-12 to catch blind crests earlier
    max_fwd = max(S["track"][6:13])
    return max_fwd < S["speedX"] * 0.70     # More conservative ratio


def is_blind_corner(S):
    """
    Detects corkscrew-style situation: forward sensor is short but we're fast
    and side sensors are still open (so is_corner() hasn't fired yet).
    """
    forward  = S["track"][9]
    max_side = max(max(S["track"][:9]), max(S["track"][10:]))
    return forward < 60.0 and S["speedX"] > 80.0 and max_side > 40.0


def calculate_corner_speed(S):
    max_fwd = max(S["track"][8:11])
    return SAFE_SHARP_CORNER_SPEED if max_fwd < SLOW_DOWN_DISTANCE else SAFE_GENTLE_CORNER_SPEED


def calculate_steering(S) -> float:
    global _bias_ema
    steer = (S["angle"] * STEER_GAIN / PI) - (S["trackPos"] * CENTERING_GAIN)

    if is_corner(S, get_min_sensor_data(S)):
        weights = np.array([0.1, 0.2, 0.4, 0.7, 1.0, 1.0, 0.7, 0.4, 0.2])
        left_weighted  = float(np.dot(weights, S["track"][:9]))
        right_weighted = float(np.dot(weights, S["track"][10:]))

        raw_bias = right_weighted - left_weighted

        _bias_ema = _BIAS_ALPHA * raw_bias + (1.0 - _BIAS_ALPHA) * _bias_ema

        corner_correction = np.clip(_bias_ema / (4.7 * 12.0), -0.8, 0.8)
        steer -= corner_correction
    else:
        _bias_ema *= 0.85

    return float(np.clip(steer, -1.0, 1.0))


def calculate_throttle(S, current_accel: float, steer: float) -> float:
    target_speed = TARGET_SPEED
    safe_speed   = calculate_corner_speed(S)

    if is_straight(S["speedX"], S["track"][9]):
        target_speed = TARGET_STRAIGHT_SPEED

    speed_error = target_speed - S["speedX"] - (abs(steer) * STEERING_EFFECT * 10)
    if speed_error > 0:
        ramp  = np.clip(speed_error / 40.0, 0.05, 0.4)
        accel = min(1.0, current_accel + ramp)
    else:
        accel = max(0.0, current_accel - 0.2)

    if hold_acceleration(S, safe_speed):
        accel = max(0.0, current_accel - 0.2)

    if is_blind_corner(S):
        blind_cap = np.clip(S["track"][9] / 60.0, 0.0, 0.5)
        accel = min(accel, blind_cap)

    if S["speedX"] < 10:
        accel = 1.0

    return float(np.clip(accel, 0.0, 1.0))


def apply_brakes(S) -> float:
    brake = 0.0

    if abs(S["angle"]) > BRAKE_THRESHOLD:
        excess = abs(S["angle"]) - BRAKE_THRESHOLD
        brake  = np.clip(excess / 0.5, 0.0, BRAKING_INTENSITY)

    if slow_down(S):
        max_fwd = max(S["track"][6:13])
        urgency = np.clip(1.0 - max_fwd / (S["speedX"] * 0.70 + 1e-6), 0.0, 1.0)
        brake  += 0.25 * urgency

    if is_blind_corner(S):
        forward  = S["track"][9]
        severity = np.clip(1.0 - forward / 60.0, 0.0, 1.0)
        brake    = max(brake, 0.5 * severity)

    return float(np.clip(brake, 0.0, 1.0))


def traction_control(S, accel: float) -> float:
    if ENABLE_TRACTION_CONTROL and S.get("wheelSpinVel") is not None:
        wsv  = S["wheelSpinVel"]
        slip = (wsv[2] + wsv[3]) - (wsv[0] + wsv[1])
        if slip > 2:
            accel -= np.clip((slip - 2) / 10.0, 0.0, 0.3)
    return float(max(0.0, accel))


# ================= MAIN POLICY =================

def drive_modular_gym(obs: np.ndarray, prev_accel: float = 0.2,
                      add_noise: bool = False) -> np.ndarray:
    """
    Tuned rule-based policy compatible with TorcsEnv.
    Set add_noise=True during data collection to randomise each episode.
    """
    S = unpack_observation(obs)

    steer = calculate_steering(S)
    accel = calculate_throttle(S, prev_accel, steer)
    accel = traction_control(S, accel)
    brake = apply_brakes(S)

    if add_noise:
        steer = float(np.clip(steer + np.random.normal(0, STEER_NOISE), -1.0,  1.0))
        accel = float(np.clip(accel + np.random.normal(0, ACCEL_NOISE),  0.0,  1.0))
        brake = float(np.clip(brake + np.random.normal(0, BRAKE_NOISE),  0.0,  1.0))

    return np.array([steer, accel, brake], dtype=np.float32)


# ================= MAIN LOOP =================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to collect")
    parser.add_argument("--output",   type=str, default="expert_data.npz",
                        help="Output file for collected data")
    parser.add_argument("--only_completed", action="store_true",
                        help="Only save episodes where the lap was completed")
    args = parser.parse_args()

    env = TorcsEnv(
        sensor_features=["speedX", "angle", "trackPos", "track"],
        truncate_limit=100_000,
        port=3001,
    )

    all_obs, all_actions, all_rewards = [], [], []
    completed, attempted = 0, 0

    for episode in range(args.episodes):
        obs, _       = env.reset()
        done         = False
        truncated    = False
        prev_accel   = 0.2
        total_reward = 0.0
        step         = 0
        ep_obs, ep_actions, ep_rewards = [], [], []

        while not (done or truncated):
            action     = drive_modular_gym(obs, prev_accel, add_noise=True)
            prev_accel = float(action[1])

            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())

            obs, reward, done, truncated, info = env.step(action)
            ep_rewards.append(reward)
            total_reward += reward
            step         += 1

        lap_done = info.get("lap_time", 0) > 0
        attempted += 1
        if lap_done:
            completed += 1

        print(
            f"Episode {episode + 1}/{args.episodes} | "
            f"steps={step} | reward={total_reward:.1f} | "
            f"dist={info.get('dist_raced', 0):.0f}m | "
            f"lap={'✓ ' + str(round(info.get('lap_time',0),2)) + 's' if lap_done else '✗'}"
        )

        if not args.only_completed or lap_done:
            all_obs.append(np.array(ep_obs,     dtype=np.float32))
            all_actions.append(np.array(ep_actions, dtype=np.float32))
            all_rewards.append(np.array(ep_rewards, dtype=np.float32))

    if all_obs:
        obs_arr     = np.concatenate(all_obs,     axis=0)
        actions_arr = np.concatenate(all_actions, axis=0)
        rewards_arr = np.concatenate(all_rewards, axis=0)
        np.savez_compressed(args.output,
                            observations=obs_arr,
                            actions=actions_arr,
                            rewards=rewards_arr)
        print(f"\nSaved {len(obs_arr):,} transitions to {args.output}")
        print(f"  Completed laps : {completed}/{attempted}")
        print(f"  obs shape      : {obs_arr.shape}")
        print(f"  actions shape  : {actions_arr.shape}")
    else:
        print("No data saved.")