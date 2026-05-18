import math

from train_ppo import RunConfig


# Config for the first 1M timesteps, for teaching the model basic car control.
# The main difference is n_steps is 512 instead of 2048. So we updated the models weights more often to help speed up the stages of training.

def corkscrew_reward(previous, current, current_action=None, previous_action=None, step_count=0) -> float:
    angle      = current.angle
    track_pos  = current.trackPos
    speed_x    = current.speedX
    speed_y    = current.speedY
    dist_raced = current.distRaced

    # ── Progress ──────────────────────────────────────────────────────────
    progress = ((dist_raced - previous.distRaced) / 10.0) * math.cos(angle)

    # ── Track-position penalty ────────────────────────────────────────────
    abs_pos     = abs(track_pos)
    pos_penalty = 0.8 * ((abs_pos - 0.6) ** 2) if abs_pos > 0.6 else 0.0

    # ── Speed bonus (Smoothed) ────────────────────────────────────────────
    # Lowered the floor to 40.0 and smoothed the multiplier. 
    # This prevents the agent from doing "all or nothing" throttle to stay above 80.
    speed_bonus = 0.002 * max(0.0, speed_x - 40.0)

    # ── Lateral & Angle Penalties ─────────────────────────────────────────
    lateral_penalty = 0.005 * abs(speed_y / 300.0)
    angle_penalty   = 0.3 * (angle ** 2)

    # ── SMOOTHNESS PENALTIES (The Jitter Killers) ─────────────────────────
    steer_mag_penalty = 0.0
    steer_delta_penalty = 0.0

    if current_action is not None:
        # Assuming action[0] is steering
        steer = current_action[0]

        # 1. Magnitude Penalty: Punish holding the wheel at 100% lock
        steer_mag_penalty = 0.1 * (steer ** 2)

        # 2. Delta Penalty: Punish rapid snapping of the wheel
        if previous_action is not None:
            prev_steer = previous_action[0]
            steer_velocity = abs(steer - prev_steer)
            # Squaring it punishes violent jerks heavily, but allows smooth turning
            steer_delta_penalty = 0.5 * (steer_velocity ** 2)

    # ── Terminal bonuses / penalties ──────────────────────────────────────
    terminal_bonus = 0.0

    lap_done = (
            current.curLapTime  > 0 and
            current.distRaced   > 10.0 and
            current.lastLapTime > 0.0 and
            current.curLapTime  < 0.5
    )
    off_track = abs_pos > 1.2

    if lap_done:
        lap_time = current.lastLapTime
        time_bonus    = max(0.0, 100.0 * (1.0 - (lap_time - 80.0) / 120.0))
        terminal_bonus = 100.0 + time_bonus

    elif off_track:
        # ── EARLY DEATH FIX (The 8m Fix) ──────────────────────────────────
        # If the car crashes immediately off the spawn, nuke the reward.
        if step_count < 50:
            terminal_bonus = -100.0
        else:
            partial_credit = min(dist_raced / 1000.0, 20.0)
            terminal_bonus = -25.0 + partial_credit

    return float(
        progress
        + speed_bonus
        - pos_penalty
        - lateral_penalty
        - angle_penalty
        - steer_mag_penalty     # NEW
        - steer_delta_penalty   # NEW
        + terminal_bonus
    )

cfg = RunConfig(
    run_name="learning-the-basics",
    reward_fn       = corkscrew_reward,
    total_timesteps = 1_000_000,

    ppo_overrides = {
        "n_steps":       512,
        "batch_size":    2048,
    },

    env_overrides = {
        "sensor_features": ["speedX", "angle", "trackPos", "track"],
        "truncate_limit":  10_000,
        "frame_skip":      4,
    },

    num_envs   = 4,
    base_port  = 3001,
)
