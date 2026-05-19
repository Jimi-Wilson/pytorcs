import math


def corkscrew_reward_1(previous, current, current_action=None, previous_action=None) -> float:
    angle = current.angle
    track_pos = current.trackPos
    speed_x = current.speedX
    speed_y = current.speedY
    dist_raced = current.distRaced

    # 1. Continuous Velocity Projection (The core driver of pace)
    # Rewards forward speed, punishes lateral sliding and poor entry angles.
    v_progress = (speed_x * math.cos(angle)) - (abs(speed_y) * math.sin(abs(angle)))
    speed_reward = v_progress * 0.01

    # 2. Track Limits (Free to use the tarmac, exponential punishment for the dirt)
    abs_pos = abs(track_pos)
    if abs_pos <= 1.0:
        pos_penalty = 0.0  # The agent is allowed to use the kerbs
    else:
        # Exponential cliff once the car drops a wheel off the track
        pos_penalty = 2.0 * ((abs_pos - 1.0) ** 2)

        # 3. Steering Smoothness (Minimalist)
    steer_penalty = 0.0
    if current_action is not None:
        steer = current_action[0]
        # Just enough to prevent binary, twitchy lock-to-lock steering
        steer_penalty = 0.05 * (steer ** 2)

    # 4. Terminal States
    terminal_bonus = 0.0

    lap_done = (
            current.curLapTime > 0 and
            current.distRaced > 10.0 and
            current.lastLapTime > 0.0 and
            current.curLapTime < 0.5
    )
    off_track = abs_pos > 1.2

    if lap_done:
        # Cap the maximum lap time evaluated to prevent weird negative exponential behavior
        # if the agent somehow takes 5 minutes to crawl around the track.
        lap_time = min(current.lastLapTime, 120.0)

        # Exponential time bonus
        # 5.0 is the multiplier (A)
        # 0.06 is the steepness (k)
        # 120.0 is the reference baseline time
        time_bonus = 5.0 * math.exp(0.06 * (120.0 - lap_time))

        terminal_bonus = 100.0 + time_bonus

    elif off_track:
        # Flat crash penalty, plus a slight credit for how far it survived
        partial_credit = min(dist_raced / 500.0, 10.0)
        terminal_bonus = -50.0 + partial_credit

    return float(
        speed_reward
        - pos_penalty
        - steer_penalty
        + terminal_bonus
    )