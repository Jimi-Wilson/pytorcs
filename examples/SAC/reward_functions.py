import math

from pytorcs.torcs_client import TorcsState


def corkscrew_reward_1(current_state: TorcsState, previous_state: TorcsState | None) -> float:
    angle = current_state.sensors.angle
    track_pos = current_state.sensors.trackPos
    speed_x = current_state.sensors.speedX
    speed_y = current_state.sensors.speedY
    dist_raced = current_state.sensors.distRaced

    v_progress = (speed_x * math.cos(angle)) - (abs(speed_y) * math.sin(abs(angle)))
    speed_reward = v_progress * 0.01

    abs_pos = abs(track_pos)
    if abs_pos <= 1.0:
        pos_penalty = 0.0
    else:
        pos_penalty = 2.0 * ((abs_pos - 1.0) ** 2)

    steer_penalty = 0.0
    if current_state.actions is not None:
        steer = current_state.actions.steering
        steer_penalty = 0.05 * (steer ** 2)

    terminal_bonus = 0.0

    lap_done = (
            current_state.sensors.curLapTime > 0 and
            current_state.sensors.distRaced > 10.0 and
            current_state.sensors.lastLapTime > 0.0 and
            current_state.sensors.curLapTime < 0.5
    )
    off_track = abs_pos > 1.2

    if lap_done:
        lap_time = min(current_state.sensors.lastLapTime, 120.0)

        time_bonus = 5.0 * math.exp(0.06 * (120.0 - lap_time))

        terminal_bonus = 100.0 + time_bonus

    elif off_track:
        partial_credit = min(dist_raced / 500.0, 10.0)
        terminal_bonus = -50.0 + partial_credit

    return float(
        speed_reward
        - pos_penalty
        - steer_penalty
        + terminal_bonus
    )