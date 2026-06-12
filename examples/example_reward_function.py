import math
from pytorcs.torcs_client import TorcsState


def reward_function(current_state: TorcsState, previous_state: TorcsState | None) -> float:
    speed_x = current_state.sensors.speedX
    track_angle = current_state.sensors.angle

    progress_reward = speed_x * math.cos(track_angle)

    track_pos = current_state.sensors.trackPos
    centering_penalty = -abs(track_pos) * 2.0

    off_track_penalty = 0.0
    if abs(track_pos) > 1.0:
        off_track_penalty = -50.0

    total_reward = progress_reward + centering_penalty + off_track_penalty

    return float(total_reward)