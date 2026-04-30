import math

def corkscrew_reward(previous, current) -> float:
    """
    Reward shaping tuned for the Corkscrew track:
    - Maximise forward progress projected onto track direction
    - Soft corridor penalty (±0.6) — tighter than default to hug the racing line
    - Speed bonus for maintaining high speed
    - Lateral slip penalty
    - Large lap-completion bonus scaled by inverse lap-time (fastest = highest reward)
    - Off-track penalty with partial distance credit
    """
    angle      = current.angle
    track_pos  = current.trackPos
    speed_x    = current.speedX
    speed_y    = current.speedY
    dist_raced = current.distRaced

    # --- Progress ---
    progress = ((dist_raced - previous.distRaced) / 10.0) * math.cos(angle)

    # --- Track-position penalty: tighter corridor than default ---
    abs_pos     = abs(track_pos)
    pos_penalty = 0.8 * ((abs_pos - 0.6) ** 2) if abs_pos > 0.6 else 0.0

    # --- Speed bonus: reward maintaining high speed (scaled to 0–0.3) ---
    speed_bonus = 0.001 * max(0.0, speed_x - 80.0)

    # --- Lateral slip penalty ---
    lateral_penalty = 0.005 * abs(speed_y / 300.0)

    # --- Angle penalty: penalise large misalignment ---
    angle_penalty = 0.3 * (angle ** 2)

    terminal_bonus = 0.0
    lap_done       = (
            current.curLapTime > 0 and
            current.distRaced   > 10.0 and
            current.lastLapTime > 0.0 and
            current.curLapTime  < 0.5
    )
    off_track = abs(track_pos) > 1.2

    if lap_done:
        lap_time       = current.lastLapTime
        terminal_bonus = 5000.0 + (10_000.0 / max(lap_time, 1.0))
        print(f"  ✓ LAP COMPLETE  time={lap_time:.2f}s  bonus={terminal_bonus:.1f}")
    elif off_track:
        partial_credit = min(dist_raced / 1000.0, 20.0)
        terminal_bonus = -150.0 + partial_credit
        print(f"  ✗ OFF TRACK  dist={dist_raced:.0f}m")

    return float(
        progress
        + speed_bonus
        - pos_penalty
        - lateral_penalty
        - angle_penalty
        + terminal_bonus
    )
