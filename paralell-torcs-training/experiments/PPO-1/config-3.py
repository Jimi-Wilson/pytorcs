import math

from train_ppo import RunConfig


# Config-3: Fine-tuning stage to push below 113s.
#
# Run with --resume to load from the latest checkpoint in corkscrew_v2/checkpoints/.
#   python train_ppo.py --config experiments/corkscrew_v2/config-3.py --resume
#
# Key changes vs config-2:
#   - Lower learning_rate (1e-4) — conservative updates near the optimum.
#   - Tighter clip_range (0.15) — smaller PPO trust-region to avoid unlearning.
#   - Higher ent_coef (0.01) — more exploration to escape the 113s plateau.
#   - Higher gamma (0.995) — longer time horizon, important for lap-level rewards.
#   - Larger n_steps (4096) — more on-policy data per update at this stage.


def corkscrew_reward_v3(previous, current) -> float:
    angle      = current.angle
    track_pos  = current.trackPos
    speed_x    = current.speedX
    speed_y    = current.speedY
    dist_raced = current.distRaced

    # ── Progress ──────────────────────────────────────────────────────────
    progress = ((dist_raced - previous.distRaced) / 10.0) * math.cos(angle)

    # ── Track-position penalty ────────────────────────────────────────────
    # Widened corridor (0.6 → 0.75) so the agent can use the full track
    # width for racing lines (apex cutting, late apex, etc.).
    abs_pos     = abs(track_pos)
    pos_penalty = 0.8 * ((abs_pos - 0.75) ** 2) if abs_pos > 0.75 else 0.0

    # ── Speed bonus ───────────────────────────────────────────────────────
    # Doubled multiplier, lower floor — creates a stronger gradient for
    # maintaining speed. At 140 km/h: +0.44 vs +0.20 in v2.
    speed_bonus = 0.004 * max(0.0, speed_x - 30.0)

    # ── Lateral & Angle Penalties ─────────────────────────────────────────
    # Halved angle penalty — the corkscrew demands body angle mid-corner.
    lateral_penalty = 0.005 * abs(speed_y / 300.0)
    angle_penalty   = 0.15 * (angle ** 2)

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
        # Steeper gradient: ~2.2 reward/s saved vs ~0.83 in v2.
        # At 113s → 104, at 100s → 133, at 80s → 178. Keeps pressure on throughout.
        time_bonus     = max(0.0, 200.0 * (1.0 - (lap_time - 70.0) / 90.0))
        terminal_bonus = 100.0 + time_bonus

    elif off_track:
        # Give partial credit based on distance travelled to encourage making it further
        partial_credit = min(dist_raced / 1000.0, 20.0)
        terminal_bonus = -30.0 + partial_credit

    return float(
        progress
        + speed_bonus
        - pos_penalty
        - lateral_penalty
        - angle_penalty
        + terminal_bonus
    )


cfg = RunConfig(
    run_name="refining-phase-2",
    reward_fn       = corkscrew_reward_v3,
    total_timesteps = 5_000_000,

    ppo_overrides={
        "learning_rate": 1e-4,
        "n_steps":       4096,
        "batch_size":    2048,
        "clip_range":    0.15,
        "ent_coef":      0.01,
        "gamma":         0.995,
        "gae_lambda":    0.95,
        "n_epochs":      10,
    },

    env_overrides = {
        "sensor_features": ["speedX", "angle", "trackPos", "track"],
        "truncate_limit":  10_000,
        "frame_skip":      4,
    },

    num_envs   = 4,
    base_port  = 3001,
)
