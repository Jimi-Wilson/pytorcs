"""TORCS wrapper config presets for the four-stage PPO Corkscrew curriculum.

The current training path is PPO-first. Stages are reward scaffolding for the
Corkscrew track (``./autostart.sh 2``), with anti-swerve terms active early and
without any Stage 1 centerline/center-track bonus.
"""

from __future__ import annotations


def _track_limit_for_stage(stage: int) -> float:
    """Hard lateral bound for wrapper ``off_track`` (|trackPos| > limit)."""
    if stage == 1:
        return 1.00
    if stage == 2:
        return 1.05
    if stage == 3:
        return 1.08
    if stage == 4:
        return 1.12
    if stage == 5:
        return 1.12
    raise ValueError(f"stage must be 1-5 for PPO, got {stage!r}")


PPO_STAGE_TRAINING_DEFAULTS: dict[int, dict[str, float | int]] = {
    1: {
        "total_steps": 1_000_000,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.990,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "ent_coef": 0.006,
    },
    2: {
        "total_steps": 1_500_000,
        "learning_rate": 2.5e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.992,
        "gae_lambda": 0.95,
        "clip_range": 0.18,
        "ent_coef": 0.005,
    },
    3: {
        "total_steps": 2_000_000,
        "learning_rate": 2.0e-4,
        "n_steps": 4096,
        "batch_size": 512,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.96,
        "clip_range": 0.16,
        "ent_coef": 0.004,
    },
    4: {
        "total_steps": 2_500_000,
        "learning_rate": 1.5e-4,
        "n_steps": 4096,
        "batch_size": 512,
        "n_epochs": 10,
        "gamma": 0.997,
        "gae_lambda": 0.97,
        "clip_range": 0.15,
        "ent_coef": 0.003,
    },
    5: {
        "total_steps": 1_000_000,
        "learning_rate": 1e-5,
        "n_steps": 8192,
        "batch_size": 512,
        "n_epochs": 4,
        "gamma": 0.997,
        "gae_lambda": 0.97,
        "clip_range": 0.08,
        "ent_coef": 5e-5,
    },
}


CORKSCREW_RACING_LINE_SECTORS: list[dict[str, float | str]] = [
    # Coarse, soft windows over TORCS distFromStart. These are intentionally
    # wider than a true apex map so a slightly inaccurate track calibration only
    # nudges the PPO policy instead of dominating progress reward.
    {
        "name": "turn2_setup",
        "start_m": 180.0,
        "end_m": 430.0,
        "target_min": -0.85,
        "target_max": -0.25,
        "wrong_half": -1.0,
        "wrong_half_weight": 0.020,
        "weight": 0.030,
    },
    {
        "name": "turn2_apex",
        "start_m": 430.0,
        "end_m": 650.0,
        "target_min": 0.10,
        "target_max": 0.65,
        "wrong_half": 1.0,
        "wrong_half_weight": 0.020,
        "weight": 0.035,
    },
    {
        "name": "turn3_setup",
        "start_m": 780.0,
        "end_m": 960.0,
        "target_min": 0.20,
        "target_max": 0.80,
        "wrong_half": 1.0,
        "wrong_half_weight": 0.018,
        "weight": 0.028,
    },
    {
        "name": "turn3_apex",
        "start_m": 960.0,
        "end_m": 1120.0,
        "target_min": -0.65,
        "target_max": -0.05,
        "wrong_half": -1.0,
        "wrong_half_weight": 0.018,
        "weight": 0.032,
    },
    {
        "name": "turn4_apex",
        "start_m": 1250.0,
        "end_m": 1460.0,
        "target_min": 0.05,
        "target_max": 0.65,
        "wrong_half": 1.0,
        "wrong_half_weight": 0.016,
        "weight": 0.030,
    },
    {
        "name": "turn5_apex",
        "start_m": 1580.0,
        "end_m": 1780.0,
        "target_min": -0.65,
        "target_max": -0.05,
        "wrong_half": -1.0,
        "wrong_half_weight": 0.018,
        "weight": 0.032,
    },
    {
        "name": "turn6_apex",
        "start_m": 1900.0,
        "end_m": 2070.0,
        "target_min": 0.05,
        "target_max": 0.60,
        "wrong_half": 1.0,
        "wrong_half_weight": 0.018,
        "weight": 0.032,
    },
    {
        "name": "corkscrew_setup",
        "start_m": 2120.0,
        "end_m": 2300.0,
        "target_min": -0.85,
        "target_max": -0.20,
        "wrong_half": -1.0,
        "wrong_half_weight": 0.025,
        "weight": 0.038,
    },
    {
        "name": "corkscrew_apex",
        "start_m": 2300.0,
        "end_m": 2480.0,
        "target_min": 0.05,
        "target_max": 0.65,
        "wrong_half": 1.0,
        "wrong_half_weight": 0.025,
        "weight": 0.042,
    },
    {
        "name": "rainey_apex",
        "start_m": 2550.0,
        "end_m": 2800.0,
        "target_min": -0.65,
        "target_max": -0.05,
        "wrong_half": -1.0,
        "wrong_half_weight": 0.022,
        "weight": 0.036,
    },
    {
        "name": "final_corner_apex",
        "start_m": 3180.0,
        "end_m": 3450.0,
        "target_min": 0.05,
        "target_max": 0.65,
        "wrong_half": 1.0,
        "wrong_half_weight": 0.020,
        "weight": 0.034,
    },
]


def sacpid_env_config_dict(
    stage: int,
    *,
    device: str = "cpu",
    port: int | None = None,
    track_limit: float | None = None,
) -> dict:
    """Return a fresh PPO env config dict for ``stage``."""
    tl = float(track_limit) if track_limit is not None else _track_limit_for_stage(stage)

    cfg: dict = {
        "track_limit": tl,
        "stage": int(stage),
        "reward_scale": 0.05,
        "reward_mode": "progress_only",
        "centerline_penalty": 0.0,
        "angle_penalty": 0.0,
        "center_reward_scale": 0.0,
        "heading_reward_scale": 0.0,
        "straight_stability_reward_scale": 0.0,
        "low_speed_threshold": 20.0,
        "low_speed_penalty": 0.0,
        "device": device,
        "watchdog_no_progress_steps": 250,
        "watchdog_dist_delta_m": 0.15,
        "watchdog_speed_kmh": 3.0,
        "loose_track_pos_limit": None,
        "auto_reset_on_loose_off_track": False,
        "obs_track_ref_length_m": 3620.0,
        "loose_track_grace_steps": 0,
        "early_episode_penalty_steps": 0,
        "early_episode_terminal_extra_penalty": 0.0,
        "early_episode_survival_bonus": 0.0,
        "brake_sector_start_m": 0.0,
        "brake_sector_end_m": 0.0,
        "brake_sector_speed_cap_kmh": 75.0,
        "brake_sector_overspeed_penalty_scale": 0.0,
        "brake_sector_brake_bonus_scale": 0.0,
        "brake_sector_min_speed_for_brake_bonus_kmh": 20.0,
        "smooth_speed_gate_lo_kmh": 20.0,
        "smooth_speed_gate_hi_kmh": 55.0,
        "smooth_curve_gate_floor": 0.30,
        "launch_phase_shaping_steps": 0,
        "launch_steer_quadratic_penalty_scale": 0.0,
        "launch_steer_straight_bonus_scale": 0.0,
        "launch_extra_lateral_penalty_scale": 0.0,
        "lateral_speed_penalty_scale": 0.0,
        "lateral_speed_penalty_threshold_kmh": 30.0,
        "steer_jerk_penalty_scale": 0.0,
        "steer_jerk_threshold": 0.4,
        "jerk_penalty_reset_grace_steps": 1,
        "jerk_penalty_recovery_trackpos_threshold": 0.35,
        "jerk_penalty_recovery_angle_threshold_rad": 0.20,
        "curve_speed_penalty_scale": 0.0,
        "curve_speed_threshold_kmh": 60.0,
        "curve_delta_min_for_penalty": 0.15,
        "curve_gate_ema_alpha": 1.0,
        "straight_stability_trackpos_threshold": 0.30,
        "straight_stability_angle_threshold_rad": 0.14,
        "line_corridor_target": 0.0,
        "line_corridor_half_width": 1.0,
        "line_corridor_penalty_scale": 0.0,
        "line_heading_penalty_scale": 0.0,
        "line_heading_tolerance_rad": 0.20,
        "sector_line_penalty_scale": 0.0,
        "sector_wrong_half_penalty_scale": 0.0,
        "corkscrew_sectors": [],
        "stable_speed_bonus_scale": 0.0,
        "stable_speed_bonus_threshold_kmh": 90.0,
        "stable_speed_max_lateral_kmh": 8.0,
        "stable_speed_max_angle_rad": 0.12,
        "stable_speed_max_abs_track_pos": 0.82,
        "action_repeat": 1,
        "steer_rate_limit_per_step": 0.0,
        "steer_smoothing_alpha": 1.0,
        "steer_jitter_flip_delta_threshold": 0.0,
        "steer_jitter_flip_delta_cap": 0.0,
        "steer_jitter_streak_gain": 0.0,
        "edge_proximity_threshold": 0.9,
        "edge_proximity_penalty_scale": 0.0,
        "unearned_decel_penalty_scale": 0.0,
        "unearned_decel_threshold_kmh_per_step": 0.5,
        "unearned_decel_min_speed_kmh": 30.0,
        "brake_utilization_bonus_scale": 0.0,
        "brake_utilization_min_brake": 0.15,
    }

    if stage == 1:
        cfg.update(
            {
                "reward_scale": 0.10,
                "watchdog_no_progress_steps": 200,
                "smooth_speed_gate_lo_kmh": 20.0,
                "smooth_speed_gate_hi_kmh": 55.0,
                "smooth_curve_gate_floor": 0.35,
                "steer_jerk_penalty_scale": 0.03,
                "steer_jerk_threshold": 0.20,
                "lateral_speed_penalty_scale": 0.015,
                "lateral_speed_penalty_threshold_kmh": 8.0,
                "edge_proximity_threshold": 0.80,
                "edge_proximity_penalty_scale": 0.05,
                "action_repeat": 3,
                "curve_gate_ema_alpha": 0.50,
                "steer_jitter_flip_delta_threshold": 0.32,
                "steer_jitter_flip_delta_cap": 0.24,
                "steer_jitter_streak_gain": 0.35,
            }
        )
    if stage == 2:
        cfg.update(
            {
                "reward_scale": 0.08,
                "line_corridor_target": 0.0,
                "line_corridor_half_width": 0.68,
                "line_corridor_penalty_scale": 0.06,
                "line_heading_penalty_scale": 0.03,
                "line_heading_tolerance_rad": 0.18,
                "smooth_speed_gate_lo_kmh": 18.0,
                "smooth_speed_gate_hi_kmh": 70.0,
                "smooth_curve_gate_floor": 0.35,
                "steer_jerk_penalty_scale": 0.05,
                "steer_jerk_threshold": 0.14,
                "lateral_speed_penalty_scale": 0.025,
                "lateral_speed_penalty_threshold_kmh": 7.0,
                "edge_proximity_threshold": 0.78,
                "edge_proximity_penalty_scale": 0.06,
                "action_repeat": 3,
                "curve_delta_min_for_penalty": 0.22,
                "curve_gate_ema_alpha": 0.35,
                "jerk_penalty_reset_grace_steps": 1,
                "jerk_penalty_recovery_trackpos_threshold": 0.45,
                "jerk_penalty_recovery_angle_threshold_rad": 0.22,
                "steer_jitter_flip_delta_threshold": 0.26,
                "steer_jitter_flip_delta_cap": 0.18,
                "steer_jitter_streak_gain": 0.50,
            }
        )
    if stage == 3:
        cfg.update(
            {
                "reward_scale": 0.075,
                "line_corridor_target": 0.0,
                "line_corridor_half_width": 0.75,
                "line_corridor_penalty_scale": 0.035,
                "line_heading_penalty_scale": 0.025,
                "line_heading_tolerance_rad": 0.20,
                "sector_line_penalty_scale": 1.0,
                "sector_wrong_half_penalty_scale": 1.0,
                "corkscrew_sectors": CORKSCREW_RACING_LINE_SECTORS,
                "slip_threshold_kmh": 18.0,
                "smooth_speed_gate_lo_kmh": 22.0,
                "smooth_speed_gate_hi_kmh": 85.0,
                "smooth_curve_gate_floor": 0.30,
                "steer_jerk_penalty_scale": 0.055,
                "steer_jerk_threshold": 0.13,
                "lateral_speed_penalty_scale": 0.025,
                "lateral_speed_penalty_threshold_kmh": 8.0,
                "edge_proximity_threshold": 0.82,
                "edge_proximity_penalty_scale": 0.05,
                "action_repeat": 2,
                "curve_delta_min_for_penalty": 0.24,
                "curve_gate_ema_alpha": 0.30,
                "jerk_penalty_reset_grace_steps": 0,
                "jerk_penalty_recovery_trackpos_threshold": 0.50,
                "jerk_penalty_recovery_angle_threshold_rad": 0.24,
                "steer_jitter_flip_delta_threshold": 0.24,
                "steer_jitter_flip_delta_cap": 0.16,
                "steer_jitter_streak_gain": 0.60,
                "brake_sector_start_m": 2120.0,
                "brake_sector_end_m": 2520.0,
                "brake_sector_speed_cap_kmh": 95.0,
                "brake_sector_overspeed_penalty_scale": 0.010,
            }
        )
    if stage == 4:
        cfg.update(
            {
                "reward_scale": 0.095,
                "line_corridor_target": 0.0,
                "line_corridor_half_width": 0.82,
                "line_corridor_penalty_scale": 0.020,
                "line_heading_penalty_scale": 0.015,
                "line_heading_tolerance_rad": 0.24,
                "sector_line_penalty_scale": 0.45,
                "sector_wrong_half_penalty_scale": 0.45,
                "corkscrew_sectors": CORKSCREW_RACING_LINE_SECTORS,
                "pace_band_v_lo": 50.0,
                "pace_band_v_hi": 220.0,
                "slip_threshold_kmh": 18.0,
                "smooth_speed_gate_lo_kmh": 35.0,
                "smooth_speed_gate_hi_kmh": 110.0,
                "smooth_curve_gate_floor": 0.25,
                "steer_jerk_penalty_scale": 0.035,
                "steer_jerk_threshold": 0.18,
                "lateral_speed_penalty_scale": 0.020,
                "lateral_speed_penalty_threshold_kmh": 12.0,
                "edge_proximity_threshold": 0.86,
                "edge_proximity_penalty_scale": 0.04,
                "action_repeat": 2,
                "curve_delta_min_for_penalty": 0.18,
                "curve_gate_ema_alpha": 0.20,
                "jerk_penalty_reset_grace_steps": 0,
                "jerk_penalty_recovery_trackpos_threshold": 0.55,
                "jerk_penalty_recovery_angle_threshold_rad": 0.28,
                "steer_jitter_flip_delta_threshold": 0.26,
                "steer_jitter_flip_delta_cap": 0.18,
                "steer_jitter_streak_gain": 0.45,
                "watchdog_no_progress_steps": 0,
                "brake_sector_start_m": 2120.0,
                "brake_sector_end_m": 2520.0,
                "brake_sector_speed_cap_kmh": 110.0,
                "brake_sector_overspeed_penalty_scale": 0.006,
                "stable_speed_bonus_scale": 0.002,
                "stable_speed_bonus_threshold_kmh": 90.0,
                "stable_speed_max_lateral_kmh": 9.0,
                "stable_speed_max_angle_rad": 0.16,
                "stable_speed_max_abs_track_pos": 0.88,
            }
        )
    if stage == 5:
        # Performance stage: warm-start from a stage-4 fine-tuned checkpoint.
        # Inherits stage-4 corridor/sector/edge shaping, then adds:
        #   - unearned-decel penalty: paying for decel only if the brake caused it
        #     (catches both lift-off coasting and steering-scrub speed-bleed)
        #   - brake-utilization bonus: rewards brake when it is causing real decel
        #   - stronger stable_speed_bonus: rewards clean high-speed motion harder
        cfg.update(
            {
                "reward_scale": 0.06,
                "line_corridor_target": 0.0,
                "line_corridor_half_width": 0.82,
                "line_corridor_penalty_scale": 0.06,
                "line_heading_penalty_scale": 0.04,
                "line_heading_tolerance_rad": 0.24,
                "sector_line_penalty_scale": 0.9,
                "sector_wrong_half_penalty_scale": 1.0,
                "corkscrew_sectors": CORKSCREW_RACING_LINE_SECTORS,
                "pace_band_v_lo": 50.0,
                "pace_band_v_hi": 220.0,
                "slip_threshold_kmh": 18.0,
                "smooth_speed_gate_lo_kmh": 35.0,
                "smooth_speed_gate_hi_kmh": 110.0,
                "smooth_curve_gate_floor": 0.25,
                "steer_jerk_penalty_scale": 0.06,
                "steer_jerk_threshold": 0.18,
                "lateral_speed_penalty_scale": 0.035,
                "lateral_speed_penalty_threshold_kmh": 12.0,
                "edge_proximity_threshold": 0.86,
                "edge_proximity_penalty_scale": 0.08,
                "action_repeat": 1,
                "curve_delta_min_for_penalty": 0.18,
                "curve_gate_ema_alpha": 0.20,
                "jerk_penalty_reset_grace_steps": 0,
                "jerk_penalty_recovery_trackpos_threshold": 0.55,
                "jerk_penalty_recovery_angle_threshold_rad": 0.28,
                "steer_jitter_flip_delta_threshold": 0.26,
                "steer_jitter_flip_delta_cap": 0.18,
                "steer_jitter_streak_gain": 0.45,
                "watchdog_no_progress_steps": 0,
                "brake_sector_start_m": 2120.0,
                "brake_sector_end_m": 2520.0,
                "brake_sector_speed_cap_kmh": 110.0,
                "brake_sector_overspeed_penalty_scale": 0.012,
                "stable_speed_bonus_scale": 0.004,
                "stable_speed_bonus_threshold_kmh": 90.0,
                "stable_speed_max_lateral_kmh": 9.0,
                "stable_speed_max_angle_rad": 0.16,
                "stable_speed_max_abs_track_pos": 0.88,
                "unearned_decel_penalty_scale": 0.08,
                "unearned_decel_threshold_kmh_per_step": 0.5,
                "unearned_decel_min_speed_kmh": 30.0,
                "brake_utilization_bonus_scale": 0.05,
                "brake_utilization_min_brake": 0.15,
            }
        )

    if port is not None:
        cfg["port"] = int(port)

    return cfg
