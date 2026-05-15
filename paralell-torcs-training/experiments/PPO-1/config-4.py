import math
import torch
from stable_baselines3 import PPO
from train_ppo import RunConfig

# ── Load the 111s teacher model once at import time ───────────────────────────
TEACHER_PATH = "/home/jw/Code/project/paralell-torcs-training/experiments/corkscrew_v2/best_models/best_lap_111.06s_step21764674"
_teacher_model = PPO.load(TEACHER_PATH, device="cpu")

def teacher_policy(obs):
    action, _ = _teacher_model.predict(obs, deterministic=True)
    return action


# ── Reward (same as v3 but with mild smoothness penalty re-added) ─────────────
def corkscrew_reward_v4(previous, current, current_action=None, previous_action=None, step_count=0) -> float:
    angle      = current.angle
    track_pos  = current.trackPos
    speed_x    = current.speedX
    speed_y    = current.speedY
    dist_raced = current.distRaced

    # Progress
    progress = ((dist_raced - previous.distRaced) / 10.0) * math.cos(angle)

    # Track position penalty (same wide corridor as v3)
    abs_pos     = abs(track_pos)
    pos_penalty = 0.8 * ((abs_pos - 0.75) ** 2) if abs_pos > 0.75 else 0.0

    # Speed bonus (same as v3)
    speed_bonus = 0.004 * max(0.0, speed_x - 30.0)

    # Angle / lateral
    lateral_penalty = 0.005 * abs(speed_y / 300.0)
    angle_penalty   = 0.15 * (angle ** 2)

    # Mild smoothness — re-added at half v2 strength
    steer_delta_penalty = 0.0
    if current_action is not None and previous_action is not None:
        steer_velocity      = abs(current_action[0] - previous_action[0])
        steer_delta_penalty = 0.25 * (steer_velocity ** 2)

    # Terminal
    terminal_bonus = 0.0
    lap_done  = (current.curLapTime > 0 and current.distRaced > 10.0
                 and current.lastLapTime > 0.0 and current.curLapTime < 0.5)
    off_track = abs_pos > 1.2

    if lap_done:
        lap_time = current.lastLapTime
        # Steeper gradient than v3: ~3.75 reward/s saved
        time_bonus     = max(0.0, 300.0 * (1.0 - (lap_time - 70.0) / 80.0))
        terminal_bonus = 100.0 + time_bonus
    elif off_track:
        if step_count < 50:
            terminal_bonus = -100.0
        else:
            partial_credit = min(dist_raced / 1000.0, 20.0)
            terminal_bonus = -30.0 + partial_credit

    return float(
        progress + speed_bonus
        - pos_penalty - lateral_penalty - angle_penalty
        - steer_delta_penalty
        + terminal_bonus
    )


cfg = RunConfig(
    reward_fn       = corkscrew_reward_v4,
    total_timesteps = 5_000_000,

    algo_overrides = {
        "learning_rate": 5e-5,
        "n_steps":       8192,
        "batch_size":    4096,
        "clip_range":    0.10,
        "ent_coef":      0.005,
        "gamma":         0.997,
        "gae_lambda":    0.97,
        "n_epochs":      15,
        "policy_kwargs": {
            "net_arch":      [512, 512, 256],
            "activation_fn": __import__("torch.nn", fromlist=["Tanh"]).Tanh,
        },
    },

    env_overrides = {
        "sensor_features": ["speedX", "angle", "trackPos", "track"],
        "truncate_limit":  10_000,
        "frame_skip":      4,
    },

    # BC: roll out the 111s teacher for 50 laps, then clone into the new net
    bc_kwargs={
        "teacher_policy": teacher_policy,
        "collection_episodes": 100,  # was 50
        "only_completed_laps": False,  # keep partial laps too
        "epochs": 30,
        "batch_size": 512,
        "lr": 1e-3,
    },

    num_envs   = 8,
    use_docker = True,
    base_port  = 3001,
)