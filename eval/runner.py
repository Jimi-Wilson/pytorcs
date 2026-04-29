"""
Model loading and single-episode execution.

No printing, no file I/O — returns EpisodeResult only.
evaluate.py owns all output.

Requires TORCS to be running before load_ppo() is called.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Allow imports from SACPID/ regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent / "SACPID"))


@dataclass
class EpisodeResult:
    # Race progress
    dist_raced_m: float
    laps_in_attempt: int            # number of laps completed in this attempt (via lastLapTime)
    lap_completed: bool             # derived: laps_in_attempt > 0
    lap_time: float | None          # best lap time in this attempt; None if no lap completed
    all_lap_times: list             # all per-lap times (for multi-lap attempts)

    # Speed
    mean_speed_kmh: float
    max_speed_kmh: float
    min_speed_kmh: float            # slowest point (cornering speed)

    # Track position quality
    max_abs_track_pos: float        # worst lateral excursion (0=centre, 1=edge)
    mean_abs_track_pos: float       # average centering quality (lower = better)

    # Driving smoothness
    steering_smoothness: float      # std dev of steer actions (lower = smoother)

    # Off-track incidents
    off_track_events: int           # rising-edge count of |trackPos| > 1.0

    # Episode info
    termination_reason: str
    step_count: int

    # REWARD_REMOVED — uncomment to restore reward monitoring
    # total_reward: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Checkpoint resolution (adapted from SACPID/run_ppo.py)
# ---------------------------------------------------------------------------

def _resolve_ppo_checkpoint(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_file() and p.suffix == ".zip":
        return p
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")
    if not p.is_dir():
        raise FileNotFoundError(f"Expected .zip file or directory, got: {p}")

    marker = p / "latest_model_path.txt"
    if marker.exists():
        target = Path(marker.read_text(encoding="utf-8").strip()).expanduser().resolve()
        if target.exists() and target.suffix == ".zip":
            return target

    cands = sorted(p.glob("**/*.zip"), key=lambda x: x.stat().st_mtime)
    if cands:
        return cands[-1]
    raise FileNotFoundError(f"No .zip checkpoint found under: {p}")


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_ppo(
    checkpoint: str,
    port: int = 3001,
    policy_action_dim: int = 3,
    torcs_host: str = "localhost",
) -> tuple[Any, Any]:
    """Load a PPO checkpoint and create the environment.

    Returns (model, env). Caller must call env.close() when done.
    """
    # snakeoil3 reads --host from sys.argv in every Client.__init__ (called each env.reset()).
    # Set it here once; the value persists for the lifetime of the process.
    sys.argv = [sys.argv[0], "--host", torcs_host]

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "stable_baselines3 is required. Install from eval/requirements.txt."
        ) from exc

    try:
        from sacpid_ppo_env import SacpidPpoEnv
    except ImportError as exc:
        raise RuntimeError(
            "sacpid_ppo_env not found. Ensure the SACPID/ directory is present "
            "at the repo root (run: git checkout install_script)."
        ) from exc

    ckpt = _resolve_ppo_checkpoint(checkpoint)

    # Stage 4 is the evaluation default (full curriculum, all manoeuvres active).
    # Suppress the _TorcsSimSlot startup prints (port/stage/brake config lines).
    with contextlib.redirect_stdout(io.StringIO()):
        env = SacpidPpoEnv(
            stage=4,
            port=int(port),
            policy_action_dim=int(policy_action_dim),
        )
    model = PPO.load(str(ckpt))
    return model, env


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    model: Any,
    env: Any,
    *,
    deterministic: bool = True,
    verbose: bool = False,
    seed: int | None = None,
    step_callback: Any = None,
) -> EpisodeResult:
    """Run one episode and return collected race statistics.

    The env's built-in episode-end print is suppressed unless verbose=True.
    """
    obs, _ = env.reset(seed=seed)

    speeds: list[float] = []
    steers: list[float] = []
    track_positions: list[float] = []
    step_count = 0
    off_track_events = 0
    prev_off_track = False

    # Lap detection via SCR lastLapTime field — TORCS updates this each time the
    # car crosses the start/finish line; we watch for changes to count laps.
    lap_times_list: list[float] = []
    prev_last_lap = 0.0

    # REWARD_REMOVED — uncomment to restore reward monitoring
    # total_reward = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)

        # Suppress the SacpidPpoEnv episode-end print unless verbose
        ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with ctx:
            obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        # REWARD_REMOVED: total_reward += float(reward)

        tele: dict[str, Any] = getattr(env, "_last_tele", {}) or {}

        speed = float(tele.get("speedX_kmh", 0.0))
        speeds.append(speed)

        steer = float(action[0]) if hasattr(action, "__len__") else float(action)
        steers.append(steer)

        track_pos = tele.get("track_pos")
        if track_pos is not None:
            tp = float(track_pos)
            track_positions.append(abs(tp))
            off = abs(tp) > 1.0
            if off and not prev_off_track:
                off_track_events += 1
            prev_off_track = off

        dist = float(tele.get("dist_raced_m", 0.0))

        # Check SCR lastLapTime — non-zero and changed means a lap was just completed
        try:
            cur_last_lap = float(env._slot._env.client.S.d.get("lastLapTime", 0.0))
        except (AttributeError, KeyError, TypeError):
            cur_last_lap = 0.0
        if cur_last_lap > 0.0 and cur_last_lap != prev_last_lap:
            lap_times_list.append(cur_last_lap)
            prev_last_lap = cur_last_lap

        if step_callback is not None:
            step_callback({
                "speed":     round(speed, 1),
                "steer":     round(steer, 4),
                "track_pos": round(float(track_pos), 4) if track_pos is not None else 0.0,
                "dist":      round(dist, 1),
                "step":      step_count,
            })

        if verbose:
            tp_str = f"{float(track_pos):.3f}" if track_pos is not None else "n/a"
            print(f"  step={step_count} speed={speed:.1f}km/h dist={dist:.1f}m trackPos={tp_str}")

        if bool(terminated) or bool(truncated):
            break

    reason = str(info.get("termination_reason", "unknown"))
    dist_m = float(info.get("sacpid_dist_raced_m", 0.0))
    laps_in_attempt = len(lap_times_list)
    lap_completed   = laps_in_attempt > 0
    lap_time        = min(lap_times_list) if lap_times_list else None

    mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
    max_speed  = max(speeds) if speeds else 0.0
    min_speed  = min(speeds) if speeds else 0.0

    max_abs_tp  = float(info.get("sacpid_max_abs_track_pos", max(track_positions) if track_positions else 0.0))
    mean_abs_tp = sum(track_positions) / len(track_positions) if track_positions else 0.0

    steer_smooth = _std(steers)

    return EpisodeResult(
        dist_raced_m=dist_m,
        laps_in_attempt=laps_in_attempt,
        lap_completed=lap_completed,
        lap_time=lap_time,
        all_lap_times=lap_times_list,
        mean_speed_kmh=mean_speed,
        max_speed_kmh=max_speed,
        min_speed_kmh=min_speed,
        max_abs_track_pos=max_abs_tp,
        mean_abs_track_pos=mean_abs_tp,
        steering_smoothness=steer_smooth,
        off_track_events=off_track_events,
        termination_reason=reason,
        step_count=step_count,
        # REWARD_REMOVED: total_reward=total_reward,
    )
