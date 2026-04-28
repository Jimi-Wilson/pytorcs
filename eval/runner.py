"""
Model loading and single-episode execution.

No printing, no file I/O — returns EpisodeResult only.
Caller (evaluate.py) owns all output.

Requires TORCS to be running before load_ppo() is called.
"""

from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Allow imports from SACPID/ regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent / "SACPID"))


@dataclass
class EpisodeResult:
    dist_raced_m: float
    lap_completed: bool
    lap_time: float | None      # None when lap was not completed
    mean_speed_kmh: float
    max_speed_kmh: float
    total_reward: float
    off_track_events: int
    termination_reason: str
    step_count: int

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


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_ppo(
    checkpoint: str,
    stage: int = 4,
    port: int = 3001,
    policy_action_dim: int = 3,
) -> tuple[Any, Any]:
    """Load a PPO checkpoint and create the SacpidPpoEnv.

    Returns (model, env). Caller must call env.close() when done.
    """
    # snakeoil3 parses sys.argv via getopt; clear args so our flags don't leak
    sys.argv = [sys.argv[0]]

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "stable_baselines3 is required. Install from SACPID/requirements.txt."
        ) from exc

    try:
        from sacpid_ppo_env import SacpidPpoEnv
    except ImportError as exc:
        raise RuntimeError(
            "sacpid_ppo_env not found. Ensure SACPID/ directory is present at the "
            "repo root (it is untracked — check out sacpid-clean once to populate it)."
        ) from exc

    ckpt = _resolve_ppo_checkpoint(checkpoint)
    env = SacpidPpoEnv(
        stage=int(stage),
        port=int(port),
        policy_action_dim=int(policy_action_dim),
    )
    model = PPO.load(str(ckpt))
    return model, env


def load_sacpid(checkpoint: str, **kwargs: Any) -> tuple[Any, Any]:
    raise NotImplementedError(
        "SACPID eval not yet implemented. Use --model-type ppo."
    )


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
) -> EpisodeResult:
    """Run one episode and return collected KPIs.

    The env's built-in episode-end print is suppressed unless verbose=True.
    """
    obs, _ = env.reset(seed=seed)

    speeds: list[float] = []
    total_reward = 0.0
    step_count = 0
    off_track_events = 0
    prev_off_track = False

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)

        # Suppress the SacpidPpoEnv episode-end print unless verbose
        ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with ctx:
            obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        total_reward += float(reward)

        tele: dict[str, Any] = getattr(env, "_last_tele", {}) or {}
        speed = float(tele.get("speedX_kmh", 0.0))
        speeds.append(speed)

        track_pos = tele.get("track_pos")
        if track_pos is not None:
            off = abs(float(track_pos)) > 1.0
            if off and not prev_off_track:
                off_track_events += 1
            prev_off_track = off

        if verbose:
            dist = float(tele.get("dist_raced_m", 0.0))
            tp = float(track_pos) if track_pos is not None else float("nan")
            print(f"  step={step_count} speed={speed:.1f}km/h dist={dist:.1f}m trackPos={tp:.3f}")

        if bool(terminated) or bool(truncated):
            break

    reason = str(info.get("termination_reason", "unknown"))
    raw_lap_time = info.get("lap_time")
    lap_completed = reason == "lap_complete"
    lap_time = float(raw_lap_time) if lap_completed and raw_lap_time else None
    dist_m = float(info.get("sacpid_dist_raced_m", 0.0))

    mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
    max_speed = max(speeds) if speeds else 0.0

    return EpisodeResult(
        dist_raced_m=dist_m,
        lap_completed=lap_completed,
        lap_time=lap_time,
        mean_speed_kmh=mean_speed,
        max_speed_kmh=max_speed,
        total_reward=total_reward,
        off_track_events=off_track_events,
        termination_reason=reason,
        step_count=step_count,
    )
