from __future__ import annotations

"""
OmniSafe CMDP wrapper for TORCS (vtorcs-RL-color).
Registers a custom environment with OmniSafe's env_register so that
omnisafe.Agent('SACPID', 'TorcsSafe-v0') works out of the box.

Implements 33 engineered features for a Corkscrew hotlap and returns
a per-step binary cost for the SACPID Lagrangian constraint. Fixed-prefix
index 13 is lap-wrapped normalized ``distFromStart`` (see ``normalized_dist_along_track_obs``).

True multi-sim (N TORCS UDP clients) is opt-in via ``TorcsVectorSafe-v0``;
default ``TorcsSafe-v0`` remains a single simulator.

``TorcsHotlapEval-v0`` subclasses the same stack with eval-oriented defaults
(``auto_reset_on_done``, ``gym_reset_relaunch``, ``reward_mode='gym_raw'``,
``wrapper_terminal``, ``gym_terminate_on_off_track``); see ``run_model.py``.

Optional ``ENV_CONFIG`` keys (training defaults preserve prior behavior):

- ``auto_reset_on_done`` (bool, default True): if False, ``step`` does not call
  ``reset`` after a terminal step (hotlap).
- ``gym_reset_relaunch`` (bool, default True): passed to ``TorcsEnv.reset(relaunch=…)``.
- ``reward_mode`` ``gym_raw``: per-step reward is the gym progress signal (clipped);
  step cost is forced to 0.
- ``wrapper_terminal`` (bool, default True): if False, skip wrapper-added
  terminations (track_limit, angle/speed, watchdog); gym ``meta`` / disconnect still apply.
- ``gym_terminate_on_off_track`` (bool | None): if set, passed to ``TorcsEnv``;
  if None, ``TORCS_OFFTRACK_TERMINATION`` env or gym default applies.
- ``policy_action_dim`` (int, default 3): OmniSafe policy width. Use ``2`` for older
  checkpoints trained as ``[steer, accel]`` (brake forced to 0 in ``_TorcsSimSlot.step``).
- ``loose_track_pos_limit`` (float | None): if set and ``|trackPos|`` exceeds it, force
  ``done`` with ``termination_reason='off_track_loose'`` (independent of ``wrapper_terminal``).
  **Training** (`train_sacpid.py`) sets this to ``None`` so tight RL terminations stay
  ``off_track`` vs ``|trackPos|`` vs ``track_limit``. **Hotlap / eval**
  (`TorcsHotlapEval-v0`, ``run_model.py``) defaults to 1.25; increase for a looser reset.
- ``corkscrew_sector_start_m`` / ``corkscrew_sector_end_m`` / ``corkscrew_sector_bonus_mult``:
  only when ``reward_mode='corkscrew_sector_shaping'`` (optional / legacy); scales line/pace bonuses
  when ``distFromStart`` lies in ``[start, end]``.
- ``brake_sector_*``: when ``brake_sector_end_m > brake_sector_start_m`` (after min/max), applies
  overspeed penalty and optional brake bonus inside that ``distFromStart`` window (meters).
- ``auto_reset_on_loose_off_track`` (bool, default False): if True, call ``reset`` after
  ``done`` only when ``termination_reason == 'off_track_loose'`` (unless
  ``auto_reset_on_done`` is already True).
- ``obs_track_ref_length_m`` (float, default 3620): reference lap length in meters for the
  14th engineered scalar (index 13): lap-wrapped normalized ``distFromStart``; see
  ``normalized_dist_along_track_obs``. Set ``<= 0`` to force that scalar to ``0.0`` (legacy).
- ``loose_track_grace_steps`` (int, default 0): skip ``off_track_loose`` for wrapper steps ``1..N``.
- ``early_episode_penalty_steps`` (int, default 0), ``early_episode_terminal_extra_penalty`` (float),
  ``early_episode_survival_bonus`` (float): cold-start shaping when not ``gym_raw`` (0 disables).
- ``launch_phase_shaping_steps`` (int, default 0): if ``>0``, for wrapper steps ``1..N`` only (not ``gym_raw``,
  not ``hard_terminal_failure``): add launch-phase steering/lateral shaping before the usual reward clip.
  ``launch_steer_quadratic_penalty_scale`` subtracts ``scale * steer^2``; ``launch_steer_straight_bonus_scale``
  adds ``scale * (1 - |steer|)``; ``launch_extra_lateral_penalty_scale`` subtracts ``scale * |trackPos|`` on
  top of the normal ``centerline_part``. Set ``N=0`` or all scales ``0`` to disable.
- ``lateral_speed_penalty_scale`` (float, default 0): penalise lateral sliding. Subtracts
  ``curve_gate * scale * max(|speedY_kmh| - threshold, 0)`` each step, where ``curve_gate`` fades
  from 1.0 on straights toward 0.0 in tighter corners.
- ``steer_jerk_penalty_scale`` (float, default 0): penalise large single-step steering changes
  (overcorrections). Subtracts ``curve_gate * scale * max(|Δsteer| - threshold, 0)``.
  ``steer_jerk_threshold`` (default 0.4).
- ``jerk_penalty_reset_grace_steps`` (int, default 1): disable jerk penalty for the first
  few wrapper steps after reset to avoid the ``_prev_steer=0`` cold-start spike.
- ``jerk_penalty_recovery_trackpos_threshold`` (float, default 0.35) and
  ``jerk_penalty_recovery_angle_threshold_rad`` (float, default 0.20): reduce jerk
  penalty when the car still needs obvious correction (large |trackPos| or |angle|).
- ``curve_gate_ema_alpha`` (float, default 1.0): optional smoothing for curvature-gated
  penalties/rewards (1.0 = no smoothing; smaller values reduce gate flicker).
- ``straight_stability_reward_scale`` (float, default 0): reward stable straight-line
  motion using ``curve_gate`` and bounded center/heading closeness.
- ``steer_rate_limit_per_step`` (float, default 0): cap per-step steer delta applied
  to the simulator (0 disables).
- ``steer_smoothing_alpha`` (float, default 1): optional steer command smoothing
  before TORCS step: ``applied = alpha * target + (1-alpha) * previous``.
- ``steer_jitter_flip_delta_threshold`` (float, default 0): only when a steer sign
  flip also has ``|delta_steer|`` above this threshold do we damp it (0 disables).
- ``steer_jitter_flip_delta_cap`` (float, default 0): cap on per-step steer delta
  only for those large sign-flip events (does not cap same-direction steering).
- ``steer_jitter_streak_gain`` (float, default 0): if >0, repeated large sign-flip
  events tighten ``steer_jitter_flip_delta_cap`` by ``1/(1+streak*gain)``.
- ``curve_speed_penalty_scale`` (float, default 0): corner-aware speed penalty using track
  sensor asymmetry (``|track[18]-track[0]|``) as a curvature proxy. Subtracts
  ``scale * curve_delta * max(v_kmh - threshold, 0)`` when curve and speed exceed their
  minimums. ``curve_speed_threshold_kmh`` (default 60), ``curve_delta_min_for_penalty``
  (default 0.15, in [0,1] — track sensors are pre-divided by 200 in gym_torcs). Stage 4: scale=0.003.
- ``edge_proximity_threshold`` (float, default 0.9): |trackPos| above which the non-linear
  edge proximity penalty activates. ``edge_proximity_penalty_scale`` (float, default 0):
  subtracts ``scale * (|trackPos| - threshold)²``.
- Env: ``SACPID_WANDB_OBS_SAMPLE_PERIOD`` (sparse ``obs/*``, 0=off); ``SACPID_COLDSTART_DIAG=1`` prints
  step-1 obs for first two episodes.
"""

import csv
import socket
import threading
import time
import json
import os
import torch
import numpy as np
from gymnasium import spaces

from omnisafe.envs.core import CMDP, env_register
from gym_torcs import TorcsEnv

# Module-level config set by train_sacpid.py before Agent() is created.
ENV_CONFIG: dict = {}

# Unified termination taxonomy (W&B, CSV, console). Codes match historical episode_end/termination_code.
TERMINATION_REASON_CODES: dict[str, int] = {
    "off_track": 0,
    "too_slow": 1,
    "backwards": 2,
    "damage": 3,
    "snakeoil_disconnected": 4,
    "watchdog_timeout": 5,
    "off_track_loose": 6,
    "angle_exceeded": 7,
    "reverse_speed": 8,
    "raw_reward_minus_one": 9,
    "unknown": -1,
}

# Set by train_sacpid before agent.learn() so episode-end rows land next to progress.csv.
_SACPID_CSV_LOG_DIR: str | None = None
_sacpid_episode_csv_initialized: set[str] = set()
_sacpid_episode_csv_lock = threading.Lock()


def termination_reason_code(reason: str) -> int:
    r = str(reason).strip()
    return int(TERMINATION_REASON_CODES.get(r, -1))


_TERMINATION_ONE_HOT_KEYS = (
    "off_track",
    "off_track_loose",
    "backwards",
    "damage",
    "watchdog_timeout",
    "snakeoil_disconnected",
    "too_slow",
    "angle_exceeded",
    "reverse_speed",
    "raw_reward_minus_one",
)


def termination_reason_one_hot(tr: str) -> dict[str, float]:
    """Float 0/1 flags for W&B / CSV (unknown reasons count as term_other)."""
    t = str(tr).strip()
    d = {f"term_{k}": 1.0 if t == k else 0.0 for k in _TERMINATION_ONE_HOT_KEYS}
    known = set(_TERMINATION_ONE_HOT_KEYS)
    d["term_other"] = 0.0 if t in known else 1.0
    return d


def normalized_dist_along_track_obs(dist_from_start_m: float, ref_len_m: float) -> float:
    """Lap-local along-track position for the 33-D engineered vector (fixed-prefix index 13).

    **distFromStart** (m): TORCS SCR arc length from the start/finish line along the
    centerline; it wraps each lap (drops back near zero at S/F), so it encodes *where on
    the current lap* the car is.

    **distRaced** (m): monotonic session odometer. It is **not** used here so the signal
    repeats each lap without needing an explicit lap index (Markov-friendly “sector” cue).

    **Wrap / normalization:** ``r = distFromStart mod L_ref`` with
    ``L_ref = obs_track_ref_length_m`` (default ~3.62 km for Laguna Seca). Scalar is
    ``2 * (r / L_ref) - 1`` in ``[-1, 1)``. If ``L_ref <= 0``, returns ``0.0`` (legacy
    placeholder behavior).
    """
    L = float(ref_len_m)
    if L <= 0.0:
        return 0.0
    d = float(dist_from_start_m)
    r = d % L
    frac = r / L
    return float(2.0 * frac - 1.0)


def set_sacpid_csv_log_dir(path: str | None) -> None:
    """Trainer calls this once the OmniSafe run directory is known (before learn())."""
    global _SACPID_CSV_LOG_DIR
    _SACPID_CSV_LOG_DIR = os.path.abspath(path) if path else None


def append_sacpid_episode_termination_csv(row: dict[str, object]) -> None:
    """Append one episode summary row to sacpid_episode_terminations.csv (same folder as progress.csv)."""
    base = _SACPID_CSV_LOG_DIR
    if not base:
        return
    path = os.path.join(base, "sacpid_episode_terminations.csv")
    fieldnames = [
        "time_unix",
        "env_label",
        "termination_reason",
        "termination_code",
        "episode_length",
        "loose_track_pos_limit",
        "max_abs_track_pos",
        "dist_raced_m",
        "dist_from_start_m",
        "term_off_track",
        "term_off_track_loose",
        "term_backwards",
        "term_damage",
        "term_watchdog_timeout",
        "term_snakeoil_disconnected",
        "term_too_slow",
        "term_other",
    ]
    with _sacpid_episode_csv_lock:
        new_file = path not in _sacpid_episode_csv_initialized
        if new_file:
            _sacpid_episode_csv_initialized.add(path)
        with open(path, "a", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow(row)


# Monotonic env-step counter for wandb (one increment per env step per sim slot).
_WANDB_GLOBAL_ENV_STEP: int = 0


def _wandb_bump_global_env_step() -> int:
    global _WANDB_GLOBAL_ENV_STEP
    _WANDB_GLOBAL_ENV_STEP += 1
    return _WANDB_GLOBAL_ENV_STEP

# For multi-client: when port_base is set, each TorcsSafeEnv gets port_base + next index.
_port_lock = threading.Lock()
_port_next_index = 0

# SCR(init …) ray string must match snakeoil3_gym.Client.setup_connection
_SCR_RAY_ANGLES = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
_UDP_RECV_MAX = 2**17


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": "2e77ba",
            "runId": "scr-precheck",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        # Primary session log path plus host-local fallback so logs are captured on remote runners too.
        candidates = [
            "/home/kerem/Desktop/University/COM21002 - AI Group Project/Team Repository/project/SACPID/.cursor/debug-2e77ba.log",
            os.path.join(os.path.dirname(__file__), ".cursor", "debug-2e77ba.log"),
            os.path.join(os.getcwd(), ".cursor", "debug-2e77ba.log"),
        ]
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        for path in candidates:
            try:
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(path, "a", encoding="utf-8") as fp:
                    fp.write(line)
                break
            except Exception:
                continue
    except Exception:
        pass
    # #endregion


def verify_scr_identify(host: str = "127.0.0.1", port: int = 3001, timeout: float = 3.0) -> None:
    """Raise RuntimeError if SCR UDP handshake does not return ***identified***."""
    # #region agent log
    _debug_log(
        "H1",
        "torcs_omnisafe_wrapper.py:verify_scr_identify:entry",
        "verify_scr_identify begin",
        {
            "pid": os.getpid(),
            "host": host,
            "port": int(port),
            "timeout": float(timeout),
        },
    )
    # #endregion
    initmsg = f"SCR(init {_SCR_RAY_ANGLES})".encode()
    so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    so.settimeout(timeout)
    t0 = time.time()
    try:
        so.sendto(initmsg, (host, port))
        # #region agent log
        _debug_log(
            "H6",
            "torcs_omnisafe_wrapper.py:verify_scr_identify:post_send",
            "verify_scr_identify packet sent",
            {
                "host": host,
                "port": int(port),
                "local_socket": repr(so.getsockname()),
            },
        )
        # #endregion
        data, _ = so.recvfrom(_UDP_RECV_MAX)
    except OSError as e:
        # #region agent log
        _debug_log(
            "H2",
            "torcs_omnisafe_wrapper.py:verify_scr_identify:exception",
            "verify_scr_identify recv/send failed",
            {
                "host": host,
                "port": int(port),
                "elapsed_ms": int((time.time() - t0) * 1000),
                "error_type": type(e).__name__,
                "error": str(e),
            },
        )
        # #endregion
        raise RuntimeError(f"SCR identify UDP I/O failed on {host}:{port}: {e}") from e
    finally:
        so.close()
    if b"***identified***" not in data:
        # #region agent log
        _debug_log(
            "H3",
            "torcs_omnisafe_wrapper.py:verify_scr_identify:unexpected_reply",
            "verify_scr_identify got non-identified response",
            {
                "host": host,
                "port": int(port),
                "elapsed_ms": int((time.time() - t0) * 1000),
                "reply_prefix": repr(data[:120]),
            },
        )
        # #endregion
        raise RuntimeError(
            f"SCR identify failed on {host}:{port}: expected ***identified*** in reply, got {data[:200]!r}"
        )
    # #region agent log
    _debug_log(
        "H4",
        "torcs_omnisafe_wrapper.py:verify_scr_identify:success",
        "verify_scr_identify success",
        {
            "host": host,
            "port": int(port),
            "elapsed_ms": int((time.time() - t0) * 1000),
            "reply_prefix": repr(data[:120]),
        },
    )
    # #endregion


def _gymnasium_action_box(policy_action_dim: int) -> spaces.Box:
    """Policy action space exposed to OmniSafe (underlying TORCS step always uses 3 pedals)."""
    d = int(policy_action_dim)
    if d == 2:
        return spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
    if d == 3:
        return spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
    raise ValueError(f"policy_action_dim must be 2 or 3, got {policy_action_dim!r}")


def verify_scr_ports(num_envs: int, host: str = "127.0.0.1", port_base: int = 3001) -> None:
    """Fail fast before training if any expected SCR port does not identify."""
    # #region agent log
    _debug_log(
        "H5",
        "torcs_omnisafe_wrapper.py:verify_scr_ports:entry",
        "verify_scr_ports begin",
        {
            "num_envs": int(num_envs),
            "host": host,
            "port_base": int(port_base),
        },
    )
    # #endregion
    for i in range(num_envs):
        # #region agent log
        _debug_log(
            "H5",
            "torcs_omnisafe_wrapper.py:verify_scr_ports:iterate",
            "verify_scr_ports checking port",
            {
                "index": int(i),
                "port": int(port_base + i),
            },
        )
        # #endregion
        verify_scr_identify(host, port_base + i)


class _TorcsSimSlot:
    """One TORCS SCR client plus reward/cost/feature state (shared semantics with TorcsSafeEnv)."""

    def __init__(self, port: int, cfg: dict, label: str = "") -> None:
        _dev = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(_dev) if isinstance(_dev, str) else _dev

        self.track_limit: float = float(cfg.get("track_limit", 1.05))
        self.stage: int = int(cfg.get("stage", 1))
        self.reward_mode: str = str(cfg.get("reward_mode", "legacy"))
        self.reward_scale: float = float(cfg.get("reward_scale", 0.05))
        self.centerline_penalty: float = float(cfg.get("centerline_penalty", 0.1))
        self.angle_penalty: float = float(cfg.get("angle_penalty", 0.05))
        self.low_speed_threshold: float = float(cfg.get("low_speed_threshold", 20.0))
        self.low_speed_penalty: float = float(cfg.get("low_speed_penalty", 0.2))
        self.min_reward_speed: float = float(cfg.get("min_reward_speed", 18.0))
        self.speed_gate_span: float = float(cfg.get("speed_gate_span", 22.0))
        # Stage 1 + centered_speed_gate: full progress at crawl, penalty ramp with speed (see sacpid_env_presets).
        self._stage1_full_progress_max_speed_kmh: float = float(
            cfg.get("stage1_full_progress_max_speed_kmh", 0.0)
        )
        self._stage1_progress_blend_span_kmh: float = float(
            cfg.get("stage1_progress_blend_span_kmh", 6.0)
        )
        _pr_lo = float(cfg.get("penalty_ramp_speed_lo_kmh", 0.0))
        _pr_hi = float(cfg.get("penalty_ramp_speed_hi_kmh", -1.0))
        self._stage1_penalty_ramp_enabled: bool = _pr_hi > _pr_lo
        self._stage1_penalty_ramp_lo_kmh: float = _pr_lo
        self._stage1_penalty_ramp_hi_kmh: float = _pr_hi if self._stage1_penalty_ramp_enabled else 0.0
        self._stage1_penalty_ramp_scale: float = 1.0
        self._stage1_progress_speed_mult: float = 0.0
        self.center_reward_band: float = float(cfg.get("center_reward_band", 0.30))
        self.heading_reward_band: float = float(cfg.get("heading_reward_band", 0.35))
        self.center_reward_scale: float = float(cfg.get("center_reward_scale", 0.45))
        self.heading_reward_scale: float = float(cfg.get("heading_reward_scale", 0.5))
        self.launch_grace_steps: int = int(cfg.get("launch_grace_steps", 35))
        self.terminal_failure_penalty: float = float(cfg.get("terminal_failure_penalty", -5.0))
        self.forward_bonus_scale: float = float(cfg.get("forward_bonus_scale", 0.0))
        self.dist_raced_bonus_scale: float = float(cfg.get("dist_raced_bonus_scale", 0.0))

        # Stage 2 `line_first_mild_pace` (scaffolding; progress-primary, bonuses + mild pace band).
        self.line_target_offset: float = float(cfg.get("line_target_offset", 0.0))
        self.line_corridor_half_width: float = float(cfg.get("line_corridor_half_width", 0.55))
        self.line_reward_scale: float = float(cfg.get("line_reward_scale", 0.22))
        self.line_heading_reward_band: float = float(cfg.get("line_heading_reward_band", 0.65))
        self.line_heading_reward_scale: float = float(cfg.get("line_heading_reward_scale", 0.25))
        self.pace_band_v_lo: float = float(cfg.get("pace_band_v_lo", 25.0))
        self.pace_band_v_hi: float = float(cfg.get("pace_band_v_hi", 95.0))
        self.pace_band_scale: float = float(cfg.get("pace_band_scale", 0.08))
        self.pace_band_high_tail: float = float(cfg.get("pace_band_high_tail", 40.0))
        self.slip_threshold_kmh: float = float(cfg.get("slip_threshold_kmh", 18.0))

        # Optional (`reward_mode == corkscrew_sector_shaping`): extra line/pace weight along distFromStart window.
        self.corkscrew_sector_start_m: float = float(cfg.get("corkscrew_sector_start_m", 600.0))
        self.corkscrew_sector_end_m: float = float(cfg.get("corkscrew_sector_end_m", 1600.0))
        self.corkscrew_sector_bonus_mult: float = float(cfg.get("corkscrew_sector_bonus_mult", 1.25))

        # Brake teaching along distFromStart (m): penalise speed above cap; reward brake when fast enough.
        _bs0 = float(cfg.get("brake_sector_start_m", 0.0))
        _bs1 = float(cfg.get("brake_sector_end_m", 0.0))
        self._brake_sector_lo_m: float = float(min(_bs0, _bs1))
        self._brake_sector_hi_m: float = float(max(_bs0, _bs1))
        self._brake_sector_speed_cap_kmh: float = float(cfg.get("brake_sector_speed_cap_kmh", 75.0))
        self._brake_sector_overspeed_penalty_scale: float = float(
            cfg.get("brake_sector_overspeed_penalty_scale", 0.0)
        )
        self._brake_sector_brake_bonus_scale: float = float(cfg.get("brake_sector_brake_bonus_scale", 0.0))
        self._brake_sector_min_speed_for_brake_bonus_kmh: float = float(
            cfg.get("brake_sector_min_speed_for_brake_bonus_kmh", 20.0)
        )

        # Anti-swerve curriculum shaping. These are speed-gated so launch and
        # low-speed recovery do not learn "do nothing" as the safest action.
        self._smooth_speed_gate_lo_kmh: float = float(cfg.get("smooth_speed_gate_lo_kmh", 20.0))
        self._smooth_speed_gate_hi_kmh: float = float(cfg.get("smooth_speed_gate_hi_kmh", 55.0))
        self._smooth_curve_gate_floor: float = float(np.clip(cfg.get("smooth_curve_gate_floor", 0.30), 0.0, 1.0))

        # Lateral speed (oversteer) penalty.
        self._lateral_speed_penalty_scale: float = float(cfg.get("lateral_speed_penalty_scale", 0.0))
        self._lateral_speed_penalty_threshold_kmh: float = float(cfg.get("lateral_speed_penalty_threshold_kmh", 30.0))

        # Steering jerk penalty (large single-step steer changes).
        self._steer_jerk_penalty_scale: float = float(cfg.get("steer_jerk_penalty_scale", 0.0))
        self._steer_jerk_threshold: float = float(cfg.get("steer_jerk_threshold", 0.4))
        self._jerk_penalty_reset_grace_steps: int = max(0, int(cfg.get("jerk_penalty_reset_grace_steps", 1)))
        self._jerk_penalty_recovery_trackpos_threshold: float = max(
            1e-6, float(cfg.get("jerk_penalty_recovery_trackpos_threshold", 0.35))
        )
        self._jerk_penalty_recovery_angle_threshold_rad: float = max(
            1e-6, float(cfg.get("jerk_penalty_recovery_angle_threshold_rad", 0.20))
        )

        # Corner-aware speed penalty (track sensor asymmetry as curvature proxy).
        self._curve_speed_penalty_scale: float = float(cfg.get("curve_speed_penalty_scale", 0.0))
        self._curve_speed_threshold_kmh: float = float(cfg.get("curve_speed_threshold_kmh", 60.0))
        self._curve_delta_min: float = float(cfg.get("curve_delta_min_for_penalty", 0.15))
        self._curve_gate_ema_alpha: float = float(np.clip(cfg.get("curve_gate_ema_alpha", 1.0), 0.0, 1.0))

        # Straight-line stability shaping: reward centered and aligned behavior mainly on straights.
        self._straight_stability_reward_scale: float = float(cfg.get("straight_stability_reward_scale", 0.0))
        self._straight_stability_trackpos_threshold: float = max(
            1e-6, float(cfg.get("straight_stability_trackpos_threshold", 0.30))
        )
        self._straight_stability_angle_threshold_rad: float = max(
            1e-6, float(cfg.get("straight_stability_angle_threshold_rad", 0.14))
        )

        # Wide-corridor and Corkscrew sector shaping. Corridor terms are
        # penalties only: they do not reward centerline chasing.
        self._line_corridor_target: float = float(cfg.get("line_corridor_target", 0.0))
        self._line_corridor_half_width: float = max(1e-6, float(cfg.get("line_corridor_half_width", 1.0)))
        self._line_corridor_penalty_scale: float = float(cfg.get("line_corridor_penalty_scale", 0.0))
        self._line_heading_penalty_scale: float = float(cfg.get("line_heading_penalty_scale", 0.0))
        self._line_heading_tolerance_rad: float = max(0.0, float(cfg.get("line_heading_tolerance_rad", 0.20)))
        self._sector_line_penalty_scale: float = float(cfg.get("sector_line_penalty_scale", 0.0))
        self._sector_wrong_half_penalty_scale: float = float(cfg.get("sector_wrong_half_penalty_scale", 0.0))
        _sectors = cfg.get("corkscrew_sectors", [])
        self._corkscrew_sectors: list[dict] = list(_sectors) if isinstance(_sectors, list) else []

        # Final speed-push bonus is only paid under stable, low-swerve motion.
        self._stable_speed_bonus_scale: float = float(cfg.get("stable_speed_bonus_scale", 0.0))
        self._stable_speed_bonus_threshold_kmh: float = float(cfg.get("stable_speed_bonus_threshold_kmh", 90.0))
        self._stable_speed_max_lateral_kmh: float = max(0.0, float(cfg.get("stable_speed_max_lateral_kmh", 8.0)))
        self._stable_speed_max_angle_rad: float = max(0.0, float(cfg.get("stable_speed_max_angle_rad", 0.12)))
        self._stable_speed_max_abs_track_pos: float = max(0.0, float(cfg.get("stable_speed_max_abs_track_pos", 0.82)))

        # Stage 5 "earn your decel" shaping. Penalises any real decel that the
        # brake did not cause (catches lift-off coasting and steering-scrub
        # speed-bleed alike); rewards brake when it is actually slowing the car.
        self._unearned_decel_penalty_scale: float = max(0.0, float(cfg.get("unearned_decel_penalty_scale", 0.0)))
        self._unearned_decel_threshold_kmh_per_step: float = max(
            0.0, float(cfg.get("unearned_decel_threshold_kmh_per_step", 0.5))
        )
        self._unearned_decel_min_speed_kmh: float = max(0.0, float(cfg.get("unearned_decel_min_speed_kmh", 30.0)))
        self._brake_utilization_bonus_scale: float = max(0.0, float(cfg.get("brake_utilization_bonus_scale", 0.0)))
        self._brake_utilization_min_brake: float = max(0.0, float(cfg.get("brake_utilization_min_brake", 0.15)))
        self._prev_speedX_kmh: float | None = None

        self._action_repeat: int = max(1, int(cfg.get("action_repeat", 1)))
        self._steer_rate_limit_per_step: float = max(0.0, float(cfg.get("steer_rate_limit_per_step", 0.0)))
        self._steer_smoothing_alpha: float = float(np.clip(cfg.get("steer_smoothing_alpha", 1.0), 0.0, 1.0))
        self._steer_jitter_flip_delta_threshold: float = max(
            0.0, float(cfg.get("steer_jitter_flip_delta_threshold", 0.0))
        )
        self._steer_jitter_flip_delta_cap: float = max(0.0, float(cfg.get("steer_jitter_flip_delta_cap", 0.0)))
        self._steer_jitter_streak_gain: float = max(0.0, float(cfg.get("steer_jitter_streak_gain", 0.0)))

        # Non-linear edge proximity penalty (quadratic repulsion near track boundary).
        self._edge_proximity_threshold: float = float(cfg.get("edge_proximity_threshold", 0.9))
        self._edge_proximity_penalty_scale: float = float(cfg.get("edge_proximity_penalty_scale", 0.0))

        self.watchdog_no_progress_steps: int = int(cfg.get("watchdog_no_progress_steps", 0))
        self.watchdog_dist_delta_m: float = float(cfg.get("watchdog_dist_delta_m", 0.15))
        self.watchdog_speed_kmh: float = float(cfg.get("watchdog_speed_kmh", 3.0))
        self._watchdog_prev_dist: float | None = None
        self._watchdog_stale: int = 0
        self._curve_gate_ema: float = 1.0

        self._policy_action_dim: int = int(cfg.get("policy_action_dim", 3))
        if self._policy_action_dim not in (2, 3):
            raise ValueError(f"policy_action_dim must be 2 or 3, got {self._policy_action_dim!r}")

        # Eval / hotlap: see run_model.py and TorcsHotlapEval-v0.
        self._gym_reset_relaunch: bool = bool(cfg.get("gym_reset_relaunch", True))
        self.wrapper_terminal: bool = bool(cfg.get("wrapper_terminal", True))
        _ltl = cfg.get("loose_track_pos_limit", None)
        if _ltl is None:
            self._loose_track_pos_limit: float | None = None
        else:
            _ltl_f = float(_ltl)
            self._loose_track_pos_limit = None if _ltl_f <= 0.0 else _ltl_f

        self._obs_track_ref_length_m: float = float(cfg.get("obs_track_ref_length_m", 3620.0))

        self._loose_track_grace_steps: int = max(0, int(cfg.get("loose_track_grace_steps", 0)))
        self._early_episode_penalty_steps: int = max(0, int(cfg.get("early_episode_penalty_steps", 0)))
        self._early_episode_terminal_extra_penalty: float = float(
            cfg.get("early_episode_terminal_extra_penalty", 0.0)
        )
        self._early_episode_survival_bonus: float = float(cfg.get("early_episode_survival_bonus", 0.0))

        self._launch_phase_shaping_steps: int = max(0, int(cfg.get("launch_phase_shaping_steps", 0)))
        self._launch_steer_quadratic_penalty_scale: float = float(
            cfg.get("launch_steer_quadratic_penalty_scale", 0.0)
        )
        self._launch_steer_straight_bonus_scale: float = float(
            cfg.get("launch_steer_straight_bonus_scale", 0.0)
        )
        self._launch_extra_lateral_penalty_scale: float = float(
            cfg.get("launch_extra_lateral_penalty_scale", 0.0)
        )

        _gym_off = cfg.get("gym_terminate_on_off_track", None)
        self._env = TorcsEnv(
            vision=False,
            throttle=True,
            gear_change=False,
            port=int(port),
            terminate_on_off_track=_gym_off,
        )
        self._port = int(port)
        self._label = label
        tag = f" [{label}]" if label else ""
        print(
            f"[TorcsSimSlot{tag}] TORCS SCR on port {port} "
            f"(stage={self.stage}, track_limit={self.track_limit}, "
            f"policy_action_dim={self._policy_action_dim})"
        )
        if self._brake_sector_hi_m > self._brake_sector_lo_m and (
            self._brake_sector_overspeed_penalty_scale > 0.0
            or self._brake_sector_brake_bonus_scale > 0.0
        ):
            print(
                f"[TorcsSimSlot{tag}] brake_teaching distFromStart_m=[{self._brake_sector_lo_m:.1f}, "
                f"{self._brake_sector_hi_m:.1f}] speed_cap_kmh={self._brake_sector_speed_cap_kmh:.1f} "
                f"overspeed_penalty_scale={self._brake_sector_overspeed_penalty_scale:g} "
                f"brake_bonus_scale={self._brake_sector_brake_bonus_scale:g} "
                f"(needs policy_action_dim=3 for brake bonus)"
            )
        if self._action_repeat > 1:
            print(f"[TorcsSimSlot{tag}] action_repeat={self._action_repeat} SCR frames per policy action")

        self._alpha = 0.2
        self._wheel_radius = 0.33
        self._num_obs = 33

        self._reset_state_tracking()
        self._last_reset_duration: float | None = None
        self._episode_count: int = 0
        self._episode_step: int = 0
        self._wandb_error_logged: bool = False
        self._wandb_ep_sum_reward: float = 0.0
        self._wandb_ep_sum_speed_kmh: float = 0.0
        self._wandb_ep_min_margin: float = float("inf")
        self._wandb_ep_sum_cost: float = 0.0
        self._wandb_ep_slip_event_count: int = 0
        self._wandb_ep_max_abs_track_pos: float = 0.0
        self._wandb_ep_sum_obs_dist_along: float = 0.0

    def close(self) -> None:
        self._env.end()

    def _reset_state_tracking(self) -> None:
        self.prev_speedX = 0.0
        self.prev_speedY = 0.0
        self.prev_speedZ = 0.0
        self.ema_accelX = 0.0
        self.ema_accelY = 0.0
        self.ema_accelZ = 0.0
        self.prev_trackPos = 0.0
        self._prev_dist_raced: float | None = None
        self._prev_steer: float = 0.0
        self._curve_gate_ema = 1.0
        self._steer_flip_streak: int = 0

    def _engineer_features(self, raw_obs) -> np.ndarray:
        norm_speedX = float(raw_obs.speedX)
        norm_speedY = float(raw_obs.speedY)
        norm_speedZ = float(raw_obs.speedZ)
        track_sensors = np.asarray(raw_obs.track, dtype=np.float32)

        try:
            track_pos = float(self._env.client.S.d["trackPos"])
        except Exception:
            track_pos = 0.0

        try:
            angle = float(self._env.client.S.d["angle"]) / np.pi
        except Exception:
            angle = 0.0

        slip_angle = float(np.arctan2(norm_speedY, norm_speedX + 1e-8)) / (np.pi / 2.0)

        avg_rear_rads = (raw_obs.wheelSpinVel[2] + raw_obs.wheelSpinVel[3]) / 2.0
        wheel_speed_ms = avg_rear_rads * self._wheel_radius
        car_speed_ms = norm_speedX * self._env.default_speed / 3.6
        if abs(car_speed_ms) < 1.4:
            tire_slip_delta = 0.0
        else:
            denom = max(abs(wheel_speed_ms), abs(car_speed_ms), 1.0)
            tire_slip_delta = float(np.clip((wheel_speed_ms - car_speed_ms) / denom, -1.0, 1.0))

        raw_ax = norm_speedX - self.prev_speedX
        self.ema_accelX = (1.0 - self._alpha) * self.ema_accelX + self._alpha * raw_ax
        self.prev_speedX = norm_speedX

        raw_ay = norm_speedY - self.prev_speedY
        self.ema_accelY = (1.0 - self._alpha) * self.ema_accelY + self._alpha * raw_ay
        self.prev_speedY = norm_speedY

        speed_scale = self._env.default_speed
        norm_accelX = float(np.clip(self.ema_accelX * speed_scale / 10.0, -1.0, 1.0))
        norm_accelY = float(np.clip(self.ema_accelY * speed_scale / 10.0, -1.0, 1.0))

        raw_az = norm_speedZ - self.prev_speedZ
        self.ema_accelZ = (1.0 - self._alpha) * self.ema_accelZ + self._alpha * raw_az
        self.prev_speedZ = norm_speedZ

        norm_accelZ = float(np.clip(self.ema_accelZ * speed_scale / 5.0, -1.0, 1.0))

        curve_delta = float(track_sensors[18] - track_sensors[0])

        trackPos_rate = track_pos - self.prev_trackPos
        self.prev_trackPos = track_pos

        dist_to_edge = max(self.track_limit - abs(track_pos), 0.0)
        moving_outward = track_pos * trackPos_rate > 0

        if dist_to_edge <= 0.0:
            ttb = 0.0
        elif moving_outward and abs(trackPos_rate) > 0.001:
            steps_to_edge = dist_to_edge / (abs(trackPos_rate) + 1e-8)
            ttb = float(np.clip(steps_to_edge / 100.0, 0.0, 1.0))
        else:
            ttb = 1.0

        rpm_norm = float(raw_obs.rpm) / 10000.0

        try:
            dfs = float(self._env.client.S.d.get("distFromStart", 0.0))
        except Exception:
            dfs = 0.0
        dist_along = normalized_dist_along_track_obs(dfs, self._obs_track_ref_length_m)

        state = np.array(
            [
                norm_speedX,
                norm_speedY,
                norm_speedZ,
                track_pos,
                angle,
                rpm_norm,
                slip_angle,
                tire_slip_delta,
                norm_accelX,
                norm_accelY,
                norm_accelZ,
                curve_delta,
                ttb,
                dist_along,
            ],
            dtype=np.float32,
        )
        return np.concatenate([state, track_sensors])

    def reset(self, relaunch: bool | None = None) -> np.ndarray:
        # #region agent log
        _debug_log(
            "H7",
            "torcs_omnisafe_wrapper.py:_TorcsSimSlot.reset:entry",
            "slot reset begin",
            {
                "label": self._label,
                "port": self._port,
                "episode_count_next": int(self._episode_count + 1),
            },
        )
        # #endregion
        self._reset_state_tracking()
        self._episode_count += 1
        self._episode_step = 0
        self._wandb_error_logged = False
        self._wandb_ep_sum_reward = 0.0
        self._wandb_ep_sum_speed_kmh = 0.0
        self._wandb_ep_min_margin = float("inf")
        self._wandb_ep_sum_cost = 0.0
        self._wandb_ep_slip_event_count = 0
        self._wandb_ep_max_abs_track_pos = 0.0
        self._wandb_ep_sum_obs_dist_along = 0.0
        t0 = time.perf_counter()
        _rl = self._gym_reset_relaunch if relaunch is None else bool(relaunch)
        try:
            raw_obs, _ = self._env.reset(relaunch=_rl)
        except Exception as e:
            # #region agent log
            _debug_log(
                "H7",
                "torcs_omnisafe_wrapper.py:_TorcsSimSlot.reset:exception",
                "slot reset exception",
                {
                    "label": self._label,
                    "port": self._port,
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
            # #endregion
            raise
        self._last_reset_duration = time.perf_counter() - t0
        # #region agent log
        _debug_log(
            "H7",
            "torcs_omnisafe_wrapper.py:_TorcsSimSlot.reset:success",
            "slot reset success",
            {
                "label": self._label,
                "port": self._port,
                "elapsed_ms": int(self._last_reset_duration * 1000),
            },
        )
        # #endregion

        self.prev_speedX = float(raw_obs.speedX)
        self._prev_speedX_kmh = None
        self.prev_speedY = float(raw_obs.speedY)
        self.prev_speedZ = float(raw_obs.speedZ)
        try:
            self.prev_trackPos = float(self._env.client.S.d["trackPos"])
        except Exception:
            self.prev_trackPos = 0.0

        try:
            self._prev_dist_raced = float(self._env.client.S.d.get("distRaced", 0.0))
        except Exception:
            self._prev_dist_raced = 0.0

        obs_np = self._engineer_features(raw_obs)
        self._watchdog_stale = 0
        try:
            self._watchdog_prev_dist = float(self._env.client.S.d.get("distRaced", 0.0))
        except Exception:
            self._watchdog_prev_dist = None
        return np.nan_to_num(obs_np)

    def step(
        self, np_action: np.ndarray
    ) -> tuple[np.ndarray, float, float, bool, bool, dict, dict]:
        """Returns engineered obs, shaped_reward, cost, terminated, truncated, base_info, telemetry_parts."""
        np_action = np.asarray(np_action, dtype=np.float32).copy().reshape(-1)
        steer_raw_policy = 0.0
        steer_after_rate_limit = 0.0
        jitter_flip_damped = 0.0
        jitter_flip_cap_used = 0.0
        jitter_flip_streak = int(self._steer_flip_streak)
        jitter_recovery_gate = 1.0
        try:
            _track_pos_now = float(self._env.client.S.d.get("trackPos", 0.0))
        except Exception:
            _track_pos_now = 0.0
        try:
            _angle_now = float(self._env.client.S.d.get("angle", 0.0))
        except Exception:
            _angle_now = 0.0
        _recovery_need = max(
            abs(_track_pos_now) / self._jerk_penalty_recovery_trackpos_threshold,
            abs(_angle_now) / self._jerk_penalty_recovery_angle_threshold_rad,
        )
        jitter_recovery_gate = float(np.clip(1.0 - _recovery_need, 0.0, 1.0))
        if self._policy_action_dim == 2:
            if np_action.size != 2:
                raise ValueError(
                    f"policy_action_dim=2 expects 2 actions [steer, accel], got shape {np_action.shape}"
                )
            steer = float(np.clip(np_action[0], -1.0, 1.0))
            steer_raw_policy = steer
            delta_raw = float(steer - self._prev_steer)
            sign_flip = bool((steer * self._prev_steer) < 0.0)
            large_flip = sign_flip and (
                abs(delta_raw) > float(self._steer_jitter_flip_delta_threshold)
            )
            if (
                large_flip
                and self._steer_jitter_flip_delta_cap > 0.0
                and jitter_recovery_gate > 0.0
            ):
                cap = float(self._steer_jitter_flip_delta_cap)
                if self._steer_jitter_streak_gain > 0.0:
                    cap = cap / (1.0 + (float(self._steer_flip_streak) * self._steer_jitter_streak_gain))
                # Relax damping during obvious recovery so valid corrective steering is preserved.
                cap = min(2.0, cap / max(0.15, jitter_recovery_gate))
                cap = float(max(1e-6, cap))
                d = float(np.clip(delta_raw, -cap, cap))
                steer = float(self._prev_steer + d)
                jitter_flip_damped = 1.0
                jitter_flip_cap_used = cap
                self._steer_flip_streak += 1
            else:
                if sign_flip:
                    self._steer_flip_streak = 0
                else:
                    self._steer_flip_streak = max(0, self._steer_flip_streak - 1)
            jitter_flip_streak = int(self._steer_flip_streak)
            if self._steer_rate_limit_per_step > 0.0:
                d = float(
                    np.clip(
                        steer - self._prev_steer,
                        -self._steer_rate_limit_per_step,
                        self._steer_rate_limit_per_step,
                    )
                )
                steer = float(self._prev_steer + d)
            steer_after_rate_limit = steer
            if self._steer_smoothing_alpha < 1.0:
                steer = float(
                    (self._steer_smoothing_alpha * steer)
                    + ((1.0 - self._steer_smoothing_alpha) * self._prev_steer)
                )
            steer = float(np.clip(steer, -1.0, 1.0))
            accel = float(np.clip(np_action[1], 0.0, 1.0))
            np_action = np.array([steer, accel, 0.0], dtype=np.float32)
        else:
            if np_action.size != 3:
                raise ValueError(
                    f"policy_action_dim=3 expects 3 actions [steer, accel, brake], got shape {np_action.shape}"
                )
            steer = float(np.clip(np_action[0], -1.0, 1.0))
            steer_raw_policy = steer
            delta_raw = float(steer - self._prev_steer)
            sign_flip = bool((steer * self._prev_steer) < 0.0)
            large_flip = sign_flip and (
                abs(delta_raw) > float(self._steer_jitter_flip_delta_threshold)
            )
            if (
                large_flip
                and self._steer_jitter_flip_delta_cap > 0.0
                and jitter_recovery_gate > 0.0
            ):
                cap = float(self._steer_jitter_flip_delta_cap)
                if self._steer_jitter_streak_gain > 0.0:
                    cap = cap / (1.0 + (float(self._steer_flip_streak) * self._steer_jitter_streak_gain))
                # Relax damping during obvious recovery so valid corrective steering is preserved.
                cap = min(2.0, cap / max(0.15, jitter_recovery_gate))
                cap = float(max(1e-6, cap))
                d = float(np.clip(delta_raw, -cap, cap))
                steer = float(self._prev_steer + d)
                jitter_flip_damped = 1.0
                jitter_flip_cap_used = cap
                self._steer_flip_streak += 1
            else:
                if sign_flip:
                    self._steer_flip_streak = 0
                else:
                    self._steer_flip_streak = max(0, self._steer_flip_streak - 1)
            jitter_flip_streak = int(self._steer_flip_streak)
            if self._steer_rate_limit_per_step > 0.0:
                d = float(
                    np.clip(
                        steer - self._prev_steer,
                        -self._steer_rate_limit_per_step,
                        self._steer_rate_limit_per_step,
                    )
                )
                steer = float(self._prev_steer + d)
            steer_after_rate_limit = steer
            if self._steer_smoothing_alpha < 1.0:
                steer = float(
                    (self._steer_smoothing_alpha * steer)
                    + ((1.0 - self._steer_smoothing_alpha) * self._prev_steer)
                )
            np_action[0] = np.clip(steer, -1.0, 1.0)  # steer
            np_action[1] = np.clip(np_action[1], 0.0, 1.0)  # throttle
            np_action[2] = np.clip(np_action[2], 0.0, 1.0)  # brake
        raw_obs = None
        reward = 0.0
        done = False
        _truncated = False
        base_info: dict = {}
        repeat_steps_executed = 0
        raw_reward_minus_one_seen = False
        for _repeat_idx in range(self._action_repeat):
            raw_obs, step_reward, done, _truncated, base_info = self._env.step(np_action)
            repeat_steps_executed += 1
            reward += float(step_reward)
            if float(step_reward) == -1.0:
                raw_reward_minus_one_seen = True
            if done or _truncated or raw_reward_minus_one_seen:
                break
        if raw_obs is None:
            raise RuntimeError("TORCS action repeat executed zero simulator steps.")
        self._episode_step += 1
        # Track per-policy-action steer change for jerk penalty; update after measuring.
        _steer_now = float(np_action[0])
        _steer_jerk = abs(_steer_now - self._prev_steer)
        self._prev_steer = _steer_now
        obs_np = self._engineer_features(raw_obs)

        cost = 0.0
        try:
            track_pos = self._env.client.S.d["trackPos"]
            angle = self._env.client.S.d["angle"]
            speedX = self._env.client.S.d["speedX"]
        except Exception:
            track_pos = 0.0
            angle = 0.0
            speedX = float(raw_obs.speedX) * self._env.default_speed

        self._wandb_ep_max_abs_track_pos = max(
            self._wandb_ep_max_abs_track_pos, abs(float(track_pos))
        )
        try:
            dist_raced_m = float(self._env.client.S.d.get("distRaced", 0.0))
        except Exception:
            dist_raced_m = 0.0
        try:
            dist_from_start_m = float(self._env.client.S.d.get("distFromStart", 0.0))
        except Exception:
            dist_from_start_m = 0.0

        termination_reason = base_info.get("termination_reason", "unknown")

        if self.wrapper_terminal:
            if abs(track_pos) > self.track_limit:
                cost = 1.0
                done = True
                base_info = dict(base_info)
                base_info["termination_reason"] = "off_track"
                termination_reason = "off_track"

            elif abs(angle) > (np.pi / 2.0):
                cost = 1.0
                done = True
                base_info = dict(base_info)
                base_info["termination_reason"] = "angle_exceeded"
                termination_reason = "angle_exceeded"

            elif speedX < -1.0:
                cost = 1.0
                done = True
                base_info = dict(base_info)
                base_info["termination_reason"] = "reverse_speed"
                termination_reason = "reverse_speed"

            if raw_reward_minus_one_seen:
                cost = 1.0
                if str(termination_reason) in ("unknown", "", "none"):
                    base_info = dict(base_info)
                    base_info["termination_reason"] = "raw_reward_minus_one"
                    termination_reason = "raw_reward_minus_one"

            if self.stage == 1 and termination_reason == "too_slow":
                cost = 1.0

        wd_n = self.watchdog_no_progress_steps
        if self.wrapper_terminal and wd_n > 0 and getattr(self._env.client, "so", None) is not None:
            try:
                dr_now = float(self._env.client.S.d.get("distRaced", 0.0))
            except Exception:
                dr_now = 0.0
            speed_kmh = float(speedX)
            delta_dr = (
                1e9
                if self._watchdog_prev_dist is None
                else (dr_now - self._watchdog_prev_dist)
            )
            self._watchdog_prev_dist = dr_now
            if delta_dr >= self.watchdog_dist_delta_m or speed_kmh >= self.watchdog_speed_kmh:
                self._watchdog_stale = 0
            else:
                self._watchdog_stale += 1
            if self._watchdog_stale >= wd_n:
                self._watchdog_stale = 0
                done = True
                base_info = dict(base_info)
                base_info["termination_reason"] = "watchdog_timeout"
                termination_reason = "watchdog_timeout"

        # Loose lateral bound for hotlap / eval (|trackPos| vs loose_track_pos_limit).
        _lg = self._loose_track_grace_steps
        if (
            self._loose_track_pos_limit is not None
            and abs(float(track_pos)) > self._loose_track_pos_limit
            and (_lg <= 0 or self._episode_step > _lg)
        ):
            done = True
            base_info = dict(base_info)
            base_info["termination_reason"] = "off_track_loose"
            termination_reason = "off_track_loose"

        reward_speed_gate = 0.0
        progress_reward = 0.0
        center_bonus = 0.0
        heading_bonus = 0.0
        line_bonus = 0.0
        pace_bonus = 0.0
        centerline_part = 0.0
        angle_part = 0.0
        low_speed_part = 0.0
        forward_bonus = 0.0
        dist_raced_bonus = 0.0
        brake_sector_overspeed_part = 0.0
        brake_sector_brake_bonus_part = 0.0
        launch_phase_steer_part = 0.0
        launch_phase_lateral_part = 0.0
        line_corridor_penalty_part = 0.0
        line_heading_penalty_part = 0.0
        sector_line_penalty_part = 0.0
        sector_wrong_half_penalty_part = 0.0
        edge_proximity_penalty_part = 0.0
        steer_jerk_penalty_part = 0.0
        lateral_penalty_part = 0.0
        straight_stability_bonus_part = 0.0
        stable_speed_bonus_part = 0.0
        unearned_decel_penalty_part = 0.0
        brake_utilization_bonus_part = 0.0
        decel_kmh_per_step = 0.0
        smooth_speed_gate = 0.0
        smooth_gate = 0.0
        curve_gate = 1.0
        curve_gate_raw = 1.0
        jerk_recovery_gate = 1.0
        lateral_speed_kmh_pre = abs(float(raw_obs.speedY) * self._env.default_speed)

        gym_raw = self.reward_mode == "gym_raw"
        hard_terminal_failure = bool(not gym_raw and cost >= 1.0 and done)
        if gym_raw:
            shaped_reward = float(np.clip(float(reward), -10.0, 10.0))
            cost = 0.0
        else:
            # Progress-only shaping for all non-gym_raw stages.
            progress_reward = float(reward) * self.reward_scale
            self._stage1_penalty_ramp_scale = 1.0
            self._stage1_progress_speed_mult = 1.0
            shaped_reward = float(np.clip(progress_reward, -10.0, 10.0))
            curve_delta = float(obs_np[11]) if obs_np.size > 11 else 0.0
            if self._curve_delta_min > 0.0:
                curve_gate_raw = float(np.clip(1.0 - (abs(curve_delta) / self._curve_delta_min), 0.0, 1.0))
                if self._curve_gate_ema_alpha < 1.0:
                    self._curve_gate_ema = float(
                        (self._curve_gate_ema_alpha * curve_gate_raw)
                        + ((1.0 - self._curve_gate_ema_alpha) * self._curve_gate_ema)
                    )
                    curve_gate = self._curve_gate_ema
                else:
                    curve_gate = curve_gate_raw
            if self._smooth_speed_gate_hi_kmh > self._smooth_speed_gate_lo_kmh:
                smooth_speed_gate = float(
                    np.clip(
                        (float(speedX) - self._smooth_speed_gate_lo_kmh)
                        / (self._smooth_speed_gate_hi_kmh - self._smooth_speed_gate_lo_kmh),
                        0.0,
                        1.0,
                    )
                )
            else:
                smooth_speed_gate = 1.0
            smooth_gate = smooth_speed_gate * max(float(curve_gate), self._smooth_curve_gate_floor)
            # Encourage stable centered + aligned behavior on straights.
            if self._straight_stability_reward_scale > 0.0:
                track_term = 1.0 - float(
                    np.clip(abs(float(track_pos)) / self._straight_stability_trackpos_threshold, 0.0, 1.0)
                )
                angle_term = 1.0 - float(
                    np.clip(abs(float(angle)) / self._straight_stability_angle_threshold_rad, 0.0, 1.0)
                )
                straight_stability_bonus_part = float(
                    self._straight_stability_reward_scale * smooth_gate * track_term * angle_term
                )
                shaped_reward += straight_stability_bonus_part
            # Wide-corridor shaping: penalize only outside the allowed band so
            # the policy is not paid to chase the centerline.
            if self._line_corridor_penalty_scale > 0.0:
                line_error = abs(float(track_pos) - self._line_corridor_target)
                line_excess = max(0.0, line_error - self._line_corridor_half_width)
                line_corridor_penalty_part = float(
                    self._line_corridor_penalty_scale * smooth_speed_gate * (line_excess ** 2)
                )
                shaped_reward -= line_corridor_penalty_part
            if self._line_heading_penalty_scale > 0.0:
                heading_excess = max(0.0, abs(float(angle)) - self._line_heading_tolerance_rad)
                line_heading_penalty_part = float(
                    self._line_heading_penalty_scale * smooth_speed_gate * (heading_excess ** 2)
                )
                shaped_reward -= line_heading_penalty_part
            if self._corkscrew_sectors and (
                self._sector_line_penalty_scale > 0.0 or self._sector_wrong_half_penalty_scale > 0.0
            ):
                for sector in self._corkscrew_sectors:
                    try:
                        start_m = float(sector.get("start_m", 0.0))
                        end_m = float(sector.get("end_m", 0.0))
                        target_min = float(sector.get("target_min", -1.0))
                        target_max = float(sector.get("target_max", 1.0))
                    except (TypeError, ValueError, AttributeError):
                        continue
                    lo_m = min(start_m, end_m)
                    hi_m = max(start_m, end_m)
                    if hi_m <= lo_m or not (lo_m <= dist_from_start_m <= hi_m):
                        continue
                    target_lo = min(target_min, target_max)
                    target_hi = max(target_min, target_max)
                    if float(track_pos) < target_lo:
                        sector_excess = target_lo - float(track_pos)
                    elif float(track_pos) > target_hi:
                        sector_excess = float(track_pos) - target_hi
                    else:
                        sector_excess = 0.0
                    sector_weight = float(sector.get("weight", 0.0))
                    sector_line_penalty_part += float(
                        self._sector_line_penalty_scale
                        * sector_weight
                        * smooth_speed_gate
                        * (sector_excess ** 2)
                    )

                    wrong_half = float(sector.get("wrong_half", 0.0))
                    if wrong_half != 0.0 and (float(track_pos) * wrong_half) > 0.0:
                        wrong_weight = float(sector.get("wrong_half_weight", 0.0))
                        sector_wrong_half_penalty_part += float(
                            self._sector_wrong_half_penalty_scale
                            * wrong_weight
                            * smooth_speed_gate
                            * min(abs(float(track_pos)), 1.0)
                        )
                shaped_reward -= sector_line_penalty_part + sector_wrong_half_penalty_part
            # Penalize unnecessary steering oscillations mainly on straights.
            if self._steer_jerk_penalty_scale > 0.0:
                in_reset_grace = self._episode_step <= self._jerk_penalty_reset_grace_steps
                recovery_need = max(
                    abs(float(track_pos)) / self._jerk_penalty_recovery_trackpos_threshold,
                    abs(float(angle)) / self._jerk_penalty_recovery_angle_threshold_rad,
                )
                jerk_recovery_gate = float(np.clip(1.0 - recovery_need, 0.0, 1.0))
                if not in_reset_grace:
                    jerk_excess = max(0.0, float(_steer_jerk) - self._steer_jerk_threshold)
                    steer_jerk_penalty_part = float(
                        self._steer_jerk_penalty_scale * smooth_gate * jerk_recovery_gate * (jerk_excess ** 2)
                    )
                    shaped_reward -= steer_jerk_penalty_part
            # Penalize straight-line lateral sliding / swerving.
            if self._lateral_speed_penalty_scale > 0.0:
                lateral_excess = max(0.0, lateral_speed_kmh_pre - self._lateral_speed_penalty_threshold_kmh)
                lateral_penalty_part = float(
                    self._lateral_speed_penalty_scale * smooth_gate * ((lateral_excess / 10.0) ** 2)
                )
                shaped_reward -= lateral_penalty_part
            if self._edge_proximity_penalty_scale > 0.0:
                edge_excess = max(0.0, abs(float(track_pos)) - self._edge_proximity_threshold)
                edge_proximity_penalty_part = float(
                    self._edge_proximity_penalty_scale * smooth_speed_gate * (edge_excess ** 2)
                )
                shaped_reward -= edge_proximity_penalty_part
            if (
                self._brake_sector_hi_m > self._brake_sector_lo_m
                and self._brake_sector_lo_m <= dist_from_start_m <= self._brake_sector_hi_m
            ):
                if self._brake_sector_overspeed_penalty_scale > 0.0:
                    overspeed = max(0.0, float(speedX) - self._brake_sector_speed_cap_kmh)
                    brake_sector_overspeed_part = float(
                        self._brake_sector_overspeed_penalty_scale * ((overspeed / 10.0) ** 2)
                    )
                    shaped_reward -= brake_sector_overspeed_part
                if self._brake_sector_brake_bonus_scale > 0.0 and np_action.size >= 3:
                    if float(speedX) >= self._brake_sector_min_speed_for_brake_bonus_kmh:
                        brake_sector_brake_bonus_part = float(
                            self._brake_sector_brake_bonus_scale * float(np.clip(np_action[2], 0.0, 1.0))
                        )
                        shaped_reward += brake_sector_brake_bonus_part
            if self._stable_speed_bonus_scale > 0.0:
                stable = (
                    lateral_speed_kmh_pre <= self._stable_speed_max_lateral_kmh
                    and abs(float(angle)) <= self._stable_speed_max_angle_rad
                    and abs(float(track_pos)) <= self._stable_speed_max_abs_track_pos
                )
                if stable:
                    stable_speed_bonus_part = float(
                        self._stable_speed_bonus_scale * max(0.0, float(speedX) - self._stable_speed_bonus_threshold_kmh)
                    )
                    shaped_reward += stable_speed_bonus_part

            # "Earn your deceleration" shaping (stage 5+). dV/dt computed from
            # frame-to-frame speedX (kmh per wrapper step). The brake pressure
            # is the throttle/brake action channel index 2 when present.
            cur_speed_kmh = float(speedX)
            if self._prev_speedX_kmh is not None:
                decel_kmh_per_step = float(self._prev_speedX_kmh - cur_speed_kmh)
            self._prev_speedX_kmh = cur_speed_kmh
            if (
                cur_speed_kmh >= self._unearned_decel_min_speed_kmh
                and decel_kmh_per_step > self._unearned_decel_threshold_kmh_per_step
            ):
                excess_decel = decel_kmh_per_step - self._unearned_decel_threshold_kmh_per_step
                brake_pressure = float(np.clip(np_action[2], 0.0, 1.0)) if np_action.size >= 3 else 0.0
                if self._unearned_decel_penalty_scale > 0.0:
                    unearned_decel_penalty_part = float(
                        self._unearned_decel_penalty_scale * excess_decel * (1.0 - brake_pressure)
                    )
                    shaped_reward -= unearned_decel_penalty_part
                if (
                    self._brake_utilization_bonus_scale > 0.0
                    and brake_pressure >= self._brake_utilization_min_brake
                ):
                    brake_utilization_bonus_part = float(
                        self._brake_utilization_bonus_scale * excess_decel * brake_pressure
                    )
                    shaped_reward += brake_utilization_bonus_part
        shaped_reward = float(shaped_reward)

        terminated = bool(done)
        truncated = False
        obs_np = np.nan_to_num(obs_np)

        if (
            os.environ.get("SACPID_COLDSTART_DIAG", "").strip() == "1"
            and self._episode_step == 1
            and self._episode_count <= 2
        ):
            print(
                f"[SACPID coldstart_diag env={self._label or 'default'}] "
                f"episode_index={self._episode_count} "
                f"obs_prefix={np.array2string(obs_np[:14], precision=4, max_line_width=120)}"
            )

        termination_reason = str(base_info.get("termination_reason", termination_reason))

        speedX_kmh = float(raw_obs.speedX) * self._env.default_speed
        speedY_kmh = float(raw_obs.speedY) * self._env.default_speed
        forward_speed_kmh = speedX_kmh
        lateral_speed_kmh = abs(speedY_kmh)
        slip_event = abs(speedY_kmh) > self.slip_threshold_kmh
        margin = float(max(0.0, self.track_limit - abs(track_pos)))
        self._wandb_ep_sum_reward += shaped_reward
        self._wandb_ep_sum_speed_kmh += speedX_kmh
        self._wandb_ep_min_margin = min(self._wandb_ep_min_margin, margin)
        self._wandb_ep_sum_cost += float(cost)
        self._wandb_ep_slip_event_count += int(slip_event)

        obs_dist_along = float(obs_np[13]) if obs_np.size > 13 else 0.0
        self._wandb_ep_sum_obs_dist_along += obs_dist_along

        tele = {
            "np_action": np_action,
            "steer_raw_policy": float(steer_raw_policy),
            "steer_after_rate_limit": float(steer_after_rate_limit),
            "steer_jitter_flip_damped": float(jitter_flip_damped),
            "steer_jitter_flip_cap_used": float(jitter_flip_cap_used),
            "steer_jitter_flip_streak": int(jitter_flip_streak),
            "steer_jitter_recovery_gate": float(jitter_recovery_gate),
            "action_repeat": int(self._action_repeat),
            "action_repeat_steps_executed": int(repeat_steps_executed),
            "raw_obs": raw_obs,
            "raw_gym_reward": float(reward),
            "raw_reward_minus_one_seen": bool(raw_reward_minus_one_seen),
            "track_pos": float(track_pos),
            "angle": float(angle),
            "shaped_reward": shaped_reward,
            "reward_speed_gate": float(reward_speed_gate),
            "stage1_penalty_ramp_scale": float(self._stage1_penalty_ramp_scale),
            "stage1_progress_speed_mult": float(self._stage1_progress_speed_mult),
            "progress_reward": progress_reward,
            "center_bonus": center_bonus,
            "heading_bonus": heading_bonus,
            "line_bonus": line_bonus,
            "pace_bonus": pace_bonus,
            "centerline_part": centerline_part,
            "angle_part": angle_part,
            "low_speed_part": low_speed_part,
            "forward_bonus": forward_bonus,
            "dist_raced_bonus": dist_raced_bonus,
            "cost": float(cost),
            "termination_reason": termination_reason,
            "terminated": terminated,
            "wrapper_episode_step": int(self._episode_step),
            "speedX_kmh": speedX_kmh,
            "speedY_kmh": speedY_kmh,
            "forward_speed_kmh": forward_speed_kmh,
            "lateral_speed_kmh": lateral_speed_kmh,
            "slip_event": bool(slip_event),
            "slip_threshold_kmh": self.slip_threshold_kmh,
            "dist_raced_m": float(dist_raced_m),
            "dist_from_start_m": float(dist_from_start_m),
            "obs_dist_along": obs_dist_along,
            "brake_sector_overspeed_part": float(brake_sector_overspeed_part),
            "brake_sector_brake_bonus_part": float(brake_sector_brake_bonus_part),
            "launch_phase_steer_part": float(launch_phase_steer_part),
            "launch_phase_lateral_part": float(launch_phase_lateral_part),
            "line_corridor_penalty_part": float(line_corridor_penalty_part),
            "line_heading_penalty_part": float(line_heading_penalty_part),
            "sector_line_penalty_part": float(sector_line_penalty_part),
            "sector_wrong_half_penalty_part": float(sector_wrong_half_penalty_part),
            "edge_proximity_penalty_part": float(edge_proximity_penalty_part),
            "steer_jerk_penalty_part": float(steer_jerk_penalty_part),
            "lateral_penalty_part": float(lateral_penalty_part),
            "smooth_speed_gate": float(smooth_speed_gate),
            "smooth_gate": float(smooth_gate),
            "curve_penalty_gate": float(curve_gate),
            "curve_penalty_gate_raw": float(curve_gate_raw),
            "jerk_recovery_gate": float(jerk_recovery_gate),
            "straight_stability_bonus_part": float(straight_stability_bonus_part),
            "stable_speed_bonus_part": float(stable_speed_bonus_part),
            "unearned_decel_penalty_part": float(unearned_decel_penalty_part),
            "brake_utilization_bonus_part": float(brake_utilization_bonus_part),
            "decel_kmh_per_step": float(decel_kmh_per_step),
            "steer_jerk": float(_steer_jerk),
        }
        return obs_np, shaped_reward, cost, terminated, truncated, base_info, tele

    def _wandb_log_episode_end(self, tele: dict, global_step: int) -> None:
        tr = str(tele.get("termination_reason", "unknown"))
        code = termination_reason_code(tr)
        n = max(int(tele.get("wrapper_episode_step", 0)), 1)
        margin_min = (
            float(self._wandb_ep_min_margin)
            if self._wandb_ep_min_margin < float("inf")
            else 0.0
        )
        loose = self._loose_track_pos_limit
        loose_csv = "" if loose is None else f"{float(loose):g}"
        max_tp = float(self._wandb_ep_max_abs_track_pos)
        dr = float(tele.get("dist_raced_m", 0.0))
        dfs = float(tele.get("dist_from_start_m", 0.0))
        oh = termination_reason_one_hot(tr)

        if os.environ.get("SACPID_EPISODE_END_CONSOLE", "1").strip() == "1":
            tag = f" [{self._label}]" if self._label else ""
            loose_s = f"{loose:g}" if loose is not None else "off"
            print(
                f"[SACPID episode end{tag}] termination_reason={tr!r} (code={code}) "
                f"length={n} loose_track_pos_limit={loose_s} max_abs_track_pos={max_tp:.4f} "
                f"dist_raced_m={dr:.2f} dist_from_start_m={dfs:.2f} "
                f"obs_dist_along={float(tele.get('obs_dist_along', 0.0)):.4f}"
            )

        csv_row: dict[str, object] = {
            "time_unix": int(time.time()),
            "env_label": self._label or "default",
            "termination_reason": tr,
            "termination_code": code,
            "episode_length": n,
            "loose_track_pos_limit": loose_csv,
            "max_abs_track_pos": max_tp,
            "dist_raced_m": dr,
            "dist_from_start_m": dfs,
        }
        for k, v in oh.items():
            csv_row[k] = int(v)
        append_sacpid_episode_termination_csv(csv_row)

        import wandb

        if wandb.run is None:
            return
        prefix = f"episode_end/{self._label}/" if self._label else "episode_end/"
        payload: dict[str, object] = {
            f"{prefix}return_mean": float(self._wandb_ep_sum_reward) / n,
            f"{prefix}speed_mean_kmh": float(self._wandb_ep_sum_speed_kmh) / n,
            f"{prefix}track_margin_min": margin_min,
            f"{prefix}termination_code": code,
            f"{prefix}termination_reason": tr,
            f"{prefix}length": n,
            f"{prefix}cost_sum": float(self._wandb_ep_sum_cost),
            f"{prefix}slip_event_count": int(self._wandb_ep_slip_event_count),
            f"{prefix}episode_index": int(self._episode_count),
            f"{prefix}loose_track_pos_limit": float(loose) if loose is not None else -1.0,
            f"{prefix}max_abs_track_pos": max_tp,
            f"{prefix}dist_raced_m": dr,
            f"{prefix}dist_from_start_m": dfs,
            f"{prefix}obs_dist_along_last": float(tele.get("obs_dist_along", 0.0)),
            f"{prefix}obs_dist_along_mean": float(self._wandb_ep_sum_obs_dist_along) / float(n),
        }
        payload["env_step"] = global_step
        wandb.log(payload, commit=True)

    def log_wandb_step(self, tele: dict) -> None:
        try:
            import wandb

            global_step = _wandb_bump_global_env_step()

            if wandb.run is not None and os.environ.get("SACPID_WANDB_STEP_TELEMETRY", "").strip() == "1":
                np_action = tele["np_action"]
                raw_obs = tele["raw_obs"]
                S = self._env.client.S.d
                speedX_kmh = float(raw_obs.speedX) * self._env.default_speed
                speedY_kmh = float(raw_obs.speedY) * self._env.default_speed

                telemetry = {
                    "agent/steer": float(np_action[0]),
                    "agent/steer_raw_policy": float(tele.get("steer_raw_policy", np_action[0])),
                    "agent/steer_after_rate_limit": float(tele.get("steer_after_rate_limit", np_action[0])),
                    "agent/steer_jitter_flip_damped": float(tele.get("steer_jitter_flip_damped", 0.0)),
                    "agent/steer_jitter_flip_cap_used": float(tele.get("steer_jitter_flip_cap_used", 0.0)),
                    "agent/steer_jitter_flip_streak": float(tele.get("steer_jitter_flip_streak", 0.0)),
                    "agent/steer_jitter_recovery_gate": float(tele.get("steer_jitter_recovery_gate", 1.0)),
                    "agent/accel": float(np_action[1]),
                    "agent/brake": float(np_action[2]),
                    "car/speedX_kmh": speedX_kmh,
                    "car/speedY_kmh": speedY_kmh,
                    "car/forward_speed_kmh": tele["forward_speed_kmh"],
                    "car/lateral_speed_kmh": tele["lateral_speed_kmh"],
                    "car/slip_event": int(tele["slip_event"]),
                    "car/slip_threshold_kmh": tele["slip_threshold_kmh"],
                    "car/trackPos": tele["track_pos"],
                    "car/angle_rad": tele["angle"],
                    "car/rpm": float(raw_obs.rpm),
                    "car/gear": int(S.get("gear", 0)),
                    "reward/total": tele["shaped_reward"],
                    "reward/speed_gate": tele["reward_speed_gate"],
                    "reward/progress": tele["progress_reward"],
                    "reward/center_bonus": tele["center_bonus"],
                    "reward/heading_bonus": tele["heading_bonus"],
                    "reward/line_bonus": tele.get("line_bonus", 0.0),
                    "reward/pace_bonus": tele.get("pace_bonus", 0.0),
                    "reward/centerline": tele["centerline_part"],
                    "reward/angle": tele["angle_part"],
                    "reward/low_speed": tele["low_speed_part"],
                    "reward/forward_bonus": tele["forward_bonus"],
                    "reward/dist_raced_bonus": tele["dist_raced_bonus"],
                    "reward/brake_sector_overspeed": float(tele.get("brake_sector_overspeed_part", 0.0)),
                    "reward/brake_sector_brake_bonus": float(tele.get("brake_sector_brake_bonus_part", 0.0)),
                    "reward/launch_phase_steer": float(tele.get("launch_phase_steer_part", 0.0)),
                    "reward/launch_phase_lateral": float(tele.get("launch_phase_lateral_part", 0.0)),
                    "reward/line_corridor_penalty": float(tele.get("line_corridor_penalty_part", 0.0)),
                    "reward/line_heading_penalty": float(tele.get("line_heading_penalty_part", 0.0)),
                    "reward/sector_line_penalty": float(tele.get("sector_line_penalty_part", 0.0)),
                    "reward/sector_wrong_half_penalty": float(
                        tele.get("sector_wrong_half_penalty_part", 0.0)
                    ),
                    "reward/edge_proximity_penalty": float(tele.get("edge_proximity_penalty_part", 0.0)),
                    "reward/steer_jerk_penalty": float(tele.get("steer_jerk_penalty_part", 0.0)),
                    "reward/lateral_penalty": float(tele.get("lateral_penalty_part", 0.0)),
                    "reward/smooth_speed_gate": float(tele.get("smooth_speed_gate", 0.0)),
                    "reward/smooth_gate": float(tele.get("smooth_gate", 0.0)),
                    "reward/curve_penalty_gate": float(tele.get("curve_penalty_gate", 1.0)),
                    "reward/curve_penalty_gate_raw": float(tele.get("curve_penalty_gate_raw", 1.0)),
                    "reward/straight_stability_bonus": float(tele.get("straight_stability_bonus_part", 0.0)),
                    "reward/stable_speed_bonus": float(tele.get("stable_speed_bonus_part", 0.0)),
                    "reward/unearned_decel_penalty": float(tele.get("unearned_decel_penalty_part", 0.0)),
                    "reward/brake_utilization_bonus": float(tele.get("brake_utilization_bonus_part", 0.0)),
                    "car/decel_kmh_per_step": float(tele.get("decel_kmh_per_step", 0.0)),
                    "reward/jerk_recovery_gate": float(tele.get("jerk_recovery_gate", 1.0)),
                    "agent/steer_jerk": float(tele.get("steer_jerk", 0.0)),
                    "cost/step": tele["cost"],
                }
                telemetry["env_step"] = global_step
                try:
                    telemetry["car/speedX_server"] = float(S.get("speedX", 0.0))
                except Exception:
                    pass
                if "track" in S:
                    track_arr = np.array(S["track"], dtype=np.float64)
                    telemetry["car/track_min"] = float(np.min(track_arr))
                    telemetry["car/track_max"] = float(np.max(track_arr))
                if "distRaced" in S:
                    telemetry["race/dist_raced_km"] = float(S["distRaced"]) / 1000.0
                if "distFromStart" in S:
                    telemetry["race/dist_from_start_km"] = float(S["distFromStart"]) / 1000.0
                telemetry["obs/dist_along_track"] = float(tele.get("obs_dist_along", 0.0))
                if "damage" in S:
                    telemetry["race/damage"] = float(S["damage"])
                if "curLapTime" in S:
                    telemetry["race/cur_lap_time_s"] = float(S["curLapTime"])
                if "lastLapTime" in S:
                    telemetry["race/last_lap_time_s"] = float(S["lastLapTime"])
                tr = str(tele.get("termination_reason", "unknown"))
                telemetry["episode/termination_reason_code"] = (
                    termination_reason_code(tr) if tele["terminated"] else -1
                )
                telemetry["train/wrapper_episode_step"] = int(tele.get("wrapper_episode_step", 0))
                if tele["terminated"]:
                    telemetry["episode/number"] = self._episode_count
                    telemetry["episode/termination_reason"] = str(
                        tele.get("termination_reason", "unknown")
                    )
                if self._last_reset_duration is not None:
                    telemetry["episode/reset_duration_s"] = self._last_reset_duration
                    self._last_reset_duration = None
                wandb.log(telemetry, commit=True)

            _obs_period = int(os.environ.get("SACPID_WANDB_OBS_SAMPLE_PERIOD", "0").strip() or "0")
            if (
                wandb.run is not None
                and _obs_period > 0
                and global_step > 0
                and global_step % _obs_period == 0
            ):
                wandb.log(
                    {
                        "obs/dist_along_track": float(tele.get("obs_dist_along", 0.0)),
                        "obs/dist_from_start_m": float(tele.get("dist_from_start_m", 0.0)),
                        "env_step": global_step,
                    },
                    commit=True,
                )

            if tele["terminated"]:
                self._wandb_log_episode_end(tele, global_step)
            elif self._last_reset_duration is not None:
                # Consume reset timing without per-step mode (already logged when SACPID_WANDB_STEP_TELEMETRY=1).
                self._last_reset_duration = None
        except Exception as _e:
            if not self._wandb_error_logged:
                print(f"[TorcsSimSlot] wandb telemetry error (logged once): {_e}")
                self._wandb_error_logged = True


# OmniSafe's OnlineAdapter creates both training `_env` and `_eval_env` with the same UDP port.
# A second `_TorcsSimSlot` on the same port opens a second snakeoil Client → SCR sees two clients
# on one port and may spam ***restart*** / never deliver clean telemetry. Share one slot per port.
_torcs_slot_pool_lock = threading.Lock()
_torcs_slot_by_port: dict[int, "_TorcsSimSlot"] = {}
_torcs_slot_refcount: dict[int, int] = {}


def _acquire_torcs_sim_slot(port: int, cfg: dict) -> "_TorcsSimSlot":
    with _torcs_slot_pool_lock:
        if port in _torcs_slot_by_port:
            _torcs_slot_refcount[port] = _torcs_slot_refcount.get(port, 0) + 1
            return _torcs_slot_by_port[port]
        slot = _TorcsSimSlot(port, cfg)
        _torcs_slot_by_port[port] = slot
        _torcs_slot_refcount[port] = 1
        return slot


def _release_torcs_sim_slot(port: int, slot: "_TorcsSimSlot") -> None:
    with _torcs_slot_pool_lock:
        n = _torcs_slot_refcount.get(port, 0)
        if n <= 1:
            _torcs_slot_refcount.pop(port, None)
            if _torcs_slot_by_port.get(port) is slot:
                _torcs_slot_by_port.pop(port, None)
            slot.close()
        else:
            _torcs_slot_refcount[port] = n - 1


@env_register
class TorcsSafeEnv(CMDP):
    """TORCS environment wrapped as an OmniSafe CMDP (single simulator)."""

    _support_envs = ["TorcsSafe-v0"]
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False

    _action_space: spaces.Box
    _observation_space: spaces.Box

    def __init__(self, env_id: str, **kwargs) -> None:
        super().__init__(env_id)
        kwargs.pop("num_envs", None)
        kwargs.pop("device", None)
        cfg = {**ENV_CONFIG, **kwargs}

        if "port" in cfg:
            _port = int(cfg["port"])
        elif "port_base" in cfg:
            global _port_lock, _port_next_index
            with _port_lock:
                _port = int(cfg["port_base"]) + _port_next_index
                _port_next_index += 1
        else:
            _port = 3001

        self._num_envs = 1
        self._torcs_slot_port: int = int(_port)
        self._auto_reset_on_done: bool = bool(cfg.get("auto_reset_on_done", True))
        self._auto_reset_on_loose_off_track: bool = bool(
            cfg.get("auto_reset_on_loose_off_track", False)
        )
        self._slot = _acquire_torcs_sim_slot(int(_port), cfg)

        self._device = self._slot._device
        self._action_space = _gymnasium_action_box(int(cfg.get("policy_action_dim", 3)))
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._slot._num_obs,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    def set_seed(self, seed: int) -> None:
        pass

    def close(self) -> None:
        _release_torcs_sim_slot(self._torcs_slot_port, self._slot)

    def render(self):
        pass

    def save(self):
        return {}

    def step(self, action: torch.Tensor):
        np_action = action.detach().cpu().numpy()
        if np_action.ndim > 1:
            np_action = np_action.squeeze(0)

        obs_np, shaped_reward, cost, terminated, truncated, base_info, tele = self._slot.step(np_action)
        self._slot.log_wandb_step(tele)

        info: dict = {"termination_reason": str(base_info.get("termination_reason", "unknown"))}
        done_now = bool(terminated) or bool(truncated)
        if done_now:
            info["sacpid_episode_length"] = int(tele["wrapper_episode_step"])
            info["sacpid_loose_track_pos_limit"] = self._slot._loose_track_pos_limit
            info["sacpid_max_abs_track_pos"] = float(self._slot._wandb_ep_max_abs_track_pos)
            info["sacpid_dist_raced_m"] = float(tele.get("dist_raced_m", 0.0))
            info["sacpid_dist_from_start_m"] = float(tele.get("dist_from_start_m", 0.0))
            # Unbatched tensors: OmniSafe's Unsqueeze wrapper adds the (1, …) batch dim for training.
            info["final_observation"] = torch.as_tensor(
                obs_np, dtype=torch.float32, device=self._device
            )
            info["_final_observation"] = torch.tensor(
                True, dtype=torch.bool, device=self._device
            )
            _reason = str(base_info.get("termination_reason", ""))
            _do_reset = self._auto_reset_on_done or (
                self._auto_reset_on_loose_off_track and _reason == "off_track_loose"
            )
            if _do_reset:
                obs_np = self._slot.reset()

        return (
            torch.as_tensor(obs_np, dtype=torch.float32, device=self._device),
            torch.as_tensor(shaped_reward, dtype=torch.float32, device=self._device),
            torch.as_tensor(cost, dtype=torch.float32, device=self._device),
            torch.as_tensor(terminated, dtype=torch.bool, device=self._device),
            torch.as_tensor(truncated, dtype=torch.bool, device=self._device),
            info,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs_np = self._slot.reset()
        return torch.as_tensor(obs_np, dtype=torch.float32, device=self._device), {}


@env_register
class TorcsHotlapEvalEnv(TorcsSafeEnv):
    """Hotlap-oriented defaults: gym_raw reward, no tight wrapper terminals, optional
    ``|trackPos|`` reset (``off_track_loose``) with selective auto-reset."""

    _support_envs = ["TorcsHotlapEval-v0"]

    def __init__(self, env_id: str, **kwargs) -> None:
        merged = {**ENV_CONFIG, **kwargs}
        merged.setdefault("auto_reset_on_done", False)
        merged.setdefault("gym_reset_relaunch", False)
        merged.setdefault("reward_mode", "gym_raw")
        merged.setdefault("wrapper_terminal", False)
        merged.setdefault("gym_terminate_on_off_track", False)
        merged.setdefault("watchdog_no_progress_steps", 0)
        merged.setdefault("loose_track_pos_limit", 1.25)
        merged.setdefault("auto_reset_on_loose_off_track", True)
        super().__init__(env_id, **merged)


@env_register
class TorcsVectorSafeEnv(CMDP):
    """N independent TORCS SCR clients; batched step/reset for OmniSafe ``vector_env_nums`` > 1."""

    _support_envs = ["TorcsVectorSafe-v0"]
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False

    _action_space: spaces.Box
    _observation_space: spaces.Box

    def __init__(self, env_id: str, **kwargs) -> None:
        super().__init__(env_id)
        num_envs = int(kwargs.pop("num_envs", 1))
        kwargs.pop("device", None)
        cfg = {**ENV_CONFIG, **kwargs}

        port_base = int(cfg.get("port_base", 3001))
        self._num_envs = num_envs
        self._auto_reset_on_done: bool = bool(cfg.get("auto_reset_on_done", True))
        self._auto_reset_on_loose_off_track: bool = bool(
            cfg.get("auto_reset_on_loose_off_track", False)
        )
        self._slots = [
            _TorcsSimSlot(port_base + i, cfg, label=f"env{i}") for i in range(num_envs)
        ]
        self._device = self._slots[0]._device

        self._action_space = _gymnasium_action_box(int(cfg.get("policy_action_dim", 3)))
        obs_n = self._slots[0]._num_obs
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_n,), dtype=np.float32
        )

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    def set_seed(self, seed: int) -> None:
        pass

    def close(self) -> None:
        for s in self._slots:
            s.close()

    def render(self):
        pass

    def save(self):
        return {}

    def _actions_to_matrix(self, action: torch.Tensor) -> np.ndarray:
        ac = action.detach().cpu().numpy()
        if ac.ndim == 1:
            ac = ac.reshape(1, -1)
        if ac.shape[0] != self._num_envs:
            raise ValueError(
                f"Expected action batch of size {self._num_envs}, got shape {ac.shape}"
            )
        return ac

    def step(self, action: torch.Tensor):
        ac = self._actions_to_matrix(action)
        if self._num_envs == 1:
            slot0 = self._slots[0]
            obs_np, shaped_reward, cost, terminated, truncated, bi, tele = slot0.step(ac[0])
            slot0.log_wandb_step(tele)
            info: dict = {"termination_reason": str(bi.get("termination_reason", "unknown"))}
            done_now = bool(terminated) or bool(truncated)
            if done_now:
                info["sacpid_episode_length"] = int(tele["wrapper_episode_step"])
                info["sacpid_loose_track_pos_limit"] = slot0._loose_track_pos_limit
                info["sacpid_max_abs_track_pos"] = float(slot0._wandb_ep_max_abs_track_pos)
                info["sacpid_dist_raced_m"] = float(tele.get("dist_raced_m", 0.0))
                info["sacpid_dist_from_start_m"] = float(tele.get("dist_from_start_m", 0.0))
                info["final_observation"] = torch.as_tensor(
                    obs_np, dtype=torch.float32, device=self._device
                )
                info["_final_observation"] = torch.tensor(
                    True, dtype=torch.bool, device=self._device
                )
                _reason0 = str(bi.get("termination_reason", ""))
                _do0 = self._auto_reset_on_done or (
                    self._auto_reset_on_loose_off_track and _reason0 == "off_track_loose"
                )
                if _do0:
                    obs_np = slot0.reset()
            return (
                torch.as_tensor(obs_np, dtype=torch.float32, device=self._device),
                torch.as_tensor(shaped_reward, dtype=torch.float32, device=self._device),
                torch.as_tensor(cost, dtype=torch.float32, device=self._device),
                torch.as_tensor(terminated, dtype=torch.bool, device=self._device),
                torch.as_tensor(truncated, dtype=torch.bool, device=self._device),
                info,
            )

        obs_rows: list[np.ndarray] = []
        r_list: list[float] = []
        c_list: list[float] = []
        term_list: list[bool] = []
        trunc_list: list[bool] = []
        reason_list: list[str] = []

        final_obs = torch.zeros((self._num_envs, self._observation_space.shape[0]), device=self._device)
        done_mask = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

        for i, slot in enumerate(self._slots):
            o_np, rw, co, te, tr, bi, tele = slot.step(ac[i])
            slot.log_wandb_step(tele)
            if te or tr:
                final_obs[i] = torch.as_tensor(o_np, dtype=torch.float32, device=self._device)
                done_mask[i] = True
                _ri = str(bi.get("termination_reason", ""))
                _doi = self._auto_reset_on_done or (
                    self._auto_reset_on_loose_off_track and _ri == "off_track_loose"
                )
                if _doi:
                    o_np = slot.reset()
            obs_rows.append(o_np)
            r_list.append(rw)
            c_list.append(co)
            term_list.append(te)
            trunc_list.append(tr)
            reason_list.append(str(bi.get("termination_reason", "unknown")))

        obs = torch.as_tensor(np.stack(obs_rows), dtype=torch.float32, device=self._device)
        info: dict = {}
        info["termination_reason"] = reason_list
        if bool(done_mask.any().item()):
            info["final_observation"] = final_obs
            info["_final_observation"] = done_mask

        return (
            obs,
            torch.as_tensor(r_list, dtype=torch.float32, device=self._device),
            torch.as_tensor(c_list, dtype=torch.float32, device=self._device),
            torch.as_tensor(term_list, dtype=torch.bool, device=self._device),
            torch.as_tensor(trunc_list, dtype=torch.bool, device=self._device),
            info,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        # #region agent log
        _debug_log(
            "H7",
            "torcs_omnisafe_wrapper.py:TorcsVectorSafeEnv.reset:entry",
            "vector reset begin",
            {
                "num_envs": self._num_envs,
            },
        )
        # #endregion
        obs_rows = []
        for i, s in enumerate(self._slots):
            # #region agent log
            _debug_log(
                "H7",
                "torcs_omnisafe_wrapper.py:TorcsVectorSafeEnv.reset:slot_begin",
                "vector reset slot begin",
                {
                    "index": int(i),
                },
            )
            # #endregion
            obs_rows.append(s.reset())
            # #region agent log
            _debug_log(
                "H7",
                "torcs_omnisafe_wrapper.py:TorcsVectorSafeEnv.reset:slot_done",
                "vector reset slot done",
                {
                    "index": int(i),
                },
            )
            # #endregion
        if self._num_envs == 1:
            return (
                torch.as_tensor(obs_rows[0], dtype=torch.float32, device=self._device),
                {},
            )
        obs_np = np.stack(obs_rows)
        return torch.as_tensor(obs_np, dtype=torch.float32, device=self._device), {}
