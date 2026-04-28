"""
Evaluation engine for PPO (and future SACPID) checkpoints.

Prerequisites:
  - TORCS must be running:  cd SACPID && ./autostart.sh 2
  - Dependencies installed: pip install -r SACPID/requirements.txt

Usage:
  python eval/evaluate.py \\
    --checkpoint SACPID/logs/ppo/stage4 \\
    --model-type ppo \\
    --episodes 10 \\
    --seed 42 \\
    --output results/eval_run_001.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _aggregate(results: list[Any]) -> dict[str, Any]:
    from runner import EpisodeResult  # local import after sys.path is set

    def _stat(vals: list[float], higher_is_better: bool = True) -> dict[str, float]:
        return {
            "mean": round(_mean(vals), 3),
            "std": round(_std(vals), 3),
            "best": round(max(vals) if higher_is_better else min(vals), 3) if vals else 0.0,
        }

    completed = [r for r in results if r.lap_completed]
    lap_times = [r.lap_time for r in completed if r.lap_time is not None]

    return {
        "dist_raced_m":     _stat([r.dist_raced_m for r in results]),
        "lap_time":         _stat(lap_times, higher_is_better=False) if lap_times else None,
        "mean_speed_kmh":   _stat([r.mean_speed_kmh for r in results]),
        "max_speed_kmh":    _stat([r.max_speed_kmh for r in results]),
        "total_reward":     _stat([r.total_reward for r in results]),
        "off_track_events": _stat([float(r.off_track_events) for r in results], higher_is_better=False),
        "completion_rate":  round(len(completed) / len(results), 3) if results else 0.0,
        "best_lap_time":    round(min(lap_times), 3) if lap_times else None,
    }


# ---------------------------------------------------------------------------
# Rich display
# ---------------------------------------------------------------------------

def _render_table(
    results: list[Any],
    agg: dict[str, Any],
    checkpoint: str,
    model_type: str,
    output_path: str,
) -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
    except ImportError:
        _fallback_print(results, agg, checkpoint)
        return

    console = Console()
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
    table.add_column("KPI", style="bold", min_width=22)
    table.add_column("Mean", justify="right", min_width=10)
    table.add_column("Std", justify="right", min_width=8)
    table.add_column("Best", justify="right", min_width=10)

    def _row(label: str, stat: dict[str, float] | None, fmt: str = ".1f", unit: str = "") -> None:
        if stat is None:
            table.add_row(label, "[dim]n/a[/dim]", "[dim]—[/dim]", "[dim]n/a[/dim]")
            return
        table.add_row(
            label,
            f"{stat['mean']:{fmt}}{unit}",
            f"{stat['std']:{fmt}}",
            f"{stat['best']:{fmt}}{unit}",
        )

    _row("Total distance (m)",   agg["dist_raced_m"],     ".1f")
    _row("Lap time (s)",         agg["lap_time"],          ".2f")
    _row("Mean speed (km/h)",    agg["mean_speed_kmh"],    ".1f")
    _row("Max speed (km/h)",     agg["max_speed_kmh"],     ".1f")
    _row("Total reward",         agg["total_reward"],      ".2f")
    _row("Off-track events",     agg["off_track_events"],  ".1f")

    cr = agg["completion_rate"]
    table.add_row("Lap completion rate", f"{cr * 100:.1f}%", "[dim]—[/dim]", "[dim]—[/dim]")

    reasons = Counter(r.termination_reason for r in results)
    reason_str = "  ".join(f"[green]{k}[/green]={v}" for k, v in sorted(reasons.items()))

    title = f"[bold]{Path(checkpoint).name}[/bold]  ([dim]{model_type}[/dim])"
    subtitle = f"[dim]{len(results)} episodes · results → {output_path}[/dim]"
    footer = f"Termination reasons: {reason_str}"

    console.print()
    console.print(Panel(table, title=title, subtitle=subtitle, expand=False))
    console.print(f"  {footer}\n")


def _fallback_print(results: list[Any], agg: dict[str, Any], checkpoint: str) -> None:
    print(f"\n=== Evaluation: {checkpoint} ===")
    print(f"Episodes: {len(results)}")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run N evaluation episodes and collect KPIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", "-c", required=True,
                        help="Path to checkpoint (.zip for PPO, .pt / dir for SACPID).")
    parser.add_argument("--model-type", default="ppo", choices=["ppo", "sacpid"],
                        help="Which model type to load.")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed passed to env.reset() (TORCS ignores it; included for reproducibility metadata).")
    parser.add_argument("--output", default=None,
                        help="Output JSON path. Defaults to results/eval_<timestamp>.json.")
    parser.add_argument("--stage", type=int, default=4, choices=[1, 2, 3, 4, 5],
                        help="Curriculum stage config for env construction.")
    parser.add_argument("--port", type=int, default=3001,
                        help="TORCS SCR UDP port.")
    parser.add_argument("--policy-action-dim", type=int, default=3, choices=[2, 3],
                        help="Action dimension width of the checkpoint policy.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-step telemetry and env debug output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve output path before loading anything so we fail fast on bad paths
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("results") / f"eval_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Import runner after sys.path is already set by runner module itself
    sys.path.insert(0, str(Path(__file__).parent))
    from runner import load_ppo, load_sacpid, run_episode

    print(f"Loading {args.model_type} checkpoint: {args.checkpoint}")
    if args.model_type == "ppo":
        model, env = load_ppo(
            args.checkpoint,
            stage=args.stage,
            port=args.port,
            policy_action_dim=args.policy_action_dim,
        )
    else:
        load_sacpid(args.checkpoint)  # raises NotImplementedError

    print(f"Running {args.episodes} episode(s)... (TORCS must be running)\n")

    try:
        from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
        _rich_available = True
    except ImportError:
        _rich_available = False

    results = []

    def _run_episodes_plain() -> None:
        for i in range(args.episodes):
            print(f"  Episode {i + 1}/{args.episodes} ...", end=" ", flush=True)
            result = run_episode(model, env, deterministic=True, verbose=args.verbose, seed=args.seed)
            results.append(result)
            lap_str = f"lap={result.lap_time:.2f}s" if result.lap_time else "no lap"
            print(f"done  reason={result.termination_reason}  {lap_str}  dist={result.dist_raced_m:.0f}m")

    if _rich_available and not args.verbose:
        from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"Evaluating ({args.model_type})", total=args.episodes)
            for i in range(args.episodes):
                result = run_episode(model, env, deterministic=True, verbose=False, seed=args.seed)
                results.append(result)
                progress.advance(task)
    else:
        _run_episodes_plain()

    env.close()

    agg = _aggregate(results)

    payload: dict[str, Any] = {
        "meta": {
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "model_type": args.model_type,
            "episodes": args.episodes,
            "seed": args.seed,
            "stage": args.stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "episodes": [r.to_dict() for r in results],
        "aggregated": agg,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _render_table(results, agg, args.checkpoint, args.model_type, str(output_path))
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
