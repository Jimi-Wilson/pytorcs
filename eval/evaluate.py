"""
Evaluation engine for PPO (and future SACPID) checkpoints.

Prerequisites:
  - TORCS must be running:  cd SACPID && ./autostart.sh 2
  - Dependencies installed: pip install -r SACPID/requirements.txt
  - For live server:        pip install flask

Usage:
  python eval/evaluate.py \\
    --checkpoint SACPID/logs/ppo/stage4 \\
    --model-type ppo \\
    --episodes 10 \\
    --seed 42 \\
    --output results/eval_run_001.json

  # With live dashboard (opens browser automatically):
  python eval/evaluate.py --checkpoint ... --serve

See docs/EVAL.md for full usage guide.
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
    def _stat(vals: list[float], higher_is_better: bool = True) -> dict[str, float]:
        return {
            "mean": round(_mean(vals), 3),
            "std":  round(_std(vals), 3),
            "best": round((max if higher_is_better else min)(vals), 3) if vals else 0.0,
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
# Rich console table
# ---------------------------------------------------------------------------

def _render_table(
    results: list[Any],
    agg: dict[str, Any],
    checkpoint: str,
    model_type: str,
    output_path: str,
    report_path: str | None = None,
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

    def _row(label: str, stat: dict[str, float] | None, fmt: str = ".1f") -> None:
        if stat is None:
            table.add_row(label, "[dim]n/a[/dim]", "[dim]—[/dim]", "[dim]n/a[/dim]")
            return
        table.add_row(label, f"{stat['mean']:{fmt}}", f"{stat['std']:{fmt}}", f"{stat['best']:{fmt}}")

    _row("Total distance (m)",  agg["dist_raced_m"],     ".1f")
    _row("Lap time (s)",        agg["lap_time"],          ".2f")
    _row("Mean speed (km/h)",   agg["mean_speed_kmh"],    ".1f")
    _row("Max speed (km/h)",    agg["max_speed_kmh"],     ".1f")
    _row("Total reward",        agg["total_reward"],      ".2f")
    _row("Off-track events",    agg["off_track_events"],  ".1f")

    cr = agg["completion_rate"]
    table.add_row("Lap completion rate", f"{cr * 100:.1f}%", "[dim]—[/dim]", "[dim]—[/dim]")

    reasons    = Counter(r.termination_reason for r in results)
    reason_str = "  ".join(f"[green]{k}[/green]={v}" for k, v in sorted(reasons.items()))
    subtitle   = f"[dim]{len(results)} episodes · {output_path}[/dim]"

    console.print()
    console.print(Panel(table, title=f"[bold]{Path(checkpoint).name}[/bold]  ([dim]{model_type}[/dim])",
                        subtitle=subtitle, expand=False))
    console.print(f"  Termination reasons: {reason_str}\n")
    if report_path:
        console.print(f"  [bold]HTML report:[/bold] {report_path}\n")


def _fallback_print(results: list[Any], agg: dict[str, Any], checkpoint: str) -> None:
    print(f"\n=== Evaluation: {checkpoint} ({len(results)} episodes) ===")
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
                        help="Seed passed to env.reset() (included for reproducibility metadata).")
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
    parser.add_argument("--serve", action="store_true",
                        help="Start live dashboard server and open browser (requires flask).")
    parser.add_argument("--serve-port", type=int, default=5001,
                        help="Port for the live dashboard server.")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip HTML report generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve output path early so we fail fast on bad paths
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("results") / f"eval_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    # ── Live server ──────────────────────────────────────────────────────────
    srv = None
    if args.serve:
        from server import EvalServer
        import webbrowser
        srv = EvalServer(
            total_episodes=args.episodes,
            checkpoint=args.checkpoint,
            stage=args.stage,
            port=args.serve_port,
        )
        srv.start()
        print(f"Live dashboard: {srv.url}  (opening browser…)")
        webbrowser.open(srv.url)

    # ── Episode loop ─────────────────────────────────────────────────────────
    print(f"Running {args.episodes} episode(s)… (TORCS must be running)\n")

    try:
        from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
        _rich_available = True
    except ImportError:
        _rich_available = False

    results = []

    def _run_one(i: int) -> None:
        if not _rich_available or args.verbose:
            print(f"  Episode {i + 1}/{args.episodes} …", end=" ", flush=True)
        result = run_episode(model, env, deterministic=True, verbose=args.verbose, seed=args.seed)
        results.append(result)
        if srv:
            srv.push_result(result)
        if not _rich_available or args.verbose:
            lap_str = f"lap={result.lap_time:.2f}s" if result.lap_time else "no lap"
            print(f"done  reason={result.termination_reason}  {lap_str}  dist={result.dist_raced_m:.0f}m")

    if _rich_available and not args.verbose:
        with Progress(SpinnerColumn(), "[progress.description]{task.description}",
                      BarColumn(), TaskProgressColumn(), TimeElapsedColumn()) as progress:
            task = progress.add_task(f"Evaluating ({args.model_type})", total=args.episodes)
            for i in range(args.episodes):
                _run_one(i)
                progress.advance(task)
    else:
        for i in range(args.episodes):
            _run_one(i)

    env.close()

    if srv:
        srv.set_complete()

    # ── Build payload ─────────────────────────────────────────────────────────
    agg = _aggregate(results)
    payload: dict[str, Any] = {
        "meta": {
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "model_type": args.model_type,
            "episodes":   args.episodes,
            "seed":       args.seed,
            "stage":      args.stage,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        },
        "episodes":   [r.to_dict() for r in results],
        "aggregated": agg,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ── HTML report ───────────────────────────────────────────────────────────
    report_path = None
    if not args.no_report:
        try:
            from report import generate_report
            report_path = generate_report(payload, output_path)
            report_path = str(report_path)
        except Exception as exc:
            print(f"  Warning: HTML report generation failed: {exc}")

    # ── Console summary ───────────────────────────────────────────────────────
    _render_table(results, agg, args.checkpoint, args.model_type,
                  str(output_path), report_path)
    print(f"Results written to: {output_path}")

    # ── Keep live server up ───────────────────────────────────────────────────
    if srv:
        print(f"\nLive dashboard still running at {srv.url}")
        print("Press Ctrl+C to stop the server.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
