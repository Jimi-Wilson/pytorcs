"""
Evaluation engine for PPO checkpoints on TORCS.

Prerequisites:
  - TORCS must be running:  cd SACPID && ./autostart.sh 2
  - Dependencies installed: pip install -r eval/requirements.txt

Usage:
  python eval/evaluate.py --checkpoint /path/to/model.zip --episodes 10

  # With live dashboard:
  python eval/evaluate.py --checkpoint /path/to/model.zip --serve

See docs/EVAL.md for the full usage guide.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _is_headless() -> bool:
    """Return True when running in a headless environment (Docker, SSH, CI)."""
    if not sys.stdout.isatty():
        return True
    if not os.environ.get("DISPLAY") and sys.platform != "win32":
        return True
    return False


# ---------------------------------------------------------------------------
# Aggregation
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
        "dist_raced_m":        _stat([r.dist_raced_m for r in results]),
        "lap_time":            _stat(lap_times, higher_is_better=False) if lap_times else None,
        "mean_speed_kmh":      _stat([r.mean_speed_kmh for r in results]),
        "max_speed_kmh":       _stat([r.max_speed_kmh for r in results]),
        "min_speed_kmh":       _stat([r.min_speed_kmh for r in results]),
        "max_abs_track_pos":   _stat([r.max_abs_track_pos for r in results], higher_is_better=False),
        "mean_abs_track_pos":  _stat([r.mean_abs_track_pos for r in results], higher_is_better=False),
        "steering_smoothness": _stat([r.steering_smoothness for r in results], higher_is_better=False),
        "off_track_events":    _stat([float(r.off_track_events) for r in results], higher_is_better=False),
        "completion_rate":     round(len(completed) / len(results), 3) if results else 0.0,
        "best_lap_time":       round(min(lap_times), 3) if lap_times else None,
        # REWARD_REMOVED: "total_reward": _stat([r.total_reward for r in results]),
    }


# ---------------------------------------------------------------------------
# Rich console table
# ---------------------------------------------------------------------------

def _render_table(
    results: list[Any],
    agg: dict[str, Any],
    checkpoint: str,
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
    table.add_column("Stat", style="bold", min_width=26)
    table.add_column("Mean", justify="right", min_width=10)
    table.add_column("Std",  justify="right", min_width=8)
    table.add_column("Best", justify="right", min_width=10)

    def _row(label: str, stat: dict[str, float] | None, fmt: str = ".1f") -> None:
        if stat is None:
            table.add_row(label, "[dim]n/a[/dim]", "[dim]—[/dim]", "[dim]n/a[/dim]")
            return
        table.add_row(label, f"{stat['mean']:{fmt}}", f"{stat['std']:{fmt}}", f"{stat['best']:{fmt}}")

    _row("Distance raced (m)",        agg["dist_raced_m"])
    _row("Lap time (s)",              agg["lap_time"],           ".2f")
    _row("Mean speed (km/h)",         agg["mean_speed_kmh"])
    _row("Max speed (km/h)",          agg["max_speed_kmh"])
    _row("Min speed (km/h)",          agg["min_speed_kmh"])
    _row("Max track pos (0–1)",       agg["max_abs_track_pos"],  ".3f")
    _row("Mean track pos (0–1)",      agg["mean_abs_track_pos"], ".3f")
    _row("Steering smoothness (σ)",   agg["steering_smoothness"],".3f")
    _row("Off-track events",          agg["off_track_events"])
    # REWARD_REMOVED: _row("Total reward", agg["total_reward"], ".2f")

    cr = agg["completion_rate"]
    table.add_row("Lap completion rate", f"{cr * 100:.1f}%", "[dim]—[/dim]", "[dim]—[/dim]")

    reasons    = Counter(r.termination_reason for r in results)
    reason_str = "  ".join(f"[green]{k}[/green]={v}" for k, v in sorted(reasons.items()))
    subtitle   = f"[dim]{len(results)} episodes · {output_path}[/dim]"

    console.print()
    console.print(Panel(
        table,
        title=f"[bold]{Path(checkpoint).name}[/bold]  [dim](PPO)[/dim]",
        subtitle=subtitle,
        expand=False,
    ))
    console.print(f"  Termination reasons: {reason_str}\n")
    if report_path:
        console.print(f"  [bold]HTML report:[/bold] {report_path}\n")


def _fallback_print(results: list[Any], agg: dict[str, Any], checkpoint: str) -> None:
    print(f"\n=== {Path(checkpoint).name} — {len(results)} episodes ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run N evaluation episodes and collect race statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", "-c", required=True,
                        help="Path to a .zip checkpoint file, or a directory containing one.")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Passed to env.reset() (recorded in metadata for reproducibility).")
    parser.add_argument("--output", default=None,
                        help="Output JSON path. Defaults to results/eval_<timestamp>.json.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-step telemetry during each episode.")
    parser.add_argument("--serve", action="store_true",
                        help="Start live dashboard server (requires flask).")
    parser.add_argument("--serve-port", type=int, default=5001,
                        help="Port for the live dashboard.")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser (useful in Docker / headless environments).")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip HTML report generation.")
    # Advanced / rarely needed
    parser.add_argument("--port", type=int, default=3001,
                        help=argparse.SUPPRESS)
    parser.add_argument("--torcs-host", default="localhost",
                        help=argparse.SUPPRESS)
    parser.add_argument("--policy-action-dim", type=int, default=3, choices=[2, 3],
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("results") / f"eval_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent))
    from runner import load_ppo, run_episode

    print(f"Loading checkpoint: {args.checkpoint}")
    model, env = load_ppo(
        args.checkpoint,
        port=args.port,
        policy_action_dim=args.policy_action_dim,
        torcs_host=args.torcs_host,
    )

    # ── Live server ──────────────────────────────────────────────────────────
    srv = None
    if args.serve:
        import socket
        print(f"[serve] importing EvalServer …", flush=True)
        from server import EvalServer

        # Check the port isn't already occupied before we even try
        try:
            with socket.create_connection(("127.0.0.1", args.serve_port), timeout=0.3):
                print(f"[serve] ERROR: port {args.serve_port} is already in use — "
                      f"kill the process using it or choose a different port with --serve-port", flush=True)
                args.serve = False  # skip server, continue with eval
        except OSError:
            pass  # port is free, good

    if args.serve:
        print(f"[serve] constructing EvalServer on port {args.serve_port} …", flush=True)
        srv = EvalServer(
            total_episodes=args.episodes,
            checkpoint=args.checkpoint,
            port=args.serve_port,
        )
        print(f"[serve] starting Flask thread …", flush=True)
        srv.start()

        # Poll until Flask accepts connections (up to 10 s)
        deadline = time.monotonic() + 10.0
        attempt  = 0
        ready    = False
        while time.monotonic() < deadline:
            attempt += 1
            try:
                with socket.create_connection(("127.0.0.1", args.serve_port), timeout=0.2):
                    ready = True
                    print(f"[serve] port open after {attempt} poll(s)", flush=True)
                    break
            except OSError as e:
                if attempt % 10 == 0:
                    print(f"[serve] still waiting … (attempt {attempt}, error: {e})", flush=True)
                # If the Flask thread already crashed, stop waiting
                if getattr(srv, "_thread_error", []):
                    print(f"[serve] thread error detected, aborting wait", flush=True)
                    break
                time.sleep(0.1)

        if ready:
            print(f"\n  Live dashboard →  {srv.url}\n", flush=True)
            if not args.no_browser:
                import webbrowser
                webbrowser.open(srv.url)
        else:
            errors = getattr(srv, "_thread_error", [])
            if errors:
                print(f"\n  [serve] Flask failed to start: {errors[0]}\n", flush=True)
            else:
                print(f"\n  [serve] Timed out waiting for port {args.serve_port} after 10 s.\n"
                      f"  Try:  lsof -i :{args.serve_port}  to see what is using it.\n", flush=True)

    # ── Episode loop ─────────────────────────────────────────────────────────
    try:
        from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
        from rich.console import Console as RichConsole
        _rich = True
    except ImportError:
        _rich = False

    results = []

    def _run_one(i: int) -> None:
        if not _rich or args.verbose:
            print(f"  Episode {i + 1}/{args.episodes} …", end=" ", flush=True)
        result = run_episode(model, env, deterministic=True, verbose=args.verbose, seed=args.seed)
        results.append(result)
        if srv:
            srv.push_result(result)
        if not _rich or args.verbose:
            lap_str = f"lap={result.lap_time:.2f}s" if result.lap_time else "no lap"
            print(f"done  {result.termination_reason}  {lap_str}  {result.dist_raced_m:.0f}m")

    if _rich and not args.verbose:
        console = RichConsole()
        # First episode: show a spinner while waiting for the TORCS UDP handshake
        with console.status(f"Connecting to TORCS ({args.torcs_host}:{args.port})…"):
            _run_one(0)
        if args.episodes > 1:
            with Progress(SpinnerColumn(), "[progress.description]{task.description}",
                          BarColumn(), TaskProgressColumn(), TimeElapsedColumn()) as progress:
                task = progress.add_task("Evaluating", total=args.episodes)
                progress.advance(task)
                for i in range(1, args.episodes):
                    _run_one(i)
                    progress.advance(task)
    else:
        print(f"Connecting to TORCS ({args.torcs_host}:{args.port})…", flush=True)
        for i in range(args.episodes):
            _run_one(i)

    env.close()
    if srv:
        srv.set_complete()

    # ── Build and write payload ───────────────────────────────────────────────
    agg = _aggregate(results)
    payload: dict[str, Any] = {
        "meta": {
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "episodes":   args.episodes,
            "seed":       args.seed,
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
            report_path = str(generate_report(payload, output_path))
        except Exception as exc:
            print(f"  Warning: HTML report generation failed: {exc}")

    # ── Console summary ───────────────────────────────────────────────────────
    _render_table(results, agg, args.checkpoint, str(output_path), report_path)
    print(f"Results written to: {output_path}")

    # ── Keep live server up ───────────────────────────────────────────────────
    if srv:
        print(f"\nLive dashboard still running at {srv.url}")
        print("Press Ctrl+C to stop the server.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
