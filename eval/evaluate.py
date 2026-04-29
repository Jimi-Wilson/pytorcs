"""
Evaluation engine for PPO checkpoints on TORCS.

Prerequisites:
  - TORCS race must be running on the Corkscrew track with scr_server1 selected.
  - Dependencies installed: pip install -r eval/requirements.txt

Usage:
  python eval/evaluate.py --checkpoint /path/to/model.zip --laps 10

  # With live dashboard:
  python eval/evaluate.py --checkpoint /path/to/model.zip --laps 10 --serve

See docs/EVAL.md for the full usage guide.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def _aggregate(attempts: list[Any]) -> dict[str, Any]:
    def _stat(vals: list[float], higher_is_better: bool = True) -> dict[str, float]:
        return {
            "mean": round(_mean(vals), 3),
            "std":  round(_std(vals), 3),
            "best": round((max if higher_is_better else min)(vals), 3) if vals else 0.0,
        }

    completed  = [r for r in attempts if r.lap_completed]
    lap_times  = [r.lap_time for r in completed if r.lap_time is not None]

    return {
        "dist_raced_m":        _stat([r.dist_raced_m for r in attempts]),
        "lap_time":            _stat(lap_times, higher_is_better=False) if lap_times else None,
        "mean_speed_kmh":      _stat([r.mean_speed_kmh for r in attempts]),
        "max_speed_kmh":       _stat([r.max_speed_kmh for r in attempts]),
        "min_speed_kmh":       _stat([r.min_speed_kmh for r in attempts]),
        "max_abs_track_pos":   _stat([r.max_abs_track_pos for r in attempts], higher_is_better=False),
        "mean_abs_track_pos":  _stat([r.mean_abs_track_pos for r in attempts], higher_is_better=False),
        "steering_smoothness": _stat([r.steering_smoothness for r in attempts], higher_is_better=False),
        "off_track_events":    _stat([float(r.off_track_events) for r in attempts], higher_is_better=False),
        "completion_rate":     round(len(completed) / len(attempts), 3) if attempts else 0.0,
        "best_lap_time":       round(min(lap_times), 3) if lap_times else None,
        # REWARD_REMOVED: "total_reward": _stat([r.total_reward for r in attempts]),
    }


# ---------------------------------------------------------------------------
# Rich console table
# ---------------------------------------------------------------------------

def _render_table(
    attempts: list[Any],
    laps_completed: int,
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
        _fallback_print(attempts, agg, checkpoint)
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

    _row("Lap time (s)  [completed only]", agg["lap_time"],           ".2f")
    _row("Distance raced (m)",             agg["dist_raced_m"])
    _row("Mean speed (km/h)",              agg["mean_speed_kmh"])
    _row("Max speed (km/h)",               agg["max_speed_kmh"])
    _row("Min speed (km/h)",               agg["min_speed_kmh"])
    _row("Max track pos (0–1)",            agg["max_abs_track_pos"],  ".3f")
    _row("Mean track pos (0–1)",           agg["mean_abs_track_pos"], ".3f")
    _row("Steering smoothness (σ)",        agg["steering_smoothness"],".3f")
    _row("Off-track events",               agg["off_track_events"])
    # REWARD_REMOVED: _row("Total reward", agg["total_reward"], ".2f")

    cr = agg["completion_rate"]
    table.add_row(
        "Completion rate",
        f"{laps_completed} laps / {len(attempts)} attempts  ({cr * 100:.1f}%)",
        "[dim]—[/dim]", "[dim]—[/dim]",
    )

    reasons    = Counter(r.termination_reason for r in attempts)
    reason_str = "  ".join(f"[green]{k}[/green]={v}" for k, v in sorted(reasons.items()))
    subtitle   = f"[dim]{laps_completed} laps · {len(attempts)} attempts · {output_path}[/dim]"

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


def _fallback_print(attempts: list[Any], agg: dict[str, Any], checkpoint: str) -> None:
    print(f"\n=== {Path(checkpoint).name} — {len(attempts)} attempts ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print()


def _open_url_silently(url: str) -> None:
    """Open URL while suppressing browser launcher output."""
    try:
        if sys.platform.startswith("linux"):
            subprocess.Popen(
                ["xdg-open", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return
        if sys.platform == "darwin":
            subprocess.Popen(
                ["open", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return
    except Exception:
        pass

    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run to N completed laps and collect race statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", "-c", required=True,
                        help="Path to a .zip checkpoint file, or a directory containing one.")
    parser.add_argument("--laps", type=int, default=10,
                        help="Target number of completed laps.")
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="Hard ceiling on total attempts before stopping "
                             "(default: laps × 5, minimum 20).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Passed to the first env.reset() for reproducibility.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path. Defaults to results/eval_<timestamp>.json.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-step telemetry during each attempt.")
    parser.add_argument("--serve", action="store_true",
                        help="Start live dashboard server (requires flask).")
    parser.add_argument("--serve-port", type=int, default=5001,
                        help="Port for the live dashboard.")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser (useful in Docker / headless environments).")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip HTML report generation.")
    parser.add_argument("--reconnect-timeout-s", type=float, default=10.0,
                        help="After first connection, reconnect wait above this is treated as race finished.")
    parser.add_argument("--lap-debug", action="store_true",
                        help="Write per-step lap detection debug logs to JSONL file.")
    parser.add_argument("--lap-debug-file", default=None,
                        help="Path for lap debug JSONL file (default: results/lap_debug_<timestamp>.jsonl).")
    # Advanced / rarely needed
    parser.add_argument("--port", type=int, default=3001,
                        help=argparse.SUPPRESS)
    parser.add_argument("--torcs-host", default="localhost",
                        help=argparse.SUPPRESS)
    parser.add_argument("--policy-action-dim", type=int, default=3, choices=[2, 3],
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def _write_results(
    attempts: list[Any],
    laps_completed: int,
    target_laps: int,
    max_attempts: int,
    args: Any,
    output_path: Path,
    interrupted: bool,
) -> tuple[dict[str, Any], str | None]:
    """Aggregate attempts, write JSON, generate HTML report. Returns (payload, report_path)."""
    agg = _aggregate(attempts)
    payload: dict[str, Any] = {
        "meta": {
            "checkpoint":     str(Path(args.checkpoint).expanduser().resolve()),
            "target_laps":    target_laps,
            "laps_completed": laps_completed,
            "total_attempts": len(attempts),
            "max_attempts":   max_attempts,
            "seed":           args.seed,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "interrupted":    interrupted,
        },
        "attempts":   [r.to_dict() for r in attempts],
        "aggregated": agg,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = None
    if not args.no_report:
        try:
            from report import generate_report
            report_path = str(generate_report(payload, output_path))
        except Exception as exc:
            print(f"  Warning: HTML report generation failed: {exc}")

    return payload, report_path


def main() -> None:
    args = parse_args()

    target_laps  = args.laps
    max_attempts = args.max_attempts if args.max_attempts else max(target_laps * 5, 20)

    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("results") / f"eval_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent))
    from runner import load_ppo, run_episode

    os.environ["TORCS_RECONNECT_TIMEOUT_S"] = str(args.reconnect_timeout_s)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = None
    env = None

    # ── Live server ──────────────────────────────────────────────────────────
    srv = None

    # ── Laps loop ────────────────────────────────────────────────────────────
    print(
        f"Target: {target_laps} completed laps  (max {max_attempts} attempts)  "
        f"— launch race on Corkscrew with scr_server1 selected\n"
    )

    attempts: list[Any] = []
    laps_completed = 0
    attempt_count  = 0
    interrupted    = False
    race_finished_due_to_timeout = False
    progress_obj: Any = None
    progress_task: Any = None
    lap_debug_path: Path | None = None
    lap_debug_fp: Any = None

    if args.lap_debug:
        if args.lap_debug_file:
            lap_debug_path = Path(args.lap_debug_file)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            lap_debug_path = Path("results") / f"lap_debug_{ts}.jsonl"
        lap_debug_path.parent.mkdir(parents=True, exist_ok=True)
        lap_debug_fp = lap_debug_path.open("a", encoding="utf-8")
        print(f"[lap-debug] writing lap diagnostics to: {lap_debug_path}")

    def _record_lap(lap_time_s: float | None = None) -> None:
        nonlocal laps_completed, progress_obj, progress_task
        laps_completed += 1
        if srv:
            srv.push_lap_event(lap_time_s)
        if progress_obj is not None and progress_task is not None:
            progress_obj.update(
                progress_task,
                description=f"Laps: {laps_completed}/{target_laps}  Attempts: {attempt_count}",
            )
            progress_obj.advance(progress_task, 1)

    def _run_one() -> None:
        nonlocal laps_completed, attempt_count, race_finished_due_to_timeout
        attempt_count += 1
        laps_seen_live = 0

        def _on_lap(lap_time_s: float) -> None:
            nonlocal laps_seen_live
            laps_seen_live += 1
            _record_lap(lap_time_s)

        seed = args.seed if attempt_count == 1 else None
        if not _rich or args.verbose:
            print(f"  Attempt {attempt_count}  (lap {laps_completed + 1}/{target_laps}) …",
                  end=" ", flush=True)
        try:
            result = run_episode(
                model,
                env,
                deterministic=True,
                verbose=args.verbose,
                seed=seed,
                step_callback=srv.push_step if srv else None,
                lap_callback=_on_lap,
                should_stop=lambda: laps_completed >= target_laps,
                lap_debug=bool(args.lap_debug),
                lap_debug_writer=(lambda line: (lap_debug_fp.write(line + "\n"), lap_debug_fp.flush()))
                if lap_debug_fp
                else None,
            )
        except Exception as exc:
            msg = str(exc)
            if "race_finished_reconnect_timeout:" in msg:
                race_finished_due_to_timeout = True
                print(
                    f"\n  Race finished: reconnect wait exceeded {args.reconnect_timeout_s:.1f}s "
                    f"after prior connection. Writing report…"
                )
                return
            print(f"\n  Attempt {attempt_count} failed: {exc}")
            return
        attempts.append(result)
        if result.laps_in_attempt > laps_seen_live:
            for _ in range(result.laps_in_attempt - laps_seen_live):
                _record_lap(None)
        if srv:
            srv.push_result(result)
        if not _rich or args.verbose:
            if result.laps_in_attempt > 0 and result.lap_time:
                lap_str = f"laps={result.laps_in_attempt} best={result.lap_time:.2f}s"
            else:
                lap_str = "no lap"
            print(f"done  {result.termination_reason}  {lap_str}  {result.dist_raced_m:.0f}m"
                  f"  [{laps_completed}/{target_laps} laps]")

    try:
        model, env = load_ppo(
            args.checkpoint,
            port=args.port,
            policy_action_dim=args.policy_action_dim,
            torcs_host=args.torcs_host,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted during setup.")
        return

    try:
        if args.serve:
            import socket
            print(f"[serve] importing EvalServer …", flush=True)
            from server import EvalServer

            try:
                with socket.create_connection(("127.0.0.1", args.serve_port), timeout=0.3):
                    print(f"[serve] ERROR: port {args.serve_port} is already in use — "
                          f"kill the process using it or choose a different port with --serve-port",
                          flush=True)
                    args.serve = False
            except OSError:
                pass

        if args.serve:
            print(f"[serve] constructing EvalServer on port {args.serve_port} …", flush=True)
            srv = EvalServer(
                target_laps=target_laps,
                max_attempts=max_attempts,
                checkpoint=args.checkpoint,
                port=args.serve_port,
            )
            print(f"[serve] starting Flask thread …", flush=True)
            srv.start()

            deadline = time.monotonic() + 10.0
            poll_n   = 0
            ready    = False
            while time.monotonic() < deadline:
                poll_n += 1
                try:
                    with socket.create_connection(("127.0.0.1", args.serve_port), timeout=0.2):
                        ready = True
                        print(f"[serve] port open after {poll_n} poll(s)", flush=True)
                        break
                except OSError as e:
                    if poll_n % 10 == 0:
                        print(f"[serve] still waiting … (attempt {poll_n}, error: {e})", flush=True)
                    if getattr(srv, "_thread_error", []):
                        print(f"[serve] thread error detected, aborting wait", flush=True)
                        break
                    time.sleep(0.1)

            if ready:
                print(f"\n  Live dashboard →  {srv.url}\n", flush=True)
                if not args.no_browser:
                    _open_url_silently(srv.url)
            else:
                errors = getattr(srv, "_thread_error", [])
                if errors:
                    print(f"\n  [serve] Flask failed to start: {errors[0]}\n", flush=True)
                else:
                    print(f"\n  [serve] Timed out waiting for port {args.serve_port} after 10 s.\n"
                          f"  Try:  lsof -i :{args.serve_port}  to see what is using it.\n", flush=True)

        try:
            from rich.progress import (
                Progress, SpinnerColumn, BarColumn, TaskProgressColumn,
                TimeElapsedColumn, TextColumn,
            )
            _rich = True
        except ImportError:
            _rich = False

        if _rich and not args.verbose:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task(
                    f"Laps: 0/{target_laps}  Attempts: 0", total=target_laps
                )
                progress_obj = progress
                progress_task = task
                while (
                    laps_completed < target_laps
                    and attempt_count < max_attempts
                    and not race_finished_due_to_timeout
                ):
                    _run_one()
                    progress.update(
                        task,
                        description=f"Laps: {laps_completed}/{target_laps}  Attempts: {attempt_count}",
                    )
                progress_obj = None
                progress_task = None
        else:
            while (
                laps_completed < target_laps
                and attempt_count < max_attempts
                and not race_finished_due_to_timeout
            ):
                _run_one()
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n  Interrupted — {laps_completed} laps completed in {attempt_count} attempts"
              f" — saving partial results…")
    finally:
        try:
            if env:
                env.close()
        except Exception:
            pass
        try:
            if lap_debug_fp:
                lap_debug_fp.close()
        except Exception:
            pass
        if srv:
            srv.set_complete()

    if not attempts:
        print("No attempts completed — nothing to report.")
        return

    if race_finished_due_to_timeout:
        print(
            f"  Race finished due to reconnect timeout (> {args.reconnect_timeout_s:.1f}s)."
        )
    elif not interrupted and attempt_count >= max_attempts and laps_completed < target_laps:
        print(f"  Max attempts ({max_attempts}) reached with {laps_completed}/{target_laps} laps completed.")

    # ── Build and write payload ───────────────────────────────────────────────
    payload, report_path = _write_results(
        attempts, laps_completed, target_laps, max_attempts, args, output_path, interrupted
    )
    agg = payload["aggregated"]

    # ── Console summary ───────────────────────────────────────────────────────
    _render_table(attempts, laps_completed, agg, args.checkpoint, str(output_path), report_path)
    label = "Partial results" if interrupted else "Results"
    print(f"{label} written to: {output_path}")

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
