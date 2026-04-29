# Evaluation Guide

Runs a trained PPO checkpoint against TORCS, collects race statistics, writes a JSON results file and a self-contained HTML report, and optionally serves a live dashboard while the run is in progress.

---

## Prerequisites

1. **TORCS must be running** before the eval script starts — it connects over UDP and will hang if nothing is listening.

   ```bash
   cd SACPID && ./autostart.sh 2
   ```

2. **Python dependencies:**

   ```bash
   # CPU-only (recommended — no CUDA required for eval)
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
   pip install -r eval/requirements.txt
   ```

---

## Basic usage

```bash
python eval/evaluate.py --checkpoint /path/to/model.zip --episodes 10
```

This will:
- Run 10 episodes using the checkpoint
- Print a summary table to the terminal when all episodes finish
- Write results to `results/eval_<timestamp>.json`
- Write a self-contained HTML report to `results/eval_<timestamp>.html`

---

## Checkpoint path resolution

`--checkpoint` accepts:

- A direct `.zip` file: `/path/to/best_model.zip`
- A directory with a `latest_model_path.txt` marker (written by the training script)
- Any directory — the most recently modified `.zip` is used

---

## Flags

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` / `-c` | *(required)* | Path to a `.zip` checkpoint or directory |
| `--episodes` | `10` | Number of episodes to run |
| `--seed` | `42` | Recorded in metadata for reproducibility |
| `--output` | `results/eval_<timestamp>.json` | Where to write the JSON results file |
| `--verbose` | off | Print per-step telemetry (speed, distance, track position) |
| `--serve` | off | Start live dashboard server |
| `--serve-port` | `5001` | Port for the live dashboard |
| `--no-browser` | off | Do not auto-open a browser (use in Docker / headless environments) |
| `--no-report` | off | Skip HTML report generation |

---

## Race statistics collected

**Per episode:**
- Distance raced (m)
- Lap completed (yes/no) and lap time (s)
- Mean, max, and min speed (km/h)
- Max track position — worst lateral excursion (0 = centreline, 1 = edge)
- Mean track position — average centering quality (lower = better)
- Steering smoothness — standard deviation of steer actions (lower = smoother)
- Off-track events — count of excursions beyond the track boundary
- Termination reason (`lap_complete`, `timeout`, `off_track`, etc.)

**Aggregated across all episodes:**
- Mean, std, and best value for each of the above
- Completion rate
- Best lap time

---

## Outputs

### JSON results file

Written to `--output`. Contains run metadata, per-episode stats, and aggregated values.

### HTML report

Written alongside the JSON file (`.html` extension). Fully self-contained:
- No internet connection required
- Open directly in any browser
- Export to PDF via browser print (`Ctrl+P` → Save as PDF)
- Safe to email as an attachment

The report includes summary cards, five charts (distance, lap time, speed, track position, termination reasons), and a full per-episode breakdown table.

---

## Live dashboard

Add `--serve` to watch the run as it progresses:

```bash
python eval/evaluate.py --checkpoint /path/to/model.zip --episodes 10 --serve
```

The browser opens automatically at `http://localhost:5001`. The dashboard updates every 3 seconds and shows a **"Run complete"** banner when all episodes finish.

### Stopping the live server

The server stays up after the eval finishes so you can review the final state. The terminal will print:

```
Live dashboard still running at http://localhost:5001
Press Ctrl+C to stop the server.
```

Press **Ctrl+C** to stop it. The HTML report remains available at the path shown in the terminal summary and does not require the server.

---

## Running with Docker

Build the eval image:

```bash
docker build -f docker/Dockerfile.eval -t torcs-eval .
```

Run (TORCS must be running on the host):

```bash
docker run --rm \
  --network=host \
  -v /path/to/checkpoints:/checkpoints \
  -v /path/to/results:/app/results \
  torcs-eval \
  --checkpoint /checkpoints/best_model.zip \
  --episodes 10
```

With live dashboard:

```bash
docker run --rm \
  --network=host \
  -v /path/to/checkpoints:/checkpoints \
  -v /path/to/results:/app/results \
  torcs-eval \
  --checkpoint /checkpoints/best_model.zip \
  --episodes 10 \
  --serve \
  --no-browser
```

Then open `http://localhost:5001` in your browser on the host.

**Notes:**
- `--network=host` gives the container access to TORCS UDP on `localhost:3001`
- `--no-browser` is required — containers have no display
- Results written to `/app/results` inside the container are accessible via the volume mount
- The live server is accessible on the host at `http://localhost:5001` via `--network=host`

---

## Advanced flags

These are hidden from `--help` but accepted for compatibility:

| Flag | Default | Description |
|---|---|---|
| `--port` | `3001` | TORCS SCR UDP port |
| `--policy-action-dim` | `3` | Action dimensions: `2` = [steer, accel], `3` = [steer, accel, brake] |
