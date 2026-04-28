# Evaluation Guide

This document covers how to run the evaluation engine (`eval/evaluate.py`) against a trained PPO checkpoint.

---

## Prerequisites

1. **TORCS must be running** before you start the eval script. The script connects to TORCS over UDP and will hang indefinitely if no server is listening.

   ```bash
   cd SACPID
   ./autostart.sh 2      # launches TORCS on the Corkscrew track
   ```

2. **Python dependencies** — install from the SACPID requirements file:

   ```bash
   pip install -r SACPID/requirements.txt
   ```

3. **Flask** — only needed for the live dashboard (`--serve`):

   ```bash
   pip install flask
   ```

---

## Basic Usage

```bash
python eval/evaluate.py \
  --checkpoint SACPID/logs/ppo/stage4 \
  --episodes 10
```

This will:
- Run 10 episodes using the most recent `.zip` checkpoint found under `SACPID/logs/ppo/stage4`
- Print a Rich summary table to the terminal when all episodes finish
- Write results to `results/eval_<timestamp>.json`
- Write a self-contained HTML report to `results/eval_<timestamp>.html`

---

## All Flags

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` / `-c` | *(required)* | Path to a `.zip` checkpoint file, or a directory containing one |
| `--model-type` | `ppo` | `ppo` or `sacpid` (SACPID support coming soon) |
| `--episodes` | `10` | Number of episodes to run |
| `--seed` | `42` | Passed to `env.reset()` (TORCS ignores it; recorded in metadata) |
| `--output` | `results/eval_<timestamp>.json` | Where to write the JSON results file |
| `--stage` | `4` | Curriculum stage (1–5); sets env reward/cost limits |
| `--port` | `3001` | TORCS SCR UDP port |
| `--policy-action-dim` | `3` | Action dimensions: `2` = [steer, accel], `3` = [steer, accel, brake] |
| `--verbose` | off | Print per-step telemetry (speed, distance, track position) |
| `--serve` | off | Start live dashboard server and open browser automatically |
| `--serve-port` | `5001` | Port for the live dashboard |
| `--no-report` | off | Skip HTML report generation |

---

## Checkpoint Path Resolution

The `--checkpoint` flag accepts:

- A direct `.zip` file: `SACPID/logs/ppo/stage4/best_model.zip`
- A directory with a `latest_model_path.txt` marker (written by the training script)
- Any directory — the most recently modified `.zip` file is used

---

## Outputs

### JSON results file

Written to `--output` (default `results/eval_<timestamp>.json`). Contains:

```json
{
  "meta": { "checkpoint": "...", "model_type": "ppo", "episodes": 10, ... },
  "episodes": [ { "dist_raced_m": 3620, "lap_completed": true, ... }, ... ],
  "aggregated": {
    "dist_raced_m":   { "mean": 3241, "std": 182, "best": 3620 },
    "lap_time":       { "mean": 94.2, "std": 3.1,  "best": 89.7 },
    "completion_rate": 0.7,
    "best_lap_time":   89.7,
    ...
  }
}
```

### HTML report

Written alongside the JSON file (same name, `.html` extension). Self-contained — no internet connection required. Can be:
- Opened directly in any browser
- Exported to PDF via browser print (`Ctrl+P` → Save as PDF)
- Emailed as an attachment

---

## Live Dashboard

Add `--serve` to start a live dashboard that updates as each episode completes:

```bash
python eval/evaluate.py \
  --checkpoint SACPID/logs/ppo/stage4 \
  --episodes 10 \
  --serve
```

The browser opens automatically at `http://localhost:5001`. The dashboard shows:

- Episode progress bar
- Last episode summary (distance, lap time, speed, termination reason)
- Distance raced per episode (bar chart, colour-coded by termination reason)
- Mean speed per episode (line chart)

Charts update every 3 seconds while the eval is running. When all episodes finish, the page shows a **"Run complete"** banner and slows polling to every 10 seconds.

### Stopping the live server

The server stays up after the eval finishes so you can review the final state. The terminal will print:

```
Live dashboard still running at http://localhost:5001
Press Ctrl+C to stop the server.
```

Press **Ctrl+C** in that terminal to shut it down. The HTML report is still available at the path printed in the console summary — you do not need the server to view it.

---

## Colour coding

All charts use a consistent colour scheme:

| Colour | Termination reason |
|---|---|
| Green | `lap_complete` — successfully finished a lap |
| Amber | `timeout` — episode reached the step limit |
| Red | `off_track` / `crash` — car left the track boundary |
| Grey | Other / unknown |

---

## Example: full run with live server and verbose output

```bash
# Terminal 1 — start TORCS
cd SACPID && ./autostart.sh 2

# Terminal 2 — run eval
python eval/evaluate.py \
  --checkpoint SACPID/logs/ppo/stage4 \
  --episodes 5 \
  --stage 4 \
  --serve \
  --verbose
```
