"""
Generates a self-contained HTML evaluation report.

All assets (Chart.js) are inlined — the file works offline and can be
emailed as an attachment or exported to PDF via browser print.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _chartjs() -> str:
    vendor = Path(__file__).parent / "vendor" / "chart.min.js"
    return vendor.read_text(encoding="utf-8")


def generate_report(payload: dict[str, Any], output_path: Path) -> None:
    """Write a standalone HTML report next to the JSON output file."""
    html = _build_html(payload, _chartjs())
    report_path = output_path.with_suffix(".html")
    report_path.write_text(html, encoding="utf-8")
    return report_path


def _colour(reason: str) -> str:
    return {
        "lap_complete": "#22c55e",
        "timeout":      "#f59e0b",
        "off_track":    "#ef4444",
    }.get(reason, "#94a3b8")


def _build_html(payload: dict[str, Any], chartjs_src: str) -> str:
    meta = payload["meta"]
    agg  = payload["aggregated"]
    eps  = payload["episodes"]

    checkpoint   = Path(meta["checkpoint"]).name
    model_type   = meta["model_type"].upper()
    stage        = meta["stage"]
    n_episodes   = meta["episodes"]
    seed         = meta["seed"]
    timestamp    = meta["timestamp"][:19].replace("T", " ")

    completion_pct = f'{agg["completion_rate"] * 100:.1f}%'
    best_lap  = f'{agg["best_lap_time"]:.2f} s'   if agg["best_lap_time"]  else "—"
    mean_lap  = f'{agg["lap_time"]["mean"]:.2f} s' if agg["lap_time"]       else "—"
    mean_spd  = f'{agg["mean_speed_kmh"]["mean"]:.1f} km/h'
    best_spd  = f'{agg["max_speed_kmh"]["best"]:.1f} km/h'
    mean_rwd  = f'{agg["total_reward"]["mean"]:.1f}'
    mean_dist = f'{agg["dist_raced_m"]["mean"]:.0f} m'

    reasons = Counter(e["termination_reason"] for e in eps)
    reason_labels = json.dumps(list(reasons.keys()))
    reason_counts = json.dumps(list(reasons.values()))
    reason_colours = json.dumps([_colour(k) for k in reasons.keys()])

    ep_labels    = json.dumps([f"Ep {i+1}" for i in range(len(eps))])
    lap_times    = json.dumps([e["lap_time"] if e["lap_time"] else None for e in eps])
    lap_colours  = json.dumps([_colour(e["termination_reason"]) for e in eps])
    mean_speeds  = json.dumps([round(e["mean_speed_kmh"], 1) for e in eps])
    distances    = json.dumps([round(e["dist_raced_m"], 0) for e in eps])
    dist_colours = json.dumps([_colour(e["termination_reason"]) for e in eps])

    rows = ""
    for i, e in enumerate(eps):
        lt   = f'{e["lap_time"]:.2f}' if e["lap_time"] else "—"
        dot  = f'<span class="dot" style="background:{_colour(e["termination_reason"])}"></span>'
        rows += (
            f'<tr>'
            f'<td>{i+1}</td>'
            f'<td>{e["dist_raced_m"]:.0f}</td>'
            f'<td>{"✓" if e["lap_completed"] else "✗"}</td>'
            f'<td>{lt}</td>'
            f'<td>{e["mean_speed_kmh"]:.1f}</td>'
            f'<td>{e["max_speed_kmh"]:.1f}</td>'
            f'<td>{e["total_reward"]:.1f}</td>'
            f'<td>{e["off_track_events"]}</td>'
            f'<td>{dot} {e["termination_reason"]}</td>'
            f'</tr>\n'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation Report — {checkpoint}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px;
    color: #1e293b;
    background: #f8fafc;
    padding: 2rem;
  }}

  .page {{ max-width: 1100px; margin: 0 auto; background: #fff;
           padding: 2.5rem 3rem; border-radius: 8px;
           box-shadow: 0 1px 3px rgba(0,0,0,.1); }}

  /* ── Header ── */
  header {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 1.25rem; margin-bottom: 2rem; }}
  header h1 {{ font-size: 1.6rem; font-weight: 700; color: #0f172a; }}
  header h1 span {{ font-weight: 400; color: #64748b; font-size: 1.1rem; margin-left: .5rem; }}
  .meta-row {{ display: flex; flex-wrap: wrap; gap: .5rem 2rem; margin-top: .6rem;
               font-size: .85rem; color: #64748b; }}
  .meta-row strong {{ color: #334155; }}

  /* ── Section headings ── */
  h2 {{ font-size: 1rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: .05em; color: #475569; margin: 2rem 0 1rem; }}

  /* ── Summary cards ── */
  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 1rem; }}
  .card {{ background: #f1f5f9; border-radius: 6px; padding: 1rem 1.25rem; }}
  .card .label {{ font-size: .75rem; text-transform: uppercase; letter-spacing: .05em;
                  color: #64748b; margin-bottom: .3rem; }}
  .card .value {{ font-size: 1.5rem; font-weight: 700; color: #0f172a; }}
  .card .sub   {{ font-size: .78rem; color: #94a3b8; margin-top: .2rem; }}
  .card.green .value {{ color: #16a34a; }}
  .card.amber .value {{ color: #d97706; }}

  /* ── Charts ── */
  .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  .chart-wide  {{ grid-column: 1 / -1; }}
  .chart-box   {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
                  padding: 1.25rem; }}
  .chart-box h3 {{ font-size: .82rem; font-weight: 600; text-transform: uppercase;
                   letter-spacing: .04em; color: #64748b; margin-bottom: .75rem; }}
  .chart-box canvas {{ max-height: 260px; }}

  /* ── Table ── */
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th {{ background: #f1f5f9; text-align: left; padding: .5rem .75rem;
        font-size: .75rem; text-transform: uppercase; letter-spacing: .04em;
        color: #64748b; border-bottom: 1px solid #e2e8f0; }}
  td {{ padding: .45rem .75rem; border-bottom: 1px solid #f1f5f9; color: #334155; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f8fafc; }}
  .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
          vertical-align: middle; margin-right: 4px; }}

  /* ── Legend ── */
  .legend {{ display: flex; gap: 1.5rem; font-size: .82rem; color: #475569; margin-top: .75rem; }}
  .legend span {{ display: flex; align-items: center; gap: .35rem; }}
  .legend .dot {{ width: 10px; height: 10px; flex-shrink: 0; }}

  /* ── Footer ── */
  footer {{ margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;
            font-size: .78rem; color: #94a3b8; }}

  /* ── Print styles ── */
  @media print {{
    body {{ background: #fff; padding: .5rem; }}
    .page {{ box-shadow: none; padding: 1rem 1.5rem; max-width: 100%; }}
    .charts-grid {{ grid-template-columns: 1fr 1fr; }}
    h2 {{ margin-top: 1.25rem; }}
    tr {{ page-break-inside: avoid; }}
    .chart-box canvas {{ max-height: 200px; }}
  }}
</style>
</head>
<body>
<div class="page">

  <!-- Header -->
  <header>
    <h1>TORCS Evaluation Report <span>— {checkpoint}</span></h1>
    <div class="meta-row">
      <div><strong>Model type:</strong> {model_type}</div>
      <div><strong>Stage:</strong> {stage}</div>
      <div><strong>Episodes:</strong> {n_episodes}</div>
      <div><strong>Seed:</strong> {seed}</div>
      <div><strong>Generated:</strong> {timestamp} UTC</div>
    </div>
  </header>

  <!-- Summary -->
  <h2>Summary</h2>
  <div class="cards">
    <div class="card green">
      <div class="label">Completion rate</div>
      <div class="value">{completion_pct}</div>
      <div class="sub">{reasons.get("lap_complete", 0)} of {n_episodes} laps</div>
    </div>
    <div class="card">
      <div class="label">Best lap time</div>
      <div class="value">{best_lap}</div>
      <div class="sub">mean {mean_lap}</div>
    </div>
    <div class="card">
      <div class="label">Mean speed</div>
      <div class="value">{mean_spd}</div>
      <div class="sub">peak {best_spd}</div>
    </div>
    <div class="card">
      <div class="label">Mean reward</div>
      <div class="value">{mean_rwd}</div>
    </div>
    <div class="card">
      <div class="label">Mean distance</div>
      <div class="value">{mean_dist}</div>
    </div>
  </div>

  <!-- Charts -->
  <h2>Performance Charts</h2>
  <div class="charts-grid">
    <div class="chart-box chart-wide">
      <h3>Distance raced per episode</h3>
      <canvas id="distChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Lap time per episode</h3>
      <canvas id="lapChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Mean speed per episode</h3>
      <canvas id="speedChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Termination reasons</h3>
      <canvas id="reasonChart"></canvas>
    </div>
  </div>

  <div class="legend">
    <span><span class="dot" style="background:#22c55e"></span>Lap complete</span>
    <span><span class="dot" style="background:#f59e0b"></span>Timeout</span>
    <span><span class="dot" style="background:#ef4444"></span>Off-track / crash</span>
    <span><span class="dot" style="background:#94a3b8"></span>Other</span>
  </div>

  <!-- Episode table -->
  <h2>Episode Breakdown</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Distance (m)</th>
          <th>Lap</th>
          <th>Lap time (s)</th>
          <th>Mean spd (km/h)</th>
          <th>Max spd (km/h)</th>
          <th>Reward</th>
          <th>Off-track</th>
          <th>Termination</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>

  <!-- Footer -->
  <footer>
    Generated by eval/evaluate.py &nbsp;·&nbsp; Chart.js 4.4.4 (offline, self-contained)
  </footer>

</div>

<script>{chartjs_src}</script>
<script>
const LABELS   = {ep_labels};
const LAP_T    = {lap_times};
const LAP_COL  = {lap_colours};
const SPEEDS   = {mean_speeds};
const DISTS    = {distances};
const DIST_COL = {dist_colours};
const R_LABELS = {reason_labels};
const R_COUNTS = {reason_counts};
const R_COLS   = {reason_colours};

const defaults = {{
  responsive: true,
  maintainAspectRatio: true,
  plugins: {{ legend: {{ display: false }} }},
}};

// Distance chart
new Chart(document.getElementById('distChart'), {{
  type: 'bar',
  data: {{
    labels: LABELS,
    datasets: [{{ data: DISTS, backgroundColor: DIST_COL, borderRadius: 3 }}]
  }},
  options: {{ ...defaults, scales: {{
    y: {{ title: {{ display: true, text: 'metres' }} }}
  }} }}
}});

// Lap time chart — null values shown as gaps
new Chart(document.getElementById('lapChart'), {{
  type: 'bar',
  data: {{
    labels: LABELS,
    datasets: [{{ data: LAP_T, backgroundColor: LAP_COL, borderRadius: 3 }}]
  }},
  options: {{ ...defaults, scales: {{
    y: {{ title: {{ display: true, text: 'seconds' }} }}
  }} }}
}});

// Speed chart
new Chart(document.getElementById('speedChart'), {{
  type: 'line',
  data: {{
    labels: LABELS,
    datasets: [{{
      data: SPEEDS,
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,.1)',
      fill: true,
      tension: 0.3,
      pointRadius: 4,
    }}]
  }},
  options: {{ ...defaults, scales: {{
    y: {{ title: {{ display: true, text: 'km/h' }} }}
  }} }}
}});

// Termination donut
new Chart(document.getElementById('reasonChart'), {{
  type: 'doughnut',
  data: {{
    labels: R_LABELS,
    datasets: [{{ data: R_COUNTS, backgroundColor: R_COLS, hoverOffset: 6 }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: true,
    plugins: {{ legend: {{ display: true, position: 'bottom' }} }},
  }}
}});
</script>
</body>
</html>"""
