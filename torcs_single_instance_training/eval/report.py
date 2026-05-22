"""
Generates a self-contained HTML evaluation report.

Chart.js and Barlow Condensed are inlined — the file works offline,
can be emailed, and exports cleanly to PDF via browser print.
"""

from __future__ import annotations

import base64
import json
from collections import Counter
from pathlib import Path
from typing import Any

_VENDOR = Path(__file__).parent / "vendor"


def _chartjs() -> str:
    return (_VENDOR / "chart.min.js").read_text(encoding="utf-8")


def _font_face_css() -> str:
    def _b64(name: str) -> str:
        return base64.b64encode((_VENDOR / name).read_bytes()).decode()

    return f"""
@font-face {{
  font-family: 'Barlow Condensed';
  font-style: normal;
  font-weight: 400;
  src: url('data:font/woff2;base64,{_b64("barlow-400.woff2")}') format('woff2');
}}
@font-face {{
  font-family: 'Barlow Condensed';
  font-style: normal;
  font-weight: 600;
  src: url('data:font/woff2;base64,{_b64("barlow-600.woff2")}') format('woff2');
}}
@font-face {{
  font-family: 'Barlow Condensed';
  font-style: normal;
  font-weight: 700;
  src: url('data:font/woff2;base64,{_b64("barlow-700.woff2")}') format('woff2');
}}"""


def generate_report(payload: dict[str, Any], output_path: Path) -> Path:
    report_path = output_path.with_suffix(".html")
    report_path.write_text(_build_html(payload, _chartjs(), _font_face_css()), encoding="utf-8")
    return report_path


def _colour(reason: str) -> str:
    return {"lap_complete": "#22c55e", "timeout": "#f59e0b", "off_track": "#ef4444"}.get(reason, "#94a3b8")


def _build_html(payload: dict[str, Any], chartjs_src: str, font_css: str) -> str:
    meta = payload["meta"]
    agg  = payload["aggregated"]
    eps  = payload["attempts"]

    checkpoint      = Path(meta["checkpoint"]).name
    target_laps     = meta.get("target_laps", meta.get("episodes", len(eps)))
    laps_completed  = meta.get("laps_completed", sum(1 for e in eps if e.get("lap_completed")))
    total_attempts  = meta.get("total_attempts", len(eps))
    seed            = meta["seed"]
    timestamp       = meta["timestamp"][:19].replace("T", " ")

    completion_pct = f'{agg["completion_rate"] * 100:.1f}%'
    reasons   = Counter(e["termination_reason"] for e in eps)
    n_complete = sum(1 for e in eps if e.get("laps_in_attempt", 0) > 0)

    best_lap  = f'{agg["best_lap_time"]:.2f}' if agg["best_lap_time"] else "—"
    mean_lap  = f'avg {agg["lap_time"]["mean"]:.2f} s' if agg["lap_time"] else "—"
    mean_spd  = f'{agg["mean_speed_kmh"]["mean"]:.1f}'
    best_spd  = f'{agg["max_speed_kmh"]["best"]:.1f}'
    min_spd   = f'{agg["min_speed_kmh"]["best"]:.1f}'
    mean_dist = f'{agg["dist_raced_m"]["mean"]:.0f}'
    max_tp    = f'{agg["max_abs_track_pos"]["best"] * 100:.1f}'
    steer_sm  = f'{agg["steering_smoothness"]["best"]:.3f}'

    ep_labels    = json.dumps([f"A{i+1}" for i in range(len(eps))])
    lap_times    = json.dumps([e["lap_time"] if e["lap_time"] else None for e in eps])
    lap_colours  = json.dumps(["#7c6fa0"] * len(eps))
    mean_speeds  = json.dumps([round(e["mean_speed_kmh"], 1) for e in eps])
    distances    = json.dumps([round(e["dist_raced_m"], 0) for e in eps])
    dist_colours = json.dumps([_colour(e["termination_reason"]) for e in eps])
    track_pos    = json.dumps([round(e["max_abs_track_pos"] * 100, 1) for e in eps])
    tp_colours   = json.dumps(["#7c6fa0"] * len(eps))
    reason_labels  = json.dumps(list(reasons.keys()))
    reason_counts  = json.dumps(list(reasons.values()))
    reason_colours = json.dumps([_colour(k) for k in reasons.keys()])

    rows = ""
    for i, e in enumerate(eps):
        lt  = f'{e["lap_time"]:.2f}' if e["lap_time"] else "—"
        dot = f'<span class="dot" style="background:{_colour(e["termination_reason"])}"></span>'
        n_laps = e.get("laps_in_attempt", 1 if e.get("lap_completed") else 0)
        if n_laps > 0:
            chk = f'<span style="color:#22c55e;font-weight:700">{n_laps} lap{"s" if n_laps > 1 else ""}</span>'
        else:
            chk = '<span style="color:#94a3b8">—</span>'
        rows += (
            f'<tr>'
            f'<td class="num">A{i+1}</td>'
            f'<td class="num">{e["dist_raced_m"]:.0f}</td>'
            f'<td style="text-align:center">{chk}</td>'
            f'<td class="num">{lt}</td>'
            f'<td class="num">{e["mean_speed_kmh"]:.1f}</td>'
            f'<td class="num">{e["max_speed_kmh"]:.1f}</td>'
            f'<td class="num">{e["min_speed_kmh"]:.1f}</td>'
            f'<td class="num">{e["max_abs_track_pos"] * 100:.1f}%</td>'
            f'<td class="num">{e["steering_smoothness"]:.3f}</td>'
            f'<td class="num">{e["off_track_events"]}</td>'
            f'<td>{dot}{e["termination_reason"]}</td>'
            f'</tr>\n'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation Report — {checkpoint}</title>
<style>
{font_css}

:root {{
  --purple:       rgb(68, 0, 153);
  --purple-light: rgb(108, 40, 213);
  --purple-dim:   rgba(68, 0, 153, 0.12);
  --purple-rule:  rgba(68, 0, 153, 0.25);
  --ink:          #0a0a14;
  --ink-mid:      #3a3550;
  --ink-soft:     #6b6585;
  --surface:      #ffffff;
  --surface-off:  #f5f3fb;
  --border:       #e0dcea;
  --green:  #22c55e;
  --amber:  #f59e0b;
  --red:    #ef4444;
  --grey:   #94a3b8;
}}

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  font-family: 'Barlow Condensed', 'Arial Narrow', Arial, sans-serif;
  font-size: 15px;
  font-weight: 400;
  color: var(--ink);
  background: #ece9f5;
  padding: 2rem;
  font-variant-numeric: tabular-nums;
}}

.page {{
  max-width: 1140px;
  margin: 0 auto;
  background: var(--surface);
  border: 1px solid var(--border);
}}

/* ── Header ── */
.report-header {{
  background: var(--purple);
  padding: 1.75rem 2.5rem 1.5rem;
  color: #fff;
}}
.report-header .eyebrow {{
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: rgba(255,255,255,.55);
  margin-bottom: .4rem;
}}
.report-header h1 {{
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: .02em;
  line-height: 1.1;
  word-break: break-all;
}}
.report-header .meta {{
  display: flex;
  flex-wrap: wrap;
  gap: .25rem 2rem;
  margin-top: .75rem;
  font-size: .8rem;
  font-weight: 400;
  color: rgba(255,255,255,.65);
  letter-spacing: .03em;
}}
.report-header .meta strong {{
  color: rgba(255,255,255,.9);
  font-weight: 600;
}}

/* ── Section headings ── */
.section {{
  padding: 2rem 2.5rem;
  border-bottom: 1px solid var(--border);
}}
.section:last-child {{ border-bottom: none; }}

h2 {{
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--purple);
  margin-bottom: 1.25rem;
  display: flex;
  align-items: center;
  gap: .6rem;
}}
h2::before {{
  content: '';
  display: inline-block;
  width: 3px;
  height: 1em;
  background: var(--purple);
}}

/* ── Summary cards ── */
.cards {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(148px, 1fr));
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
}}
.card {{
  background: var(--surface);
  padding: 1.1rem 1.25rem;
}}
.card:first-child {{
  background: var(--purple-dim);
  border-left: 3px solid var(--purple);
  padding-left: calc(1.25rem - 3px);
}}
.card .label {{
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--ink-soft);
  margin-bottom: .35rem;
}}
.card .value {{
  font-size: 2rem;
  font-weight: 700;
  color: var(--purple);
  line-height: 1;
  letter-spacing: -.01em;
}}
.card:first-child .value {{ color: var(--purple); }}
.card .sub {{
  font-size: .75rem;
  color: var(--ink-soft);
  margin-top: .3rem;
  font-weight: 400;
}}

/* ── Charts ── */
.charts-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
}}
.chart-wide {{ grid-column: 1 / -1; }}
.chart-box {{
  background: var(--surface-off);
  padding: 1.25rem 1.5rem;
}}
.chart-box h3 {{
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--ink-soft);
  margin-bottom: .9rem;
}}
.chart-box canvas {{ max-height: 220px; }}

/* ── Legend ── */
.legend {{
  display: flex;
  gap: 1.5rem;
  font-size: .78rem;
  font-weight: 600;
  color: var(--ink-mid);
  margin-top: 1rem;
  letter-spacing: .04em;
  text-transform: uppercase;
}}
.legend span {{ display: flex; align-items: center; gap: .4rem; }}
.dot {{
  display: inline-block;
  width: 8px;
  height: 8px;
  flex-shrink: 0;
  vertical-align: middle;
  margin-right: 3px;
}}

/* ── Table ── */
.table-wrap {{ overflow-x: auto; }}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: .85rem;
}}
thead tr {{
  background: var(--purple);
  color: #fff;
}}
th {{
  padding: .55rem .75rem;
  font-size: .65rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  text-align: left;
  white-space: nowrap;
}}
td {{
  padding: .45rem .75rem;
  border-bottom: 1px solid var(--border);
  color: var(--ink-mid);
}}
td.num {{ font-weight: 600; color: var(--ink); }}
tr:last-child td {{ border-bottom: none; }}
tr:hover td {{ background: var(--surface-off); }}

/* ── Footer ── */
.report-footer {{
  padding: .85rem 2.5rem;
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--ink-soft);
  background: var(--surface-off);
  border-top: 1px solid var(--border);
}}

/* ── Print ── */
@media print {{
  body {{ background: #fff; padding: 0; }}
  .page {{ border: none; max-width: 100%; }}
  .charts-grid {{ grid-template-columns: 1fr 1fr; }}
  .section {{ padding: 1.25rem 1.5rem; }}
  .chart-box canvas {{ max-height: 180px; }}
  tr {{ page-break-inside: avoid; }}
}}
</style>
</head>
<body>
<div class="page">

  <div class="report-header">
    <div class="eyebrow">TORCS · PPO · Evaluation Report</div>
    <h1>{checkpoint}</h1>
    <div class="meta">
      <span><strong>Laps</strong> {laps_completed} / {target_laps}</span>
      <span><strong>Attempts</strong> {total_attempts}</span>
      <span><strong>Seed</strong> {seed}</span>
      <span><strong>Generated</strong> {timestamp} UTC</span>
    </div>
  </div>

  <div class="section">
    <h2>Summary</h2>
    <div class="cards">
      <div class="card">
        <div class="label">Completion rate</div>
        <div class="value">{completion_pct}</div>
        <div class="sub">{laps_completed} laps in {total_attempts} attempts</div>
      </div>
      <div class="card">
        <div class="label">Best lap</div>
        <div class="value">{best_lap}<span style="font-size:1rem;font-weight:400"> s</span></div>
        <div class="sub">{mean_lap}</div>
      </div>
      <div class="card">
        <div class="label">Mean speed</div>
        <div class="value">{mean_spd}<span style="font-size:1rem;font-weight:400"> km/h</span></div>
        <div class="sub">peak {best_spd} km/h</div>
      </div>
      <div class="card">
        <div class="label">Min speed</div>
        <div class="value">{min_spd}<span style="font-size:1rem;font-weight:400"> km/h</span></div>
        <div class="sub">best cornering</div>
      </div>
      <div class="card">
        <div class="label">Mean distance</div>
        <div class="value">{mean_dist}<span style="font-size:1rem;font-weight:400"> m</span></div>
      </div>
      <div class="card">
        <div class="label">Worst track pos</div>
        <div class="value">{max_tp}<span style="font-size:1rem;font-weight:400">%</span></div>
        <div class="sub">of track width</div>
      </div>
      <div class="card">
        <div class="label">Steer smoothness</div>
        <div class="value">{steer_sm}</div>
        <div class="sub">σ — lower is smoother</div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Performance Charts</h2>
    <div class="charts-grid">
      <div class="chart-box chart-wide">
        <h3>Distance per attempt (m) — coloured by result</h3>
        <canvas id="distChart"></canvas>
      </div>
      <div class="chart-box">
        <h3>Lap time per attempt (s) — completed only</h3>
        <canvas id="lapChart"></canvas>
      </div>
      <div class="chart-box">
        <h3>Mean speed per attempt (km/h)</h3>
        <canvas id="speedChart"></canvas>
      </div>
      <div class="chart-box">
        <h3>Max track position per attempt (%)</h3>
        <canvas id="tpChart"></canvas>
      </div>
      <div class="chart-box">
        <h3>Termination reasons</h3>
        <canvas id="reasonChart"></canvas>
      </div>
    </div>
    <div class="legend">
      <span><span class="dot" style="background:var(--green)"></span>Lap complete</span>
      <span><span class="dot" style="background:var(--amber)"></span>Timeout</span>
      <span><span class="dot" style="background:var(--red)"></span>Off-track</span>
      <span><span class="dot" style="background:var(--grey)"></span>Other</span>
    </div>
  </div>

  <div class="section">
    <h2>Attempt Breakdown</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Attempt</th>
            <th>Distance (m)</th>
            <th>Lap #</th>
            <th>Lap time (s)</th>
            <th>Mean spd</th>
            <th>Max spd</th>
            <th>Min spd</th>
            <th>Max track pos</th>
            <th>Steer σ</th>
            <th>Off-track</th>
            <th>Termination</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </div>

  <div class="report-footer">
    Generated by eval/evaluate.py &nbsp;·&nbsp; Chart.js 4.4.4 (offline)
  </div>

</div>

<script>{chartjs_src}</script>
<script>
const PURPLE = 'rgb(68,0,153)';
const NEUTRAL = '#7c6fa0';
const NEUTRAL_A = 'rgba(124,111,160,0.18)';
const LABELS   = {ep_labels};
const LAP_T    = {lap_times};
const LAP_COL  = {lap_colours};
const SPEEDS   = {mean_speeds};
const DISTS    = {distances};
const DIST_COL = {dist_colours};
const TP_DATA  = {track_pos};
const TP_COL   = {tp_colours};
const R_LABELS = {reason_labels};
const R_COUNTS = {reason_counts};
const R_COLS   = {reason_colours};

const base = (ylabel) => ({{
  responsive: true,
  maintainAspectRatio: true,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    y: {{ grid: {{ color: 'rgba(0,0,0,.06)' }},
         ticks: {{ font: {{ family: "'Barlow Condensed', Arial", size: 12 }} }},
         title: {{ display: true, text: ylabel,
                   font: {{ family: "'Barlow Condensed', Arial", size: 11, weight: '700' }},
                   color: '#6b6585' }} }},
    x: {{ grid: {{ display: false }},
         ticks: {{ font: {{ family: "'Barlow Condensed', Arial", size: 12 }} }} }},
  }},
}});

new Chart(document.getElementById('distChart'), {{
  type: 'bar',
  data: {{ labels: LABELS, datasets: [{{ data: DISTS, backgroundColor: DIST_COL, borderRadius: 0, borderSkipped: false }}] }},
  options: {{ ...base('metres') }},
}});
new Chart(document.getElementById('lapChart'), {{
  type: 'bar',
  data: {{ labels: LABELS, datasets: [{{ data: LAP_T, backgroundColor: LAP_COL, borderRadius: 0, borderSkipped: false }}] }},
  options: {{ ...base('seconds') }},
}});
new Chart(document.getElementById('speedChart'), {{
  type: 'line',
  data: {{ labels: LABELS, datasets: [{{
    data: SPEEDS, borderColor: NEUTRAL, backgroundColor: NEUTRAL_A,
    fill: true, tension: 0.3, pointRadius: 4, pointBackgroundColor: NEUTRAL,
  }}] }},
  options: {{ ...base('km/h') }},
}});
new Chart(document.getElementById('tpChart'), {{
  type: 'bar',
  data: {{ labels: LABELS, datasets: [{{ data: TP_DATA, backgroundColor: TP_COL, borderRadius: 0, borderSkipped: false }}] }},
  options: {{
    ...base('% of track width'),
    scales: {{ ...base('% of track width').scales, y: {{ ...base('% of track width').scales.y, max: 100 }} }},
  }},
}});
new Chart(document.getElementById('reasonChart'), {{
  type: 'doughnut',
  data: {{ labels: R_LABELS, datasets: [{{ data: R_COUNTS, backgroundColor: R_COLS, hoverOffset: 4, borderWidth: 0 }}] }},
  options: {{
    responsive: true, maintainAspectRatio: true,
    plugins: {{ legend: {{
      display: true, position: 'bottom',
      labels: {{ font: {{ family: "'Barlow Condensed', Arial", size: 13, weight: '600' }},
                 padding: 16, boxWidth: 10, boxHeight: 10 }},
    }} }},
  }},
}});
</script>
</body>
</html>"""
