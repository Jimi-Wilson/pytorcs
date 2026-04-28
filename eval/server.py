"""
Live evaluation dashboard server.

Starts a Flask server in a background thread. evaluate.py pushes episode
results as they complete; the browser page polls /data every 3 seconds and
redraws the charts live.

The server stays up after eval finishes (showing a "Run complete" banner)
until you press Ctrl+C in the terminal.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


def _chartjs() -> str:
    vendor = Path(__file__).parent / "vendor" / "chart.min.js"
    return vendor.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Dashboard HTML (inlined Chart.js, polls /data every 3 s)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Live Eval — {checkpoint}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
          font-size: 14px; color: #1e293b; background: #0f172a; padding: 1.5rem; }}
  .page {{ max-width: 960px; margin: 0 auto; }}

  h1 {{ color: #f1f5f9; font-size: 1.1rem; font-weight: 600; margin-bottom: .25rem; }}
  .sub {{ color: #64748b; font-size: .82rem; margin-bottom: 1.5rem; }}

  /* Progress bar */
  .prog-wrap {{ background: #1e293b; border-radius: 99px; height: 10px; margin-bottom: 1.25rem; overflow: hidden; }}
  .prog-bar  {{ height: 100%; background: #3b82f6; border-radius: 99px;
                transition: width .4s ease; width: 0%; }}

  /* Status */
  .status {{ font-size: .85rem; color: #94a3b8; margin-bottom: 1.5rem; }}
  .status .complete {{ color: #22c55e; font-weight: 600; }}

  /* Last episode card */
  .last-card {{ background: #1e293b; border-radius: 6px; padding: 1rem 1.25rem;
                margin-bottom: 1.5rem; display: flex; flex-wrap: wrap; gap: .5rem 2rem; }}
  .last-card .kv {{ font-size: .83rem; }}
  .last-card .kv .k {{ color: #64748b; }}
  .last-card .kv .v {{ color: #f1f5f9; font-weight: 600; }}

  /* Charts */
  .charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  .chart-box {{ background: #1e293b; border-radius: 6px; padding: 1rem 1.25rem; }}
  .chart-box h3 {{ font-size: .75rem; text-transform: uppercase; letter-spacing: .05em;
                   color: #64748b; margin-bottom: .75rem; }}
  canvas {{ max-height: 220px; }}

  .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
          vertical-align: middle; margin-right: 3px; }}
</style>
</head>
<body>
<div class="page">
  <h1>Live Evaluation &nbsp;<span style="color:#64748b;font-weight:400">— {checkpoint}</span></h1>
  <div class="sub">Stage {stage} &nbsp;·&nbsp; {total} episodes &nbsp;·&nbsp; updates every 3 s</div>

  <div class="prog-wrap"><div class="prog-bar" id="bar"></div></div>
  <div class="status" id="status">Waiting for first episode…</div>

  <div class="last-card" id="last" style="display:none">
    <div class="kv"><div class="k">Last episode</div><div class="v" id="lEp">—</div></div>
    <div class="kv"><div class="k">Distance</div><div class="v" id="lDist">—</div></div>
    <div class="kv"><div class="k">Lap time</div><div class="v" id="lLap">—</div></div>
    <div class="kv"><div class="k">Mean speed</div><div class="v" id="lSpd">—</div></div>
    <div class="kv"><div class="k">Reward</div><div class="v" id="lRwd">—</div></div>
    <div class="kv"><div class="k">Termination</div><div class="v" id="lReason">—</div></div>
  </div>

  <div class="charts-row">
    <div class="chart-box">
      <h3>Distance raced (m)</h3>
      <canvas id="distChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Mean speed (km/h)</h3>
      <canvas id="speedChart"></canvas>
    </div>
  </div>
</div>

<script>{chartjs_src}</script>
<script>
const TOTAL = {total};
const REASON_COL = {{
  lap_complete: '#22c55e',
  timeout:      '#f59e0b',
  off_track:    '#ef4444',
}};
function col(r) {{ return REASON_COL[r] || '#94a3b8'; }}

const chartOpts = (ylabel) => ({{
  responsive: true, maintainAspectRatio: true,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{ y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }},
                   title: {{ display: true, text: ylabel, color: '#64748b' }} }},
             x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }} }}
}});

const distChart = new Chart(document.getElementById('distChart'), {{
  type: 'bar',
  data: {{ labels: [], datasets: [{{ data: [], backgroundColor: [], borderRadius: 3 }}] }},
  options: chartOpts('metres'),
}});

const speedChart = new Chart(document.getElementById('speedChart'), {{
  type: 'line',
  data: {{ labels: [], datasets: [{{
    data: [], borderColor: '#3b82f6',
    backgroundColor: 'rgba(59,130,246,.15)', fill: true, tension: 0.3, pointRadius: 4,
  }}] }},
  options: chartOpts('km/h'),
}});

function update(data) {{
  const eps = data.results;
  const n   = eps.length;
  if (n === 0) return;

  // Progress bar
  document.getElementById('bar').style.width = (n / TOTAL * 100) + '%';

  // Status line
  const statusEl = document.getElementById('status');
  if (data.complete) {{
    statusEl.innerHTML = '<span class="complete">Run complete</span> — ' + n + '/' + TOTAL + ' episodes finished.';
  }} else {{
    statusEl.textContent = n + ' / ' + TOTAL + ' episodes completed…';
  }}

  // Last episode card
  const last = eps[n - 1];
  document.getElementById('last').style.display = 'flex';
  document.getElementById('lEp').textContent    = n;
  document.getElementById('lDist').textContent  = last.dist_raced_m.toFixed(0) + ' m';
  document.getElementById('lLap').textContent   = last.lap_time ? last.lap_time.toFixed(2) + ' s' : '—';
  document.getElementById('lSpd').textContent   = last.mean_speed_kmh.toFixed(1) + ' km/h';
  document.getElementById('lRwd').textContent   = last.total_reward.toFixed(1);
  const dot = '<span class="dot" style="background:' + col(last.termination_reason) + '"></span>';
  document.getElementById('lReason').innerHTML  = dot + last.termination_reason;

  // Charts
  const labels = eps.map((_, i) => 'Ep ' + (i + 1));
  distChart.data.labels  = labels;
  distChart.data.datasets[0].data            = eps.map(e => e.dist_raced_m);
  distChart.data.datasets[0].backgroundColor = eps.map(e => col(e.termination_reason));
  distChart.update('none');

  speedChart.data.labels = labels;
  speedChart.data.datasets[0].data = eps.map(e => e.mean_speed_kmh);
  speedChart.update('none');
}}

async function poll() {{
  try {{
    const r = await fetch('/data');
    const d = await r.json();
    update(d);
    if (!d.complete) setTimeout(poll, 3000);
    else             setTimeout(poll, 10000); // slow-poll after done
  }} catch(e) {{
    setTimeout(poll, 5000);
  }}
}}
poll();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Server class
# ---------------------------------------------------------------------------

class EvalServer:
    def __init__(self, total_episodes: int, checkpoint: str, stage: int, port: int = 5001):
        self._results: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._complete = False
        self._total = total_episodes
        self._checkpoint = Path(checkpoint).name
        self._stage = stage
        self._port = port
        self._app = self._build_app()

    def _build_app(self):
        try:
            from flask import Flask, jsonify
        except ImportError as exc:
            raise RuntimeError(
                "Flask is required for the live server. "
                "Install it: pip install flask"
            ) from exc

        app = Flask(__name__)
        app.logger.disabled = True

        dashboard = _DASHBOARD_HTML.format(
            chartjs_src=_chartjs(),
            checkpoint=self._checkpoint,
            stage=self._stage,
            total=self._total,
        )

        @app.route("/")
        def index():
            return dashboard

        @app.route("/data")
        def data():
            with self._lock:
                return jsonify({
                    "results":  list(self._results),
                    "complete": self._complete,
                    "total":    self._total,
                })

        return app

    def push_result(self, result: Any) -> None:
        """Add one completed EpisodeResult (call from eval loop)."""
        with self._lock:
            self._results.append(result.to_dict())

    def set_complete(self) -> None:
        with self._lock:
            self._complete = True

    def start(self) -> None:
        """Start Flask in a daemon background thread."""
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        t = threading.Thread(
            target=self._app.run,
            kwargs={"port": self._port, "use_reloader": False, "debug": False},
            daemon=True,
        )
        t.start()

    @property
    def url(self) -> str:
        return f"http://localhost:{self._port}"
