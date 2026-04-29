"""
Live evaluation dashboard server.

Starts a Flask server in a background thread. evaluate.py pushes results
as they complete; the browser polls /data every 3 s and redraws live.

The server stays up after eval finishes until you press Ctrl+C.
"""

from __future__ import annotations

import base64
import threading
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
  font-style: normal; font-weight: 400;
  src: url('data:font/woff2;base64,{_b64("barlow-400.woff2")}') format('woff2');
}}
@font-face {{
  font-family: 'Barlow Condensed';
  font-style: normal; font-weight: 600;
  src: url('data:font/woff2;base64,{_b64("barlow-600.woff2")}') format('woff2');
}}
@font-face {{
  font-family: 'Barlow Condensed';
  font-style: normal; font-weight: 700;
  src: url('data:font/woff2;base64,{_b64("barlow-700.woff2")}') format('woff2');
}}"""


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Live Eval — {checkpoint}</title>
<style>
{font_css}

:root {{
  --purple:       rgb(68, 0, 153);
  --purple-light: rgb(108, 40, 213);
  --purple-dim:   rgba(68, 0, 153, 0.25);
  --surface:      #0c0818;
  --surface-card: #130f24;
  --surface-mid:  #1a1535;
  --border:       rgba(255,255,255,.07);
  --text:         #f0eef8;
  --text-mid:     #8c84b0;
  --text-soft:    #564e78;
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
  color: var(--text);
  background: var(--surface);
  padding: 0;
  font-variant-numeric: tabular-nums;
}}

/* ── Top bar ── */
.topbar {{
  background: var(--purple);
  padding: .7rem 2rem;
  display: flex;
  align-items: baseline;
  gap: 1.5rem;
}}
.topbar .label {{
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: rgba(255,255,255,.55);
}}
.topbar .name {{
  font-size: 1rem;
  font-weight: 700;
  color: #fff;
  letter-spacing: .02em;
}}
.topbar .ep-count {{
  margin-left: auto;
  font-size: .8rem;
  font-weight: 600;
  color: rgba(255,255,255,.7);
  letter-spacing: .06em;
  text-transform: uppercase;
}}

/* ── Progress ── */
.prog-wrap {{
  height: 4px;
  background: var(--surface-mid);
}}
.prog-bar {{
  height: 100%;
  background: var(--purple-light);
  width: 0%;
  transition: width .4s ease;
  box-shadow: 0 0 12px rgba(108,40,213,.6);
}}

/* ── Body ── */
.body {{ padding: 1.5rem 2rem; max-width: 1000px; margin: 0 auto; }}

/* ── Status ── */
.status {{
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--text-soft);
  margin-bottom: 1.25rem;
}}
.status .complete {{ color: var(--green); }}

/* ── Last episode card ── */
.last-card {{
  display: none;
  background: var(--surface-card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--purple);
  padding: 1rem 1.25rem;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
  gap: .4rem 2rem;
}}
.kv .k {{
  font-size: .65rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--text-soft);
  margin-bottom: .15rem;
}}
.kv .v {{
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text);
}}

/* ── Charts ── */
.charts-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
}}
.chart-box {{
  background: var(--surface-card);
  padding: 1.1rem 1.25rem;
}}
.chart-box h3 {{
  font-size: .65rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--text-soft);
  margin-bottom: .75rem;
}}
canvas {{ max-height: 220px; }}

.dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%;
        vertical-align: middle; margin-right: 3px; }}
</style>
</head>
<body>

<div class="topbar">
  <span class="label">Live Eval</span>
  <span class="name">{checkpoint}</span>
  <span class="ep-count" id="epCount">0 / {total}</span>
</div>
<div class="prog-wrap"><div class="prog-bar" id="bar"></div></div>

<div class="body">
  <div class="status" id="status">Waiting for first episode…</div>

  <div class="last-card" id="last">
    <div class="kv"><div class="k">Episode</div><div class="v" id="lEp">—</div></div>
    <div class="kv"><div class="k">Distance</div><div class="v" id="lDist">—</div></div>
    <div class="kv"><div class="k">Lap time</div><div class="v" id="lLap">—</div></div>
    <div class="kv"><div class="k">Mean speed</div><div class="v" id="lSpd">—</div></div>
    <div class="kv"><div class="k">Max track pos</div><div class="v" id="lTp">—</div></div>
    <div class="kv"><div class="k">Termination</div><div class="v" id="lReason">—</div></div>
  </div>

  <div class="charts-row">
    <div class="chart-box"><h3>Distance raced (m)</h3><canvas id="distChart"></canvas></div>
    <div class="chart-box"><h3>Mean speed (km/h)</h3><canvas id="speedChart"></canvas></div>
  </div>
</div>

<script>{chartjs_src}</script>
<script>
const TOTAL  = {total};
const PURPLE = 'rgb(68,0,153)';
const PURPLE_L = 'rgb(108,40,213)';
const PURPLE_A = 'rgba(68,0,153,0.3)';
const RC = {{ lap_complete:'#22c55e', timeout:'#f59e0b', off_track:'#ef4444' }};
const col = r => RC[r] || '#94a3b8';

const chartOpts = (ylabel) => ({{
  responsive: true, maintainAspectRatio: true,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    y: {{
      grid: {{ color: 'rgba(255,255,255,.04)' }},
      ticks: {{ color: '#564e78', font: {{ family: "'Barlow Condensed',Arial", size: 12 }} }},
      title: {{ display: true, text: ylabel, color: '#564e78',
                font: {{ family: "'Barlow Condensed',Arial", size: 11, weight: '700' }} }},
    }},
    x: {{
      grid: {{ display: false }},
      ticks: {{ color: '#564e78', font: {{ family: "'Barlow Condensed',Arial", size: 12 }} }},
    }},
  }},
}});

const distChart = new Chart(document.getElementById('distChart'), {{
  type: 'bar',
  data: {{ labels: [], datasets: [{{ data: [], backgroundColor: [], borderRadius: 0, borderSkipped: false }}] }},
  options: chartOpts('metres'),
}});
const speedChart = new Chart(document.getElementById('speedChart'), {{
  type: 'line',
  data: {{ labels: [], datasets: [{{
    data: [], borderColor: PURPLE_L, backgroundColor: PURPLE_A,
    fill: true, tension: 0.3, pointRadius: 4, pointBackgroundColor: PURPLE_L,
  }}] }},
  options: chartOpts('km/h'),
}});

async function poll() {{
  try {{
    const d = await (await fetch('/data')).json();
    const eps = d.results, n = eps.length;

    document.getElementById('bar').style.width = (n / TOTAL * 100) + '%';
    document.getElementById('epCount').textContent = n + ' / ' + TOTAL;

    const st = document.getElementById('status');
    if (d.complete) st.innerHTML = '<span class="complete">RUN COMPLETE</span> — ' + n + ' episodes';
    else st.textContent = n + ' / ' + TOTAL + ' EPISODES COMPLETED';

    if (n > 0) {{
      const last = eps[n-1];
      document.getElementById('last').style.display = 'flex';
      document.getElementById('lEp').textContent   = n;
      document.getElementById('lDist').textContent = last.dist_raced_m.toFixed(0) + ' m';
      document.getElementById('lLap').textContent  = last.lap_time ? last.lap_time.toFixed(2) + ' s' : '—';
      document.getElementById('lSpd').textContent  = last.mean_speed_kmh.toFixed(1) + ' km/h';
      document.getElementById('lTp').textContent   = (last.max_abs_track_pos * 100).toFixed(1) + '%';
      document.getElementById('lReason').innerHTML =
        '<span class="dot" style="background:' + col(last.termination_reason) + '"></span>' +
        last.termination_reason;

      const labels = eps.map((_, i) => 'Ep ' + (i+1));
      distChart.data.labels = labels;
      distChart.data.datasets[0].data            = eps.map(e => e.dist_raced_m);
      distChart.data.datasets[0].backgroundColor = eps.map(e => col(e.termination_reason));
      distChart.update('none');
      speedChart.data.labels = labels;
      speedChart.data.datasets[0].data = eps.map(e => e.mean_speed_kmh);
      speedChart.update('none');
    }}

    setTimeout(poll, d.complete ? 10000 : 3000);
  }} catch(e) {{ setTimeout(poll, 5000); }}
}}
poll();
</script>
</body>
</html>"""


class EvalServer:
    def __init__(self, total_episodes: int, checkpoint: str, port: int = 5001):
        self._results: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._complete = False
        self._total = total_episodes
        self._checkpoint = Path(checkpoint).name
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
            font_css=_font_face_css(),
            chartjs_src=_chartjs(),
            checkpoint=self._checkpoint,
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
        with self._lock:
            self._results.append(result.to_dict())

    def set_complete(self) -> None:
        with self._lock:
            self._complete = True

    def start(self) -> None:
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        self._thread_error: list[Exception] = []

        def _run() -> None:
            try:
                print(f"[server] Flask binding on 0.0.0.0:{self._port} …", flush=True)
                self._app.run(host="0.0.0.0", port=self._port, use_reloader=False, debug=False)
            except Exception as exc:
                self._thread_error.append(exc)
                print(f"[server] Flask thread crashed: {exc}", flush=True)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        self._thread = t

    @property
    def url(self) -> str:
        return f"http://localhost:{self._port}"
