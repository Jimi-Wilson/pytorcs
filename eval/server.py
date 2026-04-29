"""
Live evaluation dashboard server.

Starts a Flask server in a background thread. evaluate.py pushes results
as they complete and step telemetry as the car drives; the browser polls:
  /live  every 1 s  — real-time speed/steering/track-pos charts
  /data  every 3 s  — per-episode results and progress

The server stays up after eval finishes until you press Ctrl+C.
"""

from __future__ import annotations

import base64
import threading
import time
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
  --purple:      rgb(68, 0, 153);
  --purple-l:    rgb(108, 40, 213);
  --purple-dim:  rgba(68, 0, 153, 0.10);
  --neutral:     #7c6fa0;
  --neutral-a:   rgba(124, 111, 160, 0.18);
  --surface:     #ffffff;
  --surface-off: #f5f3fb;
  --bg:          #ece9f5;
  --border:      #e0dcea;
  --ink:         #0a0a14;
  --ink-mid:     #3a3550;
  --ink-soft:    #6b6585;
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
  background: var(--bg);
  font-variant-numeric: tabular-nums;
}}

/* ── Topbar ── */
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
  color: rgba(255,255,255,.5);
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
.prog-wrap {{ height: 4px; background: var(--border); }}
.prog-bar {{
  height: 100%;
  background: var(--purple-l);
  width: 0%;
  transition: width .4s ease;
  box-shadow: 0 0 10px rgba(108,40,213,.4);
}}

/* ── Body ── */
.body {{
  max-width: 1100px;
  margin: 0 auto;
  padding: 1.5rem 2rem 2rem;
}}

/* ── Status ── */
.status {{
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--ink-soft);
  margin-bottom: 1.25rem;
}}
.status .complete {{ color: var(--green); }}

/* ── Section ── */
.section {{
  background: var(--surface);
  border: 1px solid var(--border);
  margin-bottom: 1rem;
}}
.section-head {{
  padding: .7rem 1.25rem;
  border-bottom: 1px solid var(--border);
  font-size: .65rem;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--ink-soft);
  display: flex;
  align-items: center;
  gap: .75rem;
}}

/* ── Live indicator ── */
.live-dot {{
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--green);
  animation: pulse 1.4s ease infinite;
  flex-shrink: 0;
}}
@keyframes pulse {{
  0%,100% {{ opacity:1; transform:scale(1); }}
  50%      {{ opacity:.5; transform:scale(1.3); }}
}}
.attempt-pill {{
  background: var(--purple-dim);
  color: var(--purple);
  border: 1px solid rgba(68,0,153,.2);
  border-radius: 2px;
  padding: .1rem .55rem;
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
}}
.step-counter {{
  margin-left: auto;
  color: var(--ink-soft);
  font-weight: 400;
}}

/* ── Chart grids ── */
.charts-3 {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 1px;
  background: var(--border);
  border-top: 1px solid var(--border);
}}
.charts-2 {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
  background: var(--border);
  border-top: 1px solid var(--border);
}}
.chart-box {{
  background: var(--surface-off);
  padding: 1rem 1.25rem;
}}
.chart-box h3 {{
  font-size: .63rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--ink-soft);
  margin-bottom: .7rem;
}}
.charts-3 canvas {{ max-height: 160px; }}
.charts-2 canvas {{ max-height: 220px; }}

/* ── Last-episode card ── */
.last-card {{
  display: none;
  padding: .9rem 1.25rem;
  border-top: 1px solid var(--border);
  flex-wrap: wrap;
  gap: .4rem 2rem;
}}
.kv .k {{
  font-size: .62rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--ink-soft);
  margin-bottom: .12rem;
}}
.kv .v {{
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--ink);
}}

/* ── Legend ── */
.legend {{
  padding: .7rem 1.25rem;
  border-top: 1px solid var(--border);
  display: flex;
  gap: 1.5rem;
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .04em;
  text-transform: uppercase;
  color: var(--ink-mid);
}}
.legend span {{ display: flex; align-items: center; gap: .4rem; }}
.dot {{
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}}
.lap-table-wrap {{
  padding: .7rem 1.25rem 1rem;
  border-top: 1px solid var(--border);
}}
.lap-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: .82rem;
}}
.lap-table th, .lap-table td {{
  padding: .25rem .4rem;
  border-bottom: 1px solid rgba(0,0,0,.05);
  text-align: right;
}}
.lap-table th:first-child, .lap-table td:first-child {{ text-align: left; }}
</style>
</head>
<body>

<div class="topbar">
  <span class="label">Live Eval</span>
  <span class="name">{checkpoint}</span>
  <span class="ep-count" id="lapCount">0 / {target_laps} laps</span>
  <span class="ep-count" style="color:rgba(255,255,255,.5)" id="attemptCount">0 attempts</span>
</div>
<div class="prog-wrap"><div class="prog-bar" id="bar"></div></div>

<div class="body">
  <div class="status" id="status">Waiting for first attempt…</div>

  <!-- Live telemetry -->
  <div class="section" id="liveSection" style="display:none">
    <div class="section-head">
      <span class="live-dot"></span>
      Live
      <span class="attempt-pill" id="attemptPill">ATTEMPT 1</span>
      <span class="step-counter" id="stepCounter">step 0</span>
    </div>
    <div class="last-card" id="liveLapStats" style="display:flex;border-top:none;padding-top:.2rem">
      <div class="kv"><div class="k">Run laps</div><div class="v" id="liveRunLaps">0 / {target_laps}</div></div>
      <div class="kv"><div class="k">Attempt laps</div><div class="v" id="liveAttemptLaps">0</div></div>
      <div class="kv"><div class="k">Current lap time</div><div class="v" id="liveCurLap">—</div></div>
      <div class="kv"><div class="k">Last lap</div><div class="v" id="liveLastLap">—</div></div>
      <div class="kv"><div class="k">Best lap (run)</div><div class="v" id="liveBestLap">—</div></div>
      <div class="kv"><div class="k">Lap pace</div><div class="v" id="liveLapPace">—</div></div>
      <div class="kv"><div class="k">ETA to target</div><div class="v" id="liveEta">—</div></div>
      <div class="kv"><div class="k">Last lap event</div><div class="v" id="liveLastEvent">—</div></div>
      <div class="kv"><div class="k">Signal health</div><div class="v" id="liveSignalHealth">—</div></div>
    </div>
    <div class="charts-3">
      <div class="chart-box"><h3>Speed (km/h)</h3><canvas id="lSpeedChart"></canvas></div>
      <div class="chart-box"><h3>Steering</h3><canvas id="lSteerChart"></canvas></div>
      <div class="chart-box"><h3>Track position</h3><canvas id="lTpChart"></canvas></div>
    </div>
  </div>

  <!-- Per-attempt results -->
  <div class="section" id="resultsSection" style="display:none">
    <div class="section-head">Attempt Results</div>
    <div class="last-card" id="last">
      <div class="kv"><div class="k">Attempt</div><div class="v" id="lEp">—</div></div>
      <div class="kv"><div class="k">Distance</div><div class="v" id="lDist">—</div></div>
      <div class="kv"><div class="k">Lap time</div><div class="v" id="lLap">—</div></div>
      <div class="kv"><div class="k">Mean speed</div><div class="v" id="lSpd">—</div></div>
      <div class="kv"><div class="k">Max track pos</div><div class="v" id="lTp">—</div></div>
      <div class="kv"><div class="k">Termination</div><div class="v" id="lReason">—</div></div>
    </div>
    <div class="charts-2">
      <div class="chart-box"><h3>Distance per episode (m) — coloured by result</h3><canvas id="distChart"></canvas></div>
      <div class="chart-box"><h3>Mean speed per episode (km/h)</h3><canvas id="speedChart"></canvas></div>
    </div>
    <div class="legend">
      <span><span class="dot" style="background:var(--green)"></span>Lap complete</span>
      <span><span class="dot" style="background:var(--amber)"></span>Timeout</span>
      <span><span class="dot" style="background:var(--red)"></span>Off-track</span>
      <span><span class="dot" style="background:var(--grey)"></span>Other</span>
    </div>
    <div class="lap-table-wrap">
      <table class="lap-table">
        <thead><tr><th>Lap #</th><th>Lap time</th><th>Delta to best</th></tr></thead>
        <tbody id="liveLapRows"></tbody>
      </table>
    </div>
  </div>
</div>

<script>{chartjs_src}</script>
<script>
const TARGET_LAPS = {target_laps};
const NEUTRAL   = '#7c6fa0';
const NEUTRAL_A = 'rgba(124,111,160,0.2)';
const PURPLE_L  = 'rgb(108,40,213)';
const PURPLE_A  = 'rgba(108,40,213,0.15)';
const AMBER     = 'rgba(245,158,11,0.85)';
const AMBER_A   = 'rgba(245,158,11,0.15)';
const RC = {{ lap_complete:'#22c55e', timeout:'#f59e0b', off_track:'#ef4444' }};
const col = r => RC[r] || '#94a3b8';

const baseOpts = (ylabel, extra) => ({{
  responsive: true, maintainAspectRatio: true,
  animation: false,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    y: {{
      grid: {{ color: 'rgba(0,0,0,.05)' }},
      ticks: {{ color: '#6b6585', font: {{ family:"'Barlow Condensed',Arial", size:11 }} }},
      title: {{ display: true, text: ylabel, color:'#6b6585',
                font: {{ family:"'Barlow Condensed',Arial", size:10, weight:'700' }} }},
      ...extra,
    }},
    x: {{
      grid: {{ display: false }},
      ticks: {{ color: '#6b6585', font: {{ family:"'Barlow Condensed',Arial", size:11 }} }},
    }},
  }},
}});

/* ── Live charts ── */
const MAX_LIVE = 300;
const lLabels = [];
const mkLive = (id, color, colorA, ylabel, yExtra) => new Chart(document.getElementById(id), {{
  type: 'line',
  data: {{ labels: lLabels, datasets: [{{
    data: [], borderColor: color, backgroundColor: colorA,
    fill: true, tension: 0.2, pointRadius: 0, borderWidth: 1.5,
  }}] }},
  options: baseOpts(ylabel, yExtra || {{}}),
}});

const lSpeed = mkLive('lSpeedChart', NEUTRAL,   NEUTRAL_A, 'km/h');
const lSteer = mkLive('lSteerChart', PURPLE_L,  PURPLE_A,  'radians', {{ min:-1, max:1 }});
const lTp    = mkLive('lTpChart',    AMBER,     AMBER_A,   'track pos', {{ min:-1.2, max:1.2 }});

let _prevEp = -1, _prevStep = -1;
let _lapRowsRendered = 0;

function fmtHms(sec) {{
  if (!sec || sec <= 0) return '—';
  const s = Math.round(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${{h}}:${{String(m).padStart(2,'0')}}:${{String(r).padStart(2,'0')}}`;
  return `${{m}}:${{String(r).padStart(2,'0')}}`;
}}

function pushLive(step, speed, steer, tp) {{
  lLabels.push(step);
  lSpeed.data.datasets[0].data.push(speed);
  lSteer.data.datasets[0].data.push(steer);
  lTp.data.datasets[0].data.push(tp);
  if (lLabels.length > MAX_LIVE) {{
    lLabels.shift();
    lSpeed.data.datasets[0].data.shift();
    lSteer.data.datasets[0].data.shift();
    lTp.data.datasets[0].data.shift();
  }}
  lSpeed.update('none'); lSteer.update('none'); lTp.update('none');
}}

/* ── Episode charts ── */
const distChart = new Chart(document.getElementById('distChart'), {{
  type: 'bar',
  data: {{ labels: [], datasets: [{{ data: [], backgroundColor: [], borderRadius: 0, borderSkipped: false }}] }},
  options: {{ ...baseOpts('metres'), animation: false }},
}});
const speedChart = new Chart(document.getElementById('speedChart'), {{
  type: 'line',
  data: {{ labels: [], datasets: [{{
    data: [], borderColor: NEUTRAL, backgroundColor: NEUTRAL_A,
    fill: true, tension: 0.3, pointRadius: 4, pointBackgroundColor: NEUTRAL,
  }}] }},
  options: {{ ...baseOpts('km/h'), animation: false }},
}});

/* ── Live poll (1 s) ── */
async function pollLive() {{
  try {{
    const d = await (await fetch('/live')).json();
    if (d.active) {{
      document.getElementById('liveSection').style.display = 'block';
      document.getElementById('attemptPill').textContent = 'ATTEMPT ' + d.episode;
      const liveLaps = (d.laps_in_attempt || 0);
      const runLaps = (d.run_laps_completed || 0);
      const liveLastLap = (d.last_lap_time && d.last_lap_time > 0) ? ('  last ' + d.last_lap_time.toFixed(2) + 's') : '';
      document.getElementById('stepCounter').textContent = 'step ' + d.step + '  |  laps ' + liveLaps + liveLastLap;
      document.getElementById('liveRunLaps').textContent = runLaps + ' / ' + TARGET_LAPS;
      document.getElementById('liveAttemptLaps').textContent = String(liveLaps);
      document.getElementById('liveCurLap').textContent =
        (d.cur_lap_time && d.cur_lap_time > 0) ? (d.cur_lap_time.toFixed(2) + ' s') : '—';
      document.getElementById('liveLastLap').textContent =
        (d.last_lap_time && d.last_lap_time > 0) ? (d.last_lap_time.toFixed(2) + ' s') : '—';
      document.getElementById('liveBestLap').textContent =
        (d.best_lap_time && d.best_lap_time > 0) ? (d.best_lap_time.toFixed(2) + ' s') : '—';
      document.getElementById('liveSignalHealth').textContent =
        (d.raw_lap_signals_present ? 'OK' : 'MISSING');
      document.getElementById('liveLastEvent').textContent =
        d.last_lap_event_ts ? new Date(d.last_lap_event_ts * 1000).toLocaleTimeString() : '—';
      if (d.run_elapsed_s && d.run_laps_completed > 0) {{
        const lpm = (d.run_laps_completed * 60.0) / d.run_elapsed_s;
        document.getElementById('liveLapPace').textContent = lpm.toFixed(2) + ' laps/min';
        const rem = Math.max(0, TARGET_LAPS - d.run_laps_completed);
        const etaSec = lpm > 1e-6 ? (rem / lpm) * 60.0 : 0;
        document.getElementById('liveEta').textContent = fmtHms(etaSec);
      }} else {{
        document.getElementById('liveLapPace').textContent = '—';
        document.getElementById('liveEta').textContent = '—';
      }}

      if (d.episode !== _prevEp || d.step <= _prevStep) {{
        // New episode — clear buffers
        lLabels.length = 0;
        lSpeed.data.datasets[0].data.length = 0;
        lSteer.data.datasets[0].data.length = 0;
        lTp.data.datasets[0].data.length = 0;
        _lapRowsRendered = 0;
        document.getElementById('liveLapRows').innerHTML = '';
      }}
      _prevEp   = d.episode;
      _prevStep = d.step;
      pushLive(d.step, d.speed, d.steer, d.track_pos);

      const lapRows = d.run_lap_times || [];
      const body = document.getElementById('liveLapRows');
      while (_lapRowsRendered < lapRows.length) {{
        const idx = _lapRowsRendered;
        const lt = lapRows[idx];
        const best = d.best_lap_time || lt;
        const delta = lt - best;
        const tr = document.createElement('tr');
        tr.innerHTML =
          `<td>${{idx + 1}}</td>` +
          `<td>${{lt.toFixed(2)}} s</td>` +
          `<td>${{delta <= 1e-6 ? 'best' : ('+' + delta.toFixed(2) + ' s')}}</td>`;
        body.appendChild(tr);
        _lapRowsRendered += 1;
      }}
    }}
  }} catch(e) {{ console.error('[eval] /live poll error:', e); }}
  setTimeout(pollLive, 1000);
}}

/* ── Data poll (3 s) ── */
async function pollData() {{
  try {{
    const d = await (await fetch('/data')).json();
    const eps = d.results, n = eps.length;
    const laps = d.laps_completed;
    const live = d.live || {{}};
    const attemptsShown = Math.max(n, live.episode || 0);

    document.getElementById('bar').style.width = (laps / TARGET_LAPS * 100) + '%';
    document.getElementById('lapCount').textContent = laps + ' / ' + TARGET_LAPS + ' laps';
    document.getElementById('attemptCount').textContent = attemptsShown + ' attempts';

    const st = document.getElementById('status');
    if (d.complete) {{
      st.innerHTML = '<span class="complete">RUN COMPLETE</span> — ' + laps + ' laps in ' + n + ' attempts';
      document.getElementById('liveSection').style.display = 'none';
    }} else {{
      const liveStep = live.step || 0;
      st.textContent = laps + ' / ' + TARGET_LAPS + ' LAPS COMPLETED  (' + attemptsShown + ' attempts, step ' + liveStep + ')';
    }}

    if (n > 0) {{
      document.getElementById('resultsSection').style.display = 'block';
      const last = eps[n-1];
      document.getElementById('last').style.display = 'flex';
      document.getElementById('lEp').textContent   = n;
      document.getElementById('lDist').textContent = last.dist_raced_m.toFixed(0) + ' m';
      document.getElementById('lLap').textContent  = last.lap_time ? last.lap_time.toFixed(2) + ' s' : '—';
      document.getElementById('lSpd').textContent  = last.mean_speed_kmh.toFixed(1) + ' km/h';
      document.getElementById('lTp').textContent   = (last.max_abs_track_pos * 100).toFixed(1) + '%';
      document.getElementById('lReason').innerHTML =
        '<span class="dot" style="background:' + col(last.termination_reason) + ';margin-right:4px"></span>' +
        last.termination_reason;

      const labels = eps.map((_, i) => 'A' + (i+1));
      distChart.data.labels  = labels;
      distChart.data.datasets[0].data            = eps.map(e => e.dist_raced_m);
      distChart.data.datasets[0].backgroundColor = eps.map(e => col(e.termination_reason));
      distChart.update('none');
      speedChart.data.labels = labels;
      speedChart.data.datasets[0].data = eps.map(e => e.mean_speed_kmh);
      speedChart.update('none');
    }}

    setTimeout(pollData, d.complete ? 10000 : 3000);
  }} catch(e) {{ console.error('[eval] /data poll error:', e); setTimeout(pollData, 5000); }}
}}

pollLive();
pollData();
</script>
</body>
</html>"""


class EvalServer:
    def __init__(self, target_laps: int, max_attempts: int, checkpoint: str, port: int = 5001):
        self._results: list[dict[str, Any]] = []
        self._live: dict[str, Any] = {"active": False}
        self._lock = threading.Lock()
        self._complete = False
        self._target_laps = target_laps
        self._max_attempts = max_attempts
        self._laps_completed = 0
        self._best_lap_time: float | None = None
        self._run_lap_times: list[float] = []
        self._run_start_ts = time.time()
        self._last_lap_event_ts: float | None = None
        self._attempt_count = 1
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
            target_laps=self._target_laps,
        )

        @app.route("/")
        def index():
            return dashboard

        @app.route("/data")
        def data():
            with self._lock:
                return jsonify({
                    "results":        list(self._results),
                    "complete":       self._complete,
                    "laps_completed": self._laps_completed,
                    "target_laps":    self._target_laps,
                    "live":           dict(self._live),
                })

        @app.route("/live")
        def live():
            with self._lock:
                return jsonify(dict(self._live))

        return app

    def push_result(self, result: Any) -> None:
        with self._lock:
            d = result.to_dict()
            self._results.append(d)
            all_laps = d.get("all_lap_times", []) or []
            for lt in all_laps:
                try:
                    lt_f = float(lt)
                except (TypeError, ValueError):
                    continue
                if lt_f > 0.0 and (self._best_lap_time is None or lt_f < self._best_lap_time):
                    self._best_lap_time = lt_f
            self._attempt_count = len(self._results) + 1
            self._live = {"active": False}

    def push_lap_event(self, lap_time_s: float | None = None) -> None:
        with self._lock:
            self._laps_completed += 1
            self._last_lap_event_ts = time.time()
            live = dict(self._live)
            live["laps_in_attempt"] = int(live.get("laps_in_attempt", 0)) + 1
            if lap_time_s is not None:
                lap_v = float(lap_time_s)
                self._run_lap_times.append(lap_v)
                live["last_lap_time"] = lap_v
                if self._best_lap_time is None or lap_v < self._best_lap_time:
                    self._best_lap_time = lap_v
            if self._best_lap_time is not None:
                live["best_lap_time"] = self._best_lap_time
            live["run_laps_completed"] = self._laps_completed
            live["run_lap_times"] = list(self._run_lap_times)
            live["run_elapsed_s"] = max(0.0, time.time() - self._run_start_ts)
            live["last_lap_event_ts"] = self._last_lap_event_ts
            self._live = live

    def push_step(self, tele: dict[str, Any]) -> None:
        with self._lock:
            prev_live = dict(self._live)
            self._live = {
                "active":    True,
                "episode":   self._attempt_count,
                "speed":     tele.get("speed", 0.0),
                "steer":     tele.get("steer", 0.0),
                "track_pos": tele.get("track_pos", 0.0),
                "dist":      tele.get("dist", 0.0),
                "step":      tele.get("step", 0),
                "laps_in_attempt": tele.get("laps_in_attempt", prev_live.get("laps_in_attempt", 0)),
                "cur_lap_time": tele.get("cur_lap_time", prev_live.get("cur_lap_time", 0.0)),
                "last_lap_time": tele.get("last_lap_time", prev_live.get("last_lap_time", 0.0)),
                "run_laps_completed": self._laps_completed,
                "best_lap_time": self._best_lap_time if self._best_lap_time is not None else 0.0,
                "run_lap_times": list(self._run_lap_times),
                "run_elapsed_s": max(0.0, time.time() - self._run_start_ts),
                "last_lap_event_ts": self._last_lap_event_ts,
                "raw_lap_signals_present": bool(
                    "cur_lap_time" in tele and "last_lap_time" in tele
                ),
            }

    def set_complete(self) -> None:
        with self._lock:
            self._complete = True
            self._live = {"active": False}

    def start(self) -> None:
        import logging
        from flask import cli as flask_cli
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        flask_cli.show_server_banner = lambda *args, **kwargs: None

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
