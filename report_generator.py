"""
report_generator.py
===================
Generates a self-contained, interactive HTML report from benchmark results.
No external dependencies at runtime — all JS/CSS is inlined or from CDN.
"""

import json
from pathlib import Path
from dataclasses import asdict
from typing import Optional


def generate_report(summaries, all_records, out_dir: Path) -> Path:
    report_path = out_dir / "report.html"

    summaries_json = json.dumps([asdict(s) for s in summaries], indent=2)

    # Build per-frame time-series data (downsample to max 500 pts per model)
    series_data = {}
    for label, records in all_records.items():
        step = max(1, len(records) // 500)
        series_data[label] = {
            "frames":       [r.frame_idx for r in records[::step]],
            "inference_ms": [r.inference_ms for r in records[::step]],
            "total_ms":     [r.total_ms for r in records[::step]],
            "fps":          [round(1000 / max(r.total_ms, 0.001), 1) for r in records[::step]],
            "detections":   [r.detections for r in records[::step]],
            "gpu_mem":      [r.gpu_mem_mb for r in records[::step]],
        }
    series_json = json.dumps(series_data, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #ffffff;
    --surface: #f6f8fa;
    --surface2: #eef1f5;
    --border: #d0d7de;
    --text: #1f2328;
    --muted: #656d76;
    --blue: #0969da;
    --green: #1a7f37;
    --orange: #9a6700;
    --purple: #8250df;
    --red: #cf222e;
    --cyan: #1b7c83;
    --accent1: #0969da;
    --accent2: #1a7f37;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 14px; line-height: 1.6;
  }}
  header {{
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 1.5rem 2rem; display: flex; align-items: center; gap: 1rem;
  }}
  header h1 {{ font-size: 1.25rem; font-weight: 600; letter-spacing: -0.01em; }}
  header .badge {{
    background: var(--blue); color: #fff; border-radius: 999px;
    font-size: 11px; font-weight: 600; padding: 2px 10px; letter-spacing: 0.04em;
  }}
  .timestamp {{ color: var(--muted); font-size: 12px; margin-left: auto; }}

  main {{ max-width: 1280px; margin: 0 auto; padding: 2rem; }}

  /* Tabs */
  .tabs {{ display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 2rem; }}
  .tab {{
    padding: 0.6rem 1.2rem; cursor: pointer; font-size: 13px; font-weight: 500;
    color: var(--muted); border-bottom: 2px solid transparent;
    transition: color 0.15s, border-color 0.15s;
  }}
  .tab.active {{ color: var(--text); border-bottom-color: var(--blue); }}
  .tab:hover:not(.active) {{ color: var(--text); }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}

  /* Metric cards */
  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem 1.25rem;
  }}
  .card-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem; }}
  .card-value {{ font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em; line-height: 1; }}
  .card-sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .card.win .card-value {{ color: var(--green); }}
  .card.neutral .card-value {{ color: var(--blue); }}

  /* Comparison table */
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  thead tr {{ background: var(--surface2); }}
  th {{ padding: 0.65rem 1rem; text-align: left; font-weight: 600; color: var(--muted);
        font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
        border-bottom: 1px solid var(--border); white-space: nowrap; }}
  td {{ padding: 0.65rem 1rem; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface2); }}
  .best {{ color: var(--green); font-weight: 600; }}
  .model-name {{ font-weight: 600; color: var(--text); }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}

  /* Charts */
  .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
  @media (max-width: 800px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
  .chart-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.25rem;
  }}
  .chart-title {{ font-size: 13px; font-weight: 600; color: var(--muted); margin-bottom: 1rem;
                  text-transform: uppercase; letter-spacing: 0.05em; }}
  .chart-wrap {{ position: relative; height: 220px; }}
  .chart-wrap-tall {{ position: relative; height: 280px; }}

  /* Legend */
  .legend {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 13px; }}
  .legend-swatch {{ width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }}

  /* Delta badge */
  .delta-up   {{ color: var(--green); font-size: 12px; }}
  .delta-down {{ color: var(--red);   font-size: 12px; }}

  /* Speedometer widget */
  .speedometer {{ text-align: center; padding: 1rem 0; }}
  .speedometer .big {{ font-size: 3rem; font-weight: 700; letter-spacing: -0.03em; }}
  .speedometer .unit {{ font-size: 1rem; color: var(--muted); font-weight: 400; }}
  .speedometer .label {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}

  /* Info section */
  .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  @media (max-width: 700px) {{ .info-grid {{ grid-template-columns: 1fr; }} }}
  .info-block {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.25rem; }}
  .info-block h3 {{ font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 0.75rem; }}
  .info-row {{ display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; border-bottom: 1px solid var(--border); }}
  .info-row:last-child {{ border-bottom: none; }}
  .info-key {{ color: var(--muted); }}
  .info-val {{ font-weight: 500; }}

  footer {{
    text-align: center; padding: 2rem; font-size: 12px; color: var(--muted);
    border-top: 1px solid var(--border); margin-top: 2rem;
  }}
</style>
</head>
<body>

<header>
  <span class="badge">BENCHMARK</span>
  <h1>Model Performance Report</h1>
  <span class="timestamp" id="ts"></span>
</header>

<main>

<!-- Tabs -->
<div class="tabs">
  <div class="tab active" onclick="switchTab('overview')">Overview</div>
  <div class="tab" onclick="switchTab('timing')">Timing</div>
  <div class="tab" onclick="switchTab('memory')">Memory</div>
  <div class="tab" onclick="switchTab('timeseries')">Time-series</div>
  <div class="tab" onclick="switchTab('raw')">Raw Data</div>
</div>

<!-- ── TAB: OVERVIEW ── -->
<div id="tab-overview" class="tab-panel active">
  <div class="legend" id="legend"></div>
  <div class="cards" id="summary-cards"></div>
  <div class="chart-card">
    <div class="chart-title">FPS comparison</div>
    <div class="chart-wrap">
      <canvas id="chart-fps"></canvas>
    </div>
  </div>
  <br>
  <div class="table-wrap">
    <table id="comparison-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>FPS ↑</th>
          <th>Inf ms ↓</th>
          <th>Total ms ↓</th>
          <th>GPU MB ↓</th>
          <th>Peak GPU ↓</th>
          <th>CPU %</th>
          <th>Dets / Count</th>
        </tr>
      </thead>
      <tbody id="table-body"></tbody>
    </table>
  </div>
</div>

<!-- ── TAB: TIMING ── -->
<div id="tab-timing" class="tab-panel">
  <div class="charts-grid">
    <div class="chart-card">
      <div class="chart-title">Inference latency (ms) — mean ± std</div>
      <div class="chart-wrap"><canvas id="chart-inf-ms"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-title">End-to-end latency (ms) — mean ± std</div>
      <div class="chart-wrap"><canvas id="chart-total-ms"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-title">Pre / Inf / Post breakdown (stacked ms)</div>
      <div class="chart-wrap"><canvas id="chart-stacked"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-title">FPS — mean ± std</div>
      <div class="chart-wrap"><canvas id="chart-fps2"></canvas></div>
    </div>
  </div>
</div>

<!-- ── TAB: MEMORY ── -->
<div id="tab-memory" class="tab-panel">
  <div class="charts-grid">
    <div class="chart-card">
      <div class="chart-title">GPU memory — mean vs peak (MB)</div>
      <div class="chart-wrap"><canvas id="chart-gpu-mem"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-title">CPU utilisation (%)</div>
      <div class="chart-wrap"><canvas id="chart-cpu"></canvas></div>
    </div>
  </div>
</div>

<!-- ── TAB: TIME-SERIES ── -->
<div id="tab-timeseries" class="tab-panel">
  <div class="chart-card" style="margin-bottom:1.5rem">
    <div class="chart-title">Inference latency over frames (ms)</div>
    <div class="chart-wrap-tall"><canvas id="chart-ts-inf"></canvas></div>
  </div>
  <div class="chart-card" style="margin-bottom:1.5rem">
    <div class="chart-title">FPS over frames</div>
    <div class="chart-wrap-tall"><canvas id="chart-ts-fps"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">GPU memory over frames (MB)</div>
    <div class="chart-wrap-tall"><canvas id="chart-ts-gpu"></canvas></div>
  </div>
</div>

<!-- ── TAB: RAW DATA ── -->
<div id="tab-raw" class="tab-panel">
  <div class="info-grid" id="raw-info"></div>
</div>

</main>

<footer>Generated by benchmark_runner.py &nbsp;|&nbsp; <span id="footer-ts"></span></footer>

<script>
// ── Data ──────────────────────────────────────────────────────────────────────
const SUMMARIES = {summaries_json};
const SERIES    = {series_json};

// Colour palette (one per model)
const COLORS = ['#388bfd','#3fb950','#d29922','#a371f7','#f85149','#76e3ea'];
function modelColor(i) {{ return COLORS[i % COLORS.length]; }}

// ── Tabs ─────────────────────────────────────────────────────────────────────
function switchTab(name) {{
  document.querySelectorAll('.tab').forEach((t,i) => {{
    const id = ['overview','timing','memory','timeseries','raw'][i];
    t.classList.toggle('active', id === name);
  }});
  document.querySelectorAll('.tab-panel').forEach(p => {{
    p.classList.toggle('active', p.id === 'tab-'+name);
  }});
  if (name === 'timeseries') buildTimeSeries();
}}

// ── Legend ───────────────────────────────────────────────────────────────────
function buildLegend() {{
  const el = document.getElementById('legend');
  SUMMARIES.forEach((s, i) => {{
    el.innerHTML += `<span class="legend-item">
      <span class="legend-swatch" style="background:${{modelColor(i)}}"></span>
      ${{s.model_label}}
    </span>`;
  }});
}}

// ── Helper: best value index ─────────────────────────────────────────────────
function bestIdx(vals, higher_is_better) {{
  if (vals.length <= 1) return 0;
  return higher_is_better
    ? vals.indexOf(Math.max(...vals))
    : vals.indexOf(Math.min(...vals));
}}

// ── Summary cards ────────────────────────────────────────────────────────────
function buildCards() {{
  const container = document.getElementById('summary-cards');
  if (SUMMARIES.length < 2) return;

  const a = SUMMARIES[0], b = SUMMARIES[1];

  const speedup = (a.fps_mean / Math.max(b.fps_mean, 0.001)).toFixed(2);
  const faster = a.fps_mean > b.fps_mean ? a.model_label : b.model_label;
  const latencyDiff = (Math.abs(a.inference_ms_mean - b.inference_ms_mean)).toFixed(1);
  const memDiff = (Math.abs(a.gpu_mem_mb_mean - b.gpu_mem_mb_mean)).toFixed(0);

  container.innerHTML = `
    <div class="card win">
      <div class="card-label">Faster Model</div>
      <div class="card-value" style="font-size:1.1rem">${{faster}}</div>
      <div class="card-sub">${{speedup}}× FPS ratio</div>
    </div>
    <div class="card neutral">
      <div class="card-label">${{a.model_label}} FPS</div>
      <div class="card-value">${{a.fps_mean.toFixed(1)}}</div>
      <div class="card-sub">±${{a.fps_std.toFixed(1)}}</div>
    </div>
    <div class="card neutral">
      <div class="card-label">${{b.model_label}} FPS</div>
      <div class="card-value">${{b.fps_mean.toFixed(1)}}</div>
      <div class="card-sub">±${{b.fps_std.toFixed(1)}}</div>
    </div>
    <div class="card">
      <div class="card-label">Latency Delta</div>
      <div class="card-value" style="font-size:1.4rem">${{latencyDiff}} ms</div>
      <div class="card-sub">inference mean diff</div>
    </div>
    <div class="card">
      <div class="card-label">GPU Δ</div>
      <div class="card-value" style="font-size:1.4rem">${{memDiff}} MB</div>
      <div class="card-sub">mean usage diff</div>
    </div>
    <div class="card">
      <div class="card-label">Frames Tested</div>
      <div class="card-value" style="font-size:1.4rem">${{a.total_frames}}</div>
      <div class="card-sub">${{a.resolution}} · ${{a.device}}</div>
    </div>
  `;
}}

// ── Comparison table ─────────────────────────────────────────────────────────
function buildTable() {{
  const tbody = document.getElementById('table-body');
  const fields = ['fps_mean','inference_ms_mean','total_ms_mean','gpu_mem_mb_mean','gpu_mem_mb_peak','cpu_percent_mean','detections_mean'];
  const higherBetter = [true, false, false, false, false, false, false];

  const vals = fields.map(f => SUMMARIES.map(s => s[f]));
  const bests = fields.map((f, i) => bestIdx(vals[i], higherBetter[i]));

  SUMMARIES.forEach((s, si) => {{
    const row = document.createElement('tr');
    row.innerHTML = `
      <td class="model-name">
        <span class="dot" style="background:${{modelColor(si)}}"></span>${{s.model_label}}
      </td>
      ${{fields.map((f, fi) => `
        <td class="${{bests[fi] === si ? 'best' : ''}}">${{
          f === 'fps_mean' ? s[f].toFixed(1) + ' fps'
          : f.includes('ms') ? s[f].toFixed(1)
          : f.includes('mem') ? s[f].toFixed(0)
          : s[f].toFixed(1)
        }}</td>`).join('')}}
    `;
    tbody.appendChild(row);
  }});
}}

// ── Charts ───────────────────────────────────────────────────────────────────
const CHART_DEFAULTS = {{
  plugins: {{ legend: {{ display: false }} }},
  responsive: true,
  maintainAspectRatio: false,
  animation: {{ duration: 400 }},
}};

function barChart(id, labels, datasets, opts={{}}) {{
  return new Chart(document.getElementById(id), {{
    type: 'bar',
    data: {{ labels, datasets }},
    options: {{ ...CHART_DEFAULTS, ...opts }},
  }});
}}

function buildOverviewFPS() {{
  const labels = SUMMARIES.map(s => s.model_label);
  const data = SUMMARIES.map(s => s.fps_mean);
  barChart('chart-fps', labels, [{{
    label: 'FPS',
    data,
    backgroundColor: SUMMARIES.map((_, i) => modelColor(i) + 'cc'),
    borderColor: SUMMARIES.map((_, i) => modelColor(i)),
    borderWidth: 1,
    borderRadius: 4,
  }}], {{
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.parsed.y.toFixed(1)}} fps` }} }},
    }},
    scales: {{ y: {{ beginAtZero: true, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
              x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ display: false }} }} }},
  }});
}}

function buildTimingCharts() {{
  const labels = SUMMARIES.map(s => s.model_label);

  // Inference ms bar + error
  const infData = SUMMARIES.map(s => s.inference_ms_mean);
  const infErr  = SUMMARIES.map(s => s.inference_ms_std);
  barChart('chart-inf-ms', labels, [{{
    label: 'Inf ms',
    data: infData,
    backgroundColor: SUMMARIES.map((_, i) => modelColor(i) + 'aa'),
    borderColor: SUMMARIES.map((_, i) => modelColor(i)),
    borderWidth: 1,
    borderRadius: 4,
  }}], {{
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.parsed.y.toFixed(1)}} ms ±${{infErr[ctx.dataIndex].toFixed(1)}}` }} }},
    }},
    scales: {{ y: {{ beginAtZero: true, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
              x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ display: false }} }} }},
  }});

  // Total ms
  const totData = SUMMARIES.map(s => s.total_ms_mean);
  barChart('chart-total-ms', labels, [{{
    label: 'Total ms',
    data: totData,
    backgroundColor: SUMMARIES.map((_, i) => modelColor(i) + 'aa'),
    borderColor: SUMMARIES.map((_, i) => modelColor(i)),
    borderWidth: 1,
    borderRadius: 4,
  }}], {{
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ y: {{ beginAtZero: true, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
              x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ display: false }} }} }},
  }});

  // Stacked pre/inf/post
  new Chart(document.getElementById('chart-stacked'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{ label: 'Pre',  data: SUMMARIES.map(s => s.preprocess_ms_mean),  backgroundColor: '#a371f788', borderRadius: 4 }},
        {{ label: 'Inf',  data: SUMMARIES.map(s => s.inference_ms_mean),   backgroundColor: '#388bfd88' }},
        {{ label: 'Post', data: SUMMARIES.map(s => s.postprocess_ms_mean), backgroundColor: '#3fb95088', borderRadius: 4 }},
      ]
    }},
    options: {{
      ...CHART_DEFAULTS,
      plugins: {{ legend: {{ display: true, labels: {{ color: '#8b949e', boxWidth: 10 }} }} }},
      scales: {{
        x: {{ stacked: true, ticks: {{ color: '#8b949e' }}, grid: {{ display: false }} }},
        y: {{ stacked: true, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
      }}
    }}
  }});

  // FPS
  barChart('chart-fps2', labels, [{{
    label: 'FPS',
    data: SUMMARIES.map(s => s.fps_mean),
    backgroundColor: SUMMARIES.map((_, i) => modelColor(i) + 'aa'),
    borderColor: SUMMARIES.map((_, i) => modelColor(i)),
    borderWidth: 1,
    borderRadius: 4,
  }}], {{
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ y: {{ beginAtZero: true, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
              x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ display: false }} }} }},
  }});
}}

function buildMemoryCharts() {{
  const labels = SUMMARIES.map(s => s.model_label);

  new Chart(document.getElementById('chart-gpu-mem'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{ label: 'Mean GPU MB', data: SUMMARIES.map(s => s.gpu_mem_mb_mean),
           backgroundColor: SUMMARIES.map((_, i) => modelColor(i) + 'aa'),
           borderColor: SUMMARIES.map((_, i) => modelColor(i)), borderWidth: 1, borderRadius: 4 }},
        {{ label: 'Peak GPU MB', data: SUMMARIES.map(s => s.gpu_mem_mb_peak),
           backgroundColor: '#f8514933', borderColor: '#f85149', borderWidth: 1, borderRadius: 4,
           borderDash: [4,2] }},
      ]
    }},
    options: {{
      ...CHART_DEFAULTS,
      plugins: {{ legend: {{ display: true, labels: {{ color: '#8b949e', boxWidth: 10 }} }} }},
      scales: {{ y: {{ beginAtZero: true, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
                x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ display: false }} }} }},
    }}
  }});

  barChart('chart-cpu', labels, [{{
    label: 'CPU %',
    data: SUMMARIES.map(s => s.cpu_percent_mean),
    backgroundColor: SUMMARIES.map((_, i) => modelColor(i) + 'aa'),
    borderColor: SUMMARIES.map((_, i) => modelColor(i)),
    borderWidth: 1,
    borderRadius: 4,
  }}], {{
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ beginAtZero: true, max: 100, ticks: {{ color: '#8b949e', callback: v => v + '%' }}, grid: {{ color: '#30363d' }} }},
      x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ display: false }} }},
    }},
  }});
}}

let tsBuilt = false;
function buildTimeSeries() {{
  if (tsBuilt) return;
  tsBuilt = true;

  const keys = Object.keys(SERIES);

  function tsDatasets(field) {{
    return keys.map((label, i) => ({{
      label,
      data: SERIES[label][field],
      borderColor: modelColor(i),
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
    }}));
  }}

  function lineChart(id, field) {{
    const allFrames = keys.length ? SERIES[keys[0]].frames : [];
    new Chart(document.getElementById(id), {{
      type: 'line',
      data: {{ labels: allFrames, datasets: tsDatasets(field) }},
      options: {{
        ...CHART_DEFAULTS,
        plugins: {{ legend: {{ display: true, labels: {{ color: '#8b949e', boxWidth: 10, pointStyle: 'line' }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 10 }}, grid: {{ color: '#30363d30' }} }},
          y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
        }},
      }}
    }});
  }}

  lineChart('chart-ts-inf', 'inference_ms');
  lineChart('chart-ts-fps', 'fps');
  lineChart('chart-ts-gpu', 'gpu_mem');
}}

// ── Raw data info ─────────────────────────────────────────────────────────────
function buildRaw() {{
  const el = document.getElementById('raw-info');
  SUMMARIES.forEach((s, i) => {{
    const rows = [
      ['Model path', s.model_path],
      ['Device', s.device],
      ['Resolution', s.resolution],
      ['Frames', s.total_frames],
      ['Timestamp', s.timestamp],
      ['Pre ms (μ±σ)', `${{s.preprocess_ms_mean}} ± ${{s.preprocess_ms_std}}`],
      ['Inf ms (μ±σ)', `${{s.inference_ms_mean}} ± ${{s.inference_ms_std}}`],
      ['Post ms (μ±σ)', `${{s.postprocess_ms_mean}} ± ${{s.postprocess_ms_std}}`],
      ['Total ms (μ±σ)', `${{s.total_ms_mean}} ± ${{s.total_ms_std}}`],
      ['FPS (μ±σ)', `${{s.fps_mean}} ± ${{s.fps_std}}`],
      ['GPU mean MB', s.gpu_mem_mb_mean],
      ['GPU peak MB', s.gpu_mem_mb_peak],
      ['CPU %', s.cpu_percent_mean],
      ['Dets/Count mean', s.detections_mean],
      ['Conf/Count mean', s.confidence_mean],
    ];
    el.innerHTML += `
      <div class="info-block">
        <h3 style="color:${{modelColor(i)}}">${{s.model_label}}</h3>
        ${{rows.map(([k,v]) => `
          <div class="info-row">
            <span class="info-key">${{k}}</span>
            <span class="info-val">${{v}}</span>
          </div>`).join('')}}
      </div>`;
  }});
}}

// ── Timestamps ────────────────────────────────────────────────────────────────
const ts = SUMMARIES[0]?.timestamp || new Date().toISOString();
document.getElementById('ts').textContent = ts.replace('T',' ');
document.getElementById('footer-ts').textContent = ts.replace('T',' ');

// ── Init ──────────────────────────────────────────────────────────────────────
buildLegend();
buildCards();
buildTable();
buildOverviewFPS();
buildTimingCharts();
buildMemoryCharts();
buildRaw();
</script>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path
