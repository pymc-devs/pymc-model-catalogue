"""Generate a custom dashboard HTML from asv-published data.

Usage:
    python scripts/make_dashboard.py <asv_publish_dir> <output_html>

Reads ``index.json`` and ``graphs/branch-timeline/**/*.json`` from an
asv publish output directory (typically ``.asv/html`` during CI or
``.gh-pages-push`` / a freshly-cloned gh-pages checkout) and writes a
self-contained HTML file with embedded data and Plotly plots.

The generated page is a dense sparkline grid: one row per model, five
columns (model name + four metrics). No horizontal scroll, no chart
chrome, hover tooltips show (pymc tag, formatted value, commit
short-hash). Counts only pymc versions that actually have data, not
just tagged commits.

Each benchmark environment has its own series. Historical measurements
remain visible when a later run uses a different Python version.

This is purely a rendering script — it reads already-published asv
data and rewrites the HTML. It can run against stale data (no new
measurements) and produces a valid, just-stale dashboard, which makes
it safe to trigger from dashboard-only regeneration workflows.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


METRIC_ORDER = [
    ("track_rewrite_time", "rewrite_time", "seconds"),
    ("track_compile_time", "first_call_time", "seconds"),
    ("track_n_rewrites", "n_rewrites", "count"),
    ("time_eval", "eval", "seconds"),
]


def _decode_workload_name(name: str) -> str:
    """Decode workload names without importing model packages."""
    pair_prefix = "models_posteriordb__"
    if name.startswith(pair_prefix):
        posterior, mode = name[len(pair_prefix) :].rsplit("__", 1)
        return f"models.posteriordb/{posterior.replace('__', '-')} [{mode}]"
    for prefix, package in (
        ("models_discrete_", "models_discrete"),
        ("models_", "models"),
    ):
        if name.startswith(prefix):
            return package + "." + name[len(prefix) :]
    return name


def load_data(src: Path) -> tuple[dict, dict]:
    idx = json.loads((src / "index.json").read_text())
    revs = {int(k): v for k, v in idx["revision_to_hash"].items()}

    # SHA → tag name. We index by SHA (not asv revision number) so
    # orphaned measurements at commits that no current tag points to
    # are filtered out below. Without this, force-pushed timeline
    # rebuilds leave stale measurements at SHAs that look like raw
    # commits in the dashboard, polluting the view.
    sha_to_tag: dict[str, str] = {
        revs[rev]: tag for tag, rev in idx["tags"].items() if rev in revs
    }

    graphs_root = src / "graphs" / "branch-timeline"
    data: dict = {}
    for gf in sorted(graphs_root.rglob("*.json")):
        name = gf.stem
        if name == "summary":
            continue
        m = re.match(
            r"bench_models\.ModelBench(?:Build|Eval)_(.+?)\.(track_\w+|time_\w+)",
            name,
        )
        if not m:
            continue
        sanitized_model, metric = m.groups()
        model = _decode_workload_name(sanitized_model)
        series = json.loads(gf.read_text())
        environment = (
            " · ".join(
                "Python " + part.removeprefix("python-")
                if part.startswith("python-")
                else part.removeprefix("machine-")
                for part in gf.parent.relative_to(graphs_root).parts
            )
            or "default"
        )
        # Keep only measurements at SHAs that a current tag points to.
        # Orphans from previous timeline force-push generations are
        # silently dropped — the user can run reset=true to clean them
        # out of the underlying gh-pages data if they care.
        points = []
        for rev, value in series:
            if rev not in revs:
                continue
            sha = revs[rev]
            tag = sha_to_tag.get(sha)
            if tag is None:
                continue
            points.append(
                {
                    "tag": tag,
                    "hash": sha,
                    "value": value,
                    "environment": environment,
                }
            )
        if points:
            # ASV writes a separate graph per environment. Keep all of them;
            # assigning here used to replace the entire historical series
            # with whichever Python version's file happened to sort last.
            data.setdefault(model, {}).setdefault(metric, []).extend(points)

    for metrics in data.values():
        for points in metrics.values():
            # Sort by pymc version so file ordering cannot shuffle the x-axis.
            points.sort(
                key=lambda p: tuple(
                    int(x) for x in p["tag"].removeprefix("pymc-").split(".")
                )
            )
    return data, idx


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>pymc-model-catalogue &mdash; benchmark dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, sans-serif;
      background: #fff;
      color: #222;
      font-size: 13px;
    }
    header {
      padding: 10px 20px;
      border-bottom: 1px solid #eaeaea;
      display: flex;
      align-items: center;
      gap: 16px;
    }
    header h1 { margin: 0; font-size: 15px; font-weight: 600; }
    header .meta { color: #888; font-size: 11px; }
    header .environments { display: flex; gap: 12px; font-size: 11px; }
    header .links { margin-left: auto; display: flex; gap: 12px; }
    header .links a { color: #1f77b4; text-decoration: none; font-size: 12px; }
    header .links a:hover { text-decoration: underline; }
    main { padding: 16px 20px; max-width: 1400px; margin: 0 auto; }
    .model-row {
      margin-bottom: 14px;
      display: grid;
      grid-template-columns: 200px repeat(4, 1fr);
      gap: 8px;
      align-items: center;
      border-bottom: 1px solid #f1f1f1;
      padding-bottom: 10px;
    }
    .model-row h2 {
      margin: 0;
      font-size: 12px;
      font-weight: 600;
      font-family: ui-monospace, SF Mono, Menlo, monospace;
      color: #333;
      word-break: break-all;
      line-height: 1.3;
    }
    .plot-wrap { display: flex; flex-direction: column; gap: 2px; }
    .plot-title {
      font-size: 10px;
      color: #888;
      text-align: center;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }
    .plot { height: 80px; min-width: 0; }
    .empty {
      color: #bbb;
      font-size: 10px;
      text-align: center;
      padding: 28px 0;
    }
  </style>
</head>
<body>
<header>
  <h1>pymc-model-catalogue</h1>
  <span class="meta" id="meta"></span>
  <span class="environments" id="environments"></span>
  <div class="links">
    <a href="experiments.html">experiments</a>
    <a href="./">default asv view</a>
  </div>
</header>
<p>NUMBA first-call time includes JIT compilation. FAST_RUN first-call time measures evaluation.</p>
<main id="root"></main>
<script>
const PAYLOAD = __PAYLOAD__;

const LAYOUT_BASE = {
  paper_bgcolor: '#fff',
  plot_bgcolor: '#fff',
  font: { color: '#222', size: 9, family: 'system-ui' },
  margin: { l: 4, r: 4, t: 2, b: 2 },
  showlegend: false,
  xaxis: {
    visible: false, fixedrange: true, type: 'category',
    categoryorder: 'array', categoryarray: PAYLOAD.versions,
  },
  yaxis: { visible: false, fixedrange: true },
};
const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };
const COLORS = ['#1f77b4', '#d95f02', '#2ca02c', '#9467bd', '#8c564b'];

function environmentColor(environment) {
  return COLORS[PAYLOAD.environments.indexOf(environment) % COLORS.length];
}

function pointsFor(model, metric) {
  const d = PAYLOAD.data[model] && PAYLOAD.data[model][metric];
  return d || [];
}

function formatValue(metric, v) {
  // Asv stores null for failed benchmarks. Handle it here so the
  // hover-text builder doesn't throw on v.toFixed() and halt the
  // entire render loop before it reaches the next model row.
  if (v == null || Number.isNaN(v)) return 'n/a';
  if (metric === 'track_n_rewrites') return v.toFixed(0);
  if (v < 1e-3) return (v * 1e6).toFixed(1) + ' μs';
  if (v < 1) return (v * 1e3).toFixed(0) + ' ms';
  return v.toFixed(2) + ' s';
}

function makeSparkline(div, pts, metric) {
  if (pts.length === 0) {
    div.innerHTML = '<div class="empty">no data</div>';
    return;
  }
  const traces = PAYLOAD.environments.flatMap(environment => {
    const points = pts.filter(p => p.environment === environment);
    if (points.length === 0) return [];
    const color = environmentColor(environment);
    return [{
      name: environment,
      x: points.map(p => p.tag),
      y: points.map(p => p.value),
      mode: 'lines+markers',
      line: { color, width: 1.5, shape: 'linear' },
      marker: { size: 4, color },
      hovertemplate: '%{text}<br>%{fullData.name}<extra></extra>',
      text: points.map(p => `<b>${p.tag}</b><br>${formatValue(metric, p.value)}`),
    }];
  });
  Plotly.newPlot(div, traces, LAYOUT_BASE, PLOTLY_CONFIG);
}

function render() {
  const root = document.getElementById('root');
  root.innerHTML = '';
  PAYLOAD.models.forEach(model => {
    const row = document.createElement('div');
    row.className = 'model-row';
    let html = `<h2>${model}</h2>`;
    PAYLOAD.metrics.forEach(metric => {
      html += `<div class="plot-wrap"><div class="plot-title">${PAYLOAD.metric_labels[metric]}</div><div class="plot" data-metric="${metric}"></div></div>`;
    });
    row.innerHTML = html;
    root.appendChild(row);
    row.querySelectorAll('.plot').forEach(div => {
      const metric = div.dataset.metric;
      makeSparkline(div, pointsFor(model, metric), metric);
    });
  });
}

document.getElementById('meta').textContent =
  PAYLOAD.models.length + ' models · ' +
  PAYLOAD.measured_count + ' pymc versions measured';

PAYLOAD.environments.forEach(environment => {
  const label = document.createElement('span');
  label.style.color = environmentColor(environment);
  label.textContent = '● ' + environment;
  document.getElementById('environments').appendChild(label);
});

render();
</script>
</body>
</html>
"""


def render(data: dict, idx: dict, out: Path) -> None:
    points = [
        pt for metrics in data.values() for series in metrics.values() for pt in series
    ]
    measured_tags = sorted(
        {pt["tag"] for pt in points},
        key=lambda tag: tuple(int(x) for x in tag.removeprefix("pymc-").split(".")),
    )
    payload = {
        "models": sorted(data.keys()),
        "metrics": [m[0] for m in METRIC_ORDER],
        "metric_labels": {m[0]: m[1] for m in METRIC_ORDER},
        "metric_units": {m[0]: m[2] for m in METRIC_ORDER},
        "measured_count": len(measured_tags),
        "versions": measured_tags,
        "environments": sorted({pt["environment"] for pt in points}),
        "data": data,
    }
    html = _HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    if not (src / "index.json").exists():
        print(f"ERROR: {src / 'index.json'} not found", file=sys.stderr)
        return 1
    data, idx = load_data(src)
    render(data, idx, out)
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB; {len(data)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
