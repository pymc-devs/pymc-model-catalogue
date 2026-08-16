"""Generate a custom dashboard HTML from asv-published data.

Usage:
    python scripts/make_dashboard.py <asv_publish_dir> <output_html>

Reads ``index.json`` and ``graphs/branch-timeline/**/*.json`` from an
asv publish output directory (typically ``.asv/html`` during CI or
``.gh-pages-push`` / a freshly-cloned gh-pages checkout) and writes a
self-contained HTML file with embedded data and Plotly plots.

The generated page is a dense sparkline grid: one row per model, one
column per metric plus the model name (the grid sizes itself from
METRIC_ORDER). No horizontal scroll, no chart
chrome, hover tooltips show (pymc tag, formatted value, commit
short-hash). Counts only pymc versions that actually have data, not
just tagged commits.

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
    ("track_compile_time", "compile_time", "seconds"),
    ("track_n_rewrites", "n_rewrites", "count"),
    ("track_peak_rss", "peak_rss", "bytes"),
    ("time_eval", "eval", "seconds"),
]


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
        model = sanitized_model
        for prefix in ("models_discrete_", "models_"):
            if sanitized_model.startswith(prefix):
                model = prefix.rstrip("_") + "." + sanitized_model[len(prefix):]
                break
        series = json.loads(gf.read_text())
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
            points.append({"tag": tag, "hash": sha, "value": value})
        # Sort by pymc version (parsed as semver) so re-orderings on
        # the underlying data don't shuffle the x-axis.
        points.sort(
            key=lambda p: tuple(
                int(x) for x in p["tag"].removeprefix("pymc-").split(".")
            )
        )
        if points:
            data.setdefault(model, {})[metric] = points
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
    header .links { margin-left: auto; display: flex; gap: 12px; }
    header .links a { color: #1f77b4; text-decoration: none; font-size: 12px; }
    header .links a:hover { text-decoration: underline; }
    main { padding: 16px 20px; max-width: 1700px; margin: 0 auto; }
    .model-row {
      margin-bottom: 14px;
      display: grid;
      grid-template-columns: 200px repeat(var(--n-metrics), 1fr);
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
  <div class="links">
    <a href="experiments.html">experiments</a>
    <a href="./">default asv view</a>
  </div>
</header>
<main id="root"></main>
<script>
const PAYLOAD = __PAYLOAD__;

const LAYOUT_BASE = {
  paper_bgcolor: '#fff',
  plot_bgcolor: '#fff',
  font: { color: '#222', size: 9, family: 'system-ui' },
  margin: { l: 4, r: 4, t: 2, b: 2 },
  showlegend: false,
  xaxis: { visible: false, fixedrange: true },
  yaxis: { visible: false, fixedrange: true },
};
const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };

function pointsFor(model, metric) {
  const d = PAYLOAD.data[model] && PAYLOAD.data[model][metric];
  if (!d) return { x: [], y: [] };
  return {
    x: d.map(p => p.tag),
    y: d.map(p => p.value),
  };
}

function formatValue(metric, v) {
  // Asv stores null for failed benchmarks. Handle it here so the
  // hover-text builder doesn't throw on v.toFixed() and halt the
  // entire render loop before it reaches the next model row.
  if (v == null || Number.isNaN(v)) return 'n/a';
  if (metric === 'track_n_rewrites') return v.toFixed(0);
  // Without this, bytes fall through to the seconds ladder below.
  if (PAYLOAD.metric_units[metric] === 'bytes') {
    if (v >= 1073741824) return (v / 1073741824).toFixed(2) + ' GB';
    if (v >= 1048576) return (v / 1048576).toFixed(0) + ' MB';
    return (v / 1024).toFixed(0) + ' KB';
  }
  if (v < 1e-3) return (v * 1e6).toFixed(1) + ' μs';
  if (v < 1) return (v * 1e3).toFixed(0) + ' ms';
  return v.toFixed(2) + ' s';
}

function makeSparkline(div, pts, metric) {
  if (pts.x.length === 0) {
    div.innerHTML = '<div class="empty">no data</div>';
    return;
  }
  const hoverText = pts.x.map((tag, i) =>
    `<b>${tag}</b><br>${formatValue(metric, pts.y[i])}`
  );
  Plotly.newPlot(div, [{
    x: pts.x,
    y: pts.y,
    mode: 'lines+markers',
    line: { color: '#1f77b4', width: 1.5, shape: 'linear' },
    marker: { size: 4, color: '#1f77b4' },
    hovertemplate: '%{text}<extra></extra>',
    text: hoverText,
  }], LAYOUT_BASE, PLOTLY_CONFIG);
}

function render() {
  const root = document.getElementById('root');
  root.innerHTML = '';
  // One grid column per metric, so adding one needs no CSS change.
  document.documentElement.style.setProperty('--n-metrics', PAYLOAD.metrics.length);
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

render();
</script>
</body>
</html>
"""


def render(data: dict, idx: dict, out: Path) -> None:
    measured_tags = sorted(
        {pt["tag"] for metrics in data.values() for s in metrics.values() for pt in s}
    )
    payload = {
        "models": sorted(data.keys()),
        "metrics": [m[0] for m in METRIC_ORDER],
        "metric_labels": {m[0]: m[1] for m in METRIC_ORDER},
        "metric_units": {m[0]: m[2] for m in METRIC_ORDER},
        "measured_count": len(measured_tags),
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
