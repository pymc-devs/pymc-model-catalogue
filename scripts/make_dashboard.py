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
chrome. Hover tooltips show the PyMC release, measurement, Python environment,
and dependency versions when recorded. Counts only PyMC versions with data, not
just tagged commits.

Each benchmark environment has its own series. Historical measurements
remain visible when a later run uses a different Python version.
Each chart labels its newest measured point, preferring newer Python on ties.
A checkbox restricts all plots to the latest six measured versions,
with linear axes rescaled to that window.

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


def load_dependencies(src: Path) -> dict:
    path = Path(__file__).resolve().parents[1] / "versions" / "benchmark_dependencies.json"
    known = json.loads(path.read_text())
    dependencies = {}
    for path in (src / "results" / "ci").glob("*-virtualenv-py*.json"):
        result = json.loads(path.read_text())
        sha = result["commit_hash"]
        environment = f"ci · Python {result['python']}"
        for name, versions in result.get("benchmark_dependencies", {}).items():
            dependencies[sha, environment, name] = versions
        historical = known["commits"].get(sha, {}).get(environment)
        columns = result["result_columns"]
        if not historical or "started_at" not in columns:
            continue
        index = columns.index("started_at")
        for name, row in result["results"].items():
            # A later run can use different packages at the same timeline SHA.
            started_at = row[index] if len(row) > index else None
            if started_at is not None and known["started_at_min"] <= started_at <= known["started_at_max"]:
                dependencies.setdefault((sha, environment, name), historical)
    return dependencies


def load_data(src: Path) -> tuple[dict, dict]:
    idx = json.loads((src / "index.json").read_text())
    revs = {int(k): v for k, v in idx["revision_to_hash"].items()}
    dependencies = load_dependencies(src)

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
        environment = " · ".join(
            "Python " + part.removeprefix("python-")
            if part.startswith("python-") else part.removeprefix("machine-")
            for part in gf.parent.relative_to(graphs_root).parts
        ) or "default"
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
            point = {
                "tag": tag, "hash": sha, "value": value,
                "environment": environment,
            }
            if versions := dependencies.get((sha, environment, name)):
                point.update(versions)
            points.append(point)
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
      flex-wrap: wrap;
      align-items: center;
      gap: 16px;
    }
    header h1 { margin: 0; font-size: 15px; font-weight: 600; }
    header .meta { color: #888; font-size: 11px; }
    header .environments { display: flex; gap: 12px; font-size: 11px; }
    header .version-filter {
      display: flex;
      align-items: center;
      gap: 4px;
      font-size: 12px;
      white-space: nowrap;
      cursor: pointer;
    }
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
  <span class="environments" id="environments"></span>
  <label class="version-filter">
    <input type="checkbox" id="latest-versions" /> Latest 6 versions only
  </label>
  <div class="links">
    <a href="https://github.com/pymc-devs/pymc-model-catalogue">GitHub</a>
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
  margin: { l: 4, r: 44, t: 6, b: 6 },
  showlegend: false,
  xaxis: {
    visible: false, fixedrange: true, type: 'category',
    categoryorder: 'array', categoryarray: PAYLOAD.versions,
  },
  yaxis: { visible: false, fixedrange: true, type: 'linear', autorange: true },
};
const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };
const COLORS = ['#1f77b4', '#d95f02', '#2ca02c', '#9467bd', '#8c564b'];
const pythonVersion = environment => environment.match(/Python ([0-9]+(?:[.][0-9]+)*)/)?.[1] || '';

function environmentColor(environment) {
  return COLORS[PAYLOAD.environments.indexOf(environment) % COLORS.length];
}

function visibleVersions() {
  return document.getElementById('latest-versions').checked
    ? PAYLOAD.versions.slice(-6) : PAYLOAD.versions;
}

function pointsFor(model, metric, versions) {
  const d = PAYLOAD.data[model] && PAYLOAD.data[model][metric];
  return (d || []).filter(p => versions.has(p.tag));
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

function makeSparkline(div, pts, metric, versions) {
  if (pts.length === 0) {
    div.innerHTML = '<div class="empty">no data</div>';
    return;
  }
  const annotations = [];
  // Label the newest measured release, preferring the newer Python on ties.
  const last = pts.filter(p => Number.isFinite(p.value)).reduce((latest, point) => {
    if (!latest) return point;
    const releaseOrder = versions.indexOf(point.tag) - versions.indexOf(latest.tag);
    const pythonOrder = pythonVersion(point.environment).localeCompare(
      pythonVersion(latest.environment), undefined, { numeric: true },
    );
    return releaseOrder > 0 || (releaseOrder === 0 && pythonOrder > 0) ? point : latest;
  }, null);
  const traces = PAYLOAD.environments.flatMap(environment => {
    const points = pts.filter(p => p.environment === environment);
    if (points.length === 0) return [];
    const color = environmentColor(environment);
    if (last && last.environment === environment) {
      annotations.push({
        x: last.tag, y: last.value,
        xref: 'x', yref: 'y',
        text: formatValue(metric, last.value),
        font: { color, size: 8 },
        showarrow: false,
        xanchor: 'left', yanchor: 'middle', xshift: 4,
        captureevents: true,
      });
    }
    return [{
      name: environment,
      x: points.map(p => p.tag),
      y: points.map(p => p.value),
      mode: 'lines+markers',
      line: { color, width: 1.5, shape: 'linear' },
      marker: { size: 4, color },
      hovertemplate: '%{text}<extra></extra>',
      text: points.map(p => [
        `<b>${p.tag}</b>`, formatValue(metric, p.value), environment,
        p.pytensor && `PyTensor ${p.pytensor} · Numba ${p.numba}`,
      ].filter(Boolean).join('<br>')),
    }];
  });
  const layout = {
    ...LAYOUT_BASE,
    xaxis: {
      ...LAYOUT_BASE.xaxis,
      categoryarray: versions,
      range: [-0.25, versions.length - 0.5],
    },
    annotations,
  };
  Plotly.newPlot(div, traces, layout, PLOTLY_CONFIG).then(() => {
    if (!last) return;
    const curveNumber = traces.findIndex(trace => trace.name === last.environment);
    const pointNumber = traces[curveNumber].x.indexOf(last.tag);
    // The value label is another hover target for its point, with the same
    // tooltip and trace color. Rebind when Plotly redraws it after a resize.
    const bindLabelHover = () => {
      const label = div.querySelector('.annotation-text-g');
      if (!label) return;
      label.onpointerenter = () => Plotly.Fx.hover(div, [{ curveNumber, pointNumber }]);
      label.onpointerleave = () => Plotly.Fx.unhover(div);
    };
    bindLabelHover();
    div.on('plotly_afterplot', bindLabelHover);
  });
}

function render() {
  const root = document.getElementById('root');
  root.querySelectorAll('.plot').forEach(div => Plotly.purge(div));
  root.innerHTML = '';
  // One grid column per metric, so adding one needs no CSS change.
  document.documentElement.style.setProperty('--n-metrics', PAYLOAD.metrics.length);
  const versions = visibleVersions();
  const versionSet = new Set(versions);
  document.getElementById('meta').textContent =
    PAYLOAD.models.length + ' models · ' + versions.length + ' of ' +
    PAYLOAD.measured_count + ' pymc versions shown';
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
      makeSparkline(div, pointsFor(model, metric, versionSet), metric, versions);
    });
  });
}

document.getElementById('latest-versions').addEventListener('change', render);

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
    points = [pt for metrics in data.values() for series in metrics.values() for pt in series]
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
