"""Generate an experiment comparison dashboard from published asv data.

Usage:
    python scripts/make_experiment_dashboard.py <gh_pages_dir> <output_html>

Scans ``experiments/<name>/<sha>/`` directories under *gh_pages_dir*,
extracts the experiment-tagged benchmark data point from each run, and
writes a self-contained HTML comparison page.

Each experiment YAML produces a venv whose python path embeds the
experiment name (``…/.builds/<name>/venv/…``). The script identifies
the correct graph series for each experiment by matching the python
path in ``graph_param_list`` against the experiment directory name.
Within each run only the single data point at the ``exp-<name>`` tag
revision is extracted — timeline history is ignored.
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
    ("time_eval", "eval", "seconds"),
]


def _extract_exp_name(python_path: str) -> str | None:
    m = re.search(r"\.builds/([^/]+)/venv", python_path)
    return m.group(1) if m else None


def _parse_benchmark_name(stem: str) -> tuple[str, str] | None:
    m = re.match(
        r"bench_models\.ModelBench(?:Build|Eval)_(.+?)\.(track_\w+|time_\w+)",
        stem,
    )
    if not m:
        return None
    sanitized_model, metric = m.groups()
    model = sanitized_model
    for prefix in ("models_discrete_", "models_"):
        if sanitized_model.startswith(prefix):
            model = prefix.rstrip("_") + "." + sanitized_model[len(prefix) :]
            break
    return model, metric


def load_experiments(src: Path) -> dict:
    """Return ``{exp_name: {model: {metric: value}}}`` from all experiment runs."""
    experiments_root = src / "experiments"
    if not experiments_root.is_dir():
        return {}

    data: dict[str, dict[str, dict[str, float | None]]] = {}
    meta: dict[str, dict] = {}
    run_shas: dict[str, str] = {}

    for exp_dir in sorted(experiments_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            idx_path = run_dir / "index.json"
            if not idx_path.exists():
                continue

            meta_path = run_dir / "meta.json"
            if meta_path.exists():
                m = json.loads(meta_path.read_text())
                meta[m.get("name", exp_dir.name)] = m

            idx = json.loads(idx_path.read_text())
            exp_tags = {
                k: v for k, v in idx.get("tags", {}).items() if k.startswith("exp-")
            }
            if not exp_tags:
                continue

            for params in idx.get("graph_param_list", []):
                python_path = params.get("python", "")
                machine = params.get("machine", "")
                venv_name = _extract_exp_name(python_path)
                if not venv_name:
                    continue

                exp_tag = f"exp-{venv_name}"
                if exp_tag not in exp_tags:
                    continue
                exp_rev = exp_tags[exp_tag]

                sanitized_py = python_path.replace("/", "_")
                graph_dir = (
                    run_dir
                    / "graphs"
                    / "branch-timeline"
                    / f"machine-{machine}"
                    / f"python-{sanitized_py}"
                )
                if not graph_dir.is_dir():
                    continue

                for gf in sorted(graph_dir.glob("*.json")):
                    parsed = _parse_benchmark_name(gf.stem)
                    if not parsed:
                        continue
                    model, metric = parsed
                    series = json.loads(gf.read_text())
                    for rev, value in series:
                        if rev == exp_rev:
                            data.setdefault(venv_name, {}).setdefault(model, {})[
                                metric
                            ] = value
                            run_shas[venv_name] = run_dir.name
                            break

    return {"data": data, "run_shas": run_shas, "meta": meta}


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Experiment Comparison &mdash; pymc-model-catalogue</title>
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
      display: flex; align-items: center; gap: 16px;
    }
    header h1 { margin: 0; font-size: 15px; font-weight: 600; }
    header .meta { color: #888; font-size: 11px; }
    header .links { margin-left: auto; display: flex; gap: 12px; }
    header .links a { color: #1f77b4; text-decoration: none; font-size: 12px; }
    header .links a:hover { text-decoration: underline; }
    main { padding: 16px 20px; max-width: 1400px; margin: 0 auto; }

    .controls {
      display: flex; align-items: center; gap: 12px;
      margin-bottom: 16px; flex-wrap: wrap;
    }
    .controls label { font-size: 12px; font-weight: 600; }
    .controls select {
      font-size: 12px; padding: 3px 6px;
      border: 1px solid #ccc; border-radius: 4px;
    }

    .exp-info {
      display: grid; grid-template-columns: 1fr 1fr;
      gap: 12px; margin-bottom: 16px;
    }
    .exp-info-card {
      border: 1px solid #eaeaea; border-radius: 8px; padding: 10px 14px;
      font-size: 12px; line-height: 1.5;
    }
    .exp-info-card .exp-label {
      font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
      color: #888; margin-bottom: 2px;
    }
    .exp-info-card .exp-name { font-weight: 700; }
    .exp-info-card .exp-desc { color: #555; margin-top: 4px; }
    .exp-info-card a { color: #1f77b4; text-decoration: none; font-size: 11px; }
    .exp-info-card a:hover { text-decoration: underline; }

    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px; margin-bottom: 20px;
    }
    .summary-card {
      border: 1px solid #eaeaea; border-radius: 8px;
      padding: 12px; text-align: center;
    }
    .summary-card .metric-name {
      font-size: 10px; text-transform: uppercase;
      letter-spacing: 0.5px; color: #888; margin-bottom: 4px;
    }
    .summary-card .ratio {
      font-size: 24px; font-weight: 700;
    }
    .summary-card .detail {
      font-size: 10px; color: #888; margin-top: 2px;
    }

    .charts-section { margin-bottom: 24px; }
    .charts-section h2 {
      font-size: 13px; font-weight: 600; margin: 0 0 8px;
      border-bottom: 1px solid #eaeaea; padding-bottom: 4px;
    }
    .chart-container { height: 400px; margin-bottom: 16px; }

    table {
      width: 100%; border-collapse: collapse;
      font-size: 12px; table-layout: fixed;
    }
    th, td {
      padding: 5px 8px; text-align: right;
      border-bottom: 1px solid #f0f0f0;
    }
    th { font-weight: 600; font-size: 11px; text-transform: uppercase;
         letter-spacing: 0.3px; color: #666; border-bottom: 2px solid #ddd;
         cursor: pointer; user-select: none; }
    th:hover { color: #222; }
    th.metric-group { text-align: center; border-left: 2px solid #e0e0e0;
                      cursor: default; }
    th.metric-group:hover { color: #666; }
    th.grp-start, td.grp-start { border-left: 2px solid #e0e0e0; }
    th small { display: block; text-transform: none; font-weight: 400;
               font-size: 10px; color: #999; letter-spacing: 0; }
    th.sort-asc::after { content: ' \25b2'; }
    th.sort-desc::after { content: ' \25bc'; }
    td:first-child, th:first-child { text-align: left; width: 220px; }
    td:first-child {
      font-family: ui-monospace, SF Mono, Menlo, monospace;
      font-size: 11px; word-break: break-all;
    }
    .good { color: #16a34a; }
    .bad { color: #dc2626; }
    .neutral { color: #888; }
    .na { color: #ccc; }
    tr:hover { background: #f9f9f9; }
    .summary-row td { font-weight: 700; border-top: 2px solid #ddd; }
  </style>
</head>
<body>
<header>
  <h1>Experiment Comparison</h1>
  <span class="meta" id="meta"></span>
  <div class="links">
    <a href="dashboard.html">timeline dashboard</a>
    <a href="./">default asv view</a>
  </div>
</header>
<main id="root">
  <div class="controls">
    <label>Base experiment:</label>
    <select id="base-select"></select>
    <button id="swap-btn" title="Swap base and compare" style="
      font-size:16px; background:none; border:1px solid #ccc;
      border-radius:4px; cursor:pointer; padding:1px 7px; line-height:1.4;
    ">&harr;</button>
    <label>Compare:</label>
    <select id="compare-select"></select>
  </div>
  <div class="exp-info" id="exp-info"></div>
  <div class="summary" id="summary"></div>
  <div class="charts-section" id="charts"></div>
  <table id="table"><thead></thead><tbody></tbody></table>
</main>
<script>
const PAYLOAD = __PAYLOAD__;
const METRICS = PAYLOAD.metrics;
const METRIC_LABELS = PAYLOAD.metric_labels;
const METRIC_UNITS = PAYLOAD.metric_units;
const ALL_EXPERIMENTS = Object.keys(PAYLOAD.data).sort();

const baseSel = document.getElementById('base-select');
const compareSel = document.getElementById('compare-select');

ALL_EXPERIMENTS.forEach(e => {
  baseSel.add(new Option(e, e));
  compareSel.add(new Option(e, e));
});
function readHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  const b = params.get('base');
  const c = params.get('compare');
  if (b && ALL_EXPERIMENTS.includes(b)) baseSel.value = b;
  if (c && ALL_EXPERIMENTS.includes(c)) compareSel.value = c;
  else {
    const nonBase = ALL_EXPERIMENTS.find(e => e !== baseSel.value);
    if (nonBase) compareSel.value = nonBase;
  }
}

function writeHash() {
  const params = new URLSearchParams();
  params.set('base', baseSel.value);
  params.set('compare', compareSel.value);
  history.replaceState(null, '', '#' + params.toString());
}

readHash();

document.getElementById('swap-btn').addEventListener('click', () => {
  const tmp = baseSel.value;
  baseSel.value = compareSel.value;
  compareSel.value = tmp;
  update();
});
window.addEventListener('hashchange', () => { readHash(); update(); });

function formatValue(metric, v) {
  if (v == null || Number.isNaN(v)) return 'n/a';
  if (metric === 'track_n_rewrites') return v.toFixed(0);
  if (v < 1e-3) return (v * 1e6).toFixed(1) + ' µs';
  if (v < 1) return (v * 1e3).toFixed(0) + ' ms';
  return v.toFixed(3) + ' s';
}

function ratioClass(ratio, metric) {
  if (ratio == null) return 'na';
  // For n_rewrites, more is neutral (not inherently good/bad)
  if (metric === 'track_n_rewrites') {
    if (ratio > 1.05) return 'bad';
    if (ratio < 0.95) return 'good';
    return 'neutral';
  }
  // For time metrics, lower is better
  if (ratio > 1.05) return 'bad';
  if (ratio < 0.95) return 'good';
  return 'neutral';
}

function geoMean(values) {
  const valid = values.filter(v => v != null && v > 0 && isFinite(v));
  if (valid.length === 0) return null;
  const logSum = valid.reduce((s, v) => s + Math.log(v), 0);
  return Math.exp(logSum / valid.length);
}

function computeComparison(baseName, compareName) {
  const baseData = PAYLOAD.data[baseName] || {};
  const compareData = PAYLOAD.data[compareName] || {};
  const allModels = [...new Set([
    ...Object.keys(baseData), ...Object.keys(compareData)
  ])].sort();

  const rows = allModels.map(model => {
    const row = { model };
    METRICS.forEach(metric => {
      const bv = (baseData[model] || {})[metric];
      const cv = (compareData[model] || {})[metric];
      let ratio = null;
      if (bv != null && bv > 0 && cv != null && cv > 0) {
        ratio = cv / bv;
      }
      row[metric] = { base: bv, compare: cv, ratio };
    });
    return row;
  });

  const summary = {};
  METRICS.forEach(metric => {
    const ratios = rows.map(r => r[metric].ratio).filter(r => r != null);
    summary[metric] = geoMean(ratios);
  });

  return { rows, summary, allModels };
}

function renderSummary(summary) {
  const container = document.getElementById('summary');
  container.innerHTML = '';
  METRICS.forEach(metric => {
    const val = summary[metric];
    const card = document.createElement('div');
    card.className = 'summary-card';
    const cls = ratioClass(val, metric);
    const display = val != null ? val.toFixed(3) + 'x' : 'n/a';
    card.innerHTML = `
      <div class="metric-name">${METRIC_LABELS[metric]}</div>
      <div class="ratio ${cls}">${display}</div>
      <div class="detail">geometric mean ratio</div>`;
    container.appendChild(card);
  });
}

function renderCharts(comparison, baseName, compareName) {
  const container = document.getElementById('charts');
  container.innerHTML = '';
  METRICS.forEach(metric => {
    const models = [];
    const ratios = [];
    const colors = [];
    comparison.rows.forEach(row => {
      const r = row[metric].ratio;
      if (r == null) return;
      models.push(row.model);
      ratios.push(r);
      colors.push(r > 1.05 ? '#dc2626' : r < 0.95 ? '#16a34a' : '#94a3b8');
    });
    if (models.length === 0) return;

    const h2 = document.createElement('h2');
    h2.textContent = METRIC_LABELS[metric] + ' ratio (' + compareName + ' / ' + baseName + ')';
    container.appendChild(h2);

    const div = document.createElement('div');
    div.className = 'chart-container';
    div.style.height = Math.max(300, models.length * 28) + 'px';
    container.appendChild(div);

    Plotly.newPlot(div, [{
      type: 'bar',
      orientation: 'h',
      y: models,
      x: ratios,
      marker: { color: colors },
      hovertemplate: '%{y}<br>ratio: %{x:.3f}<extra></extra>',
    }], {
      paper_bgcolor: '#fff', plot_bgcolor: '#fff',
      font: { size: 11, family: 'system-ui' },
      margin: { l: 240, r: 20, t: 10, b: 40 },
      xaxis: {
        title: 'ratio (1.0 = no change)',
        zeroline: false,
        gridcolor: '#eee',
      },
      yaxis: { autorange: 'reversed', tickfont: { family: 'ui-monospace, monospace', size: 10 } },
      shapes: [{
        type: 'line', x0: 1, x1: 1, y0: -0.5, y1: models.length - 0.5,
        line: { color: '#333', width: 1.5, dash: 'dash' },
      }],
    }, { displayModeBar: false, responsive: true });
  });
}

let sortCol = null;
let sortAsc = true;

function renderTable(comparison, baseName, compareName) {
  const thead = document.querySelector('#table thead');
  const tbody = document.querySelector('#table tbody');

  // Header — two rows: metric name once (spanning its 3 sub-columns), then
  // short base/compare/ratio labels. The full experiment names are shown in
  // the base/compare dropdowns and chart titles, so repeating them per column
  // (and overflowing the fixed layout) is unnecessary.
  let h1 = '<tr><th data-col="model" rowspan="2">model</th>';
  let h2 = '<tr>';
  METRICS.forEach(metric => {
    h1 += `<th class="metric-group" colspan="3">${METRIC_LABELS[metric]}</th>`;
    h2 += `<th data-col="${metric}_base" class="grp-start"><small>base</small></th>`;
    h2 += `<th data-col="${metric}_compare"><small>compare</small></th>`;
    h2 += `<th data-col="${metric}_ratio"><small>ratio</small></th>`;
  });
  h1 += '</tr>';
  h2 += '</tr>';
  thead.innerHTML = h1 + h2;

  function renderRows(rows) {
    let html = '';
    // Summary row
    html += '<tr class="summary-row"><td>geometric mean</td>';
    METRICS.forEach(metric => {
      const gm = comparison.summary[metric];
      const cls = ratioClass(gm, metric);
      html += '<td class="grp-start"></td><td></td>';
      html += `<td class="${cls}">${gm != null ? gm.toFixed(3) + 'x' : 'n/a'}</td>`;
    });
    html += '</tr>';

    rows.forEach(row => {
      html += `<tr><td>${row.model}</td>`;
      METRICS.forEach(metric => {
        const d = row[metric];
        const cls = ratioClass(d.ratio, metric);
        html += `<td class="grp-start">${formatValue(metric, d.base)}</td>`;
        html += `<td>${formatValue(metric, d.compare)}</td>`;
        html += `<td class="${cls}">${d.ratio != null ? d.ratio.toFixed(3) + 'x' : 'n/a'}</td>`;
      });
      html += '</tr>';
    });
    tbody.innerHTML = html;
  }

  renderRows(comparison.rows);

  // Sortable columns
  thead.querySelectorAll('th').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (!col) return;  // metric-group spanning header is not sortable
      if (sortCol === col) { sortAsc = !sortAsc; }
      else { sortCol = col; sortAsc = true; }
      thead.querySelectorAll('th').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
      th.classList.add(sortAsc ? 'sort-asc' : 'sort-desc');

      const sorted = [...comparison.rows].sort((a, b) => {
        let va, vb;
        if (col === 'model') { va = a.model; vb = b.model; }
        else {
          const [metric, which] = col.match(/^(.+)_(base|compare|ratio)$/).slice(1);
          va = a[metric]?.[which]; vb = b[metric]?.[which];
        }
        if (va == null && vb == null) return 0;
        if (va == null) return 1;
        if (vb == null) return -1;
        if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
        return sortAsc ? va - vb : vb - va;
      });
      renderRows(sorted);
    });
  });
}

function renderExpInfo(baseName, compareName) {
  const container = document.getElementById('exp-info');
  container.innerHTML = '';
  [['base', baseName], ['compare', compareName]].forEach(([label, name]) => {
    const m = PAYLOAD.meta[name];
    const card = document.createElement('div');
    card.className = 'exp-info-card';
    let html = `<div class="exp-label">${label}</div>`;
    html += `<div class="exp-name">${name}</div>`;
    if (m && m.description) {
      html += `<div class="exp-desc">${m.description.replace(/\n/g, ' ').trim()}</div>`;
    }
    if (m && m.yaml_path) {
      const url = PAYLOAD.repo_url + '/blob/experiments/' + m.yaml_path;
      html += `<a href="${url}" target="_blank">${m.yaml_path}</a>`;
    }
    card.innerHTML = html;
    container.appendChild(card);
  });
}

function update() {
  const baseName = baseSel.value;
  const compareName = compareSel.value;
  writeHash();
  const comparison = computeComparison(baseName, compareName);
  document.getElementById('meta').textContent =
    ALL_EXPERIMENTS.length + ' experiments · ' +
    comparison.allModels.length + ' models';
  renderExpInfo(baseName, compareName);
  renderSummary(comparison.summary);
  renderCharts(comparison, baseName, compareName);
  renderTable(comparison, baseName, compareName);
}

baseSel.addEventListener('change', update);
compareSel.addEventListener('change', update);
update();
</script>
</body>
</html>
"""


def render(exp_data: dict, out: Path, repo_url: str) -> None:
    payload = {
        "metrics": [m[0] for m in METRIC_ORDER],
        "metric_labels": {m[0]: m[1] for m in METRIC_ORDER},
        "metric_units": {m[0]: m[2] for m in METRIC_ORDER},
        "data": exp_data["data"],
        "run_shas": exp_data["run_shas"],
        "meta": exp_data["meta"],
        "repo_url": repo_url,
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
    exp_data = load_experiments(src)
    if not exp_data["data"]:
        print("No experiment data found", file=sys.stderr)
        return 1

    repo_url = ""
    top_idx = src / "index.json"
    if top_idx.exists():
        repo_url = json.loads(top_idx.read_text()).get("project_url", "")

    render(exp_data, out, repo_url)
    n_exp = len(exp_data["data"])
    n_models = len(
        {m for d in exp_data["data"].values() for m in d}
    )
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB; {n_exp} experiments, {n_models} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
