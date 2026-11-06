# Benchmark harness & timeline

The catalogue doubles as an ASV benchmark suite that tracks PyMC / PyTensor
performance across releases. This document covers the harness, the metrics, the
branch/CI layout, and how to run or extend it. For the model set itself see the
[README](./README.md); for the curated benchmark subset see
[`BENCHMARK_CORE.md`](./BENCHMARK_CORE.md).

## Metrics

Each model is wrapped in an asv `ModelBench` class that captures four metrics:

- **`rewrite_time`** measures the call to `models._benchmark.build_logp_fn`.
  It includes graph construction, gradients, rewrites, and backend compilation before return.
  NUMBA defers JIT compilation until the first call.
- **`compile_time`** keeps its historical identifier and measures the first function call.
  NUMBA includes JIT compilation and one evaluation in this metric.
  FAST_RUN measures only its first evaluation here.
- **`n_rewrites`** — number of `rewriting: ...` lines emitted under
  `pytensor.config.optimizer_verbose = True`. A proxy for how many rewrites the
  graph went through; shifts when rewrite rules are added or removed.
- **`time_eval`** — steady-state per-call time, measured by asv's native timing
  machinery.

The four metrics use the curated core subset in [`BENCHMARK_CORE.md`](./BENCHMARK_CORE.md).
The existing core selector and historical benchmark names stay unchanged.

**Dashboard:** https://pymc-devs.github.io/pymc-model-catalogue/dashboard.html
**Experiments:** https://pymc-devs.github.io/pymc-model-catalogue/experiments.html

## Running locally

```bash
# Smoke-test a single model directly (no asv, no timeline):
python models/stochastic_volatility.py

# Run asv against the current .venv for a curated subset:
asv run --launch-method=spawn --environment=existing:$(which python) \
  --set-commit-hash $(git rev-parse HEAD) \
  --bench "$(python scripts/core_models.py)"
```

## Optional native PyMC6 workloads

The optional package contains 121 native PyMC models, 148 model/data pairs,
and 91 bundled processed datasets.
The registry supplies each workload's module, canonical posterior ID, and backend.
Model construction and dataset loading occur outside timed compilation.

```bash
# List canonical pairs without constructing models.
pixi run python -c 'from models.posteriordb import list_posteriors; print(*list_posteriors(), sep="\n")'

# Evaluate the original and frozen graphs without inference.
pixi run python -m models.posteriordb.blr --posterior sblri-blr

# Select one exact posterior/backend pair.
pixi run asv run --launch-method=spawn \
  --environment="existing:$(pixi run python -c 'import sys; print(sys.executable)')" \
  --set-commit-hash "$(git rev-parse HEAD)" \
  --bench 'ModelBench(?:Build|Eval)_models_posteriordb__sblri__blr__NUMBA\.'

# Select all 148 optional pairs.
pixi run asv run --launch-method=spawn \
  --environment="existing:$(pixi run python -c 'import sys; print(sys.executable)')" \
  --set-commit-hash "$(git rev-parse HEAD)" \
  --bench 'ModelBench(?:Build|Eval)_models_posteriordb__'
```

The ASV suffix replaces the posterior hyphen with double underscores and appends the declared backend.
For example, `sblri-blr` uses `ModelBenchBuild_models_posteriordb__sblri__blr__NUMBA`.
The dashboards restore the canonical posterior and show the backend separately.

NUMBA is the default backend. Some native operations need FAST_RUN.
The registry declares those workloads explicitly.
A backend failure never selects another backend.

`FAST_RUN` uses an explicit CVM linker because PyTensor's automatic linker can select NUMBA.
Native ODE callbacks use the same explicit CVM linker.
Their compilation occurs during model construction, before the measured phases.

FAST_RUN compiles during `rewrite_time`.
Its `compile_time` measures first evaluation, not NUMBA JIT compilation.
The historical metric key stays `track_compile_time`. Its visible dashboard label is `first_call_time`.

Keep backend identities separate when you interpret results.
New workloads derive gradient eligibility from actual value-variable dtypes.
Existing `models_discrete/` workloads retain their logp-only contract.
Observed discrete outcomes do not disable gradients.

These timings do not measure inference or posterior accuracy.


## Branches

- **`main`** — source of truth: asv infra (`asv_bench/`), the model set
  (`models/`, `models_discrete/`), CI workflows, and the experiment scaffolding
  (`experiments/build.py`). Nothing here is benchmark-result state.
  `versions/tracked_versions.json` ships empty; the ledger is advanced on
  `timeline`.
- **`timeline`** — append-only. Each commit adds one pymc X.Y.0 release to the
  ledger and becomes one data point on the dashboard. Created and advanced by
  `backfill.yml` / `detect-releases.yml`; do not commit here by hand.
- **`experiments`** — holds `experiments/*.yaml` specs and their
  `experiments/patches/` files. Rebased onto `main` whenever the schema or
  infrastructure moves. Experiment results live under
  `gh-pages/experiments/<name>/<short>/`, keyed by the last git SHA that touched
  the YAML file.
- **`gh-pages`** — the published dashboard. Written only by CI.

## Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `backfill.yml` | manual dispatch | Walks the timeline a `step` at a time. First dispatch creates the branch; `reset=true` wipes it back to main. |
| `benchmark.yml` | push on `timeline`, weekly cron | Walks any new commits with `asv run --skip-existing-commits` and publishes after each — partial progress is durable across a single failing release. |
| `detect-releases.yml` | daily cron | Polls PyPI for new pymc minors, opens a PR against `timeline`. Merging the PR kicks off `benchmark.yml`. |
| `experiment.yml` | push on `experiments`, manual dispatch | Runs each changed YAML, keyed by the YAML file's last-touching git SHA for dedup. |

Each timeline publish regenerates `dashboard.html` and, when experiment data
exists, `experiments.html` before committing and pushing the site. Intermediate
updates therefore include the custom pages throughout a benchmark or backfill
run. If rendering fails, that update is not pushed and the previous site remains
published.

The custom dashboard plots each benchmark environment separately, with labels
and consistent colors. A run on a newer Python version therefore keeps the
historical series visible rather than replacing it or joining measurements from
different Python versions into a single line.

The timeline CI uses a custom install hook (`asv_bench/_provision.py`) that
installs `pymc==<pinned>` with `uv pip install --exclude-newer <released_at + 1
day>`, so pytensor and every transitive dep resolve to what was latest on each
pymc release day — historical data points stay reproducible.

## Adding an experiment

1. Check out the `experiments` branch (rebase on main first if stale).
2. Copy `base.yaml` to `<your_experiment>.yaml` and add `revert_commits` or
   `patches` on either the `pytensor` or `pymc` block.
3. Patches are `git format-patch`-style files — generate them from a
   manually-resolved clone and commit them under
   `experiments/patches/<your_experiment>/`.
4. Optionally add a `smoke_test:` block — Python source run with the built venv
   right after it's created. A non-zero exit (e.g. a failed `assert`) fails the
   build before any benchmarking, so it's the place to verify the
   refs/reverts/patches produced the source state you expected (e.g. a feature's
   marker Op is — or, for the control arm, is not — present in a trivial compiled
   graph).
5. Push the branch; `experiment.yml` picks it up via the path filter.

Experiments layer modifications on top of the `pytensor.ref` / `pymc.ref` pins.
`build.py` clones both packages locally and installs them editable so pymc and
pytensor can be patched symmetrically.

## Requirements

Local development uses current pymc/pytensor branches:

```bash
uv pip install "pymc @ git+https://github.com/pymc-devs/pymc.git@main" \
               "pytensor @ git+https://github.com/pymc-devs/pytensor.git@main" \
               numba asv
```
