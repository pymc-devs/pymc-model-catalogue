# Benchmark harness & timeline

The catalogue doubles as an ASV benchmark suite that tracks PyMC / PyTensor
performance across releases. This document covers the harness, the metrics, the
branch/CI layout, and how to run or extend it. For the model set itself see the
[README](./README.md); for the curated benchmark subset see
[`BENCHMARK_CORE.md`](./BENCHMARK_CORE.md).

## Metrics

Each model is wrapped in an asv `ModelBench` class that captures five metrics:

- **`rewrite_time`** — wall-clock from calling the shared `build_logp_fn`
  helper to the moment it returns. Covers graph construction and the full
  rewriter pipeline.
- **`compile_time`** — wall-clock from the end of the rewrite phase to the end
  of the first `f(x)` call. Captures the NUMBA JIT cost, which is triggered on
  first call; the one eval's worth of arithmetic inside is microseconds, so the
  metric reflects compile cost in practice. The measured model has an empty
  disk cache; a trivial-model prewarm excludes compiler initialization.
- **`n_rewrites`** — number of `rewriting: ...` lines emitted under
  `pytensor.config.optimizer_verbose = True`. A proxy for how many rewrites the
  graph went through; shifts when rewrite rules are added or removed.
- **`peak_rss`** — the highest resident set size, in bytes, that building and
  compiling this model drove the process to. A whole-process high-water mark,
  not a window: the runner spawns a fresh process per model, so it covers imports,
  `_prewarm`, graph construction, rewrites and the NUMBA JIT together. That
  makes it the "will this fit on the box" number. Memory moves independently of
  time — the numba backend has had regressions where peak RSS grew several-fold
  on a model whose compile time barely moved, and on a CI runner that is the
  difference between a slow benchmark and an OOM-killed one.
- **`time_eval`** — steady-state per-call time, measured by asv's native timing
  machinery.

### On the RSS metric

Compile is where the memory goes: a model that needs gigabytes needs them
during the JIT, not while evaluating, and evaluation is dominated by the same
logp/dlogp work whether a sampler is driving it or not. So one number covers
the run rather than splitting build and eval.

It is **absolute**, deliberately — the question is whether a machine can run
the model, and the interpreter and imports are part of that. `build_and_measure`
also returns the compile-window peak and its delta over the pre-compile
baseline, for when the split matters, but those are not plotted.

The value comes from `resource.getrusage(...).ru_maxrss`, which cannot miss a
transient, floored by a `/proc/self/statm` sampler (200 ms) over the compile
window so the metric stays meaningful if a process is ever reused instead of
spawned. `tracemalloc` is no help either way: the spike is LLVM/numba native
memory, not Python allocations. On non-Linux there is no sampler and
`ru_maxrss` is used alone.

The five metrics are tracked on a curated subset of 26 models listed in
[`BENCHMARK_CORE.md`](./BENCHMARK_CORE.md), chosen to give broad coverage
(hierarchical, GP, scan, linear algebra, survival, ODE, mixtures, discrete).

**Dashboard:** https://pymc-devs.github.io/pymc-model-catalogue/dashboard.html
**Experiments:** https://pymc-devs.github.io/pymc-model-catalogue/experiments.html

## Running locally

CI pins ASV to 0.6.6 and `asv-runner` to 0.2.5. The model worker uses this
runner's native eval batching and statistics, with `sample_time = 0.1`.
Its timing sampler has no model setup hook: every sample calls the same
already-compiled function.

Both the ASV tool and the runner inside benchmark environments must use the
constraint. Before a local run, set:

```bash
export PIP_CONSTRAINT="$PWD/asv_bench/runner-constraints.txt"
export UV_CONSTRAINT="$PIP_CONSTRAINT"
uv tool install --force 'asv==0.6.6' --constraint "$PIP_CONSTRAINT"
```

CI copies that file from `main` into the runner's temporary directory so the
constraint survives switching branches and checking out historical commits.
These are tooling pins; release-date resolution of model dependencies is kept.

The pin does not refresh saved results: `--skip-existing-successful` will still
skip earlier successful measurements of the current benchmark version.
Use the **Benchmark** workflow's full rerun option, described below, to refresh
the saved measurements.

### One target build per model

`scripts/run_benchmarks.py` starts one ASV worker per model/environment with
fresh PyTensor and Numba cache directories. The worker first builds and calls
a tiny, unrelated Normal model to initialize LLVM and common compiler helpers,
outside the target timers. It then builds the target exactly once:

1. Graph construction and rewriting provide `rewrite_time` and `n_rewrites`.
2. The first call provides `compile_time` and completes the RSS measurement.
3. ASV's native eval sampler repeatedly calls that same compiled function.

The wrapper saves all five metrics under their existing ASV benchmark names,
including eval samples and statistics. Rewrite time, compile time, rewrite count,
and RSS each use one observation. Only eval is sampled repeatedly. Existing eval
round/repeat/batching settings are retained, but all rounds share the callable.
This also works on older PyTensor releases without a compiled-graph disk cache.
Caches are removed after each model; other `PYTENSOR_FLAGS` are preserved.
`spawn` is the default launch method. Use the wrapper rather than `asv run`
directly for these model benchmarks.

Compile time records cache counters and Numba compilation events. It fails if
the target was already compiled, its dispatcher loads a disk-cache entry, or
the first call produces no compilation events. Eval fails if any new Numba
compilation occurs during its samples. All five metrics use the benchmark
version `single-build-v1`, distinguishing this protocol from older measurements.
The compiler warmup is excluded from target rewrite/compile times, but included
in the process RSS high-water mark.

To rerun the whole timeline after merging, use the existing **Benchmark**
workflow from `main` with **rerun** checked, or:

```bash
gh workflow run benchmark.yml --repo pymc-devs/pymc-model-catalogue \
  --ref main -f rerun=true
```

This starts a full rerun, or resumes the unfinished full rerun. Progress is
stored in `gh-pages/results/rerun-state`, alongside the saved ASV results.
Successful measurements are recorded individually, and the result tree plus
checkpoint is published after each model. If a job times out, the unfinished
model is repeated; published successful values are skipped.
Failed benchmarks remain pending for another attempt. Normal runs also resume
an active campaign, rather than accepting older successful values as refreshed.
Completion requires both a success recorded in this campaign and a saved finite
result with the current benchmark version. ASV rounding durations or statistics
while saving another model does not invalidate completed measurements.

An optional `rerun_id` names a campaign. Reuse its name to resume it, or choose
a different name to deliberately remeasure everything. Without a name,
`rerun=true` resumes an active campaign and starts a new one after the previous
campaign completes. Changes to benchmark/model code require a new campaign;
UI-only and checkpoint-only changes do not invalidate completed measurements.
Older checkpoints are upgraded only after verifying that all other benchmark
files match; an incompatible explicitly named campaign is rejected rather than
silently restarted. No old values are deleted when a campaign starts: they are
replaced as measurements finish.

This checks out `timeline` even when dispatched from `main`. Each release
reuses its successful saved Python 3.12/3.14 environments; new releases use
Python 3.12 before PyMC 6 and Python 3.14 thereafter. The environment list is
saved in the checkpoint, so a failed attempt cannot silently drop an environment
from the next run. The timeline and its tags are preserved.

```bash
# Smoke-test a single model directly (no asv, no timeline):
python models/stochastic_volatility.py

# Run asv against the current .venv for a curated subset:
python scripts/run_benchmarks.py --bench "$(python scripts/core_models.py)" \
  --launch-method=spawn --environment=existing:$(which python) \
  --set-commit-hash $(git rev-parse HEAD)
```

## Posteriordb model/data pairs

See [`POSTERIORDB_MODELS.md`](./POSTERIORDB_MODELS.md) for the model/data index.
Timings in model docstrings come from short local NUMBA runs. Use ASV for performance comparisons.

```bash
# Check the original and frozen graphs for one fixed pair.
python models/posteriordb_blr_sblri.py

# Select one pair in ASV.
python scripts/run_benchmarks.py --launch-method=spawn --environment="existing:$(command -v python)" \
  --set-commit-hash "$(git rev-parse HEAD)" \
  --bench 'ModelBench(?:Build|Eval)_models_posteriordb_blr_sblri\.'

# Select every posteriordb pair, including the discrete entry.
python scripts/run_benchmarks.py --launch-method=spawn --environment="existing:$(command -v python)" \
  --set-commit-hash "$(git rev-parse HEAD)" \
  --bench 'ModelBench(?:Build|Eval)_models(?:_discrete)?_posteriordb_'
```

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
| `benchmark.yml` | push on `timeline`, weekly cron, manual dispatch | Builds each model once and samples eval from its compiled callable. Manual `rerun=true` starts/resumes a full timeline rerun; checkpoints after each model. |
| `detect-releases.yml` | daily cron | Polls PyPI for new pymc minors, opens a PR against `timeline`. Merging the PR kicks off `benchmark.yml`. |
| `experiment.yml` | push on `experiments`, manual dispatch | Runs each changed YAML, keyed by the YAML file's last-touching git SHA for dedup. |

Each timeline checkpoint regenerates `dashboard.html` and, when experiment data
exists, `experiments.html` before committing and pushing the site. Intermediate
updates therefore include the custom pages throughout a benchmark or backfill
run. If rendering fails, that update is not pushed and the previous site remains
published.

The custom dashboard plots each benchmark environment separately, with
consistent colors. Each chart labels its newest successful release; ties
prefer the newer Python version. A run on a newer Python version keeps the
historical series visible rather than replacing it or joining measurements from
different Python versions into a single line.
Tooltips show the PyTensor and Numba versions recorded by the benchmark worker
for each measurement. A partial rerun updates the versions for the measurements
it replaces. Older published points use the version backfill from the CI logs.

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
