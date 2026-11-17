# Benchmark harness & timeline

The catalogue doubles as an ASV benchmark suite that tracks PyMC / PyTensor
performance across releases. This document covers the harness, the metrics, the
branch/CI layout, and how to run or extend it. For the model set itself see the
[README](./README.md); for the curated benchmark subset see
[`BENCHMARK_CORE.md`](./BENCHMARK_CORE.md).

## Metrics

Each model is wrapped in an asv `ModelBench` class that captures five metrics:

- **`rewrite_time`** — wall-clock from calling `logp_dlogp_function` (or
  `compile_logp` for discrete models) to the moment it returns. Covers graph
  construction and the full rewriter pipeline.
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
  not a window: asv spawns a fresh process per benchmark, so it covers imports,
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

CI pins ASV to 0.6.6 and `asv-runner` to 0.2.5. Newer runners changed timing
benchmarks with `setup()` hooks to one call per sample, which defeats the eval
benchmark's steady-state batching. `sample_time = 0.1` averages many calls per
sample with the pinned runner; it does not fix batching with newer runners.

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
skip earlier single-call measurements. Refresh those with an eval-only run over
the selected releases, omitting that skip flag. This preserves the saved rewrite
and compile metrics and avoids repeating the Build benchmarks.

The manual **Refresh eval timings** workflow automates that refresh. It selects
successful core eval measurements with one call per sample and a per-call time
below the 0.1-second batching target, restricted to tagged releases. It verifies
that replacements are batched and that other measurements are unchanged before
publishing a complete site. A failed refresh leaves the published site intact.

After pushing these changes to `main`, run:

```bash
gh workflow run refresh-eval.yml --repo pymc-devs/pymc-model-catalogue --ref main
```

Add `-f dry_run=true` to list affected records without measuring or publishing.
The workflow uses Python 3.12 and 3.14 to repair both saved environments. Each
eval setup still compiles its function; the Build benchmarks are not run.

### Compile first, then reuse the cache

ASV's multiple timing rounds can run Eval before Build, regardless of class
name ordering. `scripts/run_benchmarks.py` therefore uses two ASV invocations
per release/environment: compile timings first with fresh PyTensor and Numba
cache directories, then all other metrics with the same populated caches.
RSS may use cached code. The directories are removed after both phases;
other `PYTENSOR_FLAGS` are preserved. `spawn` is the default launch method.

Compile time records cache counters and Numba compilation events. It fails if
the graph was already compiled, its dispatcher loads a disk-cache entry, or
the first call produces no compilation events. Its `cold-cache-v1` benchmark
version prevents reuse of earlier cached-load measurements. The trivial-model
prewarm still initializes LLVM and common helpers. Older PyTensor versions that
cannot cache a graph may compile again during other metrics.

To rerun the whole timeline after merging, use the existing **Benchmark**
workflow from `main` with **rerun** checked, or:

```bash
gh workflow run benchmark.yml --repo pymc-devs/pymc-model-catalogue \
  --ref main -f rerun=true
```

This starts a full rerun, or resumes the unfinished full rerun. Progress is
stored in `gh-pages/results/rerun-state`, alongside the saved ASV results.
Successful measurements are recorded individually, and the result tree plus
checkpoint is published after each compile/other-metrics phase. If a job times
out, the unfinished phase is repeated; published successful values are skipped.
Failed benchmarks remain pending for another attempt. Normal runs also resume
an active campaign, rather than accepting older successful values as refreshed.

An optional `rerun_id` names a campaign. Reuse its name to resume it, or choose
a different name to deliberately remeasure everything. Without a name,
`rerun=true` resumes an active campaign and starts a new one after the previous
campaign completes. Changes to benchmark/model code require a new campaign;
UI-only changes do not invalidate checkpointed measurements. No old values are
deleted when a campaign starts: they are replaced as measurements finish.

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
| `benchmark.yml` | push on `timeline`, weekly cron, manual dispatch | Compiles first, then runs other metrics using populated caches. Manual `rerun=true` starts/resumes a full timeline rerun; checkpoints after each phase. |
| `detect-releases.yml` | daily cron | Polls PyPI for new pymc minors, opens a PR against `timeline`. Merging the PR kicks off `benchmark.yml`. |
| `refresh-eval.yml` | manual dispatch | Replaces single-call core eval timings in tagged releases, verifies other measurements are unchanged, then publishes. Optional dry run lists the affected records. |
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
