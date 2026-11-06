# Repository Guidelines

## Project Overview

This catalogue contains standalone PyMC models and an ASV suite for PyMC/PyTensor performance across releases.
Model modules separate model construction from sampling and plotting. Benchmark checks compile and evaluate log-probability functions.

## Architecture & Data Flow

- Each model exposes `build_model()` and returns `(model, initial_point)`.
- `models/_benchmark.py` builds compiled logp/gradient functions for original models and models with frozen dimensions and data.
- `asv_bench/benchmarks/bench_models.py` discovers non-underscore Python modules in both model directories and generates benchmark classes.
- `asv_bench/_runner.py` measures rewrite time, first-call compilation time, rewrite counts, and evaluation. It checks finite outputs.
- `asv_bench/_provision.py` reads the release ledger and installs historical dependencies with a release-date cutoff.
- ASV writes results, then publishes graph JSON. Dashboard scripts convert these artifacts into HTML and keep benchmark environments separate.
- `experiments/build.py` builds environments from YAML specifications with pinned PyMC/PyTensor revisions, reverts, and patches.

## Key Directories

| Path | Purpose |
| --- | --- |
| `models/` | Models for the gradient benchmark path, shared benchmark helpers, and bundled data. |
| `models_discrete/` | Models for the logp-only path, including discrete free variables. |
| `asv_bench/` | ASV discovery, measurement, and historical environment provisioning. |
| `scripts/` | Benchmark selection, dashboards, release detection, and timeline commits. |
| `experiments/` | Environment builder. Experiment YAML and patches live on the `experiments` branch. |
| `tests/` | Dashboard data and payload regression tests. |
| `versions/` | Release ledger. `main` ships an empty ledger, while `timeline` advances it. |
| `.github/workflows/` | Benchmark, experiment, release, and dashboard automation. |

## Development Commands

Run commands from the repository root inside an existing project environment. Do not use system Python.
The documented development dependencies follow the current upstream branches:

```bash
uv pip install "pymc @ git+https://github.com/pymc-devs/pymc.git@main" \
  "pytensor @ git+https://github.com/pymc-devs/pytensor.git@main" numba asv
```

| Task | Command |
| --- | --- |
| Run one model | `python models/GLM_binomial_regression.py` |
| Run unit tests | `python -m unittest discover -s tests` |
| Show the default benchmark selector | `python scripts/core_models.py` |
| Publish local ASV results | `asv publish` |
| Render the timeline dashboard | `python scripts/make_dashboard.py .asv/html .asv/html/dashboard.html` |
| Render experiment comparisons | `python scripts/make_experiment_dashboard.py PATH_TO_GH_PAGES OUTPUT_HTML` |

Run the current benchmark subset against the active environment:

```bash
asv run --launch-method=spawn --environment="existing:$(command -v python)" \
  --set-commit-hash "$(git rev-parse HEAD)" \
  --bench "$(python scripts/core_models.py)"
```

The repository has no package build step or configured lint/format command. `asv publish` builds the static benchmark site.
Dashboard rendering needs published ASV data. Experiment rendering needs the experiment results tree, not only timeline results.

## Code Conventions & Common Patterns

- Preserve model docstrings with source references, authors, and extraction changes.
- Follow nearby Python style and four-space indentation. Use `snake_case` functions and preserve existing model filename families, such as `GLM_*`.
- Use `models/GLM_binomial_regression.py` as a compact builder example. Construct models inside a `pm.Model` context.
- Keep data preparation inside builders. Use seeded generators for synthetic data and module-relative paths for bundled files.
- Use coordinates, `dims`, and `pm.Data` where the model needs named axes or mutable data.
- Preserve the returned initial point. Some builders reset `rvs_to_initial_values`, so a second `initial_point()` call changes benchmark inputs.
- Place discrete-value models in `models_discrete/`. The ASV runner selects `with_grad=False` from this module prefix.
- Keep direct execution under `if __name__ == "__main__":` and use the shared `run_benchmark` helper.
- Keep compilation failures visible. The runner checks finite logp and gradient outputs rather than substituting fallback results.
- Benchmark classes use `ModelBenchBuild_<module>` and `ModelBenchEval_<module>`, with module dots replaced by underscores.
- Preserve separate build and evaluation measurements. ASV repeats builds for each metric to avoid stale cross-commit cache results.
- The benchmark path uses synchronous functions and explicit model/function objects. It has no async framework or dependency-injection container.

## Important Files

- `README.md`: purpose, model conventions, and development dependencies.
- `BENCHMARKING.md`: metric definitions, local commands, branch roles, and experiment procedure.
- `scripts/core_models.py`: executable benchmark selection. Its default lists 11 models, while `BENCHMARK_CORE.md` documents 26.
- `asv.conf.json`: `timeline` branch selection, virtualenv provisioning hook, and `.asv/{env,results,html}` paths.
- `versions/tracked_versions.json`: ordered release entries with `pymc` and `released_at` fields.
- `scripts/make_dashboard.py` and `scripts/make_experiment_dashboard.py`: published dashboard data contracts.
- `PLAN.md` and `PROGRESS.md`: model extraction history, not current tooling specifications.

## Runtime/Tooling Preferences

- Use Python with PyMC, PyTensor, Numba, and ASV. The repository has no root dependency manifest, lockfile, or general Python version pin.
- Use `uv` for the documented dependency installation and ASV provisioning. YAML experiment helpers also need PyYAML.
- ASV manages historical virtualenvs. Reuse an existing environment for local benchmarks instead of the empty `main` release ledger.
- Only the release-detection workflow explicitly pins Python 3.13. Do not treat that pin as a catalogue-wide runtime constraint.
- Source changes belong on `main`. CI manages `timeline` and `gh-pages`. Experiment specifications belong on `experiments`.
- Release and backfill scripts change the ledger or Git history. Do not use them as routine checks.
- Publishing workflows need authenticated write access. Keep local verification separate from remote publishing.

## Testing & QA

- Use standard-library `unittest`, not pytest. `tests/test_make_dashboard.py` imports the dashboard script directly by file path.
- Use `TemporaryDirectory` and small ASV-style JSON fixtures, as the existing tests do.
- Current tests check historical measurements, overlapping environments, tag filtering, version ordering, and empty dashboard payloads.
- The repository sets no coverage threshold and configures no coverage tool. These tests do not check browser rendering or model correctness.
- For model or harness changes, run an affected model directly. Use a focused ASV selector if the measurement path changes.
- Direct model checks report logp for original and frozen models. They do not establish posterior sampling quality.
- For dashboard changes, run the unit suite and inspect generated HTML with representative published results.
