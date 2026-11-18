# Progress Tracker

Tracks extraction progress across PLAN.md's notebook inventory. Note that one notebook
can produce multiple model files (e.g. `multilevel_modeling.ipynb` → 8 files,
`frailty_models.ipynb` → 5 files), so the primary metric is **model files**, not notebooks.

## Summary

| Metric | Count |
|---|---|
| Model files done (continuous) | 186 |
| Model files done (discrete) | 10 |
| **Total model files done** | **196** |
| Skipped model files (`.skip`) | 4 |
| Notebooks fully addressed | 140 / 140 |
| Notebooks remaining | 0 |

**Do not re-attempt any notebook listed under "Skipped"** — reasons are listed per item and
won't change unless a dependency or upstream bug is fixed.

**Numba compilation note:** PyTensor's NUMBA mode falls back to object mode for Ops that
can't be natively compiled (SciPy ODE/SDE solvers, custom Python `Op`s, JAX Ops, etc.),
so these models are still benchmarkable — they just won't get Numba's full speedup.
A notebook should only be skipped for a *true* blocker (missing PyMC model, missing
dependency, PyMC/PyTensor bug), not for "has a non-Numba-native Op".

## Per-category breakdown

| Category | Notebooks | Notebooks done | Model files | Skipped files | Notebooks remaining |
|---|---|---|---|---|---|
| BART | 4 | 0 | 0 | 4 | 0 |
| Case Studies | 16 | 16 | 18 | 0 | 0 |
| Causal Inference | 11 | 9 | 11 | 2 | 0 |
| Diagnostics | 4 | 4 | 8 | 0 | 0 |
| Fundamentals | 1 | 1 | 4 | 0 | 0 |
| Gaussian Processes | 17 | 12 | 17 | 1 | 0 |
| GLMs | 15 | 14 | 30 | 0 | 0 |
| How-To | 12 | 12 | 18 | 1 | 0 |
| Introductory | 1 | 1 | 1 | 0 | 0 |
| Mixture Models | 5 | 5 | 6 | 0 | 0 |
| ODE Models | 4 | 4 | 7 | 0 | 0 |
| Samplers | 8 | 8 | 10 | 0 | 0 |
| Spatial | 3 | 3 | 3 | 0 | 0 |
| Statistical Rethinking | 19 | 16 | 19 | 1 | 0 |
| Survival Analysis | 6 | 6 | 18 | 0 | 0 |
| Time Series | 9 | 9 | 22 | 0 | 0 |
| Variational Inference | 5 | 5 | 4 | 0 | 0 |
| **Total** | **140** | **125** | **196** | **9** | **0** |

Rows where "Notebooks done" < (notebooks − skipped) because one notebook can yield zero
extractable models (tutorial-only content counted as "done" when confirmed but produces 0
files — handled case-by-case below).

## Skipped — permanent

Reasons in three buckets. None should be retried without first fixing the blocker.

### Missing PyMC model (no inference graph)
- **GP-Circular** — PyMC3/theano only; not ported to v5
- **GP-MaunaLoa2** — PyMC3/theano only; not ported to v5
- **GP-SparseApprox** — PyMC3/theano only; not ported to v5
- **GP-MeansAndCovs** — reference tutorial, no complete inference model
- **interventional_distribution** — generative-only, no observed data

Also blocked on upstream dependencies (re-attempt once fixed):
- **All BART notebooks** (`bart_categorical_hawks`, `bart_heteroscedasticity`, `bart_introduction`, `bart_quantile_regression`) and **bayesian_nonparametric_causal** — `pymc_bart` is not yet compatible with the dev PyMC version used here. Numba object-mode fallback would otherwise handle the BART Op, but the package won't import. Re-attempt once pymc_bart catches up to dev PyMC.
- **Statistical Rethinking lectures 02, 06, 20** — discussion-only, no `pm.Model`

### Missing dependencies
- **wrapping_jax_function** (`wrapping_jax_function_normal.py.skip`) — requires `jax` (not installed in .venv)
- **GLM-ordinal-features** — requires R data file not available locally

### PyMC / PyTensor bugs
- **bayesian_ab_testing_introduction Bernoulli variant** (`ab_testing_bernoulli.py.skip`) — `pm.Beta(100, 100)` produces `inf` logp. Revenue variant extracted normally.
- **GP-TProcess** (`gp_tprocess_poisson.py.skip`) — internal PyMC error in `MvStudentT` rv_op
- **Statistical Rethinking sr19 child strategy** (`sr19_child_strategy.py.skip`) — broadcast shape mismatch in `CustomDist`. Other sr19 models extracted normally.

## Remaining (0 notebooks)

All 139 notebooks from PLAN.md have been addressed (extracted, tutorial-only, or permanently skipped).

Addressed with no files extracted:
- **howto_debugging** — tutorial-only. Both `pm.Model` blocks are deliberately broken
  demos illustrating `pytensor.printing.Print` (first uses `pm.Normal` for `sd` allowing
  negatives; second uses `a/b` as `sigma`). Neither is a real working model.
- **empirical-approx-overview** — tutorial-only. Toy `NormalMixture` / `MvNormal` with
  no observed data, used only to produce a NUTS trace for `pm.Empirical` API demonstration.
- **pathfinder** — duplicate. Same non-centered eight-schools model as the existing
  `eight_schools_noncentered.py` (minor prior tweak only); the notebook only demonstrates
  `pmx.fit(method="pathfinder")` which is a sampler concern orthogonal to the logp graph.

