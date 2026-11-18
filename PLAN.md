# PyMC Model Catalogue — Numba Compilation Benchmark Suite

## Context

Standalone PyMC model files extracted from ALL notebooks in `../pymc-examples/`.
Benchmarks PyTensor's NUMBA mode compilation and evaluation of logp/dlogp functions.
Every model is benchmarked both before and after `freeze_dims_and_data`, at the same initial point.

**Progress tracking:** See `PROGRESS.md` for the current state of done / permanently-skipped /
remaining notebooks and model files. Always consult it before picking a new batch so you
don't retry notebooks that were already skipped for permanent reasons (missing deps, PyMC
bugs, or no extractable model). Update it after each batch.

## Project Structure

```
pymc-model-catalogue/
├── models/                          # Continuous-only models (logp + dlogp)
│   ├── _benchmark.py                # Shared benchmark helper (run_benchmark)
│   ├── data/                        # .npz files for large datasets
│   └── *.py
├── models_discrete/                 # Models with discrete free variables (logp only)
│   ├── _benchmark.py                # Same helper (copy)
│   ├── data/
│   └── *.py
├── run_all.py                       # Parallel runner (1 CPU per model)
├── review_questions.md              # Uncertain items for user review
└── .venv/                           # Already exists
```

## Model File Template

### Continuous models (`models/`)

```python
"""
Model: <descriptive name>
Source: pymc-examples/<path/to/notebook.ipynb>, Section: "<section heading>"
Authors: <from `:author:` in notebook's `:::{post}` directive; if missing, top git contributors>
Description: <1-2 sentences>

Changes from original:
- <list changes or "None">

Benchmark results:
- Original:  logp = <value>, grad norm = <value>, <X.X> us/call (<N> evals)
- Frozen:    logp = <value>, grad norm = <value>, <X.X> us/call (<N> evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    # All data loading/generation happens inside build_model, not at module scope
    ...

    with pm.Model() as model:
        ...  # keep initval= as in the original notebook if present

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
```

### Discrete models (`models_discrete/`)

Only for models with discrete **free** variables (unobserved — they block NUTS). Observed discrete likelihoods (e.g. `pm.OrderedLogistic`, `pm.Bernoulli`, `pm.Poisson` with `observed=`) do NOT count — they're conditioned on data and the remaining continuous latents still have well-defined gradients, so such models belong in `models/`.

Same docstring but add `Has discrete variables: Yes (<which>)`. Uses `compile_logp` (no gradient):

```python
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model, discrete=True)
```

### Data handling

- **Small data** (< ~200 values): inline as numpy arrays in `build_model()`
- **Large data**: save as `.npz` in `models/data/` or `models_discrete/data/`, load via `np.load(Path(__file__).parent / "data" / "<name>.npz", allow_pickle=True)` (allow_pickle needed for string arrays like date coords)

### Assisting code (helpers, classes, etc.)

- **Inline everything inside `build_model()`** — just like data. This includes helper functions, custom classes (e.g., Statespace-style model subclasses), custom distributions, Op definitions, and any other supporting code from the original notebook.
- Each model `.py` file must contain **exactly one function** in its main body: `build_model()`. No module-level helper functions, no module-level classes, no module-level constants beyond imports.
- **No global state and no import-time side effects.** Module scope should only contain the docstring, imports, `build_model()`, and the `if __name__ == "__main__"` block. Nothing runs until `build_model()` is called.

## Faithfulness Rules

- **Stay as faithful to the original model as possible** — these represent how real users write models. Do not "improve", simplify, or remove seemingly redundant code. Keep `pm.Data` wrappers, keep `pm.Deterministic`, keep verbose parameterizations, keep coords/dims, keep `initval=`, etc.
- The **only** changes allowed are:
  1. Inlining external data (or saving to .npz)
  2. Removing sampling/plotting code
  3. Fixing genuine API breakage with the current dev codebase (documented in docstring)
  4. Adding the `ip` capture + initval clearing boilerplate at end of `build_model()`

## Model Selection — ALL notebooks (~139 total)

Extract every final/complete model from every notebook:
- **Skip intermediate models** that are only building blocks toward a final model in the same notebook
- **Skip deliberately broken models** used to demonstrate failure points
- **Include all complete, working models** — one file per model
- Multiple models from one notebook get separate files (e.g., `ar1.py`, `ar2.py`)

### Full notebook inventory by category:

**BART (4)**: bart_categorical_hawks, bart_heteroscedasticity, bart_introduction, bart_quantile_regression

**Case Studies (16)**: BEST, bayesian_sem_workflow, bayesian_workflow, binning, CFA_SEM, factor_analysis, GEV, hierarchical_partial_pooling, item_response_nba, occupancy, probabilistic_matrix_factorization, putting_workflow, reinforcement_learning, reliability_and_calibrated_prediction, rugby_analytics, ssm_hurricane_tracking

**Causal Inference (11)**: bayesian_ab_testing_introduction, bayesian_nonparametric_causal, counterfactuals_do_operator, difference_in_differences, excess_deaths, GLM-simpsons-paradox, interrupted_time_series, interventional_distribution, mediation_analysis, moderation_analysis, regression_discontinuity

**Diagnostics (4)**: Bayes_factor, Diagnosing_biased_Inference_with_Divergences, model_averaging, sampler-stats

**Fundamentals (1)**: data_container

**Gaussian Processes (17)**: gaussian_process, GP-Births, GP-Circular, GP-Heteroskedastic, GP-Kron, GP-Latent, GP-Marginal, GP-MaunaLoa, GP-MaunaLoa2, GP-MeansAndCovs, GP-smoothing, GP-SparseApprox, GP-TProcess, HSGP-Advanced, HSGP-Basic, log-gaussian-cox-process, MOGP-Coregion-Hadamard

**GLMs (15)**: GLM-binomial-regression, GLM-discrete-choice_models, GLM-hierarchical-binomial-model, GLM-missing-values-in-covariates, GLM-model-selection, GLM-negative-binomial-regression, GLM-ordinal-features, GLM-ordinal-regression, GLM-out-of-sample-predictions, GLM-poisson-regression, GLM-robust, GLM-robust-with-outlier-detection, GLM-rolling-regression, GLM-truncated-censored-regression, multilevel_modeling

**How-To (12)**: blackbox_external_likelihood_numpy, copula-estimation, howto_debugging, hypothesis_testing, LKJ, marginalizing-models, Missing_Data_Imputation, model_builder, profiling, spline, updating_priors, wrapping_jax_function

**Introductory (1)**: api_quickstart

**Mixture Models (5)**: dependent_density_regression, dirichlet_mixture_of_multinomials, dp_mix, gaussian_mixture_model, marginalized_gaussian_mixture_model

**ODE Models (4)**: ODE_API_introduction, ODE_API_shapes_and_benchmarking, ODE_Lotka_Volterra_multiple_ways, ODE_with_manual_gradients

**Samplers (8)**: DEMetropolisZ_EfficiencyComparison, DEMetropolisZ_tune_drop_fraction, fast_sampling_with_jax_and_numba, lasso_block_update, sampling_compound_step, sampling_conjugate_step, SMC2_gaussians, SMC-ABC_Lotka-Volterra_example

**Spatial (3)**: conditional_autoregressive_priors, malaria_prevalence, nyc_bym

**Statistical Rethinking (20)**: lectures 02 through 20

**Survival Analysis (6)**: bayes_param_survival, censored_data, frailty_models, hospital_length_of_stay, survival_analysis, weibull_aft

**Time Series (9)**: Air_passengers-Prophet, AR, bayesian_var_model, Euler-Maruyama_and_SDEs, Forecasting_with_structural_timeseries, longitudinal_models, MvGaussianRandomWalk_demo, stochastic_volatility, Time_Series_Generative_Graph

**Variational Inference (5)**: bayesian_neural_network_advi, empirical-approx-overview, GLM-hierarchical-advi-minibatch, pathfinder, variational_api_quickstart

## Parallel Runner (`run_all.py`)

- Subprocess isolation per model (numba compilation can't share state)
- 1 CPU enforced via env vars (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`)
- 600s timeout for numba first-compilation overhead
- Reports pass/fail summary

## Implementation Strategy

Work through notebooks in batches by category, using parallel agents.

### Agent workflow (per batch)

1. **Agent extracts models**: Reads notebooks, writes model files using the exact template above, but does NOT run them. Reports back what it extracted and any concerns.

2. **Review step**: I (the main agent) review each file the agent produced:
   - Check faithfulness to original model (no unnecessary simplifications, `pm.Data`/dims/coords preserved, etc.)
   - Check the benchmark boilerplate matches the template exactly (continuous vs discrete)
   - Check `build_model()` returns `(model, ip)` with the initval-clearing boilerplate
   - Check docstring has all fields (Model, Source, Authors, Description, Changes, Benchmark placeholder)
   - Check data handling (inline for small, .npz for large, `allow_pickle=True` if string arrays)
   - If something looks wrong, send the agent corrections via `SendMessage`
   - If unsure about something, log it to `review_questions.md` for the user

3. **Test step**: After review passes, run each file with `.venv/bin/python models/<name>.py`:
   - Verify both original and frozen produce finite logp
   - Verify logp matches between original and frozen (same initial point)
   - Fill in benchmark results in docstring

### Per notebook:
1. Read it and identify the final/complete model(s)
2. Skip intermediate/broken demo models
3. Extract model + data into `build_model()` — inline small data, .npz for large
4. Keep `initval=` in distribution calls as in original, add ip capture + clearing at end
5. Determine if model has discrete free variables -> `models/` vs `models_discrete/` (observed discrete likelihoods don't count — they still leave well-defined gradients for the continuous latents)
6. Document source, authors, description, and any changes in docstring
7. Return files for review (do NOT run them)

## API Changes for Current Dev Codebase

Document all changes in each file's docstring. Known issues:
- `pm.ConstantData` / `pm.MutableData` -> `pm.Data` (the only current API)
- `pm.distributions.transforms.univariate_ordered` -> `pm.distributions.transforms.ordered`
- `testval` -> `initval` (PyMC3 -> v5)
- npz files with string arrays (e.g., date coords) need `allow_pickle=True`
- `pm.Beta` with high concentration (e.g., alpha=beta=100) produces `inf` logp — PyMC/PyTensor numerical bug. Such models must be skipped.
- `Model.add_coord()` no longer accepts `mutable=` keyword

## Key Technical Details

### freeze_dims_and_data + initval incompatibility
`freeze_dims_and_data` calls `fgraph_from_model` which rejects models with non-None `rvs_to_initial_values`. Workaround: `build_model()` captures `model.initial_point()` *before* clearing all initvals, then returns `(model, ip)`. The frozen model is built from a second `build_model()` call (which also clears initvals), then frozen.

### value_vars ordering changes after freezing
`freeze_dims_and_data` may reorder `model.value_vars`. The raveled input array must be constructed per-model using `DictToArrayBijection.map({v.name: ip[v.name] for v in m.value_vars})`, NOT from a pre-raveled shared array. This ensures the same `ip` dict evaluates correctly regardless of variable ordering.

### Benchmarking details
- `logp_dlogp_function(ravel_inputs=True, mode="NUMBA")._pytensor_function` gives the raw pytensor function
- `compile_logp(mode="NUMBA").f` gives the raw pytensor function for discrete models
- The NUMBA mode is passed via `_benchmark.py`'s `run_benchmark(build_model, mode="NUMBA")`, NOT via `pytensor.config.linker`
- Model files should NOT set `pytensor.config.linker` — only import what they need
- `f.trust_input = True` skips input validation for accurate timing
- First call is separate (triggers numba JIT), not included in timing
- Adaptive eval count: calibrate to ~15s total, capped at 100k evals
- Assert `np.isfinite` on both logp and dlogp (all elements)

## Verification

1. Each model: `python models/<name>.py` prints logp, gradient (if continuous), and timing
2. logp MUST match between original and frozen (same initial point, same model)
3. Full suite: `python run_all.py` runs everything in parallel
4. All models must produce finite logp and gradient at the initial point
