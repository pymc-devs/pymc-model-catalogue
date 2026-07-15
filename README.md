# pymc-model-catalogue

A database of around 200 [PyMC](https://github.com/pymc-devs/pymc) models,
collected for research on the PyMC / [PyTensor](https://github.com/pymc-devs/pytensor)
stack. Each model is packaged as a `build_model()` function with the same
signature, so you can load and run them in bulk — to test work on rewrites,
compilation, sampling, or logp/grad performance against many models at once
instead of a few hand-picked ones.

The same model set also backs an ASV benchmark suite that tracks PyMC/PyTensor
performance across releases (see
[Benchmarking & the timeline](#benchmarking--the-timeline)).

## How the models are packaged

Every file under `models/` and `models_discrete/` defines one `build_model()`
that returns a PyMC model and a matching initial point:

```python
def build_model() -> tuple[pm.Model, dict]:
    ...
    return model, initial_point
```

Data is either inlined or loaded from a bundled `.npz`, so importing a file is
enough to build the model: there are no notebooks to run and nothing is fetched
over the network. Each model is checked to give a finite logp and gradient at
its initial point, both as written and after `freeze_dims_and_data` bakes in the
dims and shapes. Models keep the form the original author used (`pm.Data`,
`dims`, `pm.Deterministic`, and so on) rather than being rewritten into a
canonical style.

## Quickstart

```python
from models.eight_schools_centered import build_model

model, ip = build_model()        # a pm.Model + a valid initial point dict

with model:
    idata = pm.sample()          # ...or compile model.logp(), inspect the graph, etc.

# Frozen form (dims/data baked in), e.g. for rewrite/compile studies:
from pymc.model.transform.optimization import freeze_dims_and_data
frozen, _ = build_model()
frozen = freeze_dims_and_data(frozen)
```

For the exact logp/dlogp compilation used by the benchmarks (raveled input,
NUMBA mode, version-robust across old pymc releases) reuse
`models/_benchmark.py:build_logp_fn`.

## What it's useful for

- Seeing how a PyTensor rewrite change affects rewrite count, compile time, and
  run time across many real graphs. The
  [experiments](./BENCHMARKING.md#adding-an-experiment) setup can run the models
  against a patched or reverted build of pymc or pytensor.
- Sampling and posterior-geometry work. The set includes the reparameterisation
  cases from Gorinova et al. (2019) — Neal's funnel, German credit, Election
  '88, Electric Company — described in
  [`AUTOREPARAM_MODELS.md`](./AUTOREPARAM_MODELS.md).
- Tracking performance across pymc releases, which is what the timeline
  dashboard does. Historical points pin every dependency to what was current on
  each release day, so old numbers stay comparable.
- Examples and teaching, since the catalogue covers most common modelling
  patterns.

## What's inside

~200 model files, grouped roughly by area:

- **GLMs & regression** — logistic, Poisson, negative-binomial, ordinal, robust,
  truncated/censored, discrete-choice, rolling.
- **Hierarchical / multilevel** — radon (the full Gelman series), eight schools
  (centered + noncentered), partial pooling, varying intercepts/slopes.
- **Gaussian processes** — latent/marginal GPs, HSGP, Kronecker, coregion,
  log-Gaussian Cox.
- **Time series** — AR, structural/Prophet-style, stochastic volatility, VAR,
  Euler–Maruyama SDEs, scan-built generative graphs.
- **ODEs** — Lotka–Volterra and SIR via several integration paths.
- **Mixtures & nonparametrics** — Gaussian/Dirichlet mixtures, DP mixtures,
  dependent density regression.
- **Survival** — Cox PH, frailty, Weibull/log-logistic AFT.
- **Spatial** — CAR, BYM.
- **Causal inference** — diff-in-diff, regression discontinuity, mediation,
  do-operator counterfactuals.
- **Statistical Rethinking** — the `sr*` lecture models.
- **Reparameterisation stress-tests** — see
  [`AUTOREPARAM_MODELS.md`](./AUTOREPARAM_MODELS.md).

Layout:

- `models/` — continuous-latent models (NUTS-able; full logp + gradient).
- `models_discrete/` — models with discrete free variables (logp-only path).
  Observed discrete likelihoods stay in `models/`, since the continuous latents
  still have well-defined gradients.
- `models/data/`, `models_discrete/data/` — bundled `.npz` datasets.

Most models are extracted from
[pymc-examples](https://github.com/pymc-devs/pymc-examples); the rest come from
Statistical Rethinking and from papers (provenance is in each file's docstring).
Authoring conventions and the extraction template live in
[`PLAN.md`](./PLAN.md); extraction status in [`PROGRESS.md`](./PROGRESS.md).

## Benchmarking & the timeline

The catalogue backs an ASV suite that tracks four metrics — `rewrite_time`,
`compile_time`, `n_rewrites`, `time_eval` — on a curated 25-model core
([`BENCHMARK_CORE.md`](./BENCHMARK_CORE.md)) across every pymc release, published
to a [dashboard](https://pymc-devs.github.io/pymc-model-catalogue/dashboard.html).
A separate [experiments](https://pymc-devs.github.io/pymc-model-catalogue/experiments.html)
track runs the models against patched pymc/pytensor to A/B specific changes.

Full details — metrics, the branch/CI layout, running asv locally, and adding an
experiment — are in [`BENCHMARKING.md`](./BENCHMARKING.md).

## Requirements

Local development uses current pymc/pytensor branches:

```bash
uv pip install "pymc @ git+https://github.com/pymc-devs/pymc.git@main" \
               "pytensor @ git+https://github.com/pymc-devs/pytensor.git@main" \
               numba asv
```
