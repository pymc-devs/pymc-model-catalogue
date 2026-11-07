# pymc-model-catalogue

This catalogue contains [PyMC](https://github.com/pymc-devs/pymc) models for
research on the PyMC and [PyTensor](https://github.com/pymc-devs/pytensor) stack.
Each model builder returns a model and its initial point.
The optional native PyMC6 package also accepts canonical posterior IDs for exact dataset selection.

The same model set also backs an ASV benchmark suite that tracks PyMC/PyTensor
performance across releases (see
[Benchmarking & the timeline](#benchmarking--the-timeline)).

## How the models are packaged

The existing top-level files under `models/` and `models_discrete/` define
`build_model()` and return a PyMC model with its initial point:

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

## Native PyMC6 posterior workloads

The optional `models.posteriordb` package contains 121 native PyMC models,
148 model/data pairs, and 91 bundled processed datasets.
The registry preserves canonical posterior identities and embeds upstream metadata, source paths, and hashes.
The builders use local data without network access.

The package does not duplicate metadata in separate JSON files.

The package supports PyMC 6.4 and compatible PyTensor versions.
Choose a canonical posterior to select its exact dataset.
The module default uses its smallest bundled dataset.

Run `pixi install` to create the pinned local environment.

```python
from models.posteriordb import build_model, get_posterior, list_posteriors

pairs = list_posteriors("blr")
metadata = get_posterior("sblri-blr")
model, ip = build_model("sblri-blr")

# Reuse one model with another canonical dataset.
from models.posteriordb.blr import build_model as build_blr
other_model, other_ip = build_blr(posterior="sblrc-blr")
```

No command below samples a posterior.

```bash
pixi run python -m models.posteriordb.blr --posterior sblri-blr

pixi run asv run --launch-method=spawn \
  --environment="existing:$(pixi run python -c 'import sys; print(sys.executable)')" \
  --set-commit-hash "$(git rev-parse HEAD)" \
  --bench 'ModelBench(?:Build|Eval)_models_posteriordb__sblri__blr__NUMBA\.'
```

The ASV example uses the configured `timeline` branch.
For a different branch, use a separate ASV configuration with `--config`.

The existing core selector and historical benchmark names stay unchanged.
Each optional ASV workload includes its posterior ID and declared backend.
See [backend timing details](./BENCHMARKING.md#optional-native-pymc6-workloads).

### Intended-model corrections

The native ports correct these confirmed defects in the source definitions.

| Models | Native mathematical correction | Evidence |
| --- | --- | --- |
| `hmm_gaussian` | Use the first observation's state-specific emission density. | [Original HMM definition](https://github.com/luisdamiano/stancon18/blob/master/main.Rmd#L239-L271) |
| `iohmm_reg` | Use destination-state gating weights. Native Gaussian mixtures eliminate the conditionally independent states. | [Original description and generator](https://github.com/luisdamiano/stancon18/blob/master/main.Rmd#L840-L875) |
| `dogs_nonhierarchical` | Use the declared covariance `diag(sigma) @ Omega @ diag(sigma)`. | [Native module and source notes](models/posteriordb/dogs_nonhierarchical.py) |
| `2pl_latent_reg_irt`, `gpcm_latent_reg_irt` | Scale binary covariates by their range, not twice their sample standard deviation. | [Furr's model description, sections 1.1 and 1.3](https://mc-stan.org/learn-stan/case-studies/rasch_and_2pl.html) |

The shared scaling correction also covers `grsm_latent_reg_irt`. Its bundled intercept-only dataset does not change.
Canonical identifiers retain the source model/data identity. See the corresponding module docstrings for correction details and references.

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

The existing top-level catalogue contains about 200 model files across these areas.

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
- `models/posteriordb/` — reusable native PyMC6 models, a pair registry, and bundled processed ZIP datasets.

Most models are extracted from
[pymc-examples](https://github.com/pymc-devs/pymc-examples); the rest come from
Statistical Rethinking and from papers (provenance is in each file's docstring).
Authoring conventions and the extraction template live in
[`PLAN.md`](./PLAN.md); extraction status in [`PROGRESS.md`](./PROGRESS.md).

## Benchmarking & the timeline

The catalogue backs an ASV suite that tracks four metrics — `rewrite_time`,
`compile_time`, `n_rewrites`, `time_eval` — on a curated core subset
([`BENCHMARK_CORE.md`](./BENCHMARK_CORE.md)) across PyMC releases, published
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
