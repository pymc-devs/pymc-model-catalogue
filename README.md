# pymc-model-catalogue

This catalogue contains [PyMC](https://github.com/pymc-devs/pymc) models for
research on the PyMC and [PyTensor](https://github.com/pymc-devs/pytensor) stack.
Each model builder returns a model and its initial point.

The same model set also backs an ASV benchmark suite that tracks PyMC/PyTensor
performance across releases (see
[Benchmarking & the timeline](#benchmarking--the-timeline)).

## How the models are packaged

The top-level files under `models/` and `models_discrete/` define
`build_model()` and return a PyMC model with its initial point:

```python
def build_model() -> tuple[pm.Model, dict]:
    ...
    return model, initial_point
```

Data is either inlined or loaded from bundled `.npz` or JSON data. Importing a
file is enough to build the model: there are no notebooks to run and nothing is fetched
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

## Posteriordb models

The catalogue also includes 148 model/data pairs ported from
[posteriordb](https://github.com/stan-dev/posteriordb) at
[revision 91027f9](https://github.com/stan-dev/posteriordb/tree/91027f9cdd13b7e76283fec3f2898f59d4b83ec4),
covering 121 model definitions and 91 datasets. Each pair has an
explicit `posteriordb_<model>_<dataset>.py` file under `models/` or
`models_discrete/`, with bundled source JSON archives. These ports target PyMC 6.4.

See [`POSTERIORDB_MODELS.md`](./POSTERIORDB_MODELS.md) for the source index.
Model docstrings contain a short description, Stan comparison status, and a brief
explanation for adjustments or unresolved mismatches.

```python
from models.posteriordb_blr_sblri import build_model

model, ip = build_model()

# The source namespace re-exports the same builders for convenience.
from models.posteriordb import blr_sblrc
other_model, other_ip = blr_sblrc()
```

Run a pair with `python models/posteriordb_blr_sblri.py`.
See [benchmarking instructions](./BENCHMARKING.md) for selecting entries in ASV.

### Validation and intentional departures

The benchmark helper checks finite PyMC logp and gradients. Model docstrings
summarize the two-point Stan logp-delta checks as MATCH, MISMATCH, or NOT APPLICABLE.
Adjusted matches are distinguished from raw matches.

The recorded two-point outcomes are:

| Outcome | Model/data pairs |
| --- | ---: |
| Raw deltas agree | 137 |
| Agree only after explicit prior or likelihood adjustments | 7 |
| Numerical mismatch remains | 3 |
| No Stan reference (discrete M0) | 1 |

The remaining mismatches are `bones_model`, `lotka_volterra`, and `sir`.

Comparisons use constrained densities with automatic support-transform Jacobians
disabled, following
[A Rosetta stone for PyMC and Stan](https://gist.github.com/ricardoV94/c421ddacb3ba7a19ac46efa253ea466c).
Parameter-independent constants cancel in `delta = logp(B) - logp(A)`.
Two points do not establish global equivalence, and agreement after a prior or
likelihood adjustment does not make the original targets equivalent.

Measurements used BridgeStan 2.9.0 (Stan 2.39.0), PyMC 6.4.0 and PyTensor 3.3.7.
The tolerance is `1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC))`, using the
adjusted Stan delta where applicable.

The seven adjusted cases include the five intentional departures below and two
runtime prior discrepancies in `accel_gp` and `accel_splines`. Their docstrings
describe the integer-overflow issue affecting the measured HalfStudentT scale
priors. Those adjustments need rechecking after engine upgrades.

The following ports deliberately follow the stated model where it differs from
the published implementation. Their docstrings briefly describe the departures.
Some changes affect priors: the dog-effects covariance changes their joint prior,
and IRT covariate scaling changes the induced coefficient priors.
Comparisons must account for those changes explicitly, using a corresponding Stan
reference or documented density adjustments.

| Models | Effect | Intentional departure | Evidence |
| --- | --- | --- | --- |
| `hmm_gaussian` | Likelihood | Use the first observation's state-specific emission density. | [Original HMM definition](https://github.com/luisdamiano/stancon18/blob/master/main.Rmd#L239-L271) |
| `iohmm_reg` | Likelihood | Use destination-state gating weights. Native Gaussian mixtures eliminate the conditionally independent states. | [Original description and generator](https://github.com/luisdamiano/stancon18/blob/master/main.Rmd#L840-L875) |
| `dogs_nonhierarchical` | Prior | Use the declared covariance `diag(sigma) @ Omega @ diag(sigma)`. | [Native module and source notes](models/posteriordb_dogs_nonhierarchical_dogs.py) |
| `2pl_latent_reg_irt`, `gpcm_latent_reg_irt` | Induced prior | Scale binary covariates by their range, not twice their sample standard deviation. | [Furr's model description, sections 1.1 and 1.3](https://mc-stan.org/learn-stan/case-studies/rasch_and_2pl.html) |

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

The catalogue contains about 350 model files across these areas.

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
- `models/data/`, `models_discrete/data/` — bundled `.npz` datasets and JSON archives.
- `models/posteriordb/` — convenience exports and source attribution for the posteriordb entries.

Most models are extracted from
[pymc-examples](https://github.com/pymc-devs/pymc-examples); the rest come from
Statistical Rethinking and from papers (provenance is in each file's docstring).

## Benchmarking & the timeline

The catalogue backs an ASV suite that tracks five metrics — `rewrite_time`,
`compile_time`, `n_rewrites`, `peak_rss`, `time_eval` — on a curated core subset
([`BENCHMARK_CORE.md`](./BENCHMARK_CORE.md)) across PyMC releases, published
to a [dashboard](https://pymc-devs.github.io/pymc-model-catalogue/dashboard.html).
A separate [experiments](https://pymc-devs.github.io/pymc-model-catalogue/experiments.html)
track runs the models against patched pymc/pytensor to A/B specific changes.

Full details — metrics, the branch/CI layout, running asv locally, and adding an
experiment — are in [`BENCHMARKING.md`](./BENCHMARKING.md).

See [`PLAN.md`](./PLAN.md) for adding or adapting model examples.

## Requirements

Local development uses current pymc/pytensor branches:

```bash
uv pip install "pymc @ git+https://github.com/pymc-devs/pymc.git@main" \
               "pytensor @ git+https://github.com/pymc-devs/pytensor.git@main" \
               numba asv
```
