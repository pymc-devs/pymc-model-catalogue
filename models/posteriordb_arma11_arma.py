"""
Model: Autoregressive Moving Average
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/arma11.stan
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/arma11.py
Upstream catalogue contributor: Mans Magnusson
Description: An ARMA(1, 1) time series model.

Data: arma (upstream posterior: arma-arma11)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/arma.json.zip
    Archive SHA-256: 354e7d7bb05ee08a3deeac45ce1d92ae11964a696b933ba82a3a0cfe134eb45a
    Local archive: data/posteriordb_arma.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/arma-arma11.json
Reference URL: https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/arma

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Gaussian ARMA(1, 1) from the posteriordb arma11 probability model.

The initial prediction uses mu + phi * mu and zero previous error.
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -201.58685607383327; PyMC = -200.8937089553195.
- B: Stan = -270.99531028711385; PyMC = -270.3021631686002.
- Raw delta: Stan = -69.40845421328058; PyMC = -69.40845421328069.
- Raw comparison: MATCH; residual = 1.1368683772161603e-13.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.6940845421328068e-06
  using adjusted delta_Stan when an adjustment is present.
- Identity of mu, phi, theta, sigma. Both recursions start with nu[1]=mu+phi*mu and zero
  pre-series error; later nu[t]=mu+phi*y[t-1]+theta*(y[t-1]-nu[t-1]). Normal errors
  equal the conditional Normal observation likelihood, with a unit triangular Jacobian.
- HalfCauchy versus a Cauchy density on positive sigma contributes only log(2).
- Original Stan SHA-256: aa713e50a5154ab9efda30f2b5fe320d6a77510434ff73cef6708cbf72d1fae8
- Saved points: models/posteriordb_logp/points/arma-arma11.npz
- Full evidence: models/posteriordb_logp/records/arma-arma11.json
- Two-point evidence only; no global equivalence, support or inference claim.

Benchmark results:
- Not measured.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt


def build_model():
    def _arma11_error_step(current, previous, previous_error, mu, phi, theta):
        return current - mu - phi * previous - theta * previous_error

    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_arma.json.zip"
        ) as archive,
        archive.open("arma.json") as stream,
    ):
        data = json.load(stream)
    y = np.asarray(data["y"], dtype=float)
    with pm.Model(coords={"time": np.arange(int(data["T"]))}) as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        phi = pm.Normal("phi", mu=0, sigma=2)
        theta = pm.Normal("theta", mu=0, sigma=2)
        sigma = pm.HalfCauchy("sigma", beta=2.5)
        observations = pm.Data("observations", y, dims="time")
        first_prediction = mu + phi * mu
        first_error = observations[0] - first_prediction
        if len(y) > 1:
            later_errors, _ = pytensor.scan(
                fn=_arma11_error_step,
                sequences=[observations[1:], observations[:-1]],
                outputs_info=first_error,
                non_sequences=[mu, phi, theta],
                strict=True,
                name="arma11_errors",
            )
            errors = pt.concatenate([first_error[None], later_errors])
            prediction = pt.concatenate(
                [
                    first_prediction[None],
                    mu + phi * observations[:-1] + theta * errors[:-1],
                ]
            )
        else:
            prediction = first_prediction[None]
        pm.Normal("y", mu=prediction, sigma=sigma, observed=y, dims="time")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
