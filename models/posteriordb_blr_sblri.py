"""
Model: A Bayesian linear regression model with vague priors
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/blr.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/blr.py
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/blr.py
Upstream catalogue contributor: Mans Magnusson
Description: A Bayesian linear regression model with vague priors.

Data: sblri (upstream posterior: sblri-blr)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/sblri.json.zip
    Archive SHA-256: e975f6641b5e08d65cb1e3519f1d0370939afbc82bfbeb3c1f5a07e5bbd717a7
    Local archive: data/posteriordb_sblri.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/sblri-blr.json

Implementation notes:
Native Gaussian linear regression from posteriordb blr.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -3244257.0094579956; PyMC = -3244256.3163110074.
- B: Stan = -6068280.409912163; PyMC = -6068279.716765175.
- Raw delta: Stan = -2824023.4004541677; PyMC = -2824023.4004541673.
- Raw comparison: MATCH; residual = -4.656612873077393e-10.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 0.02824123400454168
- Stan sigma>0 with Normal(0,10) versus PyMC HalfNormal(10) adds the constant log(2).
- Original Stan SHA-256: e6407c4391ab11c0b59f17f3a04132df87aa69b7dae459e173036f16f9e8b161
- Saved points: models/posteriordb_logp/points/sblri-blr.npz
- Two-point evidence only; no global equivalence, support or inference claim.

Benchmark results:
- Original: logp = -26110.0910, grad norm = 56791.0468, 9.6 us/call (5 evals).
- Frozen: logp = -26110.0910, grad norm = 56791.0468, 9.1 us/call (5 evals).
- Local NUMBA validation; short timings are indicative.
- Environment: PyMC 6.4.0+9.gaf565aa28, PyTensor 3.3.8+30.g9a7f24ce0, Numba 0.68.0.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_sblri.json.zip"
        ) as archive,
        archive.open("sblri.json") as stream,
    ):
        data = json.load(stream)
    coords = {"observation": np.arange(data["N"]), "predictor": np.arange(data["D"])}
    with pm.Model(coords=coords) as model:
        x = pm.Data(
            "X", np.asarray(data["X"], dtype=float), dims=("observation", "predictor")
        )
        beta = pm.Normal("beta", 0, 10, dims="predictor")
        sigma = pm.HalfNormal("sigma", 10)
        pm.Normal("y", mu=x @ beta, sigma=sigma, observed=data["y"], dims="observation")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
