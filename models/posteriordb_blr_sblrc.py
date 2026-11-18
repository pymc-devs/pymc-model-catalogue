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

Data: sblrc (upstream posterior: sblrc-blr)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/sblrc.json.zip
    Archive SHA-256: 81990278b8762ac79244afcd470861560c7b28e34e411ce6da814e6f632a1b81
    Local archive: data/posteriordb_sblrc.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/sblrc-blr.json

Implementation notes:
Native Gaussian linear regression from posteriordb blr.

Stan comparison:
- A: Stan = -792509.0486571618; PyMC = -792508.3555101727.
- B: Stan = -959988.8302412318; PyMC = -959988.1370942437.
- Raw delta: Stan = -167479.78158407006; PyMC = -167479.781584071.
- Raw comparison: MATCH; residual = 9.313225746154785e-10.
- Tolerance: 0.00167579781584071
- Stan sigma>0 with Normal(0,10) versus PyMC HalfNormal(10) adds the constant log(2).
- Original Stan SHA-256: e6407c4391ab11c0b59f17f3a04132df87aa69b7dae459e173036f16f9e8b161

Benchmark results:
- Original: logp = -6873.0122, grad norm = 14663.6951, 10.5 us/call (5 evals).
- Frozen: logp = -6873.0122, grad norm = 14663.6951, 13.1 us/call (5 evals).
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
            Path(__file__).parent / "data" / "posteriordb_sblrc.json.zip"
        ) as archive,
        archive.open("sblrc.json") as stream,
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
