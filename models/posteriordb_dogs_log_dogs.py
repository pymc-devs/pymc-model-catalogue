"""
Model: Logarithmic Mixed Effects Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/dogs_log.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/dogs_log.py
Upstream catalogue contributor: Oliver Järnefelt
Description: y ~ exp(n_avoid + n_shock)

Data: dogs (upstream posterior: dogs-dogs_log)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/dogs.json.zip
    Archive SHA-256: afe0a4a0f7508de3edd7ac52db9370edc744fbfaf4ec857629912d20cb3ed75a
    Local archive: data/posteriordb_dogs.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/dogs-dogs_log.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.24/dogs_log.stan

Implementation notes:
Native constrained logistic model from posteriordb dogs_log.

Stan comparison:
- A: Stan = -730.8670110448141; PyMC = -730.867011044815.
- B: Stan = -391.2564909183006; PyMC = -391.25649091830047.
- Raw delta: Stan = 339.61052012651356; PyMC = 339.6105201265145.
- Raw comparison: MATCH; residual = -9.094947017729282e-13.
- Tolerance: 4.396105201265145e-06
- Stan declares beta unconstrained, but the Uniform(-100,0) and Uniform(0,100) prior
  supports still apply. The feasible center [-0.5,0.5] is chosen solely from support,
  not engine agreement.
- Original Stan SHA-256: f0257153c50d7ef91d9958607c159ae9097d622e3d5ed9778cd3be60fd91b56b

Benchmark results:
- Not measured.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    def _dogs_history(data):
        """Compute observed shock and avoidance histories before each trial."""
        y = np.asarray(data["y"], dtype="int64")
        shock = np.cumsum(y, axis=1) - y
        avoid = np.arange(y.shape[1])[None, :] - shock
        return y, shock, avoid

    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_dogs.json.zip"
        ) as archive,
        archive.open("dogs.json") as stream,
    ):
        data = json.load(stream)
    y, shock, avoid = _dogs_history(data)
    coords = {
        "dog": np.arange(1, data["n_dogs"] + 1),
        "trial": np.arange(1, data["n_trials"] + 1),
        "coefficient": ["avoid", "shock"],
    }
    with pm.Model(coords=coords) as model:
        previous_shock = pm.Data("previous_shock", shock, dims=("dog", "trial"))
        previous_avoid = pm.Data("previous_avoid", avoid, dims=("dog", "trial"))
        beta = pm.Uniform(
            "beta",
            lower=[-100.0, 0.0],
            upper=[0.0, 100.0],
            initval=[-0.1, 0.1],
            dims="coefficient",
        )
        pm.Bernoulli(
            "y",
            logit_p=beta[0] * previous_avoid + beta[1] * previous_shock,
            observed=y,
            dims=("dog", "trial"),
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
