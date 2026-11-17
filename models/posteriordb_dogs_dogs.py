"""
Model: Logistic Mixed Effects Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/dogs.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/dogs.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/dogs.py
Upstream catalogue contributor: Oliver Järnefelt
Description: y ~ logistic(n_avoid + n_shock)

Data: dogs (upstream posterior: dogs-dogs)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/dogs.json.zip
    Archive SHA-256: afe0a4a0f7508de3edd7ac52db9370edc744fbfaf4ec857629912d20cb3ed75a
    Local archive: data/posteriordb_dogs.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/dogs-dogs.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.24/dogs.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native logistic shock model from posteriordb dogs.

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
        "coefficient": ["intercept", "avoid", "shock"],
    }
    with pm.Model(coords=coords) as model:
        previous_shock = pm.Data("previous_shock", shock, dims=("dog", "trial"))
        previous_avoid = pm.Data("previous_avoid", avoid, dims=("dog", "trial"))
        beta = pm.Normal("beta", 0, 100, dims="coefficient")
        pm.Bernoulli(
            "y",
            logit_p=beta[0] + beta[1] * previous_avoid + beta[2] * previous_shock,
            observed=y,
            dims=("dog", "trial"),
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
