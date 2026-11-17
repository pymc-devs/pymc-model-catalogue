"""
Model: Hierarchical Logistic Mixed Effects Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/dogs_hierarchical.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/dogs_hierarchical.py
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/dogs_hierarchical.py
Upstream catalogue contributor: Mans Magnusson
Description: Hierarchical Logistic Mixed Effects Model.

Data: dogs (upstream posterior: dogs-dogs_hierarchical)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/dogs.json.zip
    Archive SHA-256: afe0a4a0f7508de3edd7ac52db9370edc744fbfaf4ec857629912d20cb3ed75a
    Local archive: data/posteriordb_dogs.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/dogs-dogs_hierarchical.json
Reference URL: https://github.com/stan-dev/example-models/tree/master/ARM/Ch.24

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native shared learning model from posteriordb dogs_hierarchical.

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
        "learning_trial": np.arange(2, data["n_trials"] + 1),
    }
    with pm.Model(coords=coords) as model:
        previous_shock = pm.Data(
            "previous_shock", shock[:, 1:], dims=("dog", "learning_trial")
        )
        previous_avoid = pm.Data(
            "previous_avoid", avoid[:, 1:], dims=("dog", "learning_trial")
        )
        a = pm.Uniform("a", 0, 1)
        b = pm.Uniform("b", 0, 1)
        probability = pm.math.exp(
            previous_shock * pm.math.log(a) + previous_avoid * pm.math.log(b)
        )
        # The first shock has probability one and does not depend on model parameters.
        pm.Bernoulli("first_shock", p=1.0, observed=y[:, 0], dims="dog")
        pm.Bernoulli(
            "y", p=probability, observed=y[:, 1:], dims=("dog", "learning_trial")
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
