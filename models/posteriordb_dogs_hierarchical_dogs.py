"""
Model: Hierarchical Logistic Mixed Effects Model
Source: posteriordb/posterior_database/models/stan/dogs_hierarchical.stan
Description: Hierarchical Logistic Mixed Effects Model.

Limitations:
- Posterior predictive y draws condition on observed trial histories. They
  are one-step conditional predictions, not Stan's recursively generated
  y_rep sequences.

Stan comparison:
- A: Stan = -858.3321218595022; PyMC = -858.3321218595022.
- B: Stan = -966.0679160441127; PyMC = -966.0679160441127.
- Raw delta: Stan = -107.73579418461054; PyMC = -107.73579418461054.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 2.0773579418461055e-06
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
