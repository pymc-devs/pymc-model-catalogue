"""
Model: Hidden Markov Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/hmm_example.stan
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/hmm_example.py
Upstream catalogue contributor: Oliver Järnefelt
Description: A Hidden Markov Model

Data: hmm_example (upstream posterior: hmm_example-hmm_example)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/hmm_example.json.zip
    Archive SHA-256: 8c346cc7627db1b5614b1eacb81ab9b0926aa63071ba2f3b6d6ec3b492892787
    Local archive: data/posteriordb_hmm_example.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/hmm_example-hmm_example.json
Reference URL: https://github.com/imadmali/bball-hmm/blob/master/models/hmm_example.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native two-state Gaussian example from posteriordb.

Source: https://github.com/imadmali/bball-hmm
The transition rows have uniform simplex densities.
The initial state weights equal one, as in the published target.

Stan comparison:
- Not recorded.

Benchmark results:
- Original: logp = -259.1863, grad norm = 721.6521, 212.5 us/call (5 evals).
- Frozen: logp = -259.1863, grad norm = 721.6521, 185.2 us/call (5 evals).
- Local NUMBA validation; short timings are indicative.
- Environment: PyMC 6.4.0+9.gaf565aa28, PyTensor 3.3.8+30.g9a7f24ce0, Numba 0.68.0.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt


def build_model():
    def _hidden_markov_loglik(log_emissions, transition, initial_logp):
        """Sum hidden paths with a symbolic log-space forward recursion."""
        log_transition = pt.log(transition)

        def step(log_emission, previous, log_transition):
            return log_emission + pt.logsumexp(
                previous[:, None] + log_transition, axis=0
            )

        filtered, _ = pytensor.scan(
            fn=step,
            sequences=[log_emissions[1:]],
            outputs_info=[initial_logp],
            non_sequences=[log_transition],
            strict=True,
            name="hidden_markov_forward",
        )
        final = pt.concatenate([initial_logp[None, :], filtered], axis=0)[-1]
        return pt.logsumexp(final)

    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_hmm_example.json.zip"
        ) as archive,
        archive.open("hmm_example.json") as stream,
    ):
        data = json.load(stream)
    n_states = data["K"]
    coords = {
        "time": np.arange(data["N"]),
        "state": np.arange(n_states),
        "next_state": np.arange(n_states),
    }
    with pm.Model(coords=coords) as model:
        y = pm.Data("y", np.asarray(data["y"], dtype=float), dims="time")
        theta1 = pm.Dirichlet("theta1", a=np.ones(n_states), dims="next_state")
        theta2 = pm.Dirichlet("theta2", a=np.ones(n_states), dims="next_state")
        theta = pm.Deterministic(
            "theta", pt.stack([theta1, theta2]), dims=("state", "next_state")
        )
        mu = pm.Normal(
            "mu",
            mu=np.array([3.0, 10.0]),
            sigma=1,
            transform=pm.distributions.transforms.Ordered(positive=True),
            initval=np.array([3.0, 10.0]),
            dims="state",
        )
        log_emissions = pm.logp(pm.Normal.dist(mu=mu, sigma=1), y[:, None])
        pm.Potential(
            "observations",
            _hidden_markov_loglik(log_emissions, theta, log_emissions[0]),
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
