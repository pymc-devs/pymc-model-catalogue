"""
Model: Hidden Markov Model
Source: posteriordb/posterior_database/models/stan/hmm_example.stan
Description: A Hidden Markov Model

Stan comparison:
- A: Stan = -1993.254877632162; PyMC = -1993.254877632162.
- B: Stan = -2108.9643080736214; PyMC = -2108.964308073621.
- Raw delta: Stan = -115.70943044145929; PyMC = -115.70943044145884.
- Raw comparison: MATCH; residual = -4.547473508864641e-13.
- Tolerance: 2.1570943044145927e-06

Benchmark results:
- Original:  logp = -259.1863, grad norm = 721.6521, 212.5 us/call (5 evals).
- Frozen:    logp = -259.1863, grad norm = 721.6521, 185.2 us/call (5 evals).
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
