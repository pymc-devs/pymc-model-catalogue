"""HMM with Gaussian emission.

Stan comparison: MISMATCH; MATCH after likelihood adjustment.
PyMC uses state-specific initial emissions; Stan sums them across states.
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
            Path(__file__).parent
            / "data"
            / "posteriordb_hmm_gaussian_simulated.json.zip"
        ) as archive,
        archive.open("hmm_gaussian_simulated.json") as stream,
    ):
        data = json.load(stream)
    n_states = data["K"]
    observed = np.asarray(data["y"], dtype=float)
    initial_means = np.quantile(observed, np.linspace(0.1, 0.9, n_states))
    coords = {
        "time": np.arange(data["T"]),
        "state": np.arange(n_states),
        "next_state": np.arange(n_states),
    }
    with pm.Model(coords=coords) as model:
        y = pm.Data("y", observed, dims="time")
        pi1 = pm.Dirichlet("pi1", a=np.ones(n_states), dims="state")
        transition = pm.Dirichlet(
            "A", a=np.ones(n_states), dims=("state", "next_state")
        )
        mu = pm.Flat(
            "mu",
            transform=pm.distributions.transforms.Ordered(),
            initval=initial_means,
            dims="state",
        )
        sigma = pm.HalfFlat(
            "sigma", initval=np.full(n_states, np.std(observed)), dims="state"
        )
        log_emissions = pm.logp(pm.Normal.dist(mu=mu, sigma=sigma), y[:, None])
        # Stan's vectorized normal_lpdf(y[1] | mu, sigma) sums over states,
        # adding the same scalar to every initial state. Each state instead
        # gets its own emission here. This changes the likelihood, not just
        # a parameter-independent logp constant.
        initial_logp = pt.log(pi1) + log_emissions[0]
        pm.Potential(
            "observations",
            _hidden_markov_loglik(log_emissions, transition, initial_logp),
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
