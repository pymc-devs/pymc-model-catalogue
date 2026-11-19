"""Hidden Markov Model.

Stan comparison: MATCH.
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
            Path(__file__).parent / "data" / "posteriordb_bball_drive_event_1.json.zip"
        ) as archive,
        archive.open("bball_drive_event_1.json") as stream,
    ):
        data = json.load(stream)
    n_states = data["K"]
    coords = {
        "time": np.arange(data["N"]),
        "state": np.arange(n_states),
        "next_state": np.arange(n_states),
    }
    with pm.Model(coords=coords) as model:
        u = pm.Data("u", np.asarray(data["u"], dtype=float), dims="time")
        v = pm.Data("v", np.asarray(data["v"], dtype=float), dims="time")
        alpha = np.asarray(data["alpha"], dtype=float)
        theta1 = pm.Dirichlet("theta1", a=alpha[0], dims="next_state")
        theta2 = pm.Dirichlet("theta2", a=alpha[1], dims="next_state")
        theta = pm.Deterministic(
            "theta", pt.stack([theta1, theta2]), dims=("state", "next_state")
        )
        phi = pm.Normal(
            "phi",
            mu=np.array([0.0, 3.0]),
            sigma=1,
            transform=pm.distributions.transforms.Ordered(),
            initval=np.array([0.0, 3.0]),
            dims="state",
        )
        mean_v = pm.Normal(
            "lambda",
            mu=np.array([0.0, 3.0]),
            sigma=1,
            transform=pm.distributions.transforms.Ordered(),
            initval=np.array([0.0, 3.0]),
            dims="state",
        )
        log_emissions = pm.logp(
            pm.Normal.dist(mu=phi, sigma=data["tau"]), u[:, None]
        ) + pm.logp(pm.Normal.dist(mu=mean_v, sigma=data["rho"]), v[:, None])
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
