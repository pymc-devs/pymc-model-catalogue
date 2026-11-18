"""
Model: Hidden Markov Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/hmm_drive_1.stan
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/hmm_drive_1.py
Upstream catalogue contributor: Oliver Järnefelt
Description: A Hidden Markov Model with normal distribution for emission probabilities

Data: bball_drive_event_1 (upstream posterior: bball_drive_event_1-hmm_drive_1)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/bball_drive_event_1.json.zip
    Archive SHA-256: 5b4332702c475f94b2d1c651e9ae1bebf155e761f2cc6683879db50129c70f4d
    Local archive: data/posteriordb_bball_drive_event_1.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/bball_drive_event_1-hmm_drive_1.json
Reference URL: https://github.com/imadmali/bball-hmm/blob/master/models/drive_1.stan

Implementation notes:
Native Gaussian drive model from the posteriordb basketball example.

Source: https://github.com/imadmali/bball-hmm
The initial state weights equal one, as in the published target.
Ordered transforms preserve the joint support of the emission means.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -215339.99545139476; PyMC = -215339.99545139473.
- B: Stan = -239662.78143514544; PyMC = -239662.78143514533.
- Raw delta: Stan = -24322.785983750684; PyMC = -24322.785983750597.
- Raw comparison: MATCH; residual = -8.731149137020111e-11.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 0.0002442278598375069
- Shared-support limitation: original Stan points enforce all ordered or positive-
  ordered constraints. PyMC ordering supplied only by a transform is not separately
  enforced after transforms are removed. These evaluations compare points on the shared
  Stan-valid support, not global support or model equivalence.
- Original Stan SHA-256: 295e47fe6cf70adb9c16af0697fb6f3803aad993c3ee800a34e9b437883abb75
- Saved points: models/posteriordb_logp/points/bball_drive_event_1-hmm_drive_1.npz
- Two-point evidence only; no global equivalence, support or inference claim.

Benchmark results:
- Not measured.
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
