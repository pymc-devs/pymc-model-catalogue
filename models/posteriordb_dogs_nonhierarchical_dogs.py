"""
Model: Non-Hierarchical Logistic Mixed Effects Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/dogs_nonhierarchical.stan
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/dogs_nonhierarchical.py
Upstream catalogue contributor: Mans Magnusson
Description: Logistic Mixed Effects Model.

Data: dogs (upstream posterior: dogs-dogs_nonhierarchical)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/dogs.json.zip
    Archive SHA-256: afe0a4a0f7508de3edd7ac52db9370edc744fbfaf4ec857629912d20cb3ed75a
    Local archive: data/posteriordb_dogs.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/dogs-dogs_nonhierarchical.json

Changes from original:
- Use the declared covariance diag(sigma) @ Omega @ diag(sigma), rather than the source row-factor product.
  This changes the joint prior on the dog effects.
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native correlated dog effects from posteriordb dogs_nonhierarchical.

The covariance is diag(sigma) @ Omega @ diag(sigma), as defined by the source Sigma_logit_ab.
The source row-factor product does not give that covariance. This model uses the declared covariance.
Evidence: posterior_database/models/stan/dogs_nonhierarchical.stan, lines 27-32, upstream revision 91027f9.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -964.1082378541431; PyMC = -962.0262129394036.
- B: Stan = -871.1298048632541; PyMC = -869.0412251422583.
- Raw delta: Stan = 92.97843299088902; PyMC = 92.98498779714532.
- Raw comparison: MISMATCH; residual = -0.006554806256303891.
- Additive Stan adjustment changed_standard_normal_effect_prior: A =
  0.002583351320819993; B = 0.009138157577196493.
- Adjusted Stan delta = 92.9849877971454.
- Adjusted comparison: MATCH; residual = 7.105427357601002e-14.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.9298498779714537e-06
  using adjusted delta_Stan when an adjustment is present.
- Original targets differ: Stan uses z @ A, PyMC z @ A.T. Map
  z_PyMC=solve(A,(z_Stan@A).T).T to preserve dog effects; add the independently computed
  standard-Normal prior difference.
- This z mapping has unit Jacobian. rho_unit=(L[1,0]+1)/2 has a constant Jacobian;
  LKJ(2) and Beta(2,2) then agree up to constants, as do Normal and HalfNormal factors.
- Original Stan SHA-256: 69dc8063e69d0451f54c5223edbb34e316742f5001a630f251bbab2be074109d
- Saved points: models/posteriordb_logp/points/dogs-dogs_nonhierarchical.npz
- Full evidence: models/posteriordb_logp/records/dogs-dogs_nonhierarchical.json
- Two-point evidence only; no global equivalence, support or inference claim.

Benchmark results:
- Original: logp = -922.6830, grad norm = 482.4855, 45.7 us/call (5 evals).
- Frozen: logp = -922.6830, grad norm = 482.4855, 43.2 us/call (5 evals).
- Local NUMBA validation; short timings are indicative.
- Environment: PyMC 6.4.0+9.gaf565aa28, PyTensor 3.3.8+30.g9a7f24ce0, Numba 0.68.0.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


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
        "learning_effect": ["shock", "avoid"],
    }
    with pm.Model(coords=coords) as model:
        previous_shock = pm.Data(
            "previous_shock", shock[:, 1:], dims=("dog", "learning_trial")
        )
        previous_avoid = pm.Data(
            "previous_avoid", avoid[:, 1:], dims=("dog", "learning_trial")
        )
        mu = pm.Logistic("mu_logit_ab", 0, 1, dims="learning_effect")
        sigma = pm.HalfNormal("sigma_logit_ab", 1, dims="learning_effect")
        # For two effects, LKJ(2) gives a Beta(2, 2) density for (rho + 1) / 2.
        rho_unit = pm.Beta("rho_unit", 2, 2)
        rho = pm.Deterministic("rho", 2 * rho_unit - 1)
        z = pm.Normal("z", 0, 1, dims=("dog", "learning_effect"))
        # With A = diag(sigma) @ L, Stan uses row vectors z @ A, whose
        # covariance is A.T @ A. These expressions instead give A @ A.T,
        # matching the declared Sigma_logit_ab. This changes the joint prior
        # on dog effects, not just its parameterization or a logp constant.
        logit_a = mu[0] + sigma[0] * z[:, 0]
        logit_b = mu[1] + sigma[1] * (rho * z[:, 0] + pt.sqrt(1 - rho**2) * z[:, 1])
        pm.Deterministic("a", pm.math.sigmoid(logit_a), dims="dog")
        pm.Deterministic("b", pm.math.sigmoid(logit_b), dims="dog")
        log_probability = -previous_shock * pt.softplus(
            -logit_a[:, None]
        ) - previous_avoid * pt.softplus(-logit_b[:, None])
        # The first shock has probability one and does not depend on model parameters.
        pm.Bernoulli("first_shock", p=1.0, observed=y[:, 0], dims="dog")
        pm.Bernoulli(
            "y",
            p=pt.exp(log_probability),
            observed=y[:, 1:],
            dims=("dog", "learning_trial"),
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
