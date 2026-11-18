"""
Model: HMM with Gaussian emission
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/hmm_gaussian.stan
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/hmm_gaussian.py
Upstream catalogue contributor: Keane Nguyen
Description: HMM with Gaussian emission

Data: hmm_gaussian_simulated (upstream posterior: hmm_gaussian_simulated-hmm_gaussian)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/hmm_gaussian_simulated.json.zip
    Archive SHA-256: af1e3e72f8d990ccaaa36517d5c1b1734d0fdadf5421de05a42d11937b63f66a
    Local archive: data/posteriordb_hmm_gaussian_simulated.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/hmm_gaussian_simulated-hmm_gaussian.json
Reference URL: https://github.com/luisdamiano/stancon18

Changes from original:
- Use state-specific first-observation emissions instead of the source scalar sum.
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native Gaussian hidden-state model from the Damiano example in posteriordb.

The published model has flat ordered means and flat positive scales.
The first observation uses its state-specific emission density.
This corrects the published scalar sum across all initial emission densities.

Evidence: the local-evidence definition and Bayesian update use one density per state.
https://github.com/luisdamiano/stancon18/blob/master/main.Rmd#L239-L271
Source model: https://github.com/luisdamiano/stancon18

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -87074.08921999834; PyMC = -86148.52959186646.
- B: Stan = -60347.245630767924; PyMC = -59397.338385377116.
- Raw delta: Stan = 26726.84358923041; PyMC = 26751.191206489348.
- Raw comparison: MISMATCH; residual = -24.347617258936225.
- Additive Stan adjustment changed_initial_emission_likelihood: A = 922.7870394095662; B
  = 947.1346566686552.
- Adjusted Stan delta = 26751.1912064895.
- Adjusted comparison: MATCH; residual = 1.5279510989785194e-10.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 0.00026851191206489505
  using adjusted delta_Stan when an adjustment is present.
- Original targets differ: Stan adds the sum of first-time state emissions to every
  state; PyMC uses state-specific emissions. Adjustment is the difference of
  independently computed forward likelihoods. Ordered means are tested on shared Stan-
  valid support.
- Original Stan SHA-256: 2f76768fec16648e6a49782166c8c70862d22fb72e12349906b25cf8e6e837d7
- Saved points: models/posteriordb_logp/points/hmm_gaussian_simulated-hmm_gaussian.npz
- Full evidence: models/posteriordb_logp/records/hmm_gaussian_simulated-hmm_gaussian.json
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
