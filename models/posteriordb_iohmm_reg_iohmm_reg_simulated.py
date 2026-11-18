"""
Model: Input Output HMM
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/iohmm_reg.stan
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/iohmm_reg.py
Upstream catalogue contributor: Keane Nguyen
Description: Input Output HMM

Data: iohmm_reg_simulated (upstream posterior: iohmm_reg_simulated-iohmm_reg)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/iohmm_reg_simulated.json.zip
    Archive SHA-256: 7f9239754222ef8f3da3296cad2b273ed5ae3b9a144d28cfa147d1b76259d2c4
    Local archive: data/posteriordb_iohmm_reg_simulated.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/iohmm_reg_simulated-iohmm_reg.json
Reference URL: https://github.com/luisdamiano/stancon18

Changes from original:
- Use destination-state gating and analytically eliminate the conditionally independent states.
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native input-output hidden-state regression from the Damiano example in posteriordb.

The original data generator selects each destination state with input-dependent softmax weights.
These weights do not depend on the previous state, so exact state elimination gives independent Gaussian mixtures.
The first observation uses the separate initial state probabilities.

The published forward recurrence indexes the source state instead of the destination state.
This implementation corrects that defect and uses the documented generative model.
Evidence: https://github.com/luisdamiano/stancon18/blob/master/main.Rmd#L840-L875

The initial probabilities have a uniform simplex density.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -12251.424541381222; PyMC = -12250.068644344123.
- B: Stan = -13444.51019509857; PyMC = -13441.722573636545.
- Raw delta: Stan = -1193.0856537173477; PyMC = -1191.6539292924226.
- Raw comparison: MISMATCH; residual = -1.4317244249250507.
- Additive Stan adjustment changed_transition_gating_likelihood: A = -1.416690903932249;
  B = 0.01503352099825861.
- Adjusted Stan delta = -1191.6539292924172.
- Adjusted comparison: MATCH; residual = 5.4569682106375694e-12.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.2916539292924226e-05
  using adjusted delta_Stan when an adjustment is present.
- Original targets differ: Stan gates source states; PyMC gates destination states.
  Adjustment is the independently computed destination-mixture likelihood minus the
  original forward likelihood.
- Original Stan SHA-256: 9203ab44e94440c2bc51f96551c4e8ef94bbebfb6481c601320883ee983d7c32
- Saved points: models/posteriordb_logp/points/iohmm_reg_simulated-iohmm_reg.npz
- Full evidence: models/posteriordb_logp/records/iohmm_reg_simulated-iohmm_reg.json
- Two-point evidence only; no global equivalence, support or inference claim.

Benchmark results:
- Not measured.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_iohmm_reg_simulated.json.zip"
        ) as archive,
        archive.open("iohmm_reg_simulated.json") as stream,
    ):
        data = json.load(stream)
    n_states = data["K"]
    coords = {
        "time": np.arange(data["T"]),
        "state": np.arange(n_states),
        "feature": np.arange(data["M"]),
    }
    with pm.Model(coords=coords) as model:
        u = pm.Data("u", np.asarray(data["u"], dtype=float), dims=("time", "feature"))
        pi1 = pm.Dirichlet("pi1", a=np.ones(n_states), dims="state")
        w = pm.Normal("w", mu=0, sigma=5, dims=("state", "feature"))
        b = pm.Normal("b", mu=0, sigma=5, dims=("state", "feature"))
        sigma = pm.HalfNormal(
            "sigma", sigma=3, initval=np.full(n_states, 3.0), dims="state"
        )
        # Stan's recurrence indexes these gates by the source state. Use the
        # destination state instead, as in the original data generator.
        # This changes the likelihood. Under the corrected gates, transitions
        # do not depend on the previous state, so the mixture below eliminates
        # the states exactly; that reduction introduces no approximation.
        later_weights = pm.math.softmax(pt.dot(u[1:], w.T), axis=1)
        weights = pt.concatenate([pi1[None, :], later_weights], axis=0)
        mean = pt.dot(u, b.T)
        pm.Mixture(
            "y",
            w=weights,
            comp_dists=pm.Normal.dist(mu=mean, sigma=sigma),
            observed=np.asarray(data["y"], dtype=float),
            dims="time",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
