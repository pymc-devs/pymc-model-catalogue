"""Input Output HMM.

Stan comparison:
- A: Stan = -12251.424541381222; PyMC = -12250.068644344123.
- B: Stan = -13444.51019509857; PyMC = -13441.722573636545.
- Raw delta: Stan = -1193.0856537173477; PyMC = -1191.6539292924226.
- Raw comparison: MISMATCH; residual = -1.4317244249250507.
- Additive Stan adjustment: A = -1.416690903932249;
  B = 0.01503352099825861.
- Adjusted Stan delta = -1191.6539292924172.
- Adjusted comparison: MATCH; residual = 5.4569682106375694e-12.
- Tolerance: 1.2916539292924226e-05
PyMC gates destination states rather than source states.
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
