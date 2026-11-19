"""
Model: Inferring population size considering immediate trap-response
Source: posteriordb/posterior_database/models/stan/Mb_model.stan
Authors: M. Kery, M. Schaub
Description: Capture-recapture population model where capture probability is modeled by a different
    parameter for p depending on whether an animal was caught immediately before or not.

Stan comparison:
- A: Stan = -801.2928365793337; PyMC = -801.2928365793337.
- B: Stan = -785.5224343593771; PyMC = -785.5224343593771.
- Raw delta: Stan = 15.770402219956509; PyMC = 15.770402219956509.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 1.1577040221995651e-06

Benchmark results:
- Original:  logp = -802.2076, grad norm = 84.2924, 38.5 us/call (5 evals).
- Frozen:    logp = -802.2076, grad norm = 84.2924, 63.3 us/call (5 evals).
- Local NUMBA validation; short timings are indicative.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    def _capture_history_likelihood(history, omega, *, p=None, logit_p=None):
        """Integrate each population inclusion indicator across the complete capture history."""
        log_capture = pm.logp(pm.Bernoulli.dist(p=p, logit_p=logit_p), history).sum(
            axis=-1
        )
        log_present = pt.log(omega) + log_capture
        detected = np.any(history != 0, axis=-1)
        # The zero component applies to each whole history, not each occasion.
        pm.Potential(
            "capture_history",
            pt.switch(
                detected,
                log_present,
                pt.logaddexp(pt.log1p(-omega), log_present),
            ),
            dims="individual",
        )

    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_Mb_data.json.zip"
        ) as archive,
        archive.open("Mb_data.json") as stream,
    ):
        data = json.load(stream)
    history = np.asarray(data["y"], dtype="int64")
    previous_capture = np.concatenate(
        [np.zeros((data["M"], 1), dtype="int64"), history[:, :-1]], axis=1
    )
    coords = {
        "individual": np.arange(data["M"]),
        "capture_period": np.arange(data["T"]),
    }
    with pm.Model(coords=coords) as model:
        omega = pm.Uniform("omega", 0, 1)
        p = pm.Uniform("p", 0, 1)
        c = pm.Uniform("c", 0, 1)
        p_eff = pm.Deterministic(
            "p_eff",
            p + previous_capture * (c - p),
            dims=("individual", "capture_period"),
        )
        _capture_history_likelihood(history, omega, p=p_eff)
        pm.Deterministic("trap_response", c - p)
        no_capture = (1 - p) ** int(data["T"])
        pm.Deterministic(
            "omega_nd", omega * no_capture / (omega * no_capture + 1 - omega)
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
