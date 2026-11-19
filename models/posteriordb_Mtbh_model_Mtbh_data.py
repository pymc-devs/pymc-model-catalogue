"""Capture-recapture model with time, behavioral and individual detection effects.

Stan comparison:
- A: Stan = -348.2499600792822; PyMC = -349.3485724197596.
- B: Stan = -343.17282999664394; PyMC = -344.27144233712136.
- Raw delta: Stan = 5.077130082638234; PyMC = 5.077130082638234.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 1.0507713008263823e-06
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
            Path(__file__).parent / "data" / "posteriordb_Mtbh_data.json.zip"
        ) as archive,
        archive.open("Mtbh_data.json") as stream,
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
        mean_p = pm.Uniform("mean_p", 0, 1, dims="capture_period")
        gamma = pm.Normal("gamma", 0, 10)
        sigma = pm.Uniform("sigma", 0, 3)
        eps_raw = pm.Normal("eps_raw", 0, 1, dims="individual")
        eps = pm.Deterministic("eps", sigma * eps_raw, dims="individual")
        alpha = pm.Deterministic("alpha", pm.math.logit(mean_p), dims="capture_period")
        logit_p = pm.Deterministic(
            "logit_p",
            alpha[None, :] + eps[:, None] + gamma * previous_capture,
            dims=("individual", "capture_period"),
        )
        _capture_history_likelihood(history, omega, logit_p=logit_p)
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
