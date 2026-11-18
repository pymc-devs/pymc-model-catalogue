"""
Model: Capture-recapture model where detection probability varies by occasion
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/Mtbh_model.stan
Authors: M Kery, M Schaub (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/Mtbh_model.py
Upstream catalogue contributor: Kane Lindsay
Description: The model Mtbh estimates the size of a closed population from capture-recapture data
    which has additive effects of time, behavior, and individual heterogeneity on the detection
    probability of a species

Data: Mtbh_data (upstream posterior: Mtbh_data-Mtbh_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/Mtbh_data.json.zip
    Archive SHA-256: 5a81ee8ba4e3723f63b053e258a6762f89c4809b351cd133820abb4a46a9f273
    Local archive: data/posteriordb_Mtbh_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/Mtbh_data-Mtbh_model.json
References: kery2011population (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/blob/master/BPA/Ch.06

Implementation notes:
Detection has additive occasion, behavior, and individual effects on log odds.

Source: Kéry and Schaub (2011), chapter 6.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -348.2499600792822; PyMC = -349.3485724197596.
- B: Stan = -343.17282999664394; PyMC = -344.27144233712136.
- Raw delta: Stan = 5.077130082638234; PyMC = 5.077130082638234.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.0507713008263823e-06
- PyMC's sigma Uniform(0,3) contributes only the constant -log(3).
- Original Stan SHA-256: 97c151a4cccca051c09de3dc1e66fe42980be9c850503db089e2bb8e7a16b451
- Saved points: models/posteriordb_logp/points/Mtbh_data-Mtbh_model.npz
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
