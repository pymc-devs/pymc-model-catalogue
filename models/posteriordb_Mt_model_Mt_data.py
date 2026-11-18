"""
Model: Capture-recapture model where detection probability varies by occasion
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/Mt_model.stan
Authors: M Kery, M Schaub (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/Mt_model.py
Upstream catalogue contributor: Kane Lindsay
Description: A capture-recapture sampling model where that assumes detection probability p varies
    only by occasion, perhaps because of weather conditions or because different traps or detection
    devices were used

Data: Mt_data (upstream posterior: Mt_data-Mt_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/Mt_data.json.zip
    Archive SHA-256: c16b3b0f162ea53c80e8e3b924969ed7ebdf76cf20138fbb680355e794246287
    Local archive: data/posteriordb_Mt_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/Mt_data-Mt_model.json
References: kery2011population (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/blob/master/BPA/Ch.06

Implementation notes:
Detection probabilities vary by occasion.

Source: Kéry and Schaub (2011), chapter 6.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -333.5695376087196; PyMC = -333.5695376087196.
- B: Stan = -330.28450715685545; PyMC = -330.28450715685545.
- Raw delta: Stan = 3.285030451864145; PyMC = 3.285030451864145.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.0328503045186415e-06
- Original Stan SHA-256: 3391120504996a0369e58658d1872f190d2932e5c7d192f63c3db2057e8d324c
- Saved points: models/posteriordb_logp/points/Mt_data-Mt_model.npz
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
            Path(__file__).parent / "data" / "posteriordb_Mt_data.json.zip"
        ) as archive,
        archive.open("Mt_data.json") as stream,
    ):
        data = json.load(stream)
    history = np.asarray(data["y"], dtype="int64")
    coords = {
        "individual": np.arange(data["M"]),
        "capture_period": np.arange(data["T"]),
    }
    with pm.Model(coords=coords) as model:
        omega = pm.Uniform("omega", 0, 1)
        p = pm.Uniform("p", 0, 1, dims="capture_period")
        _capture_history_likelihood(history, omega, p=p)
        no_capture = (1 - p).prod()
        pm.Deterministic(
            "omega_nd", omega * no_capture / (omega * no_capture + 1 - omega)
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
