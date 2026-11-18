"""
Model: Logistic-Normal Heterogeneity Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/Mh_model.stan
Authors: M Kery, M Schaub (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/Mh_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Heterogeneity is inherent difference in the behaviour of individuals. Individual
    heterogeneity is modeled as random noise around some mean on a logit-transformed scale, and the
    model for the noise is the normal distribution.

Data: Mh_data (upstream posterior: Mh_data-Mh_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/Mh_data.json.zip
    Archive SHA-256: e074aa597e15cf55f557cd898629390f26197e4a71655ad607478c662de45f7a
    Local archive: data/posteriordb_Mh_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/Mh_data-Mh_model.json
References: kery2011population (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/blob/master/BPA/Ch.06

Implementation notes:
Individual detection uses noncentered Gaussian effects on log odds.

Source: Kéry and Schaub (2011), chapter 6.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -753.2032385618709; PyMC = -754.8126765043747.
- B: Stan = -747.3552538441143; PyMC = -748.9646917866185.
- Raw delta: Stan = 5.847984717756617; PyMC = 5.847984717756276.
- Raw comparison: MATCH; residual = 3.410605131648481e-13.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.058479847177566e-06
- PyMC Uniform(0,5) for sigma adds only -log(5) relative to Stan's implicit bounded-flat
  prior.
- Original Stan SHA-256: 69dda71b0ec6857c4c5a0e81c014979f57da97c5260b6e3456305a13050b7ad5
- Saved points: models/posteriordb_logp/points/Mh_data-Mh_model.npz
- Two-point evidence only; no global equivalence, support or inference claim.

Benchmark results:
- Not measured.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_Mh_data.json.zip"
        ) as archive,
        archive.open("Mh_data.json") as stream,
    ):
        data = json.load(stream)
    counts = np.asarray(data["y"], dtype="int64")
    with pm.Model(coords={"individual": np.arange(data["M"])}) as model:
        omega = pm.Uniform("omega", 0, 1)
        mean_p = pm.Uniform("mean_p", 0, 1)
        sigma = pm.Uniform("sigma", 0, 5)
        eps_raw = pm.Normal("eps_raw", 0, 1, dims="individual")
        eps = pm.Deterministic(
            "eps", pm.math.logit(mean_p) + sigma * eps_raw, dims="individual"
        )
        pm.ZeroInflatedBinomial(
            "captures",
            psi=omega,
            n=data["T"],
            p=pm.math.sigmoid(eps),
            observed=counts,
            dims="individual",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
