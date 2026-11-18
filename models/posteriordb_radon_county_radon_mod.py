"""
Model: Hierarchical Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/radon_county.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/radon_county.py
Authors: Bales, Ben and Pourzanjani, Arya and Vehtari, Aki and Petzold, Linda (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/radon_county.py
Upstream catalogue contributor: Oliver Järnefelt
Description: y ~ (1|county)

Data: radon_mod (upstream posterior: radon_mod-radon_county)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/radon_mod.json.zip
    Archive SHA-256: 341aa024485b6a7b1d7bf1ec1ae50124a59b38387e47a88db620ae37848aac66
    Local archive: data/posteriordb_radon_mod.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/radon_mod-radon_county.json
References: bales2019selecting (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/bbbales2/cmdstan-warmup/develop/examples/radon/radon.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native county radon hierarchy from posteriordb, Bales et al. (2019).

The county model uses bounded uniform scale distributions.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -62822.486080068884; PyMC = -62831.69642056915.
- B: Stan = -62947.435689867634; PyMC = -62956.6460303676.
- Raw delta: Stan = -124.94960979875032; PyMC = -124.949609798452.
- Raw comparison: MATCH; residual = -2.9831426218152046e-10.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 2.249496097987503e-06
  using adjusted delta_Stan when an adjustment is present.
- Identity on all constrained free parameters, including original array order. No
  coordinate-change Jacobian or parameter-dependent factor correction. Stan support-
  transform and PyMC value-transform Jacobians are disabled.
- County labels are converted from Stan one-based to NumPy zero-based indices without
  reordering observations or county parameters. Residual SD and all varying-effect SDs
  retain their names and units.
- a, mu_a, sigma_a, sigma_y are unchanged. Stan implicit SD priors on (0,100) versus
  PyMC Uniform(0,100) contribute only -2*log(100).
- Original Stan SHA-256: a198eede373add8ab481a34eee83fc23410299b5c5982dd0bcbb738eb21ffd55
- Saved points: models/posteriordb_logp/points/radon_mod-radon_county.npz
- Full evidence: models/posteriordb_logp/records/radon_mod-radon_county.json
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
            Path(__file__).parent / "data" / "posteriordb_radon_mod.json.zip"
        ) as archive,
        archive.open("radon_mod.json") as stream,
    ):
        data = json.load(stream)
    coords = {"obs_id": np.arange(data["N"]), "county": np.arange(1, data["J"] + 1)}
    with pm.Model(coords=coords) as model:
        county = pm.Data(
            "county_idx", np.asarray(data["county"], dtype="int64") - 1, dims="obs_id"
        )
        mu_a = pm.Normal("mu_a", mu=0, sigma=1)
        sigma_a = pm.Uniform("sigma_a", lower=0, upper=100, initval=1)
        sigma_y = pm.Uniform("sigma_y", lower=0, upper=100, initval=1)
        a = pm.Normal(
            "a", mu=mu_a, sigma=sigma_a, dims="county", initval=np.zeros(data["J"])
        )
        pm.Normal(
            "y",
            mu=a[county],
            sigma=sigma_y,
            observed=np.asarray(data["y"], dtype=float),
            dims="obs_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
