"""
Model: Log-Log Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/logmesquite_logvash.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/logmesquite_logvash.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/logmesquite_logvash.py
Upstream catalogue contributor: Oliver Järnefelt
Description: log(weight) ~ log(canopy.volume) + log(canopy.area) + log(canopy.shape) +
    log(total.height) + group canopy_volume <- diam1 * diam2 * canopy_height canopy_area <- diam1 *
    diam2 canopy_shape <- diam1 / diam2

Data: mesquite (upstream posterior: mesquite-logmesquite_logvash)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/mesquite.json.zip
    Archive SHA-256: ec9db3a2e390892473e6172d76386ba9b26c3633a9682532eb34217189dcb0ac
    Local archive: data/posteriordb_mesquite.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/mesquite-logmesquite_logvash.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.4/mesquite_vash.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native Gaussian model for log mesquite weight with canopy geometry and height.

Source: Gelman and Hill (2006), ARM.

The likelihood uses the transformed response, without a response Jacobian.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -1020.8085671266597; PyMC = -1020.8085671266598.
- B: Stan = -669.1252463880306; PyMC = -669.1252463880307.
- Raw delta: Stan = 351.68332073862905; PyMC = 351.68332073862905.
- Raw comparison: MATCH; residual = 0.0.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 4.516833207386291e-06
  using adjusted delta_Stan when an adjustment is present.
- Identity on all constrained free parameters, including original array order. No
  coordinate-change Jacobian or parameter-dependent factor correction. Stan support-
  transform and PyMC value-transform Jacobians are disabled.
- Both use log(weight) as response; [1, log(diam1*diam2*canopy_height),
  log(diam1*diam2), log(diam1/diam2), log(total_height), group]. Beta order is
  unchanged, with Flat beta and HalfFlat sigma matching Stan's implicit priors. Response
  transformations are identical fixed-data operations, not parameter transforms
  requiring a Jacobian correction.
- Original Stan SHA-256: 100f4d97f30927b394c0284e92d62686fa5d16cb0cefc62c7e950fc4b7595296
- Saved points: models/posteriordb_logp/points/mesquite-logmesquite_logvash.npz
- Full evidence: models/posteriordb_logp/records/mesquite-logmesquite_logvash.json
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
            Path(__file__).parent / "data" / "posteriordb_mesquite.json.zip"
        ) as archive,
        archive.open("mesquite.json") as stream,
    ):
        data = json.load(stream)
    response = np.log(np.asarray(data["weight"], dtype=float))
    diam1 = np.asarray(data["diam1"], dtype=float)
    diam2 = np.asarray(data["diam2"], dtype=float)
    canopy_height = np.asarray(data["canopy_height"], dtype=float)
    log_canopy_volume = np.log(diam1 * diam2 * canopy_height)
    log_canopy_area = np.log(diam1 * diam2)
    log_canopy_shape = np.log(diam1 / diam2)
    log_total_height = np.log(np.asarray(data["total_height"], dtype=float))
    group = np.asarray(data["group"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            log_canopy_volume,
            log_canopy_area,
            log_canopy_shape,
            log_total_height,
            group,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": [
            "intercept",
            "log_canopy_volume",
            "log_canopy_area",
            "log_canopy_shape",
            "log_total_height",
            "group",
        ],
    }
    initial_beta = np.zeros(design.shape[1])
    initial_beta[0] = response.mean()
    with pm.Model(coords=coords) as model:
        x = pm.Data("design", design, dims=("obs_id", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient", initval=initial_beta)
        sigma = pm.HalfFlat("sigma", initval=max(float(response.std()), 1.0))
        pm.Normal(
            "log_weight",
            mu=pm.math.dot(x, beta),
            sigma=sigma,
            observed=response,
            dims="obs_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
