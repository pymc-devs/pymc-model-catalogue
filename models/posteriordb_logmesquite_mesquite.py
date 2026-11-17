"""
Model: Multiple Predictors Log-Log Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/logmesquite.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/logmesquite.py
Upstream catalogue contributor: Oliver Järnefelt
Description: log(weight) ~ log(diam1) + log(diam2) + log(canopy.height) + log(total.height) +
    log(density) + group

Data: mesquite (upstream posterior: mesquite-logmesquite)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/mesquite.json.zip
    Archive SHA-256: ec9db3a2e390892473e6172d76386ba9b26c3633a9682532eb34217189dcb0ac
    Local archive: data/posteriordb_mesquite.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/mesquite-logmesquite.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.4/mesquite_log.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native Gaussian model for log mesquite weight and log predictors.

Source: Gelman and Hill (2006), ARM.

The likelihood uses the transformed response, without a response Jacobian.

Stan comparison:
- Not recorded.

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
    log_diam1 = np.log(np.asarray(data["diam1"], dtype=float))
    log_diam2 = np.log(np.asarray(data["diam2"], dtype=float))
    log_canopy_height = np.log(np.asarray(data["canopy_height"], dtype=float))
    log_total_height = np.log(np.asarray(data["total_height"], dtype=float))
    log_density = np.log(np.asarray(data["density"], dtype=float))
    group = np.asarray(data["group"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            log_diam1,
            log_diam2,
            log_canopy_height,
            log_total_height,
            log_density,
            group,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": [
            "intercept",
            "log_diam1",
            "log_diam2",
            "log_canopy_height",
            "log_total_height",
            "log_density",
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
