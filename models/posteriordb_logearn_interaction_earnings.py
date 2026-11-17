"""
Model: Multiple Predictors Interacting Log-linear Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/logearn_interaction.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/logearn_interaction.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/logearn_interaction.py
Upstream catalogue contributor: Oliver Järnefelt
Description: log(earn) ~ height + male + height:male

Data: earnings (upstream posterior: earnings-logearn_interaction)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/earnings.json.zip
    Archive SHA-256: 06a67aa61a99d59aee057486dae51ffad6b4f887e7f8f17733739cf893af312f
    Local archive: data/posteriordb_earnings.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/earnings-logearn_interaction.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.4/logearn_interaction.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native Gaussian model for log earnings with a height-sex interaction.

Source: Gelman and Hill (2006), ARM.

The likelihood uses the transformed response, without a response Jacobian.

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
            Path(__file__).parent / "data" / "posteriordb_earnings.json.zip"
        ) as archive,
        archive.open("earnings.json") as stream,
    ):
        data = json.load(stream)
    response = np.log(np.asarray(data["earn"], dtype=float))
    height = np.asarray(data["height"], dtype=float)
    male = np.asarray(data["male"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            height,
            male,
            height * male,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": ["intercept", "height", "male", "height:male"],
    }
    initial_beta = np.zeros(design.shape[1])
    initial_beta[0] = response.mean()
    with pm.Model(coords=coords) as model:
        x = pm.Data("design", design, dims=("obs_id", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient", initval=initial_beta)
        sigma = pm.HalfFlat("sigma", initval=max(float(response.std()), 1.0))
        pm.Normal(
            "log_earn",
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
