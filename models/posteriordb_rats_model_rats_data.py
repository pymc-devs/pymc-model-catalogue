"""
Model: Normal Hierarchical Model to Model Rats' Weight Over Time
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/rats_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/rats_model.py
Authors: Alan E. Gelfand and Susan E. Hills and Amy Racine-Poon and Adrian F. M. Smith (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/rats_model.py
Upstream catalogue contributor: Kane Lindsay
Description: The model is essentially a random effects linear growth curve.

Data: rats_data (upstream posterior: rats_data-rats_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/rats_data.json.zip
    Archive SHA-256: 10ceca35c2f65e56b442a4adc7c4e8e26be0edb7ad11e76cd3ddc28a22cb3789
    Local archive: data/posteriordb_rats_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/rats_data-rats_model.json
References: gelfand1990illustration (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/bugs_examples/vol1/rats

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native centered growth effects from posteriordb rats_model.

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
            Path(__file__).parent / "data" / "posteriordb_rats_data.json.zip"
        ) as archive,
        archive.open("rats_data.json") as stream,
    ):
        data = json.load(stream)
    coords = {
        "rat_id": np.arange(1, data["N"] + 1),
        "observation": np.arange(data["Npts"]),
    }
    with pm.Model(coords=coords) as model:
        rat = pm.Data(
            "rat_index", np.asarray(data["rat"], dtype="int64") - 1, dims="observation"
        )
        centered_x = pm.Data(
            "centered_x",
            np.asarray(data["x"], dtype=float) - data["xbar"],
            dims="observation",
        )
        mu_alpha = pm.Normal("mu_alpha", 0, 100, initval=200.0)
        mu_beta = pm.Normal("mu_beta", 0, 100, initval=6.0)
        # These improper scale densities stay flat on positive scales.
        sigma_y = pm.HalfFlat("sigma_y", initval=10.0)
        sigma_alpha = pm.HalfFlat("sigma_alpha", initval=20.0)
        sigma_beta = pm.HalfFlat("sigma_beta", initval=1.0)
        alpha = pm.Normal(
            "alpha",
            mu_alpha,
            sigma_alpha,
            dims="rat_id",
            initval=np.full(data["N"], 200.0),
        )
        beta = pm.Normal(
            "beta", mu_beta, sigma_beta, dims="rat_id", initval=np.full(data["N"], 6.0)
        )
        pm.Normal(
            "y",
            mu=alpha[rat] + beta[rat] * centered_x,
            sigma=sigma_y,
            observed=data["y"],
            dims="observation",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
