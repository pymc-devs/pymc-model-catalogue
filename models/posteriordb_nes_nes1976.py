"""
Model: Multiple Predictor Linear Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/nes.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/nes.py
Upstream catalogue contributor: Oliver Järnefelt
Description: partyid7 ~ real_ideo + race_adj + age30_44 + age45_64 + age65up + educ1 + gender +
    income

Data: nes1976 (upstream posterior: nes1976-nes)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/nes1976.json.zip
    Archive SHA-256: 22198de6ae1a6f2f1002b7c25a2eb7bf1c78f473839b9e942e7b604c375fd1cf
    Local archive: data/posteriordb_nes1976.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/nes1976-nes.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.4/nes.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native Gaussian regression for National Election Study party identification.

Source: Gelman and Hill (2006), ARM.

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
            Path(__file__).parent / "data" / "posteriordb_nes1976.json.zip"
        ) as archive,
        archive.open("nes1976.json") as stream,
    ):
        data = json.load(stream)
    response = np.asarray(data["partyid7"], dtype=float)
    real_ideo = np.asarray(data["real_ideo"], dtype=float)
    race_adj = np.asarray(data["race_adj"], dtype=float)
    age = np.asarray(data["age_discrete"], dtype=int)
    educ1 = np.asarray(data["educ1"], dtype=float)
    gender = np.asarray(data["gender"], dtype=float)
    income = np.asarray(data["income"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            real_ideo,
            race_adj,
            age == 2,
            age == 3,
            age == 4,
            educ1,
            gender,
            income,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": [
            "intercept",
            "real_ideo",
            "race_adj",
            "age30_44",
            "age45_64",
            "age65up",
            "educ1",
            "gender",
            "income",
        ],
    }
    initial_beta = np.zeros(design.shape[1])
    initial_beta[0] = response.mean()
    with pm.Model(coords=coords) as model:
        x = pm.Data("design", design, dims=("obs_id", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient", initval=initial_beta)
        sigma = pm.HalfFlat("sigma", initval=max(float(response.std()), 1.0))
        pm.Normal(
            "partyid7",
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
