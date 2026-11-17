"""
Model: Dugong Age and Length
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/dugongs_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/dugongs_model.py
Authors: MRC Biostatistics Unit (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/dugongs_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Model of age and length data of dugongs using a nonlinear growth curve with no
    inflection point.

Data: dugongs_data (upstream posterior: dugongs_data-dugongs_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/dugongs_data.json.zip
    Archive SHA-256: 44b80942203caca9546646d14d9adce3f898b5789ecbf4f023ed25790b757b5c
    Local archive: data/posteriordb_dugongs_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/dugongs_data-dugongs_model.json
References: mrc2winbugs (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/bugs_examples/vol2

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native growth curve from posteriordb dugongs_model.

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
            Path(__file__).parent / "data" / "posteriordb_dugongs_data.json.zip"
        ) as archive,
        archive.open("dugongs_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model(coords={"animal": np.arange(1, data["N"] + 1)}) as model:
        x = pm.Data("x", np.asarray(data["x"], dtype=float), dims="animal")
        alpha = pm.Normal("alpha", 0, 1000, initval=2.5)
        beta = pm.Normal("beta", 0, 1000, initval=1.0)
        growth = pm.Uniform("lambda", 0.5, 1)
        tau = pm.Gamma("tau", alpha=0.0001, beta=0.0001, initval=1.0)
        sigma = pm.Deterministic("sigma", 1 / pm.math.sqrt(tau))
        pm.Normal(
            "Y",
            mu=alpha - beta * growth**x,
            sigma=sigma,
            observed=data["Y"],
            dims="animal",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
