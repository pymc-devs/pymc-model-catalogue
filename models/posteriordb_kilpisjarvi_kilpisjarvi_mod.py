"""
Model: Multiple Highly Correlated Predictors Linear Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/kilpisjarvi.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/kilpisjarvi.py
Authors: Bales, Ben and Pourzanjani, Arya and Vehtari, Aki and Petzold, Linda (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/kilpisjarvi.py
Upstream catalogue contributor: Oliver Järnefelt
Description: y ~ x

Data: kilpisjarvi_mod (upstream posterior: kilpisjarvi_mod-kilpisjarvi)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/kilpisjarvi_mod.json.zip
    Archive SHA-256: 232e8c55bbe5c07b79b0b28c92917f71711c713579292a2f40a50bda279d968e
    Local archive: data/posteriordb_kilpisjarvi_mod.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/kilpisjarvi_mod-kilpisjarvi.json
References: bales2019selecting (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/bbbales2/cmdstan-warmup/develop/examples/kilpisjarvi/kilpisjarvi.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Gaussian temperature regression from the posteriordb kilpisjarvi probability model.

The observation scale has the source model's improper positive flat density.
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.

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
            Path(__file__).parent / "data" / "posteriordb_kilpisjarvi_mod.json.zip"
        ) as archive,
        archive.open("kilpisjarvi_mod.json") as stream,
    ):
        data = json.load(stream)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    with pm.Model(coords={"year": np.arange(int(data["N"]))}) as model:
        alpha = pm.Normal("alpha", mu=data["pmualpha"], sigma=data["psalpha"])
        beta = pm.Normal("beta", mu=data["pmubeta"], sigma=data["psbeta"])
        sigma = pm.HalfFlat("sigma", initval=1)
        time = pm.Data("x", x, dims="year")
        pm.Normal("y", mu=alpha + beta * time, sigma=sigma, observed=y, dims="year")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
