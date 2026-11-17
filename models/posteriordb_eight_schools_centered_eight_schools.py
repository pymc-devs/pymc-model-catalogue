"""
Model: A centered hiearchical model for 8 schools
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/eight_schools_centered.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/eight_schools_centered.py
Authors: Rubin, Donald B; Gelman, Andrew and Carlin, John B and Stern, Hal S and Dunson, David B and Vehtari, Aki and Rubin, Donald B (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/eight_schools_centered.py
Upstream catalogue contributor: Mans Magnusson
Description: A centered hiearchical model for the 8 schools example of Rubin (1981)

Data: eight_schools (upstream posterior: eight_schools-eight_schools_centered)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/eight_schools.json.zip
    Archive SHA-256: 8e37cc01fe05b0e1779255e9bd386261f3d61daf7f75f02f72f38cdc9750531c
    Local archive: data/posteriordb_eight_schools.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/eight_schools-eight_schools_centered.json
References: rubin1981estimation, gelman2013bayesian (see posteriordb/source/bibliography/references.bib)
Reference URL: http://www.stat.columbia.edu/~gelman/arm/examples/schools

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native centered school effects from posteriordb eight_schools_centered.

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
            Path(__file__).parent / "data" / "posteriordb_eight_schools.json.zip"
        ) as archive,
        archive.open("eight_schools.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model(coords={"school": np.arange(1, data["J"] + 1)}) as model:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta = pm.Normal("theta", mu, tau, dims="school")
        pm.Normal(
            "y",
            theta,
            sigma=np.asarray(data["sigma"], dtype=float),
            observed=data["y"],
            dims="school",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
