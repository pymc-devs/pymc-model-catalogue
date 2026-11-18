"""
Model: A non-centered hiearchical model for 8 schools
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/eight_schools_noncentered.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/eight_schools_noncentered.py
Authors: Rubin, Donald B; Gelman, Andrew and Carlin, John B and Stern, Hal S and Dunson, David B and Vehtari, Aki and Rubin, Donald B (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/eight_schools_noncentered.py
Upstream catalogue contributor: Mans Magnusson
Description: A non-centered hiearchical model for the 8 schools example of Rubin (1981)

Data: eight_schools (upstream posterior: eight_schools-eight_schools_noncentered)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/eight_schools.json.zip
    Archive SHA-256: 8e37cc01fe05b0e1779255e9bd386261f3d61daf7f75f02f72f38cdc9750531c
    Local archive: data/posteriordb_eight_schools.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/eight_schools-eight_schools_noncentered.json
References: rubin1981estimation, gelman2013bayesian (see posteriordb/source/bibliography/references.bib)
Reference URL: http://www.stat.columbia.edu/~gelman/arm/examples/schools

Implementation notes:
Native noncentered school effects from posteriordb eight_schools_noncentered.

Stan comparison:
- A: Stan = -44.15124349654224; PyMC = -43.458096374219195.
- B: Stan = -44.22979687321376; PyMC = -43.53664975089073.
- Raw delta: Stan = -0.07855337667151474; PyMC = -0.07855337667153606.
- Raw comparison: MATCH; residual = 2.1316282072803006e-14.
- Tolerance: 1.0007855337667153e-06
- Stan tau>0 with Cauchy(0,5) versus PyMC HalfCauchy(5) differs only by the constant
  log(2).
- Original Stan SHA-256: b28d5326b1936cce0014c98c2591d1ba5c1d21987ea5a5d0d9c074c6d6ce9253

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
        theta_trans = pm.Normal("theta_trans", 0, 1, dims="school")
        theta = pm.Deterministic("theta", mu + tau * theta_trans, dims="school")
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
