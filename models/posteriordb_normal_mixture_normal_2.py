"""
Model: Two Component Gaussian Mixture Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/normal_mixture.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/normal_mixture.py
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/normal_mixture.py
Upstream catalogue contributor: Oliver Järnefelt
Description: a gaussian mixture with unknown proportion and means, known variance p(y|mu,theta) =
    theta * Normal(y|mu[1],1) + (1-theta) * Normal(y|mu[2],1)

Data: normal_2 (upstream posterior: normal_2-normal_mixture)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/normal_2.json.zip
    Archive SHA-256: c34231a6c79c1a262444d07f768c7eb65a42aa76541d309d8c93049f023da21e
    Local archive: data/posteriordb_normal_2.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/normal_2-normal_mixture.json
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/basic_estimators/normal_mixture.stan

Implementation notes:
Two-component Normal mixture with known unit standard deviations.

Source: posteriordb normal_mixture model metadata at revision 91027f9.
The likelihood uses only y. Other dataset fields describe the data generator.

Stan comparison:
- A: Stan = -52362.52849736699; PyMC = -52362.52849743095.
- B: Stan = -51359.06365560339; PyMC = -51359.06365566734.
- Raw delta: Stan = 1003.4648417635981; PyMC = 1003.4648417636054.
- Raw comparison: MATCH; residual = -7.275957614183426e-12.
- Tolerance: 1.1034648417636055e-05
- Original Stan SHA-256: 347644c84000e76f74c10314dda7774fd2d3b13cd53f37f049b6bfdc2054c87e

Benchmark results:
- Not measured.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_normal_2.json.zip"
        ) as archive,
        archive.open("normal_2.json") as stream,
    ):
        data = json.load(stream)
    y = np.asarray(data["y"], dtype=float)
    coords = {"component": np.arange(2), "observation": np.arange(data["N"])}
    with pm.Model(coords=coords) as model:
        theta = pm.Uniform("theta", lower=0, upper=1, initval=0.5)
        mu = pm.Normal(
            "mu", mu=0, sigma=10, dims="component", initval=np.array([-1.0, 1.0])
        )
        weights = pt.stack([theta, 1 - theta])
        pm.NormalMixture(
            "y", w=weights, mu=mu, sigma=1.0, observed=y, dims="observation"
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
