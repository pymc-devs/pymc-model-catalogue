"""
Model: K Component Gaussian Mixture Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/normal_mixture_k.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/normal_mixture_k.py
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/normal_mixture_k.py
Upstream catalogue contributor: Oliver Järnefelt
Description: a Gaussian mixture with unknown proportions, means and variances

Data: normal_5 (upstream posterior: normal_5-normal_mixture_k)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/normal_5.json.zip
    Archive SHA-256: a4d19461b4274bafc5600325e41121d3c25b0569ce739189ec8792ced2744425
    Local archive: data/posteriordb_normal_5.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/normal_5-normal_mixture_k.json
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/basic_estimators/normal_mixture_k.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
K-component Normal mixture with unknown means, weights, and standard deviations.

Source: posteriordb normal_mixture_k model metadata at revision 91027f9.
A uniform simplex density gives Dirichlet weights with unit concentrations.
The likelihood uses only y. Other dataset fields describe the data generator.

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
            Path(__file__).parent / "data" / "posteriordb_normal_5.json.zip"
        ) as archive,
        archive.open("normal_5.json") as stream,
    ):
        data = json.load(stream)
    y = np.asarray(data["y"], dtype=float)
    n_components = int(data["K"])
    coords = {
        "component": np.arange(n_components),
        "observation": np.arange(data["N"]),
    }
    with pm.Model(coords=coords) as model:
        theta = pm.Dirichlet(
            "theta",
            a=np.ones(n_components),
            dims="component",
            initval=np.full(n_components, 1.0 / n_components),
        )
        mu = pm.Normal(
            "mu",
            mu=0,
            sigma=10,
            dims="component",
            initval=np.linspace(-2.0, 2.0, n_components),
        )
        sigma = pm.Uniform(
            "sigma",
            lower=0,
            upper=10,
            dims="component",
            initval=np.full(n_components, 5.0),
        )
        pm.NormalMixture(
            "y", w=theta, mu=mu, sigma=sigma, observed=y, dims="observation"
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
