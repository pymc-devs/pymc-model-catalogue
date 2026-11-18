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

Implementation notes:
K-component Normal mixture with unknown means, weights, and standard deviations.

Source: posteriordb normal_mixture_k model metadata at revision 91027f9.
A uniform simplex density gives Dirichlet weights with unit concentrations.
The likelihood uses only y. Other dataset fields describe the data generator.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -5480.534220963223; PyMC = -5488.869093394437.
- B: Stan = -5514.887037286612; PyMC = -5523.221909717825.
- Raw delta: Stan = -34.35281632338956; PyMC = -34.35281632338774.
- Raw comparison: MATCH; residual = -1.8189894035458565e-12.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.3435281632338956e-06
- PyMC's Dirichlet(ones) and Uniform(0,10) normalizers differ from Stan's implicit
  bounded flat densities only by constants.
- Original Stan SHA-256: e9a39473b1e2c909097a5b8b265c5f47b8a5d6a609e9bd282f35355d770a5c5a
- Saved points: models/posteriordb_logp/points/normal_5-normal_mixture_k.npz
- Two-point evidence only; no global equivalence, support or inference claim.

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
