"""
Model: BYM2 model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/bym2_offset_only.stan
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/bym2_offset_only.py
Upstream catalogue contributor: Keane Nguyen
Description: The BYM2 model was proposed by Riebler et al 2016, following Simpson 2014. Like the BYM
    model, it includes two random effects components, and like the alternative Leroux (1999) model,
    it places a single precision (scale) parameter on the combined components, and a mixing
    parameter for the amount of spatial/non-spatial variation.

Data: traffic_accident_nyc (upstream posterior: traffic_accident_nyc-bym2_offset_only)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/traffic_accident_nyc.json.zip
    Archive SHA-256: f63246ccbdaaf129fb20f27c0f65f5a32b87cda8e53dfb211bcc78618799bebc
    Local archive: data/posteriordb_traffic_accident_nyc.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/traffic_accident_nyc-bym2_offset_only.json
Reference URL: https://github.com/stan-dev/example-models/tree/master/knitr/car-iar-poisson
Reference URL: https://mc-stan.org/users/documentation/case-studies/icar_stan.html

Implementation notes:
Native BYM2 Poisson model from the posteriordb spatial case study.

The ICAR law retains the published soft sum constraint.
Source: https://mc-stan.org/users/documentation/case-studies/icar_stan.html

Stan comparison:
- A: Stan = -6434575.605012348; PyMC = -6434574.9118651245.
- B: Stan = -7250370.332527916; PyMC = -7250369.639380687.
- Raw delta: Stan = -815794.727515568; PyMC = -815794.7275155624.
- Raw comparison: MATCH; residual = -5.587935447692871e-09.
- Tolerance: 0.00815894727515568
- HalfNormal versus a Normal density on positive sigma contributes only log(2).
- Original Stan SHA-256: 0137ba98292c5dfeeb4cefa006a740f6d362fcf87ca02aa7b668b3040147ead1

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
            Path(__file__).parent / "data" / "posteriordb_traffic_accident_nyc.json.zip"
        ) as archive,
        archive.open("traffic_accident_nyc.json") as stream,
    ):
        data = json.load(stream)
    n = data["N"]
    node1 = np.asarray(data["node1"], dtype="int64") - 1
    node2 = np.asarray(data["node2"], dtype="int64") - 1
    adjacency = np.zeros((n, n), dtype="int8")
    adjacency[node1, node2] = 1
    adjacency[node2, node1] = 1
    coords = {"area": np.arange(1, n + 1)}
    with pm.Model(coords=coords) as model:
        log_exposure = pm.Data(
            "log_E", np.log(np.asarray(data["E"], dtype=float)), dims="area"
        )
        beta0 = pm.Normal("beta0", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1, initval=1)
        rho = pm.Beta("rho", alpha=0.5, beta=0.5, initval=0.5)
        theta = pm.Normal("theta", mu=0, sigma=1, dims="area")
        phi = pm.ICAR(
            "phi",
            W=adjacency,
            sigma=1,
            zero_sum_stdev=0.001,
            dims="area",
            initval=np.zeros(n),
        )
        convolved_re = pm.Deterministic(
            "convolved_re",
            pt.sqrt(1 - rho) * theta + pt.sqrt(rho / data["scaling_factor"]) * phi,
            dims="area",
        )
        eta = pm.Deterministic(
            "eta", log_exposure + beta0 + sigma * convolved_re, dims="area"
        )
        mu = pm.Deterministic("mu", pt.exp(eta), dims="area")
        pm.Poisson(
            "y", mu=mu, observed=np.asarray(data["y"], dtype="int64"), dims="area"
        )
        pm.Deterministic("log_precision", -2 * pt.log(sigma))
        pm.Deterministic("logit_rho", pt.log(rho) - pt.log1p(-rho))
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
