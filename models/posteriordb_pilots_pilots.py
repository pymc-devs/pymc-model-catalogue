"""
Model: Linear Mixed Effects Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/pilots.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/pilots.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/pilots.py
Upstream catalogue contributor: Oliver Järnefelt
Description: y ~ 1 + (1 | group) + (1 | scenario))

Data: pilots (upstream posterior: pilots-pilots)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/pilots.json.zip
    Archive SHA-256: b596d205c5393b61e923bb6d920c3a39f1c22b3968f7e86925a931c9006cc6f9
    Local archive: data/posteriordb_pilots.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/pilots-pilots.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.14/pilots.stan

Implementation notes:
Native crossed Gaussian effects for pilot groups and scenarios.

Source: Gelman and Hill (2006).
The group and scenario effects retain their centered parameterization.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -255.80167956170123; PyMC = -269.6171903115181.
- B: Stan = -259.8135650165997; PyMC = -273.6290757664166.
- Raw delta: Stan = -4.011885454898447; PyMC = -4.011885454898504.
- Raw comparison: MATCH; residual = 5.684341886080802e-14.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.040118854548985e-06
- One-based group/scenario indices become zero-based. Implicit Stan (0,100) priors
  versus three PyMC Uniforms add only -3*log(100).
- Original Stan SHA-256: 4bf2c1e00ab66ce90a014b1f48ee7d129601b93783f0cb850b5dacc5109fae19
- Saved points: models/posteriordb_logp/points/pilots-pilots.npz
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
            Path(__file__).parent / "data" / "posteriordb_pilots.json.zip"
        ) as archive,
        archive.open("pilots.json") as stream,
    ):
        data = json.load(stream)
    response = np.asarray(data["y"], dtype=float)
    group_id = np.asarray(data["group_id"], dtype=int) - 1
    scenario_id = np.asarray(data["scenario_id"], dtype=int) - 1
    coords = {
        "obs_id": np.arange(len(response)),
        "group": np.arange(1, int(data["n_groups"]) + 1),
        "scenario": np.arange(1, int(data["n_scenarios"]) + 1),
    }
    with pm.Model(coords=coords) as model:
        group_index = pm.Data("group_id", group_id, dims="obs_id")
        scenario_index = pm.Data("scenario_id", scenario_id, dims="obs_id")
        mu_a = pm.Normal("mu_a", mu=0, sigma=1)
        mu_b = pm.Normal("mu_b", mu=0, sigma=1)
        sigma_a = pm.Uniform("sigma_a", lower=0, upper=100)
        sigma_b = pm.Uniform("sigma_b", lower=0, upper=100)
        sigma_y = pm.Uniform("sigma_y", lower=0, upper=100)
        a = pm.Normal("a", mu=10 * mu_a, sigma=sigma_a, dims="group")
        b = pm.Normal("b", mu=10 * mu_b, sigma=sigma_b, dims="scenario")
        pm.Normal(
            "y",
            mu=a[group_index] + b[scenario_index],
            sigma=sigma_y,
            observed=response,
            dims="obs_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
