"""
Model: Multiple Linearly Transformed Predictors Interacting Log-linear Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/logearn_interaction_z.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/logearn_interaction_z.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/logearn_interaction_z.py
Upstream catalogue contributor: Oliver Järnefelt
Description: log(earn) ~ z_height + male + z_height:male

Data: earnings (upstream posterior: earnings-logearn_interaction_z)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/earnings.json.zip
    Archive SHA-256: 06a67aa61a99d59aee057486dae51ffad6b4f887e7f8f17733739cf893af312f
    Local archive: data/posteriordb_earnings.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/earnings-logearn_interaction_z.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.4/logearn_interaction_z.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native Gaussian model for log earnings with a standardized height-sex interaction.

Source: Gelman and Hill (2006), ARM.

The likelihood uses the transformed response, without a response Jacobian.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -62325.701674457756; PyMC = -62325.70167445774.
- B: Stan = -38591.14592114192; PyMC = -38591.14592114192.
- Raw delta: Stan = 23734.555753315835; PyMC = 23734.55575331582.
- Raw comparison: MATCH; residual = 1.4551915228366852e-11.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 0.00023834555753315836
  using adjusted delta_Stan when an adjustment is present.
- Identity on all constrained free parameters, including original array order. No
  coordinate-change Jacobian or parameter-dependent factor correction. Stan support-
  transform and PyMC value-transform Jacobians are disabled.
- Both use log(earn) as response; [1, z_height, male, z_height*male], where
  z_height=(height-mean(height))/sd(height), using sample SD (ddof=1). Beta order is
  unchanged, with Flat beta and HalfFlat sigma matching Stan's implicit priors. Response
  transformations are identical fixed-data operations, not parameter transforms
  requiring a Jacobian correction.
- Original Stan SHA-256: 37e04ae7ae7e92d3d4fffc6720298a56d0a13d87f376ae34b19d3ba731551749
- Saved points: models/posteriordb_logp/points/earnings-logearn_interaction_z.npz
- Full evidence: models/posteriordb_logp/records/earnings-logearn_interaction_z.json
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
            Path(__file__).parent / "data" / "posteriordb_earnings.json.zip"
        ) as archive,
        archive.open("earnings.json") as stream,
    ):
        data = json.load(stream)
    response = np.log(np.asarray(data["earn"], dtype=float))
    height = np.asarray(data["height"], dtype=float)
    male = np.asarray(data["male"], dtype=float)
    z_height = (height - height.mean()) / height.std(ddof=1)
    design = np.column_stack(
        (
            np.ones(len(response)),
            z_height,
            male,
            z_height * male,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": ["intercept", "z_height", "male", "z_height:male"],
    }
    initial_beta = np.zeros(design.shape[1])
    initial_beta[0] = response.mean()
    with pm.Model(coords=coords) as model:
        x = pm.Data("design", design, dims=("obs_id", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient", initval=initial_beta)
        sigma = pm.HalfFlat("sigma", initval=max(float(response.std()), 1.0))
        pm.Normal(
            "log_earn",
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
