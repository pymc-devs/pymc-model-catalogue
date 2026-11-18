"""
Model: Multiple Predictor Linear Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/nes.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/nes.py
Upstream catalogue contributor: Oliver Järnefelt
Description: partyid7 ~ real_ideo + race_adj + age30_44 + age45_64 + age65up + educ1 + gender +
    income

Data: nes1996 (upstream posterior: nes1996-nes)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/nes1996.json.zip
    Archive SHA-256: 1ce70d32edefe5e612f2bef1cf2afa477150caa85eb4a5cddbfe53e664717672
    Local archive: data/posteriordb_nes1996.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/nes1996-nes.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.4/nes.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native Gaussian regression for National Election Study party identification.

Source: Gelman and Hill (2006), ARM.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -12871.551714773277; PyMC = -12871.55171477326.
- B: Stan = -6988.179245364434; PyMC = -6988.179245364431.
- Raw delta: Stan = 5883.372469408843; PyMC = 5883.37246940883.
- Raw comparison: MATCH; residual = 1.3642420526593924e-11.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 5.983372469408843e-05
  using adjusted delta_Stan when an adjustment is present.
- Identity on all constrained free parameters, including original array order. No
  coordinate-change Jacobian or parameter-dependent factor correction. Stan support-
  transform and PyMC value-transform Jacobians are disabled.
- Identical beta order [intercept, real_ideo, race_adj, age==2, age==3, age==4, educ1,
  gender, income], partyid7 response and flat beta/sigma>0 priors.
- Original Stan SHA-256: 260e7acec9bef905cae306ff6defb97e525c116f018aba9c687af154c10c8e70
- Saved points: models/posteriordb_logp/points/nes1996-nes.npz
- Full evidence: models/posteriordb_logp/records/nes1996-nes.json
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
            Path(__file__).parent / "data" / "posteriordb_nes1996.json.zip"
        ) as archive,
        archive.open("nes1996.json") as stream,
    ):
        data = json.load(stream)
    response = np.asarray(data["partyid7"], dtype=float)
    real_ideo = np.asarray(data["real_ideo"], dtype=float)
    race_adj = np.asarray(data["race_adj"], dtype=float)
    age = np.asarray(data["age_discrete"], dtype=int)
    educ1 = np.asarray(data["educ1"], dtype=float)
    gender = np.asarray(data["gender"], dtype=float)
    income = np.asarray(data["income"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            real_ideo,
            race_adj,
            age == 2,
            age == 3,
            age == 4,
            educ1,
            gender,
            income,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": [
            "intercept",
            "real_ideo",
            "race_adj",
            "age30_44",
            "age45_64",
            "age65up",
            "educ1",
            "gender",
            "income",
        ],
    }
    initial_beta = np.zeros(design.shape[1])
    initial_beta[0] = response.mean()
    with pm.Model(coords=coords) as model:
        x = pm.Data("design", design, dims=("obs_id", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient", initval=initial_beta)
        sigma = pm.HalfFlat("sigma", initval=max(float(response.std()), 1.0))
        pm.Normal(
            "partyid7",
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
