"""
Model: Multiple Interacting Predictors Standardized Linear Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/kidscore_interaction_z.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/kidscore_interaction_z.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/kidscore_interaction_z.py
Upstream catalogue contributor: Oliver Järnefelt
Description: kid_score ~ z_mom_hs + z_mom_iq + z_mom_hs:z_mom_iq

Data: kidiq_with_mom_work (upstream posterior: kidiq_with_mom_work-kidscore_interaction_z)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/kidiq_with_mom_work.json.zip
    Archive SHA-256: 0c82d40ace4c75bba4bc8824d4ece044e1d5f8700d479ea672001ab475254913
    Local archive: data/posteriordb_kidiq_with_mom_work.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/kidiq_with_mom_work-kidscore_interaction_z.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.4/kidiq_interaction_z.stan

Implementation notes:
Native standardized child-score interaction regression from Gelman and Hill (2006), Chapter 4.

Each predictor uses its sample mean and twice its sample standard deviation.
All parameter densities are constant on their supports.

Stan comparison:
- A: Stan = -765529.1913012981; PyMC = -765529.1913012983.
- B: Stan = -2005859.811291326; PyMC = -2005859.8112913256.
- Raw delta: Stan = -1240330.619990028; PyMC = -1240330.6199900273.
- Raw comparison: MATCH; residual = -6.984919309616089e-10.
- Tolerance: 0.012404306199900279
- Original Stan SHA-256: af2ebd6cd04a66deaf879899b548cb6401c7d250dd6cae7f55c726a8e7f734d6

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
            Path(__file__).parent / "data" / "posteriordb_kidiq_with_mom_work.json.zip"
        ) as archive,
        archive.open("kidiq_with_mom_work.json") as stream,
    ):
        data = json.load(stream)
    score = np.asarray(data["kid_score"], dtype=float)
    hs = np.asarray(data["mom_hs"], dtype=float)
    iq = np.asarray(data["mom_iq"], dtype=float)
    hs = (hs - hs.mean()) / (2.0 * hs.std(ddof=1))
    iq = (iq - iq.mean()) / (2.0 * iq.std(ddof=1))
    design = np.column_stack((np.ones(score.size), hs, iq, hs * iq))
    coords = {
        "child": np.arange(score.size),
        "coefficient": [
            "intercept",
            "standardized_mom_hs",
            "standardized_mom_iq",
            "hs_iq",
        ],
    }
    with pm.Model(coords=coords) as model:
        predictors = pm.Data("predictors", design, dims=("child", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient")
        sigma = pm.HalfFlat("sigma", initval=20.0)
        pm.Normal(
            "kid_score",
            mu=pm.math.dot(predictors, beta),
            sigma=sigma,
            observed=score,
            dims="child",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
