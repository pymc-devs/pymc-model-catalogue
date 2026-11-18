"""
Model: Hierarchical Loss Curve Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/losscurve_sislob.stan
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/losscurve_sislob.py
Upstream catalogue contributor: Keane Nguyen
Description: Hierarchical Loss Curve Model

Data: loss_curves (upstream posterior: loss_curves-losscurve_sislob)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/loss_curves.json.zip
    Archive SHA-256: 6ebe3be18c1e0459d1b3487bcf770116ea0ec2d59229e0e4604c29b4a2df7d0f
    Local archive: data/posteriordb_loss_curves.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/loss_curves-losscurve_sislob.json
Reference URL: https://mc-stan.org/users/documentation/case-studies/losscurves_casestudy.html

Implementation notes:
Hierarchical loss development from the posteriordb losscurve_sislob model.

The data select Weibull or log-logistic cumulative growth.
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.

Stan comparison:
- A: Stan = -574.1474428807884; PyMC = -574.1474428731698.
- B: Stan = -556.634734886495; PyMC = -556.6347348788763.
- Raw delta: Stan = 17.51270799429335; PyMC = 17.512707994293464.
- Raw comparison: MATCH; residual = -1.1368683772161603e-13.
- Tolerance: 1.1751270799429345e-06
- Original Stan SHA-256: b7e3d55e53d230a391b87e03fc9eec6007322a79ddec748ac1a4dee1741bc4b4

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
            Path(__file__).parent / "data" / "posteriordb_loss_curves.json.zip"
        ) as archive,
        archive.open("loss_curves.json") as stream,
    ):
        data = json.load(stream)
    cohort_index = np.asarray(data["cohort_id"], dtype=int) - 1
    time_index = np.asarray(data["t_idx"], dtype=int) - 1
    coords = {
        "cohort": np.arange(1, int(data["n_cohort"]) + 1),
        "time": np.asarray(data["t_value"], dtype=float),
        "observation": np.arange(int(data["n_data"])),
    }
    with pm.Model(coords=coords) as model:
        omega = pm.LogNormal("omega", mu=0, sigma=0.5)
        theta = pm.LogNormal("theta", mu=0, sigma=0.5)
        mu_LR = pm.Normal("mu_LR", mu=0, sigma=0.5)
        sd_LR = pm.LogNormal("sd_LR", mu=0, sigma=0.5)
        LR = pm.LogNormal("LR", mu=mu_LR, sigma=sd_LR, dims="cohort")
        loss_sd = pm.LogNormal("loss_sd", mu=0, sigma=0.7)
        time = pm.Data("t_value", np.asarray(data["t_value"], dtype=float), dims="time")
        premium = pm.Data(
            "premium", np.asarray(data["premium"], dtype=float), dims="cohort"
        )
        if int(data["growthmodel_id"]) == 1:
            growth = -pt.expm1(-pt.power(time / theta, omega))
        else:
            growth = pt.sigmoid(omega * (pt.log(time) - pt.log(theta)))
        gf = pm.Deterministic("gf", growth, dims="time")
        expected = LR[cohort_index] * premium[cohort_index] * gf[time_index]
        pm.Normal(
            "loss",
            mu=expected,
            sigma=loss_sd * premium[cohort_index],
            observed=np.asarray(data["loss"], dtype=float),
            dims="observation",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
