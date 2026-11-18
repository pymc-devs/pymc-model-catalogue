"""
Model: Gamma Length of Stay, Recorded Days as Exact Stays (New York COVID-19, 2024)
Source: pymc-examples/examples/survival_analysis/hospital_length_of_stay.ipynb, Section: "The model for recorded stays"
Authors: Andres Hernandez
Description: The model of hospital_length_of_stay_sparcs_interval with every recorded
    length of stay taken as the exact stay: count-weighted Gamma logp in a Potential (the
    comparison the notebook makes).
Data: the 18,290 stays of the 2024 SPARCS de-identified inpatient discharges (Health Data
    NY dataset sf4k-39ay) whose principal diagnosis is COVID-19 (CCSR INF012), one row per
    stay, four columns, as in pymc-examples/examples/data/sparcs_2024_covid19_stays.csv.
    The raw SPARCS data used to produce this publication was provided by the New York State
    Department of Health. However, the calculations, metrics, conclusions derived, and views
    expressed herein are those of the author(s) and do not reflect the work, conclusions, or
    views of New York State. The New York State Department of Health, its employees,
    officers, and agents make no representation, warranty or guarantee as to the accuracy,
    completeness, currency, or suitability of the information provided here.

Changes from original:
- Saved the notebook's data file to .npz (same rows; a redacted hospital is "" here).
- Wrote out the notebook's build_ny_model(interval) for interval=False.
- Dropped the interval bounds and row groups the notebook computes for the other model.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -51068.0594, grad norm = 4640.4707, 175.0 us/call (82409 evals)
- Frozen:    logp = -51068.0594, grad norm = 4640.4707, 171.8 us/call (80541 evals)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "sparcs_2024_covid19_stays.npz")
    records = pd.DataFrame({name: data[name] for name in data.files})
    records["permanent_facility_id"] = records["permanent_facility_id"].replace("", np.nan)

    covid = records.dropna(subset=["permanent_facility_id"])  # a few stays have the hospital redacted
    covid = covid[covid.groupby("permanent_facility_id")["age_group"].transform("size") > 10].copy()
    covid["days"] = covid["length_of_stay"].replace("120+", "120").astype(int)
    covid["severity"] = np.where(covid["apr_severity_of_illness_code"] == 4, "extreme", "up to major")
    stay_counts = (
        covid.groupby(["permanent_facility_id", "age_group", "severity", "days"])
        .size()
        .rename("n")
        .reset_index()
    )

    ny_ages = ["0-17", "18-29", "30-49", "50-69", "70 or Older"]
    ny_severities = ["up to major", "extreme"]
    ny_groups = [f"{age}, {severity}" for age in ny_ages for severity in ny_severities]
    group_idx = (
        stay_counts["age_group"].map(ny_ages.index).to_numpy() * 2
        + stay_counts["severity"].map(ny_severities.index).to_numpy()
    )
    hospital_ids, ny_hospital_idx = np.unique(stay_counts["permanent_facility_id"], return_inverse=True)
    days = stay_counts["days"].to_numpy(dtype=float)
    n_stays = stay_counts["n"].to_numpy(dtype=float)

    with pm.Model(coords={"group": ny_groups, "hospital": hospital_ids}) as model:
        mu0 = pm.Normal("mu0", 1.5, 1.0)
        sigma_mu = pm.HalfNormal("sigma_mu", 1.0)
        m = pm.Normal("m", mu0, sigma_mu, dims="group")
        sigma_h = pm.HalfNormal("sigma_h", 1.0)
        eta = pm.Normal("eta", 0.0, sigma_h, dims="hospital")
        kappa0 = pm.Normal("kappa0", 0.0, 1.0)
        sigma_k = pm.HalfNormal("sigma_k", 1.0)
        log_k = pm.Normal("log_k", kappa0, sigma_k, dims="group")

        group_mean = pm.Deterministic("group_mean", pm.math.exp(m), dims="group")
        hospital_effect = pm.Deterministic("hospital_effect", pm.math.exp(eta), dims="hospital")
        k = pm.Deterministic("k", pm.math.exp(log_k), dims="group")

        shape = k[group_idx]
        rate = shape / (group_mean[group_idx] * hospital_effect[ny_hospital_idx])

        gamma = pm.Gamma.dist(alpha=shape, beta=rate)
        loglik = (n_stays * pm.logp(gamma, days)).sum()
        pm.Potential("stays", loglik)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
