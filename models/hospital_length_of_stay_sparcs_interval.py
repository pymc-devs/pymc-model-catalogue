"""
Model: Interval-Censored Gamma Length of Stay (New York COVID-19, 2024)
Source: pymc-examples/examples/survival_analysis/hospital_length_of_stay.ipynb, Section: "The model for recorded stays"
Authors: Andres Hernandez
Description: Lengths of stay recorded in whole days, each fitted as the interval it stands
    for: a Gamma with a mean and shape per group of age and severity (centered hierarchy)
    and a factor per hospital on the mean, with count-weighted log differences of gammainc
    (below the median) and gammaincc (above it) in a Potential.
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
- Wrote out the notebook's build_ny_model(interval) for interval=True.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -48392.9952, grad norm = 2854.9231, 4070.8 us/call (3977 evals)
- Frozen:    logp = -48392.9952, grad norm = 2854.9231, 4274.0 us/call (3327 evals)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


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

    # each recorded stay of d days lasted between d - 1.5 and d - 0.5 days
    interval_start = np.maximum(days - 1.5, 0.0)
    interval_end = days - 0.5
    median_days = np.median(covid["days"])
    first_day = np.flatnonzero(days == 1)
    top_coded = np.flatnonzero(days == 120)
    below_median = np.flatnonzero((days > 1) & (days < 120) & (days <= median_days))
    above_median = np.flatnonzero((days > 1) & (days < 120) & (days > median_days))

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

        def cdf(rows, t):
            return pt.gammainc(shape[rows], rate[rows] * t[rows])

        def survival(rows, t):
            return pt.gammaincc(shape[rows], rate[rows] * t[rows])

        log_prob = [
            (first_day, pt.log(cdf(first_day, interval_end))),
            (
                below_median,
                pt.log(cdf(below_median, interval_end) - cdf(below_median, interval_start)),
            ),
            (
                above_median,
                pt.log(survival(above_median, interval_start) - survival(above_median, interval_end)),
            ),
            (top_coded, pt.log(survival(top_coded, interval_start))),
        ]
        loglik = sum((n_stays[rows] * lp).sum() for rows, lp in log_prob)
        pm.Potential("stays", loglik)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
