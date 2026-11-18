"""
Model: Hierarchical Gamma Length of Stay, Completed Stays Only
Source: pymc-examples/examples/survival_analysis/hospital_length_of_stay.ipynb, Section: "The model"
Authors: Andres Hernandez
Description: The model of hospital_length_of_stay_censored fitted to the discharged
    patients only, ignoring those still in hospital (the biased shortcut the notebook
    compares against).

Changes from original:
- Inlined the simulated wave (same code, seed 2026 as in the notebook).
- Wrote out the notebook's build_model(censored) for censored=False.
- Removed sampling, plotting and the bed forecast.

Benchmark results:
- Original:  logp = -5636.2376, grad norm = 270.4152, 61.0 us/call (100000 evals)
- Frozen:    logp = -5636.2376, grad norm = 270.4152, 67.4 us/call (100000 evals)
"""

import numpy as np
import pymc as pm


def build_model():
    rng = np.random.default_rng(2026)

    n_hospitals = 30
    n_patients = 3000
    growth = 0.05  # per day
    analysis_day = 60.0
    age_groups = ["<50", "50-69", "70+"]

    hospital_size = rng.lognormal(0.0, 0.8, n_hospitals)
    hospital_idx = rng.choice(n_hospitals, n_patients, p=hospital_size / hospital_size.sum())
    age_idx = rng.choice(3, n_patients, p=[0.3, 0.35, 0.35])

    # admission times from the exponential growth curve, by inverse transform sampling
    quantile = rng.uniform(size=n_patients)
    admitted = np.log1p(quantile * np.expm1(growth * analysis_day)) / growth

    # true parameters: log mean stay per hospital, shape per hospital, age effects
    true_log_mean = np.log(8.0) + 0.25 * rng.normal(size=n_hospitals)
    true_shape = np.exp(np.log(2.5) + 0.3 * rng.normal(size=n_hospitals))
    true_age_effect = np.array([0.0, 0.25, 0.45])

    true_mean = np.exp(true_log_mean[hospital_idx] + true_age_effect[age_idx])
    patient_shape = true_shape[hospital_idx]
    true_stay = rng.gamma(patient_shape, true_mean / patient_shape)

    elapsed = analysis_day - admitted
    in_hospital = true_stay > elapsed
    stay = np.where(in_hospital, elapsed, true_stay)

    keep = ~in_hospital
    coords = {
        "hospital": np.arange(n_hospitals),
        "age": age_groups,
        "older": age_groups[1:],
        "patient": np.flatnonzero(keep),
    }
    with pm.Model(coords=coords) as model:
        mu0 = pm.Normal("mu0", 2.0, 0.5)
        beta = pm.Normal("beta", 0.0, 0.5, dims="older")
        sigma_mu = pm.HalfNormal("sigma_mu", 0.5)
        z_mu = pm.Normal("z_mu", 0.0, 1.0, dims="hospital")
        kappa0 = pm.Normal("kappa0", np.log(2.0), 0.5)
        sigma_k = pm.HalfNormal("sigma_k", 0.5)
        z_k = pm.Normal("z_k", 0.0, 1.0, dims="hospital")

        alpha = pm.Deterministic("alpha", mu0 + sigma_mu * z_mu, dims="hospital")
        k = pm.Deterministic("k", pm.math.exp(kappa0 + sigma_k * z_k), dims="hospital")
        age_effect = pm.Deterministic("age_effect", pm.math.concatenate([[0.0], beta]), dims="age")

        mu = pm.math.exp(alpha[hospital_idx[keep]] + age_effect[age_idx[keep]])
        patient_k = k[hospital_idx[keep]]
        pm.Gamma("stay", alpha=patient_k, beta=patient_k / mu, observed=stay[keep], dims="patient")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
