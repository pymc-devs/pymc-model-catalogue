"""
Model: Bayesian A/B Testing - Revenue Model
Source: pymc-examples/examples/causal_inference/bayesian_ab_testing_introduction.ipynb, Section: "Value Conversions"
Authors: Cuong Duong
Description: Two-variant A/B test for revenue, combining Binomial conversion rates and Gamma
    revenue distributions to estimate revenue per visitor uplift using best-of-rest comparison.

Changes from original:
- Data generated from scenario 3 (A: rate=0.1, mp=10; B: rate=0.11, mp=10.5) with 100k samples
- Extracted from class-based structure into flat model definition
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -2084130.3660, grad norm = 1496585.6284, 2.9 us/call (100000 evals)
- Frozen:    logp = -2129694.2235, grad norm = 1528933.5423, 3.0 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

from scipy.stats import bernoulli, expon

def build_model():
    # Generate synthetic data faithfully (Scenario 3: higher purchase rate and mean)
    RANDOM_SEED = 4000
    rng = np.random.default_rng(RANDOM_SEED)

    variants = ["A", "B"]
    true_conversion_rates = [0.1, 0.11]
    true_mean_purchase = [10, 10.5]
    samples_per_variant = 100000

    # Generate revenue data
    converted_data = {}
    mean_purchase_data = {}
    for variant, p, mp in zip(variants, true_conversion_rates, true_mean_purchase):
        converted_data[variant] = bernoulli.rvs(
            p, size=samples_per_variant, random_state=rng
        )
        mean_purchase_data[variant] = expon.rvs(
            scale=mp, size=samples_per_variant, random_state=rng
        )

    # Aggregate
    visitors = []
    purchased = []
    total_revenue = []
    for v in variants:
        visitors.append(int(converted_data[v].shape[0]))
        purchased.append(int(converted_data[v].sum()))
        revenue = converted_data[v] * mean_purchase_data[v]
        total_revenue.append(float(revenue.sum()))

    num_variants = len(variants)

    # Priors from notebook
    c_alpha, c_beta = 5000, 5000  # Beta prior for conversion rate
    mp_alpha, mp_beta = 9000, 900  # Gamma prior for mean purchase

    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=c_alpha, beta=c_beta, shape=num_variants)
        lam = pm.Gamma("lam", alpha=mp_alpha, beta=mp_beta, shape=num_variants)
        converted = pm.Binomial(
            "converted", n=visitors, p=theta, observed=purchased, shape=num_variants
        )
        revenue = pm.Gamma(
            "revenue", alpha=purchased, beta=lam, observed=total_revenue, shape=num_variants
        )
        revenue_per_visitor = pm.Deterministic("revenue_per_visitor", theta * (1 / lam))

        # best_of_rest comparison method
        theta_reluplift = []
        reciprocal_lam_reluplift = []
        reluplift = []
        for i in range(num_variants):
            others_theta = [theta[j] for j in range(num_variants) if j != i]
            others_lam = [1 / lam[j] for j in range(num_variants) if j != i]
            others_rpv = [revenue_per_visitor[j] for j in range(num_variants) if j != i]
            # Only 2 variants, so others has length 1
            comparison_theta = others_theta[0]
            comparison_lam = others_lam[0]
            comparison_rpv = others_rpv[0]
            theta_reluplift.append(
                pm.Deterministic(f"theta_reluplift_{i}", theta[i] / comparison_theta - 1)
            )
            reciprocal_lam_reluplift.append(
                pm.Deterministic(
                    f"reciprocal_lam_reluplift_{i}", (1 / lam[i]) / comparison_lam - 1
                )
            )
            reluplift.append(
                pm.Deterministic(f"reluplift_{i}", revenue_per_visitor[i] / comparison_rpv - 1)
            )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
