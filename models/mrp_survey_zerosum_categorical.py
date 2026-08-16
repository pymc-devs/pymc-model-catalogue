"""
Model: Survey MRP step - many scaled zero-sum tables with a categorical rating likelihood
Source: synthetic stress test written for this catalogue, shaped after an MRP-style survey
    step; sized as a numba compile-cost sentinel.
Authors: pymc-model-catalogue
Description: N = 1000 respondents rate G = 2 questions on a K = 3 scale. The latent
    rating per (respondent, question) sums 8 binary demographic main effects and all 28
    pairwise interactions: 36 tables of at most 4 rows, each a ZeroSumNormal scaled by
    its own HalfNormal and gathered at the respondent's category. A monotone loading
    built from the cumulative sums of a Dirichlet maps that latent onto K logits, added
    to zero-sum per-question base rates and fed to a Categorical likelihood over
    (N, G, K). 170 parameters.

    Sizing note: numba compile cost is superlinear in the count of scaled gathered
    tables, ``(sd_t * table_t)[idx_t]``, and nearly independent of their sizes. 36 tables
    with per-table scales is the smallest form with an unambiguous signal (317 s to build
    against 168 s on a fixed stack); sharing one scale across tables defuses it.

Benchmark results:
- Original:  logp = -2348.8118, grad norm = 33.9790, 1431.2 us/call (10559 evals)
- Frozen:    logp = -2348.8118, grad norm = 33.9790, 1596.3 us/call (9454 evals)
"""

from itertools import combinations

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    N, G, K = 1000, 2, 3  # respondents, rating questions each answers, scale points
    DEMOS = ["urban", "employed", "female", "married",
             "parent", "student", "religious", "homeowner"]  # all binary

    rng = np.random.default_rng(0)
    codes = {d: rng.integers(0, 2, size=N) for d in DEMOS}
    y = rng.integers(0, K, size=(N, G))  # respondent i's scale point for question j

    with pm.Model(check_bounds=False) as model:
        # 1. hierarchical block: one scaled zero-sum table per demographic and per pair
        mu = pt.zeros((N, G))
        tables = [(d, 2, codes[d]) for d in DEMOS]
        tables += [(f"{a}_x_{b}", 4, codes[a] * 2 + codes[b])
                   for a, b in combinations(DEMOS, 2)]
        for name, levels, idx in tables:
            sd = pm.HalfNormal(f"sd_{name}", 0.3)
            off = pm.ZeroSumNormal(f"a_{name}", sigma=1.0, shape=(levels, G), n_zerosum_axes=1)
            mu = mu + (sd * off)[idx]  # (N, G) latent rating

        # No per-question intercept: it would duplicate a dof already in alpha's span.

        # 2. likelihood: phi_g = (0, ..., 1) from a Dirichlet's cumulative sums, so mu
        # orders the scale points instead of shifting them all (which softmax cancels).
        phi_diffs = pm.Dirichlet("phi_diffs", np.ones(K - 1), shape=(G, K - 1))
        phi = pt.concatenate([pt.zeros((G, 1)), pt.cumsum(phi_diffs, axis=-1)], axis=-1)  # (G, K)
        # Per-question base rates, zero-sum over the scale so they don't fight the softmax.
        alpha = pm.ZeroSumNormal("alpha", sigma=2.0, shape=(G, K), n_zerosum_axes=1)
        pm.Categorical("y", logit_p=alpha[None, :, :] + phi[None, :, :] * mu[:, :, None],
                       observed=y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
