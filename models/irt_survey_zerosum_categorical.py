"""
Model: Survey IRT - many scaled zero-sum tables with a stereotype-ordinal categorical likelihood
Source: synthetic stress test written for this catalogue, shaped after an explanatory-IRT
    survey battery; sized as a numba compile-cost sentinel.
Authors: pymc-model-catalogue
Description: N = 1000 respondents rate G = 5 questions on a K = 5 scale. The latent rating
    per (respondent, question) sums 33 scaled gathered tables - 6 three-level demographics,
    all 15 pairwise interactions, and 2 continuous inputs' slopes varying by each
    demographic - each a ZeroSumNormal (zero-sum across its levels) scaled by its own
    HalfNormal; plus a batched continuous block (age and income as linear + square
    features) and a one-factor IRT term (unit-normal scores, question 1's loading fixed
    to 1). A monotone loading built from the cumulative sums of a Dirichlet maps the
    latent onto K logits, added to zero-sum per-question base rates and fed to a
    Categorical over (N, G, K). 2013 parameters.

    Sizing note: numba compile cost is superlinear in the count of scaled gathered
    tables, ``(sd_t * table_t)[idx_t]``, and nearly independent of their sizes. 33 tables
    with per-table scales keeps an unambiguous signal (297 s build / 2180 MB peak RSS on
    pytensor 3.2.4 + numba 0.66 against 174 s / 1758 MB on 3.0.7 + numba 0.65.1); sharing
    one scale across tables defuses it.

Benchmark results:
- Original:  logp = -9898.4974, grad norm = 91.2656, 2598.1 us/call (5420 evals)
- Frozen:    logp = -9898.4974, grad norm = 91.2656, 2515.8 us/call (5753 evals)
"""

from itertools import combinations

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    N, G, K, L, C, P = 1000, 5, 5, 3, 2, 2  # respondents, questions, scale pts, levels, cont inputs, powers
    DEMOS = ["urban", "education", "gender", "region", "employment", "nationality"]
    CONT = ["age", "income"]

    rng = np.random.default_rng(0)
    codes = {d: rng.integers(0, L, size=N) for d in DEMOS}
    z = 0.5 * rng.normal(size=(N, C))  # half-z-scored continuous inputs
    feats = np.power(z[:, :, None], np.arange(1, P + 1)).reshape(N, C * P)  # age^1, age^2, income^1, ...
    y = rng.integers(0, K, size=(N, G))  # respondent i's scale point for question j

    with pm.Model(check_bounds=False) as model:
        # 1. gathered block: mains, pairwise interactions, and continuous-x-demographic
        # varying slopes - all the same scaled-gather shape, zero-sum across the levels axis
        tables = [(d, L, codes[d], None) for d in DEMOS]
        tables += [(f"{a}_x_{b}", L * L, codes[a] * L + codes[b], None)
                   for a, b in combinations(DEMOS, 2)]
        tables += [(f"{cc}_x_{d}", L, codes[d], feats[:, c * P:(c + 1) * P])
                   for c, cc in enumerate(CONT) for d in DEMOS]
        mu = pt.zeros((N, G))
        for name, levels, idx, w in tables:
            sd = pm.HalfNormal(f"sd_{name}", 0.3)
            if w is None:
                off = pm.ZeroSumNormal(f"a_{name}", sigma=1.0, shape=(G, levels), n_zerosum_axes=1)
                mu = mu + (sd * off.T)[idx]
            else:  # gather, weight by the input's features, sum over powers
                off = pm.ZeroSumNormal(f"a_{name}", sigma=1.0, shape=(P, G, levels), n_zerosum_axes=1)
                mu = mu + ((sd * off.dimshuffle(2, 0, 1))[idx] * w[:, :, None]).sum(1)

        # 2. continuous mains: one batched offset with a per-(power, question) scale per input
        b_off = pm.Normal("b_cont", 0.0, 1.0, shape=(C * P, G))
        b_sd = pt.concatenate([pm.HalfNormal(f"sd_{cc}", 0.3, shape=(P, G)) for cc in CONT])
        mu = mu + feats @ (b_sd * b_off)

        # 3. IRT factor, reference indicator: W[0] == 1 with a free factor scale.
        # A sign-only anchor does not identify the reflection when question 1 loads weakly.
        W = pt.concatenate([pt.ones((1, 1)), pm.Normal("f_loadings", 0.0, 1.0, shape=(G - 1, 1))])
        mu = mu + (pm.HalfNormal("f_sd", 1.0) * pm.Normal("f_scores", 0.0, 1.0, shape=(N, 1))) @ W.T

        # No per-question intercept: it would duplicate a dof already in alpha's span.

        # 4. likelihood: phi_g = (0, ..., 1) from a Dirichlet's cumulative sums, so mu
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
