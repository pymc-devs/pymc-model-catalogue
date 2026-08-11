"""
Model: Survey MRP step - zero-sum demographic effects with a categorical rating likelihood
Source: synthetic stress test written for this catalogue, shaped after an MRP-style survey
    step: a "feeling thermometer" battery where every respondent rates every party on a
    fixed answer scale.
Authors: pymc-model-catalogue
Description: N = 2500 respondents rate G = 8 parties on a K = 10 point scale. The latent
    rating per (respondent, question) sums 10 demographic main effects and 9 pairwise
    interactions, each a ZeroSumNormal (levels x G) table scaled by its own HalfNormal and
    gathered at the respondent's category. A monotone loading built from the cumulative sums
    of a Dirichlet maps that latent onto K logits, which are added to zero-sum per-question
    base rates and fed to a Categorical likelihood over (N, G, K). 1429 parameters.

Benchmark results:
- Original:  logp = -47408.0753, grad norm = 146.9804, 7460.3 us/call (2069 evals)
- Frozen:    logp = -47408.0753, grad norm = 146.9804, 7655.0 us/call (2036 evals)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    N, G, K = 2500, 8, 10  # respondents, rating questions each answers, scale points
    MAIN = {"age": 7, "gender": 2, "education": 4, "nationality": 3, "urban": 2,
            "income": 6, "region": 24, "methods": 3, "vote_intent": 8, "party_pref": 10}
    INTER = [("age", "education"), ("gender", "nationality"), ("age", "gender"),
             ("age", "nationality"), ("education", "nationality"), ("nationality", "urban"),
             ("gender", "urban"), ("age", "urban"), ("education", "urban")]

    rng = np.random.default_rng(0)
    codes = {d: rng.integers(0, L, size=N) for d, L in MAIN.items()}
    y = rng.integers(0, K, size=(N, G))  # respondent i's scale point for question j

    with pm.Model(check_bounds=False) as model:
        # 1. hierarchical block: one zero-sum effect table per demographic, gathered per respondent
        mu = pt.zeros((N, G))
        tables = [(d, MAIN[d], codes[d]) for d in MAIN]
        tables += [(f"{a}_x_{b}", MAIN[a] * MAIN[b], codes[a] * MAIN[b] + codes[b])
                   for a, b in INTER]
        for name, levels, idx in tables:
            sd = pm.HalfNormal(f"sd_{name}", 0.3)  # how much this demographic matters at all
            off = pm.ZeroSumNormal(f"a_{name}", sigma=1.0, shape=(levels, G), n_zerosum_axes=1)
            mu = mu + (sd * off)[idx]  # (N, G) latent rating
        # No per-question intercept: adding one shifts the logits by c_g * phi_g, whose
        # mean over the scale cancels under the softmax and whose remainder is already
        # in alpha's zero-sum space, so it would be a redundant degree of freedom.

        # 2. likelihood: a monotone loading phi turns the single latent into K logits.
        # phi_g = (0, ..., 1) from a Dirichlet's cumulative sums, so mu orders the scale points
        # rather than shifting all of them equally (a constant shift cancels under the softmax).
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
