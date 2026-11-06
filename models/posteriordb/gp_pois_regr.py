"""Latent Gaussian-process Poisson regression from posteriordb.

The native latent GP uses a noncentered Normal representation and the source jitter.
Source: https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/gp_pois_regr
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("gp_pois_regr", posterior)
    coords = {"observation": np.arange(data["N"]), "input": ["x"]}
    with pm.Model(coords=coords) as model:
        x = pm.Data(
            "x",
            np.asarray(data["x"], dtype=float)[:, None],
            dims=("observation", "input"),
        )
        rho = pm.Gamma("rho", alpha=25, beta=4)
        alpha = pm.HalfNormal("alpha", sigma=2)
        process = pm.gp.Latent(cov_func=alpha**2 * pm.gp.cov.ExpQuad(1, ls=rho))
        f = process.prior(
            "f", X=x, reparameterize=True, jitter=1e-10, dims="observation"
        )
        pm.Poisson(
            "k",
            mu=pm.math.exp(f),
            observed=np.asarray(data["k"], dtype=int),
            dims="observation",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("gp_pois_regr")
