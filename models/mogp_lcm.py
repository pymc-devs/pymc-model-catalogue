"""
Model: Multi-output GP with LCM (Linear Coregionalization Model)
Source: pymc-examples/examples/gaussian_processes/MOGP-Coregion-Hadamard.ipynb, Section: "Linear Coregionalization Model (LCM)"
Authors: Danh Phan, Bill Engels, Chris Fonnesbeck
Description: Multi-output GP using the Linear Coregionalization Model (LCM)
    with two base kernels (ExpQuad + Matern32) combined via Hadamard product.
    Models average spin rates of 5 MLB pitchers.

Changes from original:
- Saved preprocessed fastball data to .npz instead of loading CSV
- Removed sampling, plotting, conditional predictions
- Added ip capture and initval clearing

Benchmark results:
- Original:  logp = -2781450.2847, grad norm = 4265263.9398, 7549.5 us/call (2036 evals)
- Frozen:    logp = -2781450.2847, grad norm = 4265263.9398, 8073.3 us/call (2217 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    rng = np.random.default_rng(8927)

    def get_icm(input_dim, kernel, W=None, kappa=None, B=None, active_dims=None):
        coreg = pm.gp.cov.Coregion(
            input_dim=input_dim, W=W, kappa=kappa, B=B, active_dims=active_dims
        )
        return kernel * coreg

    def get_lcm(
        input_dim, active_dims, num_outputs, kernels, W=None, B=None, name="ICM"
    ):
        if B is None:
            kappa = pm.Gamma(f"{name}_kappa", alpha=5, beta=1, shape=num_outputs)
            if W is None:
                W = pm.Normal(
                    f"{name}_W",
                    mu=0,
                    sigma=5,
                    shape=(num_outputs, 1),
                    initval=rng.standard_normal((num_outputs, 1)),
                )
        else:
            kappa = None

        cov_func = 0
        for kernel in kernels:
            cov_func += get_icm(input_dim, kernel, W, kappa, B, active_dims)
        return cov_func

    data = np.load(
        Path(__file__).parent / "data" / "fastball_spin_rates.npz",
        allow_pickle=True,
    )
    X = data["X"]
    Y = data["Y"]
    n_outputs = data["n_outputs"].item()

    with pm.Model() as model:
        # Priors
        ell = pm.Gamma("ell", alpha=2, beta=0.5, shape=2)
        eta = pm.Gamma("eta", alpha=3, beta=1, shape=2)
        kernels = [pm.gp.cov.ExpQuad, pm.gp.cov.Matern32]
        sigma = pm.HalfNormal("sigma", sigma=3)

        # Define a list of covariance functions
        cov_list = [
            eta[idx] ** 2
            * kernel(input_dim=2, ls=ell[idx], active_dims=[0])
            for idx, kernel in enumerate(kernels)
        ]

        # Get the LCM kernel
        cov_lcm = get_lcm(
            input_dim=2,
            active_dims=[1],
            num_outputs=n_outputs,
            kernels=cov_list,
        )

        # Define a Multi-output GP
        mogp = pm.gp.Marginal(cov_func=cov_lcm)
        y_ = mogp.marginal_likelihood("f", X, Y, sigma=sigma)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
