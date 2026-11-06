"""The survey count has a discrete uniform distribution over one through nmax.

Source: Lee and Wagenmakers (2013), binomial rate examples.
Exact enumeration integrates the shared survey count across all observed returns.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("Survey_model", posterior)
    returns = np.asarray(data["k"], dtype="int64")
    nmax = int(data["nmax"])
    nmin = max(1, int(returns.max()))
    possible_counts = np.arange(nmin, nmax + 1, dtype="int64")
    coords = {"survey_count": np.arange(1, nmax + 1)}
    with pm.Model(coords=coords) as model:
        theta = pm.Uniform("theta", 0, 1)
        component_logp = pm.logp(
            pm.Binomial.dist(n=possible_counts[:, None], p=theta),
            returns[None, :],
        ).sum(axis=-1)
        pm.Potential("survey_returns", pm.math.logsumexp(component_logp) - np.log(nmax))
        pm.Deterministic(
            "prob_n",
            pt.concatenate(
                [pt.zeros(nmin - 1), pm.math.softmax(component_logp, axis=0)]
            ),
            dims="survey_count",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("Survey_model")
