"""Inferring the Return Rate and Number of Surveys from Observed Returns.

Stan comparison: MATCH.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_Survey_data.json.zip"
        ) as archive,
        archive.open("Survey_data.json") as stream,
    ):
        data = json.load(stream)
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
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
