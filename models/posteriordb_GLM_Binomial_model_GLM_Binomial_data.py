"""Success Rate of Peregrine broods.

Stan comparison: MATCH.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_GLM_Binomial_data.json.zip"
        ) as archive,
        archive.open("GLM_Binomial_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model(coords={"year_id": np.arange(data["nyears"])}) as model:
        year = pm.Data("year", np.asarray(data["year"], dtype=float), dims="year_id")
        alpha = pm.Normal("alpha", 0, 100)
        beta1 = pm.Normal("beta1", 0, 100)
        beta2 = pm.Normal("beta2", 0, 100)
        pm.Binomial(
            "C",
            n=data["N"],
            logit_p=alpha + year * (beta1 + year * beta2),
            observed=data["C"],
            dims="year_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
