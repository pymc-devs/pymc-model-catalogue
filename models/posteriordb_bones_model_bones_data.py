"""Latent Trait Model for Multiple Ordered Categorical Responses.

Stan comparison: MISMATCH.
The numerical discrepancy remains unexplained.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_bones_data.json.zip"
        ) as archive,
        archive.open("bones_data.json") as stream,
    ):
        data = json.load(stream)
    grade = np.asarray(data["grade"], dtype="int64")
    category_count = np.asarray(data["ncat"], dtype="int64")
    discrimination = np.asarray(data["delta"], dtype=float)
    thresholds = np.asarray(data["gamma"], dtype=float)
    coords = {"child": np.arange(1, data["nChild"] + 1)}
    with pm.Model(coords=coords) as model:
        theta = pm.Normal(
            "theta",
            mu=0,
            sigma=36,
            dims="child",
            initval=np.linspace(0, 16, data["nChild"]),
        )
        for count in np.unique(category_count):
            child, indicator = np.nonzero(
                (grade != -1) & (category_count[None, :] == count)
            )
            # Stan uses sigmoid(delta * (theta-gamma)) directly, whereas
            # OrderedLogistic uses complementary cumulative probabilities.
            # These are algebraically equivalent but can differ in floating-
            # point tails; the recorded delta does not pass the strict tolerance.
            pm.OrderedLogistic(
                f"grade_{count}_categories",
                eta=discrimination[indicator] * theta[child],
                cutpoints=(
                    discrimination[indicator, None] * thresholds[indicator, : count - 1]
                ),
                observed=grade[child, indicator] - 1,
                compute_p=False,
            )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
