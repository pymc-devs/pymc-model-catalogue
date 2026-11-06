"""Native skeletal maturity model from the BUGS bones example.

Missing grades do not contribute a likelihood factor.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("bones_model", posterior)
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
            pm.OrderedLogistic(
                f"grade_{count}_categories",
                eta=discrimination[indicator] * theta[child],
                cutpoints=(
                    discrimination[indicator, None] * thresholds[indicator, : count - 1]
                ),
                observed=grade[child, indicator] - 1,
                compute_p=False,
            )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("bones_model")
