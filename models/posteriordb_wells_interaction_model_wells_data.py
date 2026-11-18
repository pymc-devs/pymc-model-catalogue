"""
Model: 2-Predictor Logistic regression model with interactions for decision to switch wells
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/wells_interaction_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/wells_interaction_model.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/wells_interaction_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Performs logistic regression using two predictors (distance from nearest safe well and
    arsenic levels) and their interaction.

Data: wells_data (upstream posterior: wells_data-wells_interaction_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/wells_data.json.zip
    Archive SHA-256: 0ecd7efe3e64c63d849c562e61658fe322509942a882bc515176e016cc57c969
    Local archive: data/posteriordb_wells_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/wells_data-wells_interaction_model.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/ARM/Ch.5

Implementation notes:
Native well-switching logistic regression from Gelman and Hill (2006), Chapter 5.

All coefficient densities are constant. Distance uses hundreds of meters.

Stan comparison:
- A: Stan = -2032.6199086607278; PyMC = -2032.619908660728.
- B: Stan = -2309.673060142438; PyMC = -2309.673060142438.
- Raw delta: Stan = -277.0531514817101; PyMC = -277.05315148170985.
- Raw comparison: MATCH; residual = -2.2737367544323206e-13.
- Tolerance: 3.770531514817101e-06
- Original Stan SHA-256: 42d51e3a6d1325703f4ec4f8f3b934f7a2d5b1ab8dc8be96e9771d229d92a087

Benchmark results:
- Not measured.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_wells_data.json.zip"
        ) as archive,
        archive.open("wells_data.json") as stream,
    ):
        data = json.load(stream)
    switched = np.asarray(data["switched"], dtype=np.int64)
    dist = np.asarray(data["dist"], dtype=float) / 100.0
    arsenic = np.asarray(data["arsenic"], dtype=float)
    design = np.column_stack((dist, arsenic, dist * arsenic))
    coords = {
        "household": np.arange(switched.size),
        "coefficient": ["dist100", "arsenic", "distance_arsenic"],
    }
    with pm.Model(coords=coords) as model:
        predictors = pm.Data("predictors", design, dims=("household", "coefficient"))
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", dims="coefficient")
        pm.Bernoulli(
            "switched",
            logit_p=alpha + pm.math.dot(predictors, beta),
            observed=switched,
            dims="household",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
