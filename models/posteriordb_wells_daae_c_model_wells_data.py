"""
Model: 4-Predictor logistic regression model with centered inputs for decision to switch wells
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/wells_daae_c_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/wells_daae_c_model.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/wells_daae_c_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Performs logistic regression using four centered predictors: distance from nearest safe
    well, arsenic levels, years in education, and if the individual is associated with an active
    community project. The association predictor is not actually predictive of switching.

Data: wells_data (upstream posterior: wells_data-wells_daae_c_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/wells_data.json.zip
    Archive SHA-256: 0ecd7efe3e64c63d849c562e61658fe322509942a882bc515176e016cc57c969
    Local archive: data/posteriordb_wells_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/wells_data-wells_daae_c_model.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/ARM/Ch.5

Implementation notes:
Native well-switching logistic regression from Gelman and Hill (2006), Chapter 5.

All coefficient densities are constant. Distance and arsenic use their sample means for centering.
Education uses four-year units without centering. Association retains its binary coding.

Stan comparison:
- A: Stan = -2039.2934246892526; PyMC = -2039.2934246892526.
- B: Stan = -1994.319657791627; PyMC = -1994.3196577916267.
- Raw delta: Stan = 44.97376689762564; PyMC = 44.973766897625865.
- Raw comparison: MATCH; residual = -2.2737367544323206e-13.
- Tolerance: 1.4497376689762585e-06
- Original Stan SHA-256: 0791694f69f5c63c9aff64c4b88a857df9d14cbca01594c4de5df290a6553708

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
    dist = np.asarray(data["dist"], dtype=float)
    arsenic = np.asarray(data["arsenic"], dtype=float)
    dist = (dist - dist.mean()) / 100.0
    arsenic = arsenic - arsenic.mean()
    assoc = np.asarray(data["assoc"], dtype=float)
    educ = np.asarray(data["educ"], dtype=float) / 4.0
    design = np.column_stack((dist, arsenic, dist * arsenic, assoc, educ))
    coords = {
        "household": np.arange(switched.size),
        "coefficient": [
            "centered_dist100",
            "centered_arsenic",
            "distance_arsenic",
            "assoc",
            "educ4",
        ],
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
