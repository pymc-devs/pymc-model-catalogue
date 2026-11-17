"""
Model: 3-Input logistic regression model with interactions and centered inputs for decision to switch wells
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/wells_dae_inter_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/wells_dae_inter_model.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/wells_dae_inter_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Performs logistic regression using three centered inputs (distance from nearest safe
    well, log-scaled arsenic levels, and years in education) and the interactions between each pair
    of inputs.

Data: wells_data (upstream posterior: wells_data-wells_dae_inter_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/wells_data.json.zip
    Archive SHA-256: 0ecd7efe3e64c63d849c562e61658fe322509942a882bc515176e016cc57c969
    Local archive: data/posteriordb_wells_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/wells_data-wells_dae_inter_model.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/ARM/Ch.5

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native well-switching logistic regression from Gelman and Hill (2006), Chapter 5.

All coefficient densities are constant. All three predictors use sample-mean centering.
Education uses four-year units. The design includes each pairwise interaction.

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
    educ = np.asarray(data["educ"], dtype=float)
    dist = (dist - dist.mean()) / 100.0
    arsenic = arsenic - arsenic.mean()
    educ = (educ - educ.mean()) / 4.0
    design = np.column_stack(
        (dist, arsenic, educ, dist * arsenic, dist * educ, arsenic * educ)
    )
    coords = {
        "household": np.arange(switched.size),
        "coefficient": [
            "centered_dist100",
            "centered_arsenic",
            "centered_educ4",
            "distance_arsenic",
            "distance_education",
            "arsenic_education",
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
