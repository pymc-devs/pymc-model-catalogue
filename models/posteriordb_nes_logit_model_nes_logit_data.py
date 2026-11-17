"""
Model: Logistic Regression Model for Voting Preference based on Income
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/nes_logit_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/nes_logit_model.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/nes_logit_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Logistic regression to estimate the probability of supporting a candidate inferring on
    income.

Data: nes_logit_data (upstream posterior: nes_logit_data-nes_logit_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/nes_logit_data.json.zip
    Archive SHA-256: 76af25eb4b46c1aa35c2e6c8c98810148d3de1042266336721494ee5f083d06e
    Local archive: data/posteriordb_nes_logit_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/nes_logit_data-nes_logit_model.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/ARM/Ch.5
Reference URL: http://www.stat.columbia.edu/~gelman/arm/

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native logistic regression for National Election Study votes.

Source: Gelman and Hill (2006), ARM.

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
            Path(__file__).parent / "data" / "posteriordb_nes_logit_data.json.zip"
        ) as archive,
        archive.open("nes_logit_data.json") as stream,
    ):
        data = json.load(stream)
    income = np.asarray(data["income"], dtype=float)
    vote = np.asarray(data["vote"], dtype=int)
    coords = {"obs_id": np.arange(len(vote)), "coefficient": ["income"]}
    with pm.Model(coords=coords) as model:
        x = pm.Data("income", income, dims="obs_id")
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", dims="coefficient")
        pm.Bernoulli("vote", logit_p=alpha + beta[0] * x, observed=vote, dims="obs_id")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
