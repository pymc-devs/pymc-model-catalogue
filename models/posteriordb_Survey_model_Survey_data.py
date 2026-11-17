"""
Model: Inferring the Return Rate and Number of Surveys from Observed Returns
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/Survey_model.stan
Authors: M D Lee, E Wagenmakers (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/Survey_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Inference for the joint probability distribution of the rate of return and the number
    of surveys by observing the number of returned surveys.

Data: Survey_data (upstream posterior: Survey_data-Survey_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/Survey_data.json.zip
    Archive SHA-256: cd837313fec3609c673256cb74fe5b9786e4c3e08bb6cd827dc14b2574653757
    Local archive: data/posteriordb_Survey_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/Survey_data-Survey_model.json
References: lee2013cognitive (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/Bayesian_Cognitive_Modeling/ParameterEstimation/Binomial

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
The survey count has a discrete uniform distribution over one through nmax.

Source: Lee and Wagenmakers (2013), binomial rate examples.
Exact enumeration integrates the shared survey count across all observed returns.

Stan comparison:
- Not recorded.

Benchmark results:
- Not measured.
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
