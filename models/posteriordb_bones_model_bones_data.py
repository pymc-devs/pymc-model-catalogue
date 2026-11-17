"""
Model: Latent Trait Model for Multiple Ordered Categorical Responses
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/bones_model.stan
Authors: Alex F. Roche and Howard Wainer and David Thissen (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/bones_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Model to predict skeletal age by calibrating 34 indicators of skeletal maturity.

Data: bones_data (upstream posterior: bones_data-bones_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/bones_data.json.zip
    Archive SHA-256: bbb0539b10740c5863fd2329e6496b38bb34f7a175b67ec7fb368fbfdb1ed62e
    Local archive: data/posteriordb_bones_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/bones_data-bones_model.json
References: roche1975rwt (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/bugs_examples/vol1/bones
Reference URL: http://www.mrc-bsu.cam.ac.uk/wp-content/uploads/WinBUGS_Vol1.pdf

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native skeletal maturity model from the BUGS bones example.

Missing grades do not contribute a likelihood factor.

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
