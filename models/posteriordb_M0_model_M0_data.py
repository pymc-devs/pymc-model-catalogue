"""
Model: Inferring population size with constant detection probability
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/M0_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/M0_model.py
Authors: M Kery, M Schaub (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/M0_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Detection probability of an individual is assumed constant over individuals and over
    time periods

Data: M0_data (upstream posterior: M0_data-M0_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/M0_data.json.zip
    Archive SHA-256: c063c34068a1bab975a768d39d9a4416882bbb4c82be09eb6b404f2dc3478ec2
    Local archive: data/posteriordb_M0_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/M0_data-M0_model.json
References: kery2011population (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/blob/master/BPA/Ch.06

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Constant detection with exact population-inclusion marginalization.

Source: Kéry and Schaub (2011), chapter 6.

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
            Path(__file__).parent / "data" / "posteriordb_M0_data.json.zip"
        ) as archive,
        archive.open("M0_data.json") as stream,
    ):
        data = json.load(stream)
    counts = np.asarray(data["y"], dtype="int64").sum(axis=1)
    occasions = int(data["T"])
    with pm.Model(coords={"individual": np.arange(data["M"])}) as model:
        omega = pm.Uniform("omega", 0, 1)
        p = pm.Uniform("p", 0, 1)
        pm.ZeroInflatedBinomial(
            "captures",
            psi=omega,
            n=occasions,
            p=p,
            observed=counts,
            dims="individual",
        )
        no_capture = (1 - p) ** occasions
        pm.Deterministic(
            "omega_nd", omega * no_capture / (omega * no_capture + 1 - omega)
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
