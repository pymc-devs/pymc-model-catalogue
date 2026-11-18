"""
Model: LDA with 5 topics
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/ldaK5.stan
Authors: Blei, David M and Ng, Andrew Y and Jordan, Michael I (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/ldaK5.py
Upstream catalogue contributor: Mans Magnusson
Description: Latent Dirichlet Allocation (LDA) with token indicators marginalized out.

Data: prideprejudice_chapter (upstream posterior: prideprejudice_chapter-ldaK5)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/prideprejudice_chapter.json.zip
    Archive SHA-256: d10c53005ae15aa53597378b60c162de32ed3abdcf25ae95242c1208bfa83c0e
    Local archive: data/posteriordb_prideprejudice_chapter.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/prideprejudice_chapter-ldaK5.json
References: blei2003latent (see posteriordb/source/bibliography/references.bib)
Reference URL: https://mc-stan.org/docs/2_18/stan-users-guide/latent-dirichlet-allocation.html

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Latent Dirichlet allocation with five topics and dataset concentration parameters.

Source: Blei, Ng, and Jordan (2003), as specified by posteriordb.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -205189.15496980018; PyMC = -205189.15496980195.
- B: Stan = -205072.78648196885; PyMC = -205072.78648197.
- Raw delta: Stan = 116.36848783132154; PyMC = 116.36848783196183.
- Raw comparison: MATCH; residual = -6.402842700481415e-10.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 2.1636848783196183e-06
  using adjusted delta_Stan when an adjustment is present.
- Identity of theta with shape (M,5) and phi with shape (5,V), each row a simplex with
  the supplied alpha/beta concentration vectors. Each token contributes logsumexp over
  its five topics; no categorical latent variable or document-word count normalization
  is introduced. Only document/word indices shift to zero-based.
- Original Stan SHA-256: 514ebdd2564b2cbb90f64ba59f252a2a8b4447a6008685b04141ddec820933a6
- Saved points: models/posteriordb_logp/points/prideprejudice_chapter-ldaK5.npz
- Full evidence: models/posteriordb_logp/records/prideprejudice_chapter-ldaK5.json
- Two-point evidence only; no global equivalence, support or inference claim.

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
            Path(__file__).parent
            / "data"
            / "posteriordb_prideprejudice_chapter.json.zip"
        ) as archive,
        archive.open("prideprejudice_chapter.json") as stream,
    ):
        data = json.load(stream)
    topics = 5
    alpha = data["alpha"]
    beta = data["beta"]
    words = np.asarray(data["w"], dtype="int64") - 1
    documents = np.asarray(data["doc"], dtype="int64") - 1
    coords = {
        "document": range(data["M"]),
        "topic": range(topics),
        "word": range(data["V"]),
        "token": range(data["N"]),
    }
    with pm.Model(coords=coords) as model:
        word = pm.Data("word_index", words, dims="token")
        document = pm.Data("document_index", documents, dims="token")
        theta = pm.Dirichlet(
            "theta", a=np.asarray(alpha, dtype="float64"), dims=("document", "topic")
        )
        phi = pm.Dirichlet(
            "phi", a=np.asarray(beta, dtype="float64"), dims=("topic", "word")
        )
        # Sum over topics without a token-by-vocabulary probability tensor.
        token_topic_logp = pt.log(theta[document]) + pt.log(phi[:, word].T)
        pm.Potential("word_likelihood", pt.logsumexp(token_topic_logp, axis=-1).sum())
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
