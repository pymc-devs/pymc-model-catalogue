"""
Model: LDA with 5 topics
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/ldaK5.stan
Authors: Blei, David M and Ng, Andrew Y and Jordan, Michael I (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/ldaK5.py
Upstream catalogue contributor: Mans Magnusson
Description: Latent Dirichlet Allocation (LDA) with token indicators marginalized out.

Data: prideprejudice_paragraph (upstream posterior: prideprejudice_paragraph-ldaK5)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/prideprejudice_paragraph.json.zip
    Archive SHA-256: 8ce84f0b08d2f015180f5521dbe6209a4e36218a64bb6737ffc5b3856fb72c06
    Local archive: data/posteriordb_prideprejudice_paragraph.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/prideprejudice_paragraph-ldaK5.json
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
- A: Stan = -211921.01610942537; PyMC = -211921.01610942508.
- B: Stan = -211895.96462831134; PyMC = -211895.96462831122.
- Raw delta: Stan = 25.05148111403105; PyMC = 25.051481113856426.
- Raw comparison: MATCH; residual = 1.7462298274040222e-10.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.2505148111403103e-06
  using adjusted delta_Stan when an adjustment is present.
- Identity of theta with shape (M,5) and phi with shape (5,V), each row a simplex with
  the supplied alpha/beta concentration vectors. Each token contributes logsumexp over
  its five topics; no categorical latent variable or document-word count normalization
  is introduced. Only document/word indices shift to zero-based.
- Original Stan SHA-256: 514ebdd2564b2cbb90f64ba59f252a2a8b4447a6008685b04141ddec820933a6
- Saved points: models/posteriordb_logp/points/prideprejudice_paragraph-ldaK5.npz
- Full evidence: models/posteriordb_logp/records/prideprejudice_paragraph-ldaK5.json
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
            / "posteriordb_prideprejudice_paragraph.json.zip"
        ) as archive,
        archive.open("prideprejudice_paragraph.json") as stream,
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
