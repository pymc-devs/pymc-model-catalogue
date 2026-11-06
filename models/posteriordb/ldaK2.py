"""Latent Dirichlet allocation with two topics and uniform simplex densities.

Source: Blei, Ng, and Jordan (2003), as specified by posteriordb.
"""

import numpy as np

from ._data import load_data
from ._topic_neural_lda import build_topic_neural_lda


def build_model(posterior: str | None = None):
    """Build every published model and data pair with two topics."""
    data = load_data("ldaK2", posterior)
    return build_topic_neural_lda(data, 2, np.ones(2), np.ones(data["V"]))


if __name__ == "__main__":
    from ._benchmark import main

    main("ldaK2")
