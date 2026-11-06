"""Observed histories for the posteriordb dog learning models."""

import numpy as np


def _dogs_history(data):
    """Compute observed shock and avoidance histories before each trial."""
    y = np.asarray(data["y"], dtype="int64")
    shock = np.cumsum(y, axis=1) - y
    avoid = np.arange(y.shape[1])[None, :] - shock
    return y, shock, avoid
