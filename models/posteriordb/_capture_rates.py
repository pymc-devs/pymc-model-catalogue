"""Native capture-recapture models from Kéry and Schaub (2011)."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _capture_history_likelihood(history, omega, *, p=None, logit_p=None):
    """Integrate each population inclusion indicator across the complete capture history."""
    log_capture = pm.logp(pm.Bernoulli.dist(p=p, logit_p=logit_p), history).sum(axis=-1)
    log_present = pt.log(omega) + log_capture
    detected = np.any(history != 0, axis=-1)
    # The zero component applies to each whole history, not each occasion.
    pm.Potential(
        "capture_history",
        pt.switch(
            detected,
            log_present,
            pt.logaddexp(pt.log1p(-omega), log_present),
        ),
        dims="individual",
    )
