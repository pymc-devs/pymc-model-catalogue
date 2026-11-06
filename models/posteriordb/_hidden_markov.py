"""Native hidden-state likelihoods for the posteriordb model family."""

import pytensor
import pytensor.tensor as pt


def _hidden_markov_loglik(log_emissions, transition, initial_logp):
    """Sum hidden paths with a symbolic log-space forward recursion."""
    log_transition = pt.log(transition)

    def step(log_emission, previous, log_transition):
        return log_emission + pt.logsumexp(previous[:, None] + log_transition, axis=0)

    filtered, _ = pytensor.scan(
        fn=step,
        sequences=[log_emissions[1:]],
        outputs_info=[initial_logp],
        non_sequences=[log_transition],
        strict=True,
        name="hidden_markov_forward",
    )
    final = pt.concatenate([initial_logp[None, :], filtered], axis=0)[-1]
    return pt.logsumexp(final)
