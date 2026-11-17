"""Benchmark helper for the PyMC model catalogue.

``build_logp_fn`` is the shared primitive used by both the ad-hoc
``run_benchmark`` runner here and the ASV wrapper in
``asv_bench/_runner.py``. It reimplements what
``model.logp_dlogp_function(ravel_inputs=True, mode=...)`` does today
without depending on that API (``ravel_inputs`` didn't exist in older
pymc releases, which breaks the historical timeline we backfill).

The reimplementation walks the same steps pymc does internally:

1. ``model.logp()`` — the scalar logp graph.
2. ``rewrite_pregrad`` — canonicalize + stabilize so ``pt.grad`` sees
   a clean graph (matches what pymc does before taking gradients).
3. ``pt.grad(logp, value_vars)`` — per-value-var gradients, each
   ravelled and joined into a single vector via ``pt.join(0, ...)``.
4. ``join_nonshared_inputs`` — replace the per-value-var inputs with
   a single raveled input vector so the resulting function takes one
   numpy array.
5. ``pymc.pytensorf.compile`` — pymc's wrapper over
   ``pytensor.function`` that wires in pymc's default rewrites and
   RandomVariable RNG updates.

For discrete models we skip the gradient (``pt.grad`` can't handle int
value vars) and the compiled function outputs only the scalar logp.
"""

from __future__ import annotations

import timeit

import numpy as np
import pytensor.tensor as pt
from pymc.blocking import DictToArrayBijection
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.pytensorf import join_nonshared_inputs, rewrite_pregrad

try:
    from pymc.pytensorf import compile as pymc_compile
except ImportError:
    from pymc.pytensorf import compile_pymc as pymc_compile


def build_logp_fn(model, ip, *, mode: str = "NUMBA", with_grad: bool = True):
    """Build a compiled (logp[, dlogp]) function taking a raveled vector.

    Parameters
    ----------
    model :
        A pymc ``Model`` instance.
    ip :
        The initial point dict, as returned by ``build_model()``. Passed
        through instead of re-calling ``model.initial_point()`` because
        some model files clear ``rvs_to_initial_values`` after building
        — a second ``initial_point()`` call would then return different
        values than the ones the model was authored with.
    mode :
        PyTensor compile mode to pass through to ``pymc.compile``.
    with_grad :
        If True, output is ``[logp, dlogp_raveled]``; if False, output is
        ``[logp]`` only. Discrete models must pass ``with_grad=False``.

    Returns
    -------
    fn :
        Compiled pytensor function with ``trust_input=True``. Takes a
        single 1-D array of length ``sum(v.size for v in value_vars)``.
    x :
        The raveled initial-point array, ready to pass to ``fn``.
    """
    value_vars = model.value_vars

    logp = rewrite_pregrad(model.logp())

    if with_grad:
        grads = pt.grad(logp, value_vars)
        dlogp = pt.join(0, *(g.ravel() for g in grads))
        outputs = [logp, dlogp]
    else:
        outputs = [logp]

    new_outputs, joined = join_nonshared_inputs(
        point=ip, outputs=outputs, inputs=value_vars,
    )
    fn = pymc_compile([joined], new_outputs, mode=mode)
    fn.trust_input = True

    raveled = DictToArrayBijection.map({v.name: ip[v.name] for v in value_vars})
    x = raveled.data

    return fn, x


def run_benchmark(build_model, *, discrete: bool = False, mode: str = "NUMBA"):
    model, ip = build_model()
    frozen_model, frozen_ip = build_model()
    frozen_model = freeze_dims_and_data(frozen_model)

    for label, m, p in [("original", model, ip), ("frozen", frozen_model, frozen_ip)]:
        print(f"\n=== {label} ===")
        f, x = build_logp_fn(m, p, mode=mode, with_grad=not discrete)

        out = f(x)
        if discrete:
            (logp,) = out
            assert np.isfinite(logp), f"logp is not finite: {logp}"
            print(f"logp = {logp:.4f}")
        else:
            logp, dlogp = out
            assert np.isfinite(logp), f"logp is not finite: {logp}"
            assert np.all(np.isfinite(dlogp)), "dlogp has non-finite values"
            print(f"logp = {logp:.4f}, grad norm = {np.linalg.norm(dlogp):.4f}")

        call = lambda: f(x)
        single = timeit.timeit(call, number=1)
        n_evals = max(1, min(100_000, int(15.0 / max(single, 1e-8))))
        total = timeit.timeit(call, number=n_evals)
        print(f"Time per call: {total / n_evals * 1e6:.1f} us ({n_evals} evals)")
