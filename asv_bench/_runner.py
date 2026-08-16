"""ASV-side wrapper around ``models._benchmark.build_logp_fn``.

Splits the shared ``build_logp_fn`` call into measured phases so asv can
capture each as a metric. ``build_logp_fn`` itself is intentionally
API-version-agnostic (no ``logp_dlogp_function``, no ``ravel_inputs``),
so the historical timeline can reach pymc releases that predate those.

Metric definitions (intentional, don't "correct" these):

- ``rewrite_time`` — wall-clock from calling ``build_logp_fn`` (which
  internally runs ``rewrite_pregrad``, takes gradients, joins inputs,
  and calls ``pymc.compile``) to the moment it returns. Covers graph
  construction and the full rewriter pipeline. Pytensor's NUMBA
  backend does not JIT in this phase; code generation is deferred to
  the first call.
- ``compile_time`` — wall-clock from the end of ``rewrite_time`` to the
  end of the first ``f(x)`` call. This deliberately includes one eval's
  arithmetic because that is when numba JIT compilation happens. For
  practical models the JIT dominates (seconds) and the arithmetic is
  microseconds, so calling this ``compile_time`` reflects what it
  actually measures.
- ``eval_time`` — steady-state per-call timing via asv's native timing
  machinery, implemented in ``bench_models.py``.
- ``n_rewrites`` — count of ``rewriting: ...`` lines printed by pytensor
  under ``config.optimizer_verbose = True``.
- ``peak_rss`` — highest RSS during the build, in bytes.
- ``peak_rss_delta`` — the same, minus the RSS the window started at.
  ``ModelBenchBuild.setup`` calls ``_prewarm()`` first, so this excludes
  interpreter, imports and NUMBA's LLVM init.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import resource
import sys
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
from pytensor import config

# The shared build_logp_fn lives in models/_benchmark.py — add the repo
# root to sys.path so the import works both when asv runs this module
# directly and when test scripts run it cwd-relative.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models._benchmark import build_logp_fn  # noqa: E402

_REWRITE_PREFIX = "rewriting: "
_PAGE_SIZE = None
try:
    import os as _os

    _PAGE_SIZE = _os.sysconf("SC_PAGE_SIZE")
except (AttributeError, ValueError, OSError):  # pragma: no cover - non-POSIX
    pass


def _ru_maxrss_bytes() -> int:
    """Process-lifetime RSS high-water mark, normalised to bytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage if sys.platform == "darwin" else usage * 1024


def _rss_bytes() -> int | None:
    """Current RSS in bytes, or None where /proc is unavailable."""
    if _PAGE_SIZE is None:
        return None
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * _PAGE_SIZE
    except (OSError, IndexError, ValueError):
        return None


class PeakRSS:
    """Max RSS over a window: /proc sampling, plus ru_maxrss when it rose in-window."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ru_at_entry = 0
        self.baseline = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            cur = _rss_bytes()
            if cur is not None and cur > self.peak:
                self.peak = cur
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakRSS":
        self._ru_at_entry = _ru_maxrss_bytes()
        cur = _rss_bytes()
        if cur is None:  # no /proc: ru_maxrss alone, no sampler
            self.baseline = self._ru_at_entry
            return self
        self.baseline = self.peak = cur
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._thread is None:
            self.peak = _ru_maxrss_bytes()
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        cur = _rss_bytes()  # in case the peak landed between samples
        if cur is not None and cur > self.peak:
            self.peak = cur
        ru = _ru_maxrss_bytes()
        if ru > self._ru_at_entry:  # a rise can only have happened in-window
            self.peak = max(self.peak, ru)

    @property
    def delta(self) -> int:
        return max(0, self.peak - self.baseline)


def build_and_measure(model_path: str, *, mode: str = "NUMBA") -> dict:
    discrete = model_path.startswith("models_discrete.")
    buf = io.StringIO()

    with PeakRSS() as tracker:
        module = importlib.import_module(model_path)
        model, ip = module.build_model()

        t0 = perf_counter()
        with config.change_flags(optimizer_verbose=True), contextlib.redirect_stdout(buf):
            fn, x = build_logp_fn(model, ip, mode=mode, with_grad=not discrete)
        rewrite_time = perf_counter() - t0

        # covers the first call on purpose: that is when NUMBA JITs
        t1 = perf_counter()
        out = fn(x)
        compile_time = perf_counter() - t1

    n_rewrites = sum(
        1 for line in buf.getvalue().splitlines() if line.startswith(_REWRITE_PREFIX)
    )

    if discrete:
        (logp,) = out
        assert np.isfinite(logp), f"logp is not finite: {logp}"
    else:
        logp, dlogp = out
        assert np.isfinite(logp), f"logp is not finite: {logp}"
        assert np.all(np.isfinite(dlogp)), "dlogp has non-finite values"

    return {
        "rewrite_time": rewrite_time,
        "compile_time": compile_time,
        "n_rewrites": n_rewrites,
        "peak_rss": tracker.peak,
        "peak_rss_delta": tracker.delta,
        "call": lambda: fn(x),
    }
