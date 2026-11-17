"""ASV benchmark module: per-model rewrite/compile/eval timings.

One pair of classes is generated per discovered model so each lands on
its own dashboard page with its own auto-scaled y-axis. Model timings
span microseconds (eight schools) to seconds (scan/ODE/VAR), which is
unreadable on a single shared plot even in log scale.

Each model gets two sibling classes:

- ``ModelBenchBuild_<model>`` — regular ``setup()`` calls
  ``build_and_measure`` to collect four scalar metrics
  (``rewrite_time``, ``compile_time``, ``n_rewrites``, ``peak_rss``). Each
  ``track_*`` method runs in its own asv subprocess. Compile time runs
  first; later metrics rebuild using its populated disk cache where
  supported by that PyTensor version.
- ``ModelBenchEval_<model>`` — regular ``setup()`` rebuilds the
  compiled function and keeps it on ``self`` so ``time_eval`` can
  call it repeatedly for asv's native steady-state timing.

Only the compile-time phase requires a cold model compilation. Later
Build setups and Eval timing rounds may load the compiled graph from disk.

scripts/run_benchmarks.py runs compile time first with fresh caches, then
runs all other metrics using those populated caches. ASV's timing rounds
can otherwise run Eval before Build, regardless of class name ordering.
The trivial-model prewarm initializes the compiler. Only compile time
requires a cold model compilation; RSS and the other metrics may use cache.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asv_bench._cache import COLD_COMPILE_VERSION  # noqa: E402
from asv_bench._runner import _ru_maxrss_bytes, build_and_measure  # noqa: E402


def _prewarm() -> None:
    """Build + call a trivial pymc model to warm the fresh-process caches.

    asv runs every benchmark in a forked subprocess, so each
    ``ModelBenchBuild`` setup pays cold-start cost for pytensor's
    rewriter and NUMBA's LLVM init. On small models that cold-start
    overhead (~200ms rewrite, ~2s NUMBA init) dominates the measurement
    and drowns out the model-specific signal.

    Running the full pipeline (``build_logp_fn`` + first call) on a
    throwaway one-var model exercises the same code paths as the real
    build, so the subsequent ``build_and_measure`` call runs against
    warm caches and the reported metrics reflect model-specific cost
    rather than process startup.

    Only ``ModelBenchBuild`` uses this — ``ModelBenchEval`` measures
    steady-state arithmetic after its setup's first call, which doesn't
    care whether NUMBA was cold or warm.
    """
    import pymc as pm

    from models._benchmark import build_logp_fn
    with pm.Model() as m:
        pm.Normal("x")
    fn, x = build_logp_fn(m, m.initial_point(), mode="NUMBA", with_grad=True)
    fn(x)


def _discover_models() -> list[str]:
    paths: list[str] = []
    for subdir in ("models", "models_discrete"):
        for py in sorted((REPO_ROOT / subdir).glob("*.py")):
            if py.name.startswith("_"):
                continue
            paths.append(f"{subdir}.{py.stem}")
    return paths


class _BaseModelBenchBuild:
    MODEL: str = ""  # overridden per generated subclass
    timeout = 1200.0

    def setup(self):
        _prewarm()
        self._result = build_and_measure(self.MODEL)

    def track_rewrite_time(self):
        return self._result["rewrite_time"]

    track_rewrite_time.unit = "seconds"

    def track_compile_time(self):
        self._check_cold_compile()
        return self._result["compile_time"]

    track_compile_time.unit = "seconds"
    track_compile_time.version = COLD_COMPILE_VERSION

    def track_n_rewrites(self):
        return self._result["n_rewrites"]

    track_n_rewrites.unit = "count"

    def track_peak_rss(self):
        """Whole-process high-water mark: asv spawns per benchmark, so it is this model's."""
        return max(self._result["peak_rss"], _ru_maxrss_bytes())

    track_peak_rss.unit = "bytes"

    def _check_cold_compile(self):
        cache = self._result["compile_cache"]
        print("Compile cache: " + json.dumps(cache), flush=True)
        if (cache["signatures_before_first_call"] or cache["cache_hits"]
                or not cache["compile_events"]):
            raise RuntimeError(f"Expected a cold model compilation: {cache}")


class _BaseModelBenchEval:
    MODEL: str = ""  # overridden per generated subclass
    # The heavy VAR-family models compile (in setup) for 15-20 min on the
    # 2-vCPU CI runner — bayesian_var_hierarchical sat right at the old 1200 s
    # ceiling — so give the per-benchmark budget generous headroom.
    timeout = 2400.0

    # Tighten steady-state timing for the small (~10-40 us) models whose
    # run-to-run scatter otherwise swamps the effect under test. We raise
    # `sample_time` so asv auto-fills a larger `number` — i.e. each sample
    # averages many more calls — which shrinks the estimate's variance WITHOUT
    # re-running setup. We deliberately do NOT raise `repeat`: under
    # --launch-method=spawn more samples can re-run the (very expensive) numba
    # compile in setup, which is what pushed a VAR model past the timeout.
    sample_time = 0.1

    def setup(self):
        self._call = build_and_measure(self.MODEL)["call"]

    def time_eval(self):
        self._call()


def _sanitize(model_path: str) -> str:
    return model_path.replace(".", "_")


# Generate one ModelBenchBuild_<model> + ModelBenchEval_<model> pair
# per discovered model. The class name is what asv uses as the
# dashboard page id, so sanitizing dots to underscores keeps the name
# filesystem- and URL-safe.
for _model in _discover_models():
    _name = _sanitize(_model)
    globals()[f"ModelBenchBuild_{_name}"] = type(
        f"ModelBenchBuild_{_name}",
        (_BaseModelBenchBuild,),
        {"MODEL": _model},
    )
    globals()[f"ModelBenchEval_{_name}"] = type(
        f"ModelBenchEval_{_name}",
        (_BaseModelBenchEval,),
        {"MODEL": _model},
    )

del _model, _name
