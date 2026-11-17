"""Model metric descriptors for ASV and its single-build worker.

scripts/run_benchmarks.py invokes only track_compile_time for each model.
That worker warms the compiler, builds the target once, and samples eval
using the retained callable. The wrapper imports all five result rows.
"""

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asv_bench._cache import MODEL_BENCHMARK_VERSION  # noqa: E402
from asv_bench._measure import write_measurements  # noqa: E402
from asv_bench._runner import _ru_maxrss_bytes, build_and_measure  # noqa: E402


def _prewarm() -> None:
    """Initialize LLVM and common helpers on an unrelated one-variable model.

    This runs once per model worker, outside the target's measured phases.
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
        if not os.environ.get("CATALOGUE_BUNDLE_DIR"):
            raise RuntimeError("Use scripts/run_benchmarks.py to measure each model once")
        self._started_at = time.time() * 1000
        start = time.perf_counter()
        _prewarm()
        self._result = build_and_measure(self.MODEL)
        self._result["peak_rss"] = max(self._result["peak_rss"], _ru_maxrss_bytes())
        self._build_seconds = time.perf_counter() - start

    def track_rewrite_time(self):
        return self._result["rewrite_time"]

    track_rewrite_time.unit = "seconds"
    track_rewrite_time.version = MODEL_BENCHMARK_VERSION

    def track_compile_time(self):
        self._check_cold_compile()
        eval_class = globals()[f"ModelBenchEval_{_sanitize(self.MODEL)}"]
        write_measurements(self.MODEL, self._result, eval_class,
                           self._started_at, self._build_seconds)
        return self._result["compile_time"]

    track_compile_time.unit = "seconds"
    track_compile_time.version = MODEL_BENCHMARK_VERSION

    def track_n_rewrites(self):
        return self._result["n_rewrites"]

    track_n_rewrites.unit = "count"
    track_n_rewrites.version = MODEL_BENCHMARK_VERSION

    def track_peak_rss(self):
        """Whole-process high-water mark: asv spawns per benchmark, so it is this model's."""
        return self._result["peak_rss"]

    track_peak_rss.unit = "bytes"
    track_peak_rss.version = MODEL_BENCHMARK_VERSION

    def _check_cold_compile(self):
        cache = self._result["compile_cache"]
        print("Compile cache: " + json.dumps(cache), flush=True)
        if (cache["signatures_before_first_call"] or cache["cache_hits"]
                or not cache["compile_events"]):
            raise RuntimeError(f"Expected a cold model compilation: {cache}")


class _BaseModelBenchEval:
    MODEL: str = ""  # overridden per generated subclass
    timeout = 2400.0

    # ASV's native batching averages many calls per sample. The single-build
    # worker retains these settings but has no model setup hook to repeat.
    sample_time = 0.1

    # Tighten steady-state timing, especially for the small (~10-40 us) models
    # whose run-to-run scatter otherwise swamps the effect under test. asv
    # auto-fills `number` so each sample spans `sample_time`; `repeat`
    # ``(min, max, max_time_s)`` then takes many samples (capped at 30 s) and
    # reports their median. Cheap models reach the 100-sample cap in well under
    # a second, so they get the most averaging — exactly where it's needed.
    # Both arms of a paired experiment run in the same CI job, so the shared
    # between-run drift cancels and this directly shrinks the cross-arm ratio's
    # residual error.
    repeat = (20, 100, 30.0)
    sample_time = 0.02
    warmup_time = 0.2

    def setup(self):
        raise RuntimeError("Eval is sampled by scripts/run_benchmarks.py's single-build worker")

    def time_eval(self):
        self._call()

    time_eval.version = MODEL_BENCHMARK_VERSION


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
