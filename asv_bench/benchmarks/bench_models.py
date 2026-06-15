"""ASV benchmark module: per-model rewrite/compile/eval timings.

One pair of classes is generated per discovered model so each lands on
its own dashboard page with its own auto-scaled y-axis. Model timings
span microseconds (eight schools) to seconds (scan/ODE/VAR), which is
unreadable on a single shared plot even in log scale.

Each model gets two sibling classes:

- ``ModelBenchBuild_<model>`` — regular ``setup()`` calls
  ``build_and_measure`` to collect three scalar metrics
  (``rewrite_time``, ``compile_time``, ``n_rewrites``). Each
  ``track_*`` method runs in its own asv subprocess, so the rebuild
  cost is paid three times per (commit, model). We deliberately
  accept that 3× cost instead of caching: every caching strategy we
  tried (asv ``setup_cache``, disk cache keyed by git SHA, disk cache
  keyed by pymc version) had a subtle scoping trap that produced flat
  identical timelines. Re-measuring is slower but obviously correct.
- ``ModelBenchEval_<model>`` — regular ``setup()`` rebuilds the
  compiled function and keeps it on ``self`` so ``time_eval`` can
  call it repeatedly for asv's native steady-state timing.

Total rebuilds per (commit, model): 3 (Build) + 1 (Eval) = 4.

Alphabetical class ordering puts ``Build`` before ``Eval`` (``B < E``),
so ``track_compile_time`` is measured against a cold pytensor NUMBA
disk cache. ``ModelBenchEval`` subsequently rebuilds with a warm cache,
which is fine — ``time_eval`` only cares about steady-state arithmetic.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asv_bench._runner import build_and_measure  # noqa: E402


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
        return self._result["compile_time"]

    track_compile_time.unit = "seconds"

    def track_n_rewrites(self):
        return self._result["n_rewrites"]

    track_n_rewrites.unit = "count"


class _BaseModelBenchEval:
    MODEL: str = ""  # overridden per generated subclass
    timeout = 1200.0

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
