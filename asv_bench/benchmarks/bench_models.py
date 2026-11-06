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

Optional registry workloads add a canonical posterior ID and explicit backend to each class name.
Their dataset loading occurs before the measured phases.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asv_bench._runner import build_and_measure  # noqa: E402


def _prewarm(mode: str = "NUMBA") -> None:
    """Warm the selected backend with a one-variable graph.

    ASV starts each benchmark in a fresh subprocess.
    This call reduces process-start costs before the measured build phase.
    It uses the workload backend without a fallback.

    Only build benchmarks use this warmup.
    Evaluation benchmarks measure steady-state calls after setup.
    """
    import pymc as pm

    from models._benchmark import build_logp_fn

    with pm.Model() as m:
        pm.Normal("x")
    fn, x = build_logp_fn(m, m.initial_point(), mode=mode, with_grad=True)
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
    POSTERIOR: str | None = None
    MODE: str = "NUMBA"
    timeout = 1200.0

    def setup(self):
        _prewarm(self.MODE)
        self._result = build_and_measure(
            self.MODEL,
            mode=self.MODE,
            posterior=self.POSTERIOR,
        )

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
    POSTERIOR: str | None = None
    MODE: str = "NUMBA"
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
        self._call = build_and_measure(
            self.MODEL,
            mode=self.MODE,
            posterior=self.POSTERIOR,
        )["call"]

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


# Each optional workload names its posterior pair and explicit backend.
# Double underscores preserve the canonical posterior's hyphen.
from models.posteriordb import get_posterior, list_posteriors  # noqa: E402

for _posterior in list_posteriors():
    _record = get_posterior(_posterior)
    _mode = _record["benchmark_mode"]
    _name = f"models_posteriordb__{_posterior.replace('-', '__')}__{_mode}"
    for _phase, _base in (
        ("Build", _BaseModelBenchBuild),
        ("Eval", _BaseModelBenchEval),
    ):
        _class_name = f"ModelBench{_phase}_{_name}"
        globals()[_class_name] = type(
            _class_name,
            (_base,),
            {
                "MODEL": _record["module"],
                "POSTERIOR": _posterior,
                "MODE": _mode,
            },
        )

del _posterior, _record, _mode, _name, _phase, _base, _class_name
