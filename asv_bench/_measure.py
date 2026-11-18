"""Collect all model metrics while retaining a single compiled callable."""

import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace


def metric_names(model):
    suffix = model.replace(".", "_")
    build = f"bench_models.ModelBenchBuild_{suffix}."
    return {
        "rewrite_time": build + "track_rewrite_time",
        "compile_time": build + "track_compile_time",
        "n_rewrites": build + "track_n_rewrites",
        "peak_rss": build + "track_peak_rss",
        "eval": f"bench_models.ModelBenchEval_{suffix}.time_eval",
    }


def measure_eval(call, eval_class):
    """Use ASV's sampler and statistics, with no model setup hook to repeat."""
    from asv_runner.benchmarks.time import TimeBenchmark
    from asv_runner.statistics import compute_stats
    from numba.core import event

    attributes = {
        key: getattr(eval_class, key)
        for key in ("rounds", "processes", "repeat", "number", "sample_time",
                    "warmup_time", "min_run_count", "timeout", "timer")
        if hasattr(eval_class, key)
    }
    settings = SimpleNamespace(**attributes)
    benchmark = TimeBenchmark("time_eval", call, [settings])
    samples = []

    class CompileCounter(event.Listener):
        count = 0

        def on_start(self, evt):
            self.count += 1

        def on_end(self, evt):
            pass

    counter = CompileCounter()
    with event.install_listener("numba:compile", counter):
        benchmark.do_setup()
        for _ in range(benchmark.rounds):
            measured = benchmark.run()
            samples.extend(measured["samples"])
            # ASV normally carries this calibration into its later rounds.
            settings.number = measured["number"]
            benchmark.number = settings.number
    if counter.count:
        raise RuntimeError(f"Eval unexpectedly triggered {counter.count} Numba compilations")
    value, stats = compute_stats(samples, settings.number)
    return {"result": [value], "samples": [samples],
            **{"stats_" + key: [value] for key, value in stats.items()}}


def write_measurements(model, result, eval_class, started_at, build_seconds):
    import numba
    import pytensor

    from asv_bench._cache import MODEL_BENCHMARK_VERSION

    commit = os.environ.get("CATALOGUE_COMMIT") or os.environ.get("ASV_COMMIT")
    environment = os.environ.get("ASV_ENV_NAME")
    if not environment:
        # ASV's existing-environment launcher omits its usual ASV_* variables.
        environment = "existing-py" + os.path.abspath(sys.executable).replace(os.path.sep, "_")
    if not commit or not environment:
        raise RuntimeError("ASV result identity missing; use --commit or --set-commit-hash")
    names = metric_names(model)
    rows = {}
    for key in ("rewrite_time", "compile_time", "n_rewrites", "peak_rss"):
        rows[names[key]] = {
            "result": [result[key]], "params": [], "version": MODEL_BENCHMARK_VERSION,
            "started_at": started_at, "duration": build_seconds if key == "compile_time" else 0,
        }
    start = time.perf_counter()
    eval_error = None
    try:
        evaluation = measure_eval(result["call"], eval_class)
    except Exception:
        eval_error = traceback.format_exc()
        print(eval_error, flush=True)
        evaluation = {"result": None}
    rows[names["eval"]] = {
        "params": [], "version": MODEL_BENCHMARK_VERSION,
        "started_at": started_at, "duration": time.perf_counter() - start, **evaluation,
    }
    report = {
        "model": model, "commit": commit, "env_name": environment,
        "rows": rows, "compile_cache": result["compile_cache"],
        "target_builds": 1, "eval_error": eval_error,
        "dependencies": {"pytensor": pytensor.__version__, "numba": numba.__version__},
    }
    directory = Path(os.environ["CATALOGUE_BUNDLE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model}-{os.getpid()}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report))
    temporary.replace(path)
    if eval_error:
        summary = "eval failed; build measurements retained"
    else:
        summary = (f"eval {evaluation['result'][0]:.6g} s/call, "
                   f"{evaluation['stats_repeat'][0]} samples, "
                   f"{evaluation['stats_number'][0]} calls/sample; no further compilations")
    print(f"Single model build: {model}; {summary}", flush=True)
