import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_dashboard.py"
spec = importlib.util.spec_from_file_location("make_dashboard", SCRIPT)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)

EXPERIMENT_SCRIPT = SCRIPT.with_name("make_experiment_dashboard.py")
experiment_spec = importlib.util.spec_from_file_location(
    "make_experiment_dashboard",
    EXPERIMENT_SCRIPT,
)
experiment_dashboard = importlib.util.module_from_spec(experiment_spec)
experiment_spec.loader.exec_module(experiment_dashboard)


class DashboardEnvironmentsTest(unittest.TestCase):
    def test_preserves_history_and_overlapping_environment_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder)
            idx = {
                "revision_to_hash": {
                    "1": "old",
                    "2": "shared",
                    "3": "new",
                    "4": "orphan",
                },
                "tags": {"pymc-5.16.0": 1, "pymc-6.0.0": 2, "pymc-6.4.0": 3},
            }
            (src / "index.json").write_text(json.dumps(idx))
            graphs = src / "graphs" / "branch-timeline" / "machine-ci"
            benchmark = (
                "bench_models.ModelBenchBuild_models_BEST.track_compile_time.json"
            )
            for python, series in {
                "3.12": [[2, 1.6], [1, 6.5], [4, 999]],
                "3.14": [[3, 0.03], [2, 0.97], [1, None], [99, 999]],
            }.items():
                graph = graphs / f"python-{python}" / benchmark
                graph.parent.mkdir(parents=True)
                graph.write_text(json.dumps(series))

            data, loaded_idx = dashboard.load_data(src)
            points = data["models.BEST"]["track_compile_time"]
            self.assertEqual(
                {(p["tag"], p["environment"]): p["value"] for p in points},
                {
                    ("pymc-5.16.0", "ci · Python 3.12"): 6.5,
                    ("pymc-5.16.0", "ci · Python 3.14"): None,
                    ("pymc-6.0.0", "ci · Python 3.12"): 1.6,
                    ("pymc-6.0.0", "ci · Python 3.14"): 0.97,
                    ("pymc-6.4.0", "ci · Python 3.14"): 0.03,
                },
            )
            self.assertEqual(
                [p["tag"] for p in points],
                [
                    "pymc-5.16.0",
                    "pymc-5.16.0",
                    "pymc-6.0.0",
                    "pymc-6.0.0",
                    "pymc-6.4.0",
                ],
            )
            output = src / "dashboard.html"
            dashboard.render(data, loaded_idx, output)
            payload = json.loads(
                re.search(
                    r"const PAYLOAD = (.*);",
                    output.read_text(),
                ).group(1)
            )
            self.assertEqual(payload["measured_count"], 3)
            self.assertEqual(
                payload["versions"],
                [
                    "pymc-5.16.0",
                    "pymc-6.0.0",
                    "pymc-6.4.0",
                ],
            )
            self.assertEqual(
                payload["environments"],
                [
                    "ci · Python 3.12",
                    "ci · Python 3.14",
                ],
            )

    def test_empty_dashboard(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "dashboard.html"
            dashboard.render({}, {}, output)
            payload = json.loads(
                re.search(
                    r"const PAYLOAD = (.*);",
                    output.read_text(),
                ).group(1)
            )
            self.assertEqual(payload["measured_count"], 0)
            self.assertEqual(payload["environments"], [])

    def test_preserves_nested_posterior_and_backend_labels(self):
        workloads = {
            "models_posteriordb__nes1972__nes__NUMBA": "models.posteriordb/nes1972-nes [NUMBA]",
            "models_posteriordb__nes1976__nes__NUMBA": "models.posteriordb/nes1976-nes [NUMBA]",
            "models_posteriordb__nes1972__nes__FAST_RUN": "models.posteriordb/nes1972-nes [FAST_RUN]",
        }
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder)
            (src / "index.json").write_text(
                json.dumps(
                    {
                        "revision_to_hash": {"1": "current"},
                        "tags": {"pymc-6.4.0": 1},
                    }
                )
            )
            graphs = src / "graphs" / "branch-timeline"
            graphs.mkdir(parents=True)
            for workload in workloads:
                stem = f"bench_models.ModelBenchBuild_{workload}.track_compile_time"
                (graphs / f"{stem}.json").write_text(json.dumps([[1, 0.5]]))
                self.assertEqual(
                    experiment_dashboard._parse_benchmark_name(stem),
                    (workloads[workload], "track_compile_time"),
                )
            data, _ = dashboard.load_data(src)
            self.assertEqual(set(data), set(workloads.values()))
            for metrics in data.values():
                self.assertEqual(metrics["track_compile_time"][0]["value"], 0.5)


if __name__ == "__main__":
    unittest.main()
