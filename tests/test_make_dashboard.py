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


class DashboardEnvironmentsTest(unittest.TestCase):
    def test_preserves_history_and_overlapping_environment_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder)
            idx = {
                "revision_to_hash": {
                    "1": "old", "2": "shared", "3": "new", "4": "orphan",
                },
                "tags": {"pymc-5.16.0": 1, "pymc-6.0.0": 2, "pymc-6.4.0": 3},
            }
            (src / "index.json").write_text(json.dumps(idx))
            graphs = src / "graphs" / "branch-timeline" / "machine-ci"
            benchmark = "bench_models.ModelBenchBuild_models_BEST.track_compile_time.json"
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
            self.assertEqual([p["tag"] for p in points], [
                "pymc-5.16.0", "pymc-5.16.0", "pymc-6.0.0",
                "pymc-6.0.0", "pymc-6.4.0",
            ])
            output = src / "dashboard.html"
            dashboard.render(data, loaded_idx, output)
            payload = json.loads(re.search(
                r"const PAYLOAD = (.*);", output.read_text(),
            ).group(1))
            self.assertEqual(payload["measured_count"], 3)
            self.assertEqual(payload["versions"], [
                "pymc-5.16.0", "pymc-6.0.0", "pymc-6.4.0",
            ])
            self.assertEqual(payload["environments"], [
                "ci · Python 3.12", "ci · Python 3.14",
            ])

    def test_empty_dashboard(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "dashboard.html"
            dashboard.render({}, {}, output)
            payload = json.loads(re.search(
                r"const PAYLOAD = (.*);", output.read_text(),
            ).group(1))
            self.assertEqual(payload["measured_count"], 0)
            self.assertEqual(payload["environments"], [])


if __name__ == "__main__":
    unittest.main()
