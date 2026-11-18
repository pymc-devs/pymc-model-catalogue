"""Import a single-build report into the existing ASV v2 result schema."""

import hashlib
import json
import math

from asv_bench._cache import MODEL_BENCHMARK_VERSION
from asv_bench._measure import metric_names


def rows(data):
    return {name: {key: value for key, value in zip(data["result_columns"], row)
                   if value is not None}
            for name, row in data["results"].items()}


def snapshot(results_dir):
    return {path: json.loads(path.read_text())
            for path in results_dir.glob("*/*.json") if path.name != "machine.json"}


def save(path, data, measurements):
    columns = list(data["result_columns"])
    for row in measurements.values():
        for key in row:
            if key not in columns:
                columns.append(key)
    data["result_columns"] = columns
    data["results"] = {
        name: [row.get(key) for key in columns] for name, row in measurements.items()
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data))
    temporary.replace(path)


def valid_result(row):
    values = row.get("result")
    return bool(values) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def update_dependencies(data, old, selected, versions=None):
    # ASV discards extra top-level fields on save. Restore the metadata for
    # untouched measurements so a partial rerun cannot change their tooltips.
    dependencies = {name: value for name, value in old.get("benchmark_dependencies", {}).items()
                    if name not in selected}
    if versions:
        dependencies.update(dict.fromkeys(selected, versions))
    data["benchmark_dependencies"] = dependencies


def import_reports(directory, results_dir, model, selected, before, machine=None):
    failed = False
    imported = set()
    names = metric_names(model)
    compile_name = names["compile_time"]
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text())
        if (report["model"] != model or report["target_builds"] != 1
                or set(report["rows"]) != set(names.values())
                or any(row["version"] != MODEL_BENCHMARK_VERSION for row in report["rows"].values())):
            raise RuntimeError(f"Invalid single-build report: {path}")
        environment = report["env_name"]
        if len(environment) >= 128:
            environment = "env-" + hashlib.md5(environment.encode()).hexdigest()
        candidates = []
        for record in results_dir.glob(f"*/{report['commit'][:8]}-{environment}.json"):
            data = json.loads(record.read_text())
            if data["commit_hash"] != report["commit"] or data["env_name"] != report["env_name"]:
                continue
            if machine and data["params"]["machine"] != machine:
                continue
            candidates.append((record, data))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one ASV result for {report['model']}: {candidates}")
        record, data = candidates[0]
        measurements = rows(data)
        # The compile benchmark transports the report; preserve it if only eval was requested.
        if compile_name not in selected:
            old = rows(before[record]).get(compile_name) if record in before else None
            if old is None:
                measurements.pop(compile_name, None)
            else:
                measurements[compile_name] = old
        for name in selected:
            measurements[name] = report["rows"][name]
            failed |= not valid_result(measurements[name])
        update_dependencies(data, before.get(record, {}),
                            selected, report.get("dependencies"))
        save(record, data, measurements)
        imported.add(record)
    # If compilation failed, publish failed values instead of stale successful values.
    for record, data in snapshot(results_dir).items():
        if record in imported or (machine and data["params"]["machine"] != machine):
            continue
        measurements = rows(data)
        value = measurements.get(compile_name)
        old = rows(before[record]).get(compile_name) if record in before else None
        if value is None or value == old:
            continue
        if valid_result(value):
            raise RuntimeError(f"Compile succeeded without a single-build report: {record}")
        for name in selected:
            measurements[name] = {"result": None, "params": [],
                                  "version": MODEL_BENCHMARK_VERSION,
                                  "started_at": value.get("started_at")}
        if compile_name not in selected:
            if old is None:
                measurements.pop(compile_name, None)
            else:
                measurements[compile_name] = old
        update_dependencies(data, before.get(record, {}), selected)
        save(record, data, measurements)
        failed = True
    return failed or not imported
