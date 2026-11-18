"""Configure fresh compiler caches for one model worker."""

import os
from pathlib import Path


MODEL_BENCHMARK_VERSION = "single-build-v1"


def cache_environment(directory):
    root = Path(directory)
    # Retain unrelated settings (e.g. floatX and NUMBA fastmath flags).
    flags = [
        flag for flag in os.environ.get("PYTENSOR_FLAGS", "").split(",")
        if flag.strip() and flag.split("=", 1)[0].strip()
        not in {"compiledir", "base_compiledir"}
    ]
    flags.extend([
        f"base_compiledir={root / 'pytensor'}",
        f"compiledir={root / 'pytensor'}",
    ])
    return os.environ | {
        "PYTENSOR_FLAGS": ",".join(flags),
        "NUMBA_CACHE_DIR": str(root / "numba"),
    }
