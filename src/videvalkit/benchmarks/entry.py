"""Universal subprocess entry point for benchmark adapters.

Invoked by CondaEnvDispatcher inside a benchmark's own conda env::

    python -m videvalkit.benchmarks.entry --benchmark vbench --method evaluate

reads a JSON payload from stdin, dispatches to the adapter's method,
writes a JSON result to stdout. Any uncaught exception exits non-zero
with the traceback on stderr.

This is the *only* code path that runs upstream code, so it is the right
place to hook in process-wide setup (HF cache dir, torch flags, logging).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from typing import Any


_REGISTRY = {
    "vbench":        "videvalkit.benchmarks.vbench:VBenchBenchmark",
    "vbench2":       "videvalkit.benchmarks.vbench2:VBench2Benchmark",
    "videobench":    "videvalkit.benchmarks.videobench:VideoBenchBenchmark",
    "worldjen":      "videvalkit.benchmarks.worldjen:WorldJenBenchmark",
    "t2vcompbench":  "videvalkit.benchmarks.t2vcompbench:T2VCompBenchBenchmark",
    "physics_iq":    "videvalkit.benchmarks.physics_iq:PhysicsIQBenchmark",
    "vbench_pp":     "videvalkit.benchmarks.vbench_pp:VBenchPPBenchmark",
    "v_reasonbench": "videvalkit.benchmarks.v_reasonbench:VReasonBenchBenchmark",
    "worldscore":    "videvalkit.benchmarks.worldscore:WorldScoreBenchmark",
    "semantics_axis": "videvalkit.benchmarks.semantics_axis:SemanticsAxisBenchmark",
}


def _load_adapter(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown benchmark {name!r}; known: {list(_REGISTRY)}")
    module_path, cls_name = _REGISTRY[name].split(":")
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls()


def _dispatch(adapter: Any, method: str, payload: dict[str, Any]) -> Any:
    fn = getattr(adapter, method, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"adapter {adapter.name!r} has no method {method!r}")
    return fn(**payload)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="videvalkit.benchmarks.entry")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--method", required=True)
    args = ap.parse_args(argv)

    try:
        payload = json.loads(sys.stdin.read() or "{}")
        adapter = _load_adapter(args.benchmark)
        result = _dispatch(adapter, args.method, payload)
        # Pydantic models / dataclasses are JSON-serialized via default=str fallback.
        sys.stdout.write(
            json.dumps(result, default=lambda o: getattr(o, "model_dump", lambda: str(o))())
        )
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
