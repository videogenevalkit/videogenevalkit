"""Diagnostics — health checks for `videvalkit doctor`.

Probes:
  * Conda envs that adapters expect to exist
  * VLM judge endpoints (HTTP reachability)
  * API keys referenced by SUPPORTED_JUDGES
  * Workspace layout (videos / prompts presence)
  * Upstream packages importable inside each env

Returns a structured report; CLI renders it as a table.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import aiohttp


def check_conda_env(env_name: str) -> dict[str, Any]:
    if shutil.which("conda") is None:
        return {"ok": False, "reason": "conda not on PATH"}
    out = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    if out.returncode != 0:
        return {"ok": False, "reason": f"conda env list failed: {out.stderr[:200]}"}
    present = any(line.split()[0] == env_name for line in out.stdout.splitlines()
                  if line and not line.startswith("#"))
    return {"ok": present, "reason": "" if present else "env missing"}


async def _ping_endpoint(url: str, timeout: float = 3.0) -> bool:
    try:
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_cfg) as s:
            # /models is a common OpenAI-compat probe; many vLLM servers respond.
            probe_url = url.rstrip("/") + "/models"
            async with s.get(probe_url) as r:
                return r.status < 500   # 4xx is "alive but unauthenticated"
    except Exception:
        return False


def check_judge(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "kind": cfg.get("kind"), "model": cfg.get("model")}
    # API key check
    env = cfg.get("api_key_env")
    if env:
        out["api_key_present"] = bool(os.environ.get(env))
    else:
        out["api_key_present"] = True   # local endpoint, no key needed
    # Reachability check (only meaningful for openai_compatible with explicit endpoint)
    if cfg.get("kind") == "openai_compatible" and cfg.get("endpoint"):
        try:
            ok = asyncio.run(_ping_endpoint(cfg["endpoint"]))
        except Exception as e:
            ok = False
            out["reachability_error"] = str(e)[:200]
        out["reachable"] = ok
    else:
        out["reachable"] = None   # SDK-based; can't probe without API call
    return out


def check_workspace(workspace: Path) -> dict[str, Any]:
    if not workspace.exists():
        return {"ok": False, "reason": "workspace does not exist"}
    layout_keys = {
        "videos":   workspace / "videos",
        "prompts":  workspace / "prompts",
        "results":  workspace / "results",
        "api_logs": workspace / "api_logs",
    }
    out = {"ok": True, "dirs": {}}
    for k, p in layout_keys.items():
        out["dirs"][k] = {"exists": p.exists(), "n_entries": len(list(p.iterdir())) if p.exists() else 0}
    return out


def check_adapter_imports() -> dict[str, dict[str, Any]]:
    """Try to import each adapter; reports import errors (helps diagnose env mismatches).

    Module paths are derived from the entry registry so doctor auto-tracks
    every benchmark added to ``benchmarks/entry.py``.
    """
    from videvalkit.benchmarks import entry as _entry
    out: dict[str, dict[str, Any]] = {}
    for name, target in _entry._REGISTRY.items():
        modpath = target.split(":", 1)[0]
        try:
            __import__(modpath)
            out[name] = {"ok": True}
        except Exception as e:
            out[name] = {"ok": False, "error": str(e)[:200]}
    return out


def check_devices() -> dict[str, Any]:
    """Detect available compute devices [CUDA / NPU / MPS / CPU]."""
    out: dict[str, Any] = {}
    try:
        import torch
        out["cuda"] = {
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        mps = getattr(torch.backends, "mps", None)
        out["mps"] = {"available": bool(mps and mps.is_available())}
    except ImportError:
        out["torch"] = {"available": False, "reason": "torch not importable"}
    try:
        import torch_npu  # noqa
        import torch
        out["npu"] = {"available": torch.npu.is_available()}
    except Exception:
        out["npu"] = {"available": False}
    return out


def check_plugins() -> dict[str, Any]:
    try:
        from videvalkit.plugins import plugin_sources_report
        return plugin_sources_report()
    except Exception as e:
        return {"error": str(e)}


def check_metrics() -> dict[str, Any]:
    try:
        from videvalkit.metrics import SUPPORTED_METRICS
    except Exception as e:
        return {"error": str(e)}
    by_kind: dict[str, int] = {}
    for c in SUPPORTED_METRICS.values():
        k = c.get("kind", "?")
        by_kind[k] = by_kind.get(k, 0) + 1
    return {
        "total": len(SUPPORTED_METRICS),
        "judge_free": sum(1 for c in SUPPORTED_METRICS.values()
                          if not c.get("needs_judge", False)),
        "by_kind": by_kind,
    }


def check_profiles() -> dict[str, Any]:
    try:
        from videvalkit.core.profile import SUPPORTED_PROFILES
    except Exception as e:
        return {"error": str(e)}
    return {
        name: {
            "subset": p.subset,
            "frames": p.frame_sampling.n_frames,
            "samples_per_prompt": p.samples_per_prompt,
            "est_wallclock_min": p.estimated.wallclock_min,
        }
        for name, p in SUPPORTED_PROFILES.items()
    }


def check_capability_coverage() -> dict[str, Any]:
    try:
        from videvalkit.configs.capability_taxonomy import (
            ALL_TAGS, SUB_TAGS_BY_TOP, TOP_LEVEL_TAGS,
        )
        from videvalkit.core.capability import coverage_report
    except Exception as e:
        return {"error": str(e)}
    cov = coverage_report()
    covered = sum(1 for t in ALL_TAGS if cov.get(t))
    uncovered_top = [
        t for t in TOP_LEVEL_TAGS
        if not cov.get(t) and not any(cov.get(s) for s in SUB_TAGS_BY_TOP.get(t, []))
    ]
    return {
        "total_tags": len(ALL_TAGS),
        "covered_tags": covered,
        "uncovered_top_level": uncovered_top,
    }


def run_all(workspace: Path | None = None) -> dict[str, Any]:
    from videvalkit.configs import SUPPORTED_BENCHMARKS, SUPPORTED_JUDGES

    rep: dict[str, Any] = {
        "devices":   check_devices(),
        "envs":      {b: check_conda_env(cfg["env"]) for b, cfg in SUPPORTED_BENCHMARKS.items()},
        "adapters":  check_adapter_imports(),
        "judges":    {name: check_judge(name, cfg) for name, cfg in SUPPORTED_JUDGES.items()},
        "plugins":   check_plugins(),
        "metrics":   check_metrics(),
        "profiles":  check_profiles(),
        "capability_coverage": check_capability_coverage(),
        "benchmarks": {
            "total": len(SUPPORTED_BENCHMARKS),
            "judge_free": sorted(b for b, c in SUPPORTED_BENCHMARKS.items()
                                 if not c.get("needs_judge", False)),
            "needs_judge": sorted(b for b, c in SUPPORTED_BENCHMARKS.items()
                                  if c.get("needs_judge", False)),
        },
    }
    if workspace is not None:
        rep["workspace"] = check_workspace(workspace)
    return rep
