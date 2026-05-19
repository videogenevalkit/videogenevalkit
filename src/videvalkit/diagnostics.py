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


def run_all(workspace: Path | None = None) -> dict[str, Any]:
    from videvalkit.configs import SUPPORTED_BENCHMARKS, SUPPORTED_JUDGES

    rep: dict[str, Any] = {
        "envs":     {b: check_conda_env(cfg["env"]) for b, cfg in SUPPORTED_BENCHMARKS.items()},
        "adapters": check_adapter_imports(),
        "judges":   {name: check_judge(name, cfg) for name, cfg in SUPPORTED_JUDGES.items()},
    }
    if workspace is not None:
        rep["workspace"] = check_workspace(workspace)
    return rep
