"""Tests for the shared device resolver (incl. NPU fallback)."""

from __future__ import annotations

# Canonical module + the historical re-export path must both work.
from videvalkit.core.device import ensure_npu_runtime
from videvalkit.metrics.utils.device import resolve_device


def test_cpu_passthrough():
    assert resolve_device("cpu") == "cpu"


def test_auto_returns_a_known_backend():
    assert resolve_device("auto") in ("cuda", "npu", "mps", "cpu")


def test_cuda_falls_back_when_absent():
    # On a CPU-only box, an explicit cuda request degrades to cpu, never raises.
    out = resolve_device("cuda")
    assert out in ("cuda", "cpu")
    assert out.split(":")[0] in ("cuda", "cpu")


def test_npu_falls_back_without_torch_npu():
    # torch_npu is not installed in the test/CI env → must fall back to cpu,
    # not crash on the missing module.
    assert resolve_device("npu") == "cpu"


def test_unknown_passthrough():
    assert resolve_device("xpu") == "xpu"


def test_ensure_npu_runtime_no_crash_without_torch_npu():
    # torch_npu is not installed on the CUDA/CPU dev+CI box → must return False,
    # never raise (and never activate the .cuda()->npu shim here).
    assert ensure_npu_runtime() is False
