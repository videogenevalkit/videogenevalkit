# NPU Adaptation — Design (Deferred)

> **Status: deferred.** Not on the short-term roadmap. This stub records the intent
> and the constraints so the work can be picked up later without re-discovery.

---

## Intent

Run the device-agnostic parts of the toolkit (most metrics, judge-free benches,
VLM-judge calls over HTTP) on Ascend NPU, with explicit, honest degradation where a
benchmark depends on CUDA-only kernels.

## Why deferred

The current focus is Linux + CUDA. NPU adds an environment matrix and per-op
porting that is not justified until there is concrete demand; nothing in the v0.2
design blocks it later.

## Constraints to respect when resumed

- **Device selection** should auto-prefer CUDA > NPU > MPS > CPU, with explicit
  override (`--device npu`) and fail-fast when a bench is only partially supported
  on NPU (offer: switch device, restrict dimensions, or accept a partial run).
- **CUDA-only kernels** (e.g. detectron2, lietorch, droid_backends) have no NPU
  build — the affected benchmark dimensions are unsupported, not silently wrong.
  CPU/NPU builds exist for SAM-2, GroundingDINO, segment-anything.
- **A separate env** (`videvalkit-npu`) keeps the CUDA env clean; the plugin loader
  and registries are device-agnostic and need no change.
- **VLM judges** are unaffected — they are HTTP calls to an endpoint, independent of
  the local device.
- **Profiles/subsets** carry over unchanged; only the compute backend differs.

When demand materializes, scope this as its own milestone with a device-coverage
matrix per benchmark dimension.
