# NPU Adaptation — Design (Mostly deferred)

> **Status: mostly deferred.** Full benchmark NPU support is not on the
> short-term roadmap. **Partial plumbing has landed**: the device-agnostic
> canonical metrics (FVD / VFID / KVD / CLIP-FVD / CLIP-Score / ViCLIP-Score)
> resolve `--device npu` through `metrics/utils/device.py` (`resolve_device`),
> importing `torch_npu` to register the backend and falling back to cpu with a
> warning when it is absent. This is **untested on real Ascend hardware** (the
> dev box is CUDA-only) — verify on-device before reporting NPU numbers. The rest
> of this stub records the intent and constraints for the remaining work.

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

## Environment (910B)

A separate env keeps the CUDA env clean. Draft template + installer:

- `envs/videvalkit-npu.yaml` — conda env (python 3.10 + ffmpeg + device-agnostic
  pip deps); versions are placeholders to pin to the host's CANN release.
- `scripts/post_install_npu.sh` — installs torch + `torch_npu` (CANN-matched) +
  the torch-dependent deps (openai-clip, pyiqa, decord/eva-decord); `INSTALL_VBENCH=1`
  adds the lifted vbench dims.
- `scripts/npu_smoke.py` — on-device PASS/FAIL report for the easy-tier metrics
  (run after install).

Easy-tier on NPU = the 6 canonical metrics + 5 vbench lifts (temporal-flickering,
subject/background-consistency, aesthetic/imaging-quality). The `.cuda()`→npu
redirect is handled by `core.device.ensure_npu_runtime()`
(`torch_npu.contrib.transfer_to_npu`), activated when a device resolves to npu.
