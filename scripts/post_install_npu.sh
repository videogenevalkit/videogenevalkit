#!/usr/bin/env bash
# Post-conda-env install for the Ascend NPU (910B) variant.  DRAFT / TEMPLATE.
#
# Installs torch + torch_npu (CANN-matched) and the torch-dependent app deps for
# the device-agnostic metric subset. Run AFTER creating + activating the env:
#   conda activate /opt/videvalkit-npu
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   bash scripts/post_install_npu.sh
#
# ⚠️ UNVERIFIED on real 910B hardware. Pin TORCH_VER / TORCH_NPU_VER to match
#    your CANN release (see the Ascend torch_npu compatibility matrix). Override
#    without editing the file:
#      TORCH_NPU_VER=2.3.1.post4 bash scripts/post_install_npu.sh

set -eu

TORCH_VER="${TORCH_VER:-2.3.1}"            # plain CPU wheel; torch_npu adds the NPU backend
TORCHVISION_VER="${TORCHVISION_VER:-0.18.1}"
TORCH_NPU_VER="${TORCH_NPU_VER:-2.3.1}"    # PLACEHOLDER — pin to your CANN version
INSTALL_VBENCH="${INSTALL_VBENCH:-0}"      # 1 = also install vbench for the lifted dims

echo "=== 0/5: CANN prerequisite check ==="
if ! command -v npu-smi >/dev/null 2>&1; then
  echo "  WARN: npu-smi not found — is this an Ascend host with CANN sourced?"
  echo "        run: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
else
  npu-smi info | head -3 || true
fi
echo

echo "=== 1/5: torch ${TORCH_VER} (CPU wheel) + torchvision ${TORCHVISION_VER} ==="
# The plain CPU build of torch; torch_npu provides the 'npu' device backend.
pip install "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" \
  --index-url https://download.pytorch.org/whl/cpu
echo

echo "=== 2/5: torch_npu ${TORCH_NPU_VER} (Ascend backend) ==="
# If this exact version is unavailable on PyPI for your platform, fetch the wheel
# from https://gitee.com/ascend/pytorch/releases matching CANN + torch ${TORCH_VER}.
pip install "torch_npu==${TORCH_NPU_VER}" || {
  echo "  torch_npu pip install failed — install the matching wheel manually:"
  echo "    https://gitee.com/ascend/pytorch/releases  (CANN + torch ${TORCH_VER})"
  exit 1
}
echo

echo "=== 3/5: verify torch_npu + NPU visibility ==="
python - <<'PY'
import torch, torch_npu  # noqa: F401
print(f"  torch={torch.__version__}  torch_npu ok  npu_available={torch.npu.is_available()}")
assert torch.npu.is_available(), "NPU not visible — check driver/CANN/device assignment"
PY
echo

echo "=== 4/5: torch-dependent app deps (easy-tier metrics) ==="
# CLIP (clip-score / clip-fvd / aesthetic-quality / background-consistency)
pip install git+https://github.com/openai/CLIP.git
# MUSIQ IQA (imaging-quality)
pip install pyiqa
# decord (video IO). pip wheel is x86_64-only; on aarch64 use eva-decord.
if [ "$(uname -m)" = "aarch64" ]; then
  echo "  aarch64 host → installing eva-decord (decord has no aarch64 wheel)"
  pip install eva-decord || echo "  WARN: eva-decord failed; build decord from source"
else
  pip install decord
fi
echo

echo "=== 5/5: vbench (optional — the 5 easy lifted dims) ==="
if [ "$INSTALL_VBENCH" = "1" ]; then
  # vbench's full dep set pulls detectron2 (CUDA-only) for detection dims we do
  # NOT run on NPU — install without deps and rely on the easy dims' subset.
  pip install --no-deps vbench || echo "  WARN: vbench install failed (lifts unavailable)"
  echo "  NOTE: detection dims (object_class, etc.) stay CUDA-only by design."
else
  echo "  skipped (set INSTALL_VBENCH=1 to enable the lifted vbench dims)"
fi
echo

echo "=== done ==="
echo "Next:  pip install --no-deps -e .  &&  python scripts/npu_smoke.py"
