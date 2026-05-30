#!/usr/bin/env bash
# Set up a shared videvalkit checkpoint directory and pre-download all the
# backbone weights so other users / containers can reuse them without
# re-downloading. Idempotent.
#
# Usage:
#   bash scripts/setup_ckpt_dir.sh [CKPT_ROOT]
#
# Default CKPT_ROOT: /pub/evaluation_group/yue/videvalkit_ckpts
#
# After running, point your shell at it:
#   export VIDEVALKIT_CKPT_HOME=/pub/evaluation_group/yue/videvalkit_ckpts
#   export HF_HOME=$VIDEVALKIT_CKPT_HOME/huggingface
set -eu

CKPT_ROOT="${1:-/pub/evaluation_group/yue/videvalkit_ckpts}"
mkdir -p "$CKPT_ROOT"/{huggingface,viclip,i3d,torchvision}
echo "ckpt root: $CKPT_ROOT"

# 1. VideoMAE-base (FVD/KVD optional backbone, 340MB)
echo
echo "=== 1/4: VideoMAE-base [MCG-NJU/videomae-base-finetuned-kinetics, ~340MB] ==="
HF_HOME="$CKPT_ROOT/huggingface" python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="MCG-NJU/videomae-base-finetuned-kinetics",
                       allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt"])
print(f"  cached at: {p}")
PY

# 2. ViCLIP-L (viclip-score, 1.7GB) — reuse existing vbench cache via symlink
echo
echo "=== 2/4: ViCLIP-L [reuse existing vbench cache via symlink] ==="
SRC=/root/.cache/vbench/ViCLIP/ViClip-InternVid-10M-FLT.pth
TOK=/root/.cache/vbench/ViCLIP/bpe_simple_vocab_16e6.txt.gz
if [ -f "$SRC" ]; then
  ln -sf "$SRC" "$CKPT_ROOT/viclip/ViClip-InternVid-10M-FLT.pth"
  ln -sf "$TOK" "$CKPT_ROOT/viclip/bpe_simple_vocab_16e6.txt.gz"
  echo "  symlinked $SRC -> $CKPT_ROOT/viclip/"
else
  echo "  ViCLIP weight not found at $SRC; fetch with huggingface_hub if needed"
fi

# 3. S3D-K400 (torchvision auto, ~22MB) — prime torchvision cache so first FVD is fast
echo
echo "=== 3/4: S3D-K400 [torchvision auto-download, ~22MB] ==="
TORCH_HOME="$CKPT_ROOT/torchvision" python - <<'PY'
import os, torch
os.environ.setdefault("TORCH_HOME", os.environ["TORCH_HOME"])
from torchvision.models.video import s3d, S3D_Weights
m = s3d(weights=S3D_Weights.KINETICS400_V1)
print("  S3D-K400 ready (torchvision)")
PY

# 4. InceptionV3 (VFID, ~95MB) — prime torchvision cache
echo
echo "=== 4/4: InceptionV3 [VFID, torchvision auto, ~95MB] ==="
TORCH_HOME="$CKPT_ROOT/torchvision" python - <<'PY'
import os
from torchvision.models import inception_v3, Inception_V3_Weights
m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
print("  InceptionV3 ready")
PY

# I3D-K400 paper weights are NOT auto-fetchable (hosted as i3d_torchscript.pt
# on the StyleGAN-V tree). Note where to drop it if/when available:
echo
echo "=== I3D-K400 [paper-canonical, MANUAL] ==="
echo "  Drop i3d_torchscript.pt at: $CKPT_ROOT/i3d/i3d_torchscript.pt"
echo "  (otherwise FVD auto-falls-back to s3d-k400)"

cat > "$CKPT_ROOT/README.md" <<EOF
# videvalkit shared checkpoint dir

Pre-downloaded backbone weights so toolkit users don't each pay the HF
download cost.

Layout:
- \`huggingface/\` — HF transformers cache (HF_HOME points here)
  - MCG-NJU/videomae-base-finetuned-kinetics (FVD/KVD videomae backbone)
- \`viclip/\` — ViCLIP-L weights (symlink to vbench cache)
- \`i3d/\` — manual: drop i3d_torchscript.pt here for paper-canonical FVD
- \`torchvision/\` — torchvision cache (TORCH_HOME points here)
  - S3D-K400 (FVD default), InceptionV3 (VFID)

Activate from any conda env:

    export VIDEVALKIT_CKPT_HOME=$CKPT_ROOT
    export HF_HOME=\$VIDEVALKIT_CKPT_HOME/huggingface
    export TORCH_HOME=\$VIDEVALKIT_CKPT_HOME/torchvision

Refresh / verify with: \`bash <repo>/scripts/setup_ckpt_dir.sh\`
EOF

echo
echo "=== done ==="
du -sh "$CKPT_ROOT"/* 2>/dev/null | head -10
echo
echo "Activate in your shell:"
echo "  export VIDEVALKIT_CKPT_HOME=$CKPT_ROOT"
echo "  export HF_HOME=\$VIDEVALKIT_CKPT_HOME/huggingface"
echo "  export TORCH_HOME=\$VIDEVALKIT_CKPT_HOME/torchvision"
