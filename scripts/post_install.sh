#!/usr/bin/env bash
# Post-conda-env install: 7 packages that need torch already present
# (build-from-source or git-only — can't go in envs/videvalkit.yaml).
#
# Prerequisite: the conda env must be active.
#   conda activate /tmp/videvalkit-env
#
# Usage:
#   bash scripts/post_install.sh             # install everything (recommended)
#   bash scripts/post_install.sh --minimal   # skip the 3 heavy CUDA builds
#                                            #  (detectron2, lietorch, droid_backends)
#                                            #  saves ~20 min but loses VBench-2.0
#                                            #  Human_Anatomy + WorldScore
#                                            #  camera_control/3d_consistency.

set -eu

MINIMAL=0
for arg in "$@"; do
  case "$arg" in
    --minimal) MINIMAL=1 ;;
    -h|--help)
      head -16 "$0" | sed 's/^# //; s/^#//'
      exit 0
      ;;
  esac
done

echo "=== verifying torch is present ==="
python -c "import torch; print(f'  torch={torch.__version__}  cuda={torch.cuda.is_available()}')" \
  || { echo "ERROR: torch not importable. Activate the conda env first."; exit 1; }
echo

# ──────────────────────────────────────────────────────────────────────────────
# Group 1 — light wheels and pure-Python from git (always installed)
# ──────────────────────────────────────────────────────────────────────────────

echo "=== 1/4: en_core_web_sm (spaCy English model, ~12 MB; WorldScore motion_accuracy) ==="
pip install \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

echo
echo "=== 2/4: segment-anything (Meta SAM v1) + GroundingDINO + SAM-2 ==="
pip install --no-build-isolation \
  segment-anything @ git+https://github.com/facebookresearch/segment-anything.git
pip install --no-build-isolation \
  groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git
pip install --no-build-isolation \
  SAM-2 @ git+https://github.com/facebookresearch/sam2.git

if [ "$MINIMAL" -eq 1 ]; then
  echo
  echo "=== --minimal: SKIPPING detectron2 + DROID-SLAM (saves ~20 min) ==="
  echo
  echo "  To install later:"
  echo "    pip install --no-build-isolation detectron2 @ git+https://github.com/facebookresearch/detectron2.git@b599f139756b"
  echo "    bash scripts/install_droid_slam.sh   # builds lietorch + droid_backends"
  exit 0
fi

# ──────────────────────────────────────────────────────────────────────────────
# Group 2 — heavy CUDA builds
# ──────────────────────────────────────────────────────────────────────────────

echo
echo "=== 3/4: detectron2 (VBench-2.0 Human_Anatomy ViTDetector) ==="
echo "  building from source — takes ~5 min"
pip install --no-build-isolation \
  detectron2 @ git+https://github.com/facebookresearch/detectron2.git@b599f139756b

echo
echo "=== 4/4: DROID-SLAM (builds lietorch + droid_backends; WorldScore camera_control / 3d_consistency) ==="
echo "  cloning + building from source — takes ~10 min"
CACHE_HOME="${VIDEVALKIT_CACHE_HOME:-$HOME/.cache/videvalkit}"
DROID_DIR="${CACHE_HOME}/upstream/DROID-SLAM"
if [ ! -d "$DROID_DIR" ]; then
  mkdir -p "$(dirname "$DROID_DIR")"
  git clone --depth 1 --recursive https://github.com/princeton-vl/DROID-SLAM.git "$DROID_DIR"
fi
# lietorch is a submodule under thirdparty/lietorch — install it first
pip install --no-build-isolation -e "$DROID_DIR/thirdparty/lietorch"
# then DROID-SLAM itself, which builds the droid_backends C++ extension
pip install --no-build-isolation -e "$DROID_DIR"

echo
echo "=== post-install complete ==="
echo "Verify with:  videvalkit doctor"
