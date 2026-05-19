#!/usr/bin/env bash
# WorldJen smoke run on 50 paper-headline Kling videos.
# Requirements: smoke-data fetched, Gemma-4-31B-IT running on http://localhost:8003/v1
set -e -u

CACHE="${VIDEVALKIT_CACHE_HOME:-$HOME/.cache/videvalkit}"
RUN_DIR="${1:-runs/worldjen_smoke}"

videvalkit eval \
  --bench worldjen \
  --videos "${CACHE}/smoke-data/worldjen/videos/fal-ai_kling-video_v2.6_pro_text-to-video" \
  --workspace "${RUN_DIR}" \
  --model Kling \
  --judge gemma-4-31b-local \
  --aggregator phas

echo
echo "=== summary ==="
cat "${RUN_DIR}/results/summary/worldjen/Kling.json"
echo
echo "Done. Cross-benchmark aggregate:"
echo "  videvalkit aggregate --workspace ${RUN_DIR}"
