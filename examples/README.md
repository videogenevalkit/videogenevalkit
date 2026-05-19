# Examples

End-to-end shell scripts you can copy and run after `pip install -e .` from the repo root. Each example assumes:

1. Conda env active: `conda activate videvalkit`
2. Smoke data fetched: `videvalkit fetch-smoke-data`
3. Checkpoints fetched for the relevant bench: `videvalkit fetch-checkpoints --bench <name>`
4. A judge endpoint reachable (default `gemma-4-31b-local` on `http://localhost:8003/v1`)

## Quick demos (one-GPU, ~30 min)

| Script | Bench | Model | What it does |
|---|---|---|---|
| `run_worldjen_smoke.sh` | WorldJen | Kling-v2.6 | 50 prompts × 16 dims via Gemma local; PHAS aggregator → cross-bench score |
| `run_videobench_smoke.sh` | Video-Bench | CogVideoX-5B | 9-dim VLM-judge sweep on the bundled action subset |
| `run_t2vcompbench_smoke_cv.sh` | T2V-CompBench | CogVideoX-5B | CV-only dims (numeracy, spatial, motion_binding); skips the 68 GB LLaVA-1.6-34B path |

## Full reproductions (multi-hour, multi-GPU)

| Script | Bench | Model | What it does |
|---|---|---|---|
| `run_vbench_hunyuan.sh` | VBench v1 | HunyuanVideo | Full 16-dim sweep on 360 paper-released videos → `paper Δ` table |
| `run_vbench2_hunyuan.sh` | VBench-2.0 | HunyuanVideo | Full 18-dim sweep on ~1300 paper-released videos; dim-parallel across 3 GPUs |
| `run_t2vcompbench_cogvideox5b.sh` | T2V-CompBench | CogVideoX-5B | All 7 dims including LLaVA-1.6-34B paper-mode (needs ≥ 80 GB VRAM) |

See `docs/USER_MANUAL_en.md` for per-bench prerequisites and `docs/TEST_MANUAL.md` for the validation results these reproduce.
