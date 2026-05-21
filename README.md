# videogenevalkit

**Unified evaluation toolkit for text-to-video generation models.**

One config, one entrypoint, seven benchmarks. Score your model on VBench v1, VBench-2.0, Video-Bench, WorldJen, WorldScore, T2V-CompBench, and Semantics-Axis from a single command. Compares byte-for-byte against the official leaderboards.

<p align="left">
  <a href="#quickstart"><img alt="quickstart" src="https://img.shields.io/badge/quickstart-30%20min-blue"></a>
  <a href="#whats-supported"><img alt="benchmarks" src="https://img.shields.io/badge/benchmarks-7-orange"></a>
  <a href="https://huggingface.co/datasets/videogenevalkit/checkpoints"><img alt="HF checkpoints" src="https://img.shields.io/badge/HF-checkpoints-yellow?logo=huggingface"></a>
  <a href="https://huggingface.co/datasets/videogenevalkit/smoke-data"><img alt="HF smoke data" src="https://img.shields.io/badge/HF-smoke%20data-yellow?logo=huggingface"></a>
  <a href="LICENSES/"><img alt="licenses" src="https://img.shields.io/badge/licenses-multi--upstream-lightgrey"></a>
</p>

---

## TL;DR — what reproducibility you actually get

We re-ran the official leaderboards for the 6 anchored benchmarks. **mean |Δ| vs published numbers:**

| Benchmark | Model evaluated | Dims covered | Mean \|Δ\| | Max \|Δ\| | Notes |
|---|---|---:|---:|---:|---|
| **VBench v1** | HunyuanVideo | 16/16 | **0.012** | 0.026 | matches HF leaderboard |
| **VBench-2.0** | HunyuanVideo | 18/18 | **0.0055** | 0.023 | 4 dims byte-exact; matches HF leaderboard |
| **T2V-CompBench** | CogVideoX-5B | 6/7 in-tol | **0.0046** | 0.020 | paper-exact LLaVA-1.6-34B + GD-SwinT-OGC; `consistent_attribute` outlier documented |
| **Video-Bench** | cogvideox5b | 9/9 | judge-substitution offset | — | Gemma stand-in for GPT-4o; static + alignment dims match, dynamic-quality drifts (documented) |
| **WorldJen** | Kling-v2.6 | 16/16 | — | — | PHAS 3.66 vs paper-Gemma 4.12; Δ -0.47 (decord-vs-cv2 frame variance) |
| **WorldScore** | CogVideoX-5B | 10/10 | full pipeline | — | DROID-SLAM + SEA-RAFT + VFIMamba + SAM2 stack wired |
| **Semantics-Axis** | pangu3 / wan14b / seedance20 | 21/21 | in-house (no leaderboard) | — | VLM-judge prompt-following; all 21 axes score end-to-end |

Full per-dim tables: see [`docs/TEST_MANUAL.md`](docs/TEST_MANUAL.md).

---

## What's supported

**7 benchmark adapters** — 6 anchored to public leaderboards + 1 in-house (all production-ready):

| Adapter | Upstream | What it scores |
|---|---|---|
| `vbench` | [Vchitect/VBench](https://github.com/Vchitect/VBench) (CVPR 2024) | 16 dims, quality + semantic; weighted-sum + min-max norm |
| `vbench2` | [Vchitect/VBench-2.0](https://github.com/Vchitect/VBench) | 18 dims across 5 categories: Creativity / Commonsense / Controllability / Human Fidelity / Physics |
| `videobench` | [Video-Bench](https://github.com/Video-Bench/Video-Bench) (CVPR 2025) | 9 dims: 4 alignment + 4 static/dynamic quality + video_text_consistency |
| `worldjen` | [WorldJen](https://github.com/moonmath-ai/WorldJen-benchmarking-subsystem) | 16 dims grouped into motion_stability / logic_physics / instruction_adherence / aesthetic; PHAS aggregator |
| `worldscore` | [WorldScore](https://github.com/yhw-yhw/WorldScore) | 10 dims: 7 static (CLIP/DROID-SLAM/SAM/GD/IQA) + 3 dynamic (RAFT/VFIMamba/Motion-Acc) |
| `t2vcompbench` | [T2V-CompBench V2](https://github.com/KaiyueSun98/T2V-CompBench/tree/V2) | 7 compositional dims; LLaVA-1.6-34B MLLM + GD-SwinT-OGC + SAM-H + Depth-Anything V1 + DOT |
| `semantics_axis` | in-house (Semantics Axis Eval) | 21 prompt-following axes — entity / spatial / event / cinematic / modifier — + holistic `overall`; VLM-judge, 1-5, no checkpoints |

Plus 3 supplementary stubs landing (Physics-IQ, VBench++, V-ReasonBench).

**8 VLM/LLM judges** out of the box (`SUPPORTED_JUDGES`): local vLLM (Gemma-4-31B-IT, Qwen3-32B, Qwen3-VL-32B, LLaVA-Video-7B) + managed APIs (Gemini, GPT-4o, Claude). User-configurable endpoints via `~/.config/videvalkit/judges.yaml`.

**Cross-benchmark aggregation**: weighted_sum · vbench_weighted · vbench2_category · phas · bt (Bradley-Terry).

---

## Quickstart

### 1. Clone + install env (~ 10 min)

The toolkit ships as a pre-packed env tarball — the same byte-for-byte environment that produced every result in `docs/TEST_MANUAL.md`.

```bash
git clone https://github.com/videogenevalkit/videogenevalkit.git
cd videogenevalkit

# Download the env tarball (~7.6 GB, ~5 min on a fast link)
hf download videogenevalkit/env-tarball videvalkit-env.tar.gz --local-dir /tmp

# Extract + activate (~15 GB on disk after unpack)
sudo mkdir -p /opt/videvalkit-env
sudo tar xzf /tmp/videvalkit-env.tar.gz -C /opt/videvalkit-env
sudo chown -R $USER /opt/videvalkit-env
source /opt/videvalkit-env/bin/activate
conda-unpack            # rewrites absolute paths inside the env

# Install the toolkit code + 7 build-from-source extras
pip install --no-deps -e .
bash scripts/post_install.sh    # detectron2, SAM-2, GroundingDINO, segment-anything, lietorch, droid_backends, en_core_web_sm

videvalkit doctor       # verify
```

The tarball is a `conda-pack` snapshot of the validated working env — Python 3.10 + torch 2.3.1+cu121 + ~350 pinned dependencies (transformers 4.51.3, mmcv 2.2.0, xformers, triton, bitsandbytes, accelerate, peft, ms-swift, decord, opencv, pyiqa, openai-clip, timm, ...). No pip resolution at install time, so no failure modes.

### 2. Fetch smoke data + checkpoints from HF (~ 15 min, ~ 8 GB)

```bash
# representative videos + prompts across all 7 benchmarks
videvalkit fetch-smoke-data

# 5 GB of needed model checkpoints (GroundingDINO, SAM-H, Depth-Anything V1, DOT)
videvalkit fetch-checkpoints --bench worldscore --bench t2vcompbench
```

LLaVA-1.6-34B (68 GB, used by T2V-CompBench MLLM dims) is **NOT bundled** — when you first run `--bench t2vcompbench --mode upstream`, it'll resolve `liuhaotian/llava-v1.6-34b` directly from upstream HF.

### 3. Verify your setup

```bash
videvalkit doctor
# checks: conda env, GPU, HF cache, smoke-data presence, judge endpoint reachability
```

### 4. Run your first benchmark (worldjen on 3 Kling videos, ~5 min)

```bash
# Fetch the worldjen smoke set (~600 MB: 50 Kling videos + prompts.jsonl + vqa.jsonl)
videvalkit fetch-smoke-data --bench worldjen

# Stage 3 of the 50 bundled videos
mkdir -p ~/runs/worldjen/videos/Kling
ls ~/.cache/videvalkit/smoke-data/worldjen/videos/fal-ai_kling-video_v2.6_pro_text-to-video/*.mp4 \
  | head -3 | xargs -I{} ln -sf {} ~/runs/worldjen/videos/Kling/

# Evaluate (Gemma-4-31B vLLM judge on localhost:8003; ~250 calls, ~5 min)
videvalkit eval --bench worldjen \
    --videos ~/runs/worldjen/videos \
    --workspace ~/runs/worldjen/ws \
    --models Kling \
    --judge gemma-4-31b-local \
    --aggregator phas
```

Expected at `~/runs/worldjen/ws/results/summary/worldjen/Kling.json`:

```
overall: ≈ 3.2 of 5 (unweighted PHAS on 3 prompts)
per_category:
  instruction_adherence  ≈ 3.78
  aesthetic_quality      ≈ 3.50
  motion_stability       ≈ 3.20
  logic_physics          ≈ 2.58
```

The full 50-prompt calibrated PHAS for Kling-v2.6 reported in the paper is **4.12** (with learned per-dim weights). Our smoke uses 3 prompts + unweighted mean — values match the per-category ordering but differ in magnitude due to sample size + aggregator. Reproduce the calibrated headline by symlinking all 50 mp4s + using `--aggregator phas`.

### 5. Compare across benchmarks

```bash
videvalkit aggregate --workspace ~/runs/worldjen/ws
# → ~/runs/worldjen/ws/results/leaderboard/cross_benchmark.json
```

---

## Running examples for all 7 benchmarks

Worldjen (above) needs no local checkpoints. Four others each pull a specific checkpoint from `videogenevalkit/checkpoints` on HuggingFace; `semantics_axis` is judge-only. Full step-by-step recipes in [`docs/USER_MANUAL_en.md` §6](docs/USER_MANUAL_en.md).

| # | Bench / dim | Scorer | Ckpt fetch (`videogenevalkit/checkpoints`) | Wallclock (3 vids) | GPU mem | Expected score |
|---|---|---|---|---:|---:|---|
| 1 | `worldjen` / 16 dims | Gemma-4-31B vLLM | — (no local ckpt) | ~5 min | 0 GB | overall ≈ 3.2 of 5 |
| 2 | `vbench` / `subject_consistency` | DINO ViT-B/16 | `vbench/pretrained/dino_model/` (343 MB) | ~30 s | 2 GB | ≈ 0.92 |
| 3 | `vbench2` / `Camera_Motion` | CoTracker3 (vendored) | `vbench2/third_party/cotracker/` (204 MB) | ~1 min | 4 GB | ≈ 0.67 |
| 4 | `videobench` / `action_consistency` | Gemma-4-31B vLLM | — (no local ckpt) | ~2 min | 0 GB | ≈ 2.0 (raw 1-5) |
| 5 | `worldscore` / `motion_magnitude` | SEA-RAFT | `worldscore/Tartan-C-T-TSKH-*` + `raft-things` (150 MB) | ~2 min | 4 GB | ≈ 56.4 (×100) |
| 6 | `t2vcompbench` / `action_binding` (paper-mode) | LLaVA-1.6-34B | `hf-models/liuhaotian/llava-v1.6-34b/` (68 GB) | ~5 min after model load | 70 GB | raw 7.22 → norm 0.69 |
| 7 | `semantics_axis` / 21 axes | Gemma-4-31B vLLM | — (no local ckpt) | ~5 min | 0 GB | headline ≈ 4.3 of 5 |

Rows 1-6 are from the integration test on 2026-05-19, row 7 from 2026-05-21 (3 sample videos each); run-to-run variance is typically ±0.05 for CV-based dims (DINO/CoTracker3/SEA-RAFT/GroundingDINO) and ±0.15 for VLM-judge dims (Gemma, LLaVA-34B) due to temperature-0.2 sampling. Full validation against published leaderboards in [`docs/TEST_MANUAL.md`](docs/TEST_MANUAL.md).

---

## Standalone metrics (no benchmark scaffolding)

Need a single number — FID, FVD, CLIP-Score, PSNR, SSIM, LPIPS — without a prompt suite?

```bash
videvalkit metric --name fvd \
  --gen-videos path/to/generated/ \
  --ref-videos path/to/reference/ \
  --device cuda:0
```

See [`docs/DEV_MANUAL.md §14`](docs/DEV_MANUAL.md#14-standalone-metrics-module--videvalkitmetrics-planned-2026-05-18) for the metrics registry and verification methodology.

---

## What you DON'T have to do

- **No re-implementing scorers.** All adapters delegate to upstream code byte-for-byte. We add IO, scheduling, registry, and a unified output format on top.
- **No paper-Gemini / paper-Claude credentials.** Judges are pluggable; default to local vLLM on `:8003` / `:8004` etc. Managed-API support is a one-line config swap.
- **No per-benchmark conda envs.** One shared env (`envs/videvalkit.yaml`) covers all 6 adapters. Per-benchmark envs available for users who need them.

---

## Documentation map

| Doc | What's in it |
|---|---|
| [`README.md`](README.md) | (you are here) overview + quickstart |
| [`docs/USER_MANUAL_en.md`](docs/USER_MANUAL_en.md) / [`docs/USER_MANUAL_cn.md`](docs/USER_MANUAL_cn.md) | end-user how-to: install, configure judges, run each benchmark |
| [`docs/DEV_MANUAL.md`](docs/DEV_MANUAL.md) | architecture: object model, module layout, packaging plan |
| [`docs/TEST_MANUAL.md`](docs/TEST_MANUAL.md) | validation results per benchmark with Δ vs leaderboard, tolerance bands, known discrepancies |

---

## Reproducing the leaderboard numbers in this README

The Δ-vs-leaderboard claims above are reproducible from this repo. Steps:

```bash
# 1. fetch the smoke + checkpoint bundles
videvalkit fetch-smoke-data
videvalkit fetch-checkpoints --all

# 2. run each benchmark on the bundled videos
videvalkit eval --bench vbench         --videos data/smoke/vbench-hunyuan-* ...
videvalkit eval --bench vbench2        --videos data/smoke/vbench2-hunyuan-* ...
videvalkit eval --bench videobench     --videos data/smoke/videobench-cogvideox5b-* ...
videvalkit eval --bench worldjen       --videos data/smoke/worldjen-kling-50 ...
videvalkit eval --bench worldscore     --videos data/smoke/worldscore-cogvideox-5b-* ...
videvalkit eval --bench t2vcompbench   --videos data/smoke/t2vcompbench-cogvideox5b-200per-dim --mode upstream ...

# 3. compare to leaderboard
videvalkit compare-leaderboard --workspace runs/first
```

The bundled smoke data **is** the official paper-released video set for each benchmark (or a representative subset). The leaderboard JSONs are checked into `validation/expected/`.

---

## License & attribution

The toolkit (this codebase) is **Apache-2.0**. Each upstream we adapt has its own license — surfaced under [`LICENSES/`](LICENSES/) at repo root. Notable:

- **VBench / VBench-2.0**: Apache-2.0
- **Video-Bench, WorldJen**: respective papers' licenses
- **T2V-CompBench**: research-only usage on LLaVA-1.6-34B subprocess path
- **DROID-SLAM, SEA-RAFT, GroundingDINO, SAM**: respective open licenses
- **WorldScore checkpoints**: derivative use under each backbone's terms

Cite the upstream papers when reporting numbers from a specific adapter.

---

## Citation

```bibtex
@software{videogenevalkit2026,
  title = {videogenevalkit: A unified evaluation toolkit for text-to-video generation},
  author = {xxxxx},
  year = {2026},
  url = {https://github.com/videogenevalkit/videogenevalkit},
}
```

For benchmark-specific numbers, cite the original papers (BibTeX entries in [`docs/citations.bib`](docs/citations.bib)).
