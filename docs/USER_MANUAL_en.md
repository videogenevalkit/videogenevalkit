# videvalkit User Manual

| Field | Value |
|---|---|
| Title | videvalkit User Manual (English) |
| Package | `videvalkit` (Python) |
| Repository | `videogenevalkit` (GitHub) |
| Audience | Researchers, integrators, and downstream teams running text-to-video evaluation |
| Companion docs | [DEV_MANUAL.md](DEV_MANUAL.md) (architecture) - [TEST_MANUAL.md](TEST_MANUAL.md) (paper-Delta validation) |

This manual is an end-user how-to. It explains what `videvalkit` does, how to install it, how to fetch the data and weights you need, how to run each of the six anchored benchmarks, and how to configure judges and GPUs. For the internal architecture see DEV_MANUAL; for paper-Delta reproducibility tables see TEST_MANUAL.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Setting up data and weights](#4-setting-up-data-and-weights)
5. [Your first evaluation](#5-your-first-evaluation)
6. [Per-benchmark how-to](#6-per-benchmark-how-to)
7. [Configuring scorers (VLM/LLM judges)](#7-configuring-scorers-vlmllm-judges)
8. [Configuring GPUs](#8-configuring-gpus)
9. [Cross-benchmark aggregation](#9-cross-benchmark-aggregation)
10. [Standalone metrics](#10-standalone-metrics)
11. [Generating your own videos](#11-generating-your-own-videos)
12. [Custom prompts (auto-labeler)](#12-custom-prompts-auto-labeler)
13. [Troubleshooting and FAQ](#13-troubleshooting-and-faq)
14. [License and citation](#14-license-and-citation)

---

## 1. Introduction

`videvalkit` is a unified evaluation toolkit for text-to-video (T2V) generation models. One configuration, one entrypoint, six benchmarks. Score your model on VBench v1, VBench-2.0, Video-Bench, WorldJen, WorldScore, and T2V-CompBench from a single CLI, and compare cross-benchmark from a single workspace.

The toolkit ships six **anchored** benchmark adapters that are production-ready:

| Adapter | Upstream | Scores |
|---|---|---|
| `vbench` | Vchitect/VBench (CVPR 2024) | 16 dims: 7 quality + 9 semantic |
| `vbench2` | Vchitect/VBench-2.0 (CVPR 2025) | 18 dims across 5 categories (Creativity, Commonsense, Controllability, Human Fidelity, Physics) |
| `videobench` | Han et al. Video-Bench (CVPR 2025) | 9 dims: 4 alignment + 4 static/dynamic quality + video-text consistency |
| `worldjen` | moonmath-ai/WorldJen | 16 dims grouped by motion_stability, logic_physics, instruction_adherence, aesthetic_quality; PHAS aggregator |
| `worldscore` | Duan et al. WorldScore | 10 dims: 7 static + 3 dynamic; CV-only |
| `t2vcompbench` | Sun et al. T2V-CompBench V2 (ECCV 2024) | 7 compositional dims; LLaVA-1.6-34B + GD/SAM/DOT |

Three supplementary adapter stubs are landing in upcoming releases: Physics-IQ, VBench++, and V-ReasonBench. They are listed in `videvalkit list benchmarks --include-stubs` but not yet production-ready.

The toolkit does **not** reimplement the upstream scoring formulas. Each adapter delegates to the upstream package byte-for-byte and contributes only IO, scheduling, registry, judge plumbing, and a unified output format. The validation sweep in TEST_MANUAL.md records mean |Delta| against published leaderboards: VBench v1 = 0.012, VBench-2.0 = 0.006, T2V-CompBench = 0.013 (6/7 dims within +/-0.020). Reproducibility tables and per-dim Deltas are documented per benchmark in TEST_MANUAL section 4.

---

## 2. System requirements

| Requirement | Value |
|---|---|
| OS | Linux x86_64 (CentOS/RHEL/Ubuntu validated) |
| CUDA driver | 12.1 or later |
| GPU | At least 1 NVIDIA GPU with at least 24 GB VRAM. At least 80 GB for T2V-CompBench paper-mode LLaVA-1.6-34B |
| RAM | At least 64 GB |
| Disk | At least 200 GB free for full smoke data plus all six benchmarks' checkpoints |
| conda | miniforge or miniconda, recent |
| Network | Outbound HTTPS to huggingface.co and github.com |

The 24 GB minimum covers WorldJen, Video-Bench, VBench v1, VBench-2.0, and WorldScore in toolkit mode. T2V-CompBench paper-mode (`--mode upstream` with the bundled LLaVA-1.6-34B MLLM) requires an 80 GB-class GPU (A100/H100). If you do not have that GPU, run T2V-CompBench in `--mode toolkit` with a different VLM judge, accepting the paper-Delta caveat in DEV_MANUAL section 15.3.2.

CPU-only runs are possible for the 3 paired metrics (PSNR, SSIM) but not for the benchmarks.

---

## 3. Installation

The toolkit ships as a pre-packed env tarball — the same byte-for-byte environment that produced every result in `TEST_MANUAL.md`. Download, extract, run.

```bash
# 1. Clone the repo (source code + scripts + docs)
git clone https://github.com/videogenevalkit/videogenevalkit.git
cd videogenevalkit

# 2. Download the env tarball (~7.6 GB)
hf download videogenevalkit/env-tarball videvalkit-env.tar.gz \
    --local-dir /tmp

# 3. Extract to a persistent location (~15 GB on disk after unpack)
sudo mkdir -p /opt/videvalkit-env
sudo tar xzf /tmp/videvalkit-env.tar.gz -C /opt/videvalkit-env
sudo chown -R $USER /opt/videvalkit-env

# 4. Activate the env and rewrite its internal absolute paths
source /opt/videvalkit-env/bin/activate
conda-unpack

# 5. Install the toolkit source against this clone
pip install --no-deps -e .

# 6. Post-install: 7 build-from-source / git-only deps not in the tarball
bash scripts/post_install.sh
# Or, if you don't need WorldScore camera_control or VBench-2.0 Human_Anatomy:
# bash scripts/post_install.sh --minimal   (~10 min faster)
```

What's in the tarball:
- Python 3.10 + torch 2.3.1+cu121
- ~350 pinned dependencies (transformers 4.51.3, mmcv 2.2.0, decord, opencv, pyiqa, openai-clip, timm, xformers, triton, bitsandbytes, accelerate, peft, ms-swift, qwen-vl-utils, and the validated combinations of every transitive dep)
- All checkpoints fetched at first benchmark run, not bundled (~125 GB if you want everything; per-bench fetch via `videvalkit fetch-checkpoints --bench <name>`)

What `scripts/post_install.sh` adds: `detectron2`, `SAM-2`, `GroundingDINO`, `segment-anything`, `lietorch`, `droid_backends` (built via DROID-SLAM clone), and the `en_core_web_sm` spaCy model. These are git/source-build packages excluded from the tarball to keep its size predictable; they're rebuilt against your local CUDA at post-install time.

### 3.1 Verify your setup

```bash
videvalkit doctor
```

`doctor` checks: env activation, CUDA visibility, GPU memory, HF cache location, smoke-data presence, and reachability of any configured judge endpoint. Typical good output ends with all green checks; anything flagged red blocks an eval that depends on it.

---

## 4. Setting up data and weights

### 4.1 HuggingFace authentication (optional)

Most checkpoints fetched by the toolkit are public. A few upstream-published reference video sets are gated and require a HuggingFace read-scope token (auto-approve or manual approval, depending on the dataset). To register one:

```bash
hf auth login
# paste a read-scope token from https://huggingface.co/settings/tokens
```

If you are behind a corporate proxy or are using a HuggingFace mirror, set `HF_ENDPOINT` before fetching:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 4.2 Fetch smoke data

The toolkit ships a representative subset of every benchmark's prompts plus paper-released reference videos under one HF repo (`videogenevalkit/smoke-data`). Fetch all six at once (default), or restrict to one bench:

```bash
# Fetch smoke data for all 6 benchmarks (about 3 GB)
videvalkit fetch-smoke-data

# Or fetch a single bench's smoke data
videvalkit fetch-smoke-data --bench worldjen
```

By default, smoke data lands under `~/.cache/videvalkit/smoke-data/<bench>/`. Each bench's smoke set is the official paper-released video set or a representative subset (50-200 videos per bench), enough to verify the full pipeline end-to-end without a multi-day GPU job.

### 4.3 Fetch checkpoints

Pretrained checkpoints are fetched per bench. The per-bench sizes (see DEV_MANUAL section 15.3 for the full per-dim model dependency map) are:

| Bench | Approx size | Notes |
|---|---:|---|
| `vbench` | 7.7 GB | DINO, CLIP, AMT, RAFT, MUSIQ, GRiT, UMT, tag2text, ViCLIP |
| `vbench2` | 23 GB | LLaVA-Video-7B + Qwen2.5-7B + Qwen2.5-VL-3B + CV stack |
| `videobench` | 0 GB native | dataset only; uses your configured VLM judge for all 9 dims |
| `worldjen` | 16 GB | Gemma-4-31B + Qwen2.5-7B (bundled under `hf-models/`) |
| `worldscore` | 6.3 GB | DROID-SLAM, SEA-RAFT, VFIMamba, SAM2, GroundingDINO, LAION |
| `t2vcompbench` | 72 GB | LLaVA-1.6-34B (68 GB) + GD + SAM-H + Depth-Anything V1 + DOT |

```bash
# Fetch a single benchmark's checkpoints
videvalkit fetch-checkpoints --bench worldscore

# Fetch t2vcompbench but skip the 68 GB LLaVA-1.6-34B MLLM stack
videvalkit fetch-checkpoints --bench t2vcompbench --skip-mllm-upstream

# Fetch everything (about 110 GB)
videvalkit fetch-checkpoints --all

# Preview what would download without fetching
videvalkit fetch-checkpoints --bench t2vcompbench --dry-run
```

Note: `LLaVA-1.6-34B` is bundled but optional. If you have less than 80 GB of GPU memory, pass `--skip-mllm-upstream` and run T2V-CompBench in toolkit mode against a smaller VLM judge.

Checkpoints land under `~/.cache/videvalkit/checkpoints/<bench>/` by default. You can override with `VIDEVALKIT_CHECKPOINT_ROOT`. Downloads are resumable.

---

## 5. Your first evaluation

This is the canonical end-to-end smoke run on WorldJen. WorldJen is the safest first benchmark: 50 prompts, 1 video per prompt, ~1 hour wall-clock on a single GPU with the local Gemma judge. The smoke data ships videos from `fal-ai_kling-video_v2.6_pro_text-to-video` (Kling v2.6).

```bash
videvalkit eval --bench worldjen \
  --videos ~/.cache/videvalkit/smoke-data/worldjen/videos/fal-ai_kling-video_v2.6_pro_text-to-video \
  --workspace runs/first \
  --judge gemma-4-31b-local

videvalkit aggregate --workspace runs/first

cat runs/first/results/summary/worldjen/Kling.json
```

The first command runs the WorldJen adapter on the bundled Kling smoke videos with the local Gemma-4-31B vLLM judge on port 8003. WorldJen runs two phases: phase A generates per-prompt VQA questions with an LLM, phase B answers them with the VLM. The smoke data ships a prebuilt `vqa_questions_50prompts.jsonl` so phase A is skipped by default; if you want to run phase A from scratch, see section 7.

The `Summary` JSON has three top-level groups:

```json
{
  "benchmark": "worldjen",
  "model": "Kling",
  "n_videos": 50,
  "headline": {"metric": "phas", "score": 3.6561},
  "per_dimension": {
    "subject_consistency": 3.782,
    "scene_consistency":   3.718,
    "motion_smoothness":   3.134,
    "...": "..."
  },
  "overall": {
    "PHAS": 3.6561,
    "unweighted_dim_mean": 3.577,
    "n_records": 800
  },
  "meta": {
    "toolkit_commit": "abcd1234",
    "upstream_pkg":   "worldjen==in-tree",
    "judge":          "openai_compatible:google/gemma-4-31b-it",
    "ckpt_checksums": {"...": "..."},
    "runtime":        {"python": "3.10.13", "torch": "2.3.1+cu121"},
    "scorers_used":   {"default": "gemma-4-31b-local"}
  }
}
```

`per_dimension` is the 16 WorldJen dim scores (1-5 scale). `overall.PHAS` is the headline (PHAS aggregator with paper-tuned weights). `meta.scorers_used` records the actual judge used so downstream tools can flag judge-substituted runs.

For TEST_MANUAL section 4.3, this Kling smoke run yields PHAS ~3.66 vs the paper-reported Gemma-judge headline of 4.12 (Delta -0.47). The shift is decord-vs-cv2 frame variance plus Gemma sampling differences; pipeline correctness is verified at the per-dim level.

The `aggregate` step writes `runs/first/results/leaderboard/cross_benchmark.json` (useful when you have multiple benches in the same workspace; for one bench it just rolls up the single bench's per-model headlines).

---

## 6. Per-benchmark how-to

Each subsection below is a complete copy-pasteable running example: fetch the smoke data, fetch the specific checkpoint from `videogenevalkit/checkpoints` on HuggingFace, stage 3-5 sample videos, run the eval, see the score. All examples were validated end-to-end on 2026-05-19; expected scores below come from those runs (3 sample videos each).

**Quick summary table** — all 6 in one view:

| # | Bench / dim | Scorer | Checkpoint (from `videogenevalkit/checkpoints`) | Wallclock (3 vids) | GPU mem | Expected score |
|---|---|---|---|---:|---:|---|
| 6.4 | `worldjen` / 16 dims | Gemma-4-31B vLLM | — (no local ckpt) | ~5 min | 0 GB | overall ≈ 3.2 of 5 |
| 6.1 | `vbench` / `subject_consistency` | DINO ViT-B/16 | `vbench/pretrained/dino_model/` (343 MB) | ~30 s | 2 GB | ≈ 0.92 |
| 6.2 | `vbench2` / `Camera_Motion` | CoTracker3 (vendored) | `vbench2/third_party/cotracker/` (204 MB) | ~1 min | 4 GB | ≈ 0.67 |
| 6.3 | `videobench` / `action_consistency` | Gemma-4-31B vLLM | — (no local ckpt) | ~2 min | 0 GB | ≈ 2.0 (raw 1-5) |
| 6.5 | `worldscore` / `motion_magnitude` | SEA-RAFT | `worldscore/Tartan-C-T-TSKH-*` + `raft-things` (150 MB) | ~2 min | 4 GB | ≈ 56.4 (×100) |
| 6.6 | `t2vcompbench` / `action_binding` (paper-mode) | LLaVA-1.6-34B | `hf-models/liuhaotian/llava-v1.6-34b/` (68 GB) | ~5 min after model load | 70 GB | raw 7.22 → norm 0.69 |

Run-to-run variance: ±0.05 on CV-based dims (DINO, CoTracker3, SEA-RAFT, GroundingDINO), ±0.15 on VLM-judge dims (Gemma-4-31B, LLaVA-1.6-34B). Tolerance bands per dim in [`TEST_MANUAL.md`](TEST_MANUAL.md) §3.

Source material for the dim definitions and per-bench design: DEV_MANUAL section 15.3.1 (per-dim deps) and TEST_MANUAL section 4 (validation results).

### 6.1 VBench v1

**Overview.** 16 dims, 7 quality + 9 semantic. All dims are pure-CV (no VLM judge). Default scorers per dim: DINO ViT-B/16 for `subject_consistency`, CLIP ViT-B/32 for `background_consistency`, AMT-S for `motion_smoothness`, RAFT for `dynamic_degree`, MUSIQ for `imaging_quality`, GRiT for `object_class`/`color`/`multiple_objects`/`spatial_relationship`, UMT for `human_action`, tag2text for `scene`, ViCLIP for `appearance_style`/`temporal_style`/`overall_consistency`, LAION-aesthetic+CLIP ViT-L/14 for `aesthetic_quality`. Aggregator is `vbench_weighted`: `Total = 0.54*Quality + 0.46*Semantic`.

**Running example: `subject_consistency` on 3 HunyuanVideo videos (~30 s, 2 GB GPU)**

```bash
# 1. Fetch smoke videos + DINO ViT-B/16 weight + the VBench prompt registry
videvalkit fetch-smoke-data  --bench vbench
videvalkit fetch-checkpoints --bench vbench

# 2. Stage 3 of the bundled HunyuanVideo videos
mkdir -p ~/runs/vbench/videos/HunyuanVideo
ls ~/.cache/videvalkit/smoke-data/vbench/videos/HunyuanVideo/*.mp4 \
  | head -3 | xargs -I{} ln -sf {} ~/runs/vbench/videos/HunyuanVideo/

# 3. Evaluate
videvalkit eval --bench vbench \
    --videos ~/runs/vbench/videos \
    --workspace ~/runs/vbench/ws \
    --models HunyuanVideo \
    --dimensions subject_consistency \
    --prompts-file ~/.cache/videvalkit/smoke-data/vbench/prompts.jsonl
```

**Expected:** `per_dimension.subject_consistency ≈ 0.92` (DINO frame-pair cosine averaged over 8 frames × 3 videos). Variance across runs: ±0.02.

**Checkpoint:** `vbench/pretrained/dino_model/dino_vitbase16_pretrain.pth` (343 MB) — equivalent to `huggingface_hub.snapshot_download(repo_id="videogenevalkit/checkpoints", repo_type="dataset", allow_patterns=["vbench/pretrained/dino_model/*", "vbench/VBench_full_info.json"])`.

**Gotchas:** `dynamic_degree` is the noisiest dim because of RAFT GPU non-determinism; widen tolerance to ±0.025 for that dim. `human_action` depends on YOLOv5x weights; verify the SHA-256 in `checksums.json`. For prompt-dependent dims (`object_class`, `color`, `spatial_relationship`, `scene`, `human_action`, `multiple_objects`) custom prompts must carry `auxiliary_info` tags; use the auto-labeler (section 12) if your prompts do not.

**Validation:** HunyuanVideo full-set sweep, 16/16 dims within ±0.025; mean |Δ| 0.012 vs HF leaderboard.

### 6.2 VBench-2.0

**Overview.** 18 dims across 5 categories (Creativity, Commonsense, Controllability, Human Fidelity, Physics). The 12 reasoning dims use LLaVA-Video-7B-Qwen2 as the VLM scorer; 5 dims add Qwen2.5-7B-Instruct as an LLM judge. The remaining 6 use vendored CV stacks: `Camera_Motion` and `Multi-View_Consistency` use CoTracker3; `Diversity` uses VGG-19; `Human_Anatomy` uses a vendored ViTDetector; `Human_Identity` uses ArcFace + RetinaFace; `Instance_Preservation` uses Qwen2.5-VL-3B via ms-swift. Aggregator is `vbench2_category`: 5 category means averaged into `Overall`.

**Running example: `Camera_Motion` on 3 HunyuanVideo videos (~1 min, 4 GB GPU)**

```bash
# 1. Fetch smoke videos + CoTracker3 weight + the VBench-2.0 prompt registry
videvalkit fetch-smoke-data  --bench vbench2
videvalkit fetch-checkpoints --bench vbench2

# 2. Stage 3 HunyuanVideo Camera_Motion-tagged videos
mkdir -p ~/runs/vbench2/videos/HunyuanVideo/Camera_Motion
ls ~/.cache/videvalkit/smoke-data/vbench2/videos/HunyuanVideo/Camera_Motion/*.mp4 \
  | head -3 | xargs -I{} ln -sf {} ~/runs/vbench2/videos/HunyuanVideo/Camera_Motion/

# 3. Evaluate
videvalkit eval --bench vbench2 \
    --videos ~/runs/vbench2/videos \
    --workspace ~/runs/vbench2/ws \
    --models HunyuanVideo \
    --dimensions Camera_Motion \
    --extra-kwarg 'mode="vbench2_standard"'
```

**Expected:** `per_dimension.Camera_Motion ≈ 0.67` (binary pass/fail per video, averaged across 3 videos = either 0, 0.33, 0.67, or 1.0 depending on which Kling-camera classes Kling reproduces). Run-to-run is deterministic for CoTracker3 (no temperature).

**Checkpoint:** `vbench2/third_party/cotracker/cotracker2.pth` (204 MB) — `allow_patterns=["vbench2/third_party/cotracker/*", "vbench2/VBench2_full_info.json"]`.

**Gotchas:** `Human_Anatomy` and `Human_Identity` are NOT swappable to a different VLM (vendored detectors). `Diversity` needs at least 2 seeds per prompt. The full sweep on HunyuanVideo yields mean |Δ| 0.0055 across 18/18 dims after the cv2 sequential-read fix and the `-1` sentinel filter. Default judge is `local-llava-video-7b`; you can substitute with `--scorer-vlm gemma-4-31b-local` for stronger reasoning (paper-Δ widens; see section 7).

**Validation:** HunyuanVideo full-set sweep, 18/18 dims within ±0.025; mean |Δ| 0.0055 vs HF leaderboard (2025-03-28 row).

### 6.3 Video-Bench

**Overview.** 9 dims: 4 alignment (`video_text_consistency`, `object_class_consistency`, `color_consistency`, `action_consistency`, `scene_consistency`) on a 1-3 scale + 4 quality (`imaging_quality`, `aesthetic_quality`, `temporal_consistency`, `motion_effects`) on a 1-5 scale. All 9 dims use the **same** VLM judge — this is the simplest benchmark by model count. Paper uses GPT-4o (`gpt-4o-2024-08-06`); the toolkit registry pins `gpt-4o-2024-11-20`. Aggregator is `videobench_per_dim`: arithmetic mean across the chain-of-query responses.

**Running example: `action_consistency` on 3 CogVideoX-5B videos (~2 min, Gemma judge — no local ckpt)**

```bash
# 1. Fetch smoke videos + prompts.jsonl (no checkpoint — VLM judge does the scoring)
videvalkit fetch-smoke-data --bench videobench

# 2. Stage 3 of the bundled action_consistency-tagged videos
mkdir -p ~/runs/videobench/videos/cogvideox5b
ls ~/.cache/videvalkit/smoke-data/videobench/videos/cogvideox5b/action_consistency/*.mp4 \
  | head -3 | xargs -I{} ln -sf {} ~/runs/videobench/videos/cogvideox5b/

# 3. Evaluate (Gemma-4-31B vLLM judge on localhost:8003)
videvalkit eval --bench videobench \
    --videos ~/runs/videobench/videos \
    --workspace ~/runs/videobench/ws \
    --models cogvideox5b \
    --judge gemma-4-31b-local \
    --dimensions action_consistency \
    --prompts-file ~/.cache/videvalkit/smoke-data/videobench/prompts.jsonl
```

**Expected:** `per_dimension.action_consistency ≈ 2.0` (raw 1-5 scale). Variance: ±0.5 across runs due to Gemma temperature-0.2 sampling.

**Checkpoint:** none — the entire scorer is the VLM endpoint. To use GPT-4o instead: `--judge gpt-4o` with `OPENAI_API_KEY` set; for the paper-exact `gpt-4o-2024-08-06` snapshot, register that pin in `~/.config/videvalkit/judges.yaml` (see section 7).

**Gotchas:** GPT-4o snapshots drift; for paper reproduction, register a `gpt-4o-2024-08-06` entry and pass `--judge gpt-4o-2024-08-06`. Substituting Gemma for GPT-4o produces dynamic-quality drift of ±2.0 points on the 1-5 scale (`temporal_consistency` and `motion_effects` are bimodal under Gemma). Static + alignment dims match within ±0.2 of paper under Gemma.

### 6.4 WorldJen

**Overview.** 16 dims grouped into 4 macro-categories (motion_stability, logic_physics, instruction_adherence, aesthetic_quality). Two-phase: phase A generates VQA questions with an LLM (`qwen3-32b-local` on port 8004 by default); phase B answers them with a VLM (`gemma-4-31b-local` on port 8003 by default). Aggregator is `phas`: weighted sum of mean dim scores minus a variance penalty, with paper-calibrated `(w, λ)`.

**Running example: all 16 dims on 3 Kling videos (~5 min, Gemma judge — no local ckpt)**

```bash
# 1. Fetch the worldjen smoke set (50 Kling videos + prompts.jsonl + vqa.jsonl)
videvalkit fetch-smoke-data --bench worldjen

# 2. Stage 3 of the bundled Kling videos
mkdir -p ~/runs/worldjen/videos/Kling
ls ~/.cache/videvalkit/smoke-data/worldjen/videos/fal-ai_kling-video_v2.6_pro_text-to-video/*.mp4 \
  | head -3 | xargs -I{} ln -sf {} ~/runs/worldjen/videos/Kling/

# 3. Evaluate (~250 Gemma calls: phase A VQA gen + phase B VQA answering)
videvalkit eval --bench worldjen \
    --videos ~/runs/worldjen/videos \
    --workspace ~/runs/worldjen/ws \
    --models Kling \
    --judge gemma-4-31b-local \
    --aggregator phas
```

**Expected:** `~/runs/worldjen/ws/results/summary/worldjen/Kling.json`:

```
overall ≈ 3.2 of 5  (unweighted mean of 16 dim means on 3 prompts)
per_category:
  instruction_adherence  ≈ 3.78    (Kling's strongest area)
  aesthetic_quality      ≈ 3.50
  motion_stability       ≈ 3.20
  logic_physics          ≈ 2.58    (Kling's weakest area)
```

Per-dim breakdown (means across the 3 sample prompts):

| Dim                  | Score | Dim                  | Score |
|---|---:|---|---:|
| `semantic_adherence`     | 4.47 | `composition_framing` | 3.60 |
| `semantic_drift`         | 4.07 | `lighting_volumetric` | 3.60 |
| `color_harmony`          | 3.83 | `subject_consistency` | 3.43 |
| `scene_consistency`      | 3.73 | `structural_gestalt`  | 2.97 |
| `temporal_flickering`    | 3.73 | `spatial_relationship`| 2.80 |
| `human_fidelity`         | 2.70 | `motion_smoothness`   | 2.60 |
| `inertial_consistency`   | 2.50 | `dynamic_degree`      | 2.57 |
| `physical_mechanics`     | 2.57 | `object_permanence`   | 2.50 |

**Checkpoint:** none — both phase A (LLM) and phase B (VLM) are remote endpoints. The default `gemma-4-31b-local` is `google/gemma-4-31b-it` served by vLLM on `http://localhost:8003/v1`. Swap with `--judge gemini-3-flash` (managed API, requires `GOOGLE_API_KEY`) for the paper-Gemini configuration.

**Calibrated headline.** The paper-reported PHAS for Kling-v2.6 on the full 50-prompt headline split is **4.12** — uses the *calibrated* per-dim weights (non-neg ridge fit against human ratings) instead of the unweighted mean. To reproduce, symlink all 50 mp4s into the videos dir and re-run with `--aggregator phas` (the calibrated weights are loaded automatically).

**Gotchas:** If the workspace already has `vqa_questions_50prompts.jsonl`, phase A is silently skipped — the log line "judge_llm not given; defaulting to judge for VQA gen" fires unconditionally before the existence check. To confirm phase A actually ran, inspect `<ws>/api_logs/calls/Qwen/`. Keep `max_concurrency=2` against Gemma to avoid broken-pipe spam; phase A on Qwen is light and can run alongside heavier phase-B work without contention.

### 6.5 WorldScore

**Overview.** 10 dims: 7 static (`camera_control`, `object_control`, `content_alignment`, `3d_consistency`, `photometric_consistency`, `style_consistency`, `subjective_quality`) + 3 dynamic (`motion_accuracy`, `motion_magnitude`, `motion_smoothness`). All CV-based, no VLM judge required. The stack is DROID-SLAM (camera + 3D), SEA-RAFT (optical flow + photometric), VFIMamba (motion smoothness via interpolation), SAM2 (motion-accuracy mask propagation), GroundingDINO + SAM-H (object detection), VGG-19 (style Gram), LAION + CLIP-IQA+ (subjective quality), torchmetrics CLIPScore (content alignment). Headlines: `WorldScore-Static` (mean of 7 static dims × 100) and `WorldScore-Dynamic` (mean of all 10 dims × 100).

**Running example: `motion_magnitude` on 3 CogVideoX-5B dynamic videos (~2 min, 4 GB GPU)**

WorldScore's upstream code expects its weights at a relative path inside the upstream repo, not in the toolkit's HF cache. After `fetch-checkpoints` we symlink them into the expected location.

```bash
# 1. Fetch smoke videos + upstream repo + the SEA-RAFT weights
videvalkit fetch-smoke-data  --bench worldscore
videvalkit fetch-upstream    --bench worldscore   # git-clones the WorldScore repo
videvalkit fetch-checkpoints --bench worldscore

# 2. Symlink ckpts into upstream's expected layout (one-time)
WS_ROOT=~/.cache/videvalkit/upstream/WorldScore
mkdir -p $WS_ROOT/worldscore/benchmark/metrics/checkpoints
ln -sf ~/.cache/videvalkit/checkpoints/worldscore/*.pth \
       $WS_ROOT/worldscore/benchmark/metrics/checkpoints/
ln -sf ~/.cache/videvalkit/checkpoints/worldscore/*.pkl \
       $WS_ROOT/worldscore/benchmark/metrics/checkpoints/
export VIDEVALKIT_WORLDSCORE_ROOT=$WS_ROOT

# 3. Stage 3 of the bundled dynamic videos
mkdir -p ~/runs/worldscore/videos/cogvideox-5b/dynamic
ls ~/.cache/videvalkit/smoke-data/worldscore/videos/cogvideox-5b/dynamic/*.mp4 \
  | head -3 | xargs -I{} ln -sf {} ~/runs/worldscore/videos/cogvideox-5b/dynamic/

# 4. Evaluate (SEA-RAFT optical flow on 49 frames per video, median magnitude)
videvalkit eval --bench worldscore \
    --videos ~/runs/worldscore/videos \
    --workspace ~/runs/worldscore/ws \
    --models cogvideox-5b \
    --dimensions motion_magnitude \
    --prompts-file ~/.cache/videvalkit/smoke-data/worldscore/prompts/dynamic.jsonl
```

**Expected:** `per_dimension.motion_magnitude ≈ 56.4` (×100 scale per upstream convention). Headline `WorldScore-Dynamic ≈ 56.4` when only this dim is run. Run-to-run variance: deterministic for SEA-RAFT (no temperature).

**Checkpoints:**
- `worldscore/Tartan-C-T-TSKH-spring540x960-M.pth` (130 MB) — SEA-RAFT optical flow
- `worldscore/raft-things.pth` (20 MB) — RAFT inside DROID-SLAM
- Equivalent: `allow_patterns=["worldscore/Tartan*", "worldscore/raft-things*"]`.

For all 10 dims you also need: `groundingdino_swint_ogc.pth`, `sam_vit_h_4b8939.pth`, `sam2.1_hiera_large.pt`, `VFIMamba.pkl`, `droid.pth`, `sac+logos+ava1-l14-linearMSE.pth` (full set fetched by `--bench worldscore` without an `allow_patterns` filter; ~6.3 GB total).

**Gotchas:** `style_consistency` measures against an `input_image.png` from the HF dataset, NOT the generated video's frame 0; for T2V models the reference image is generated by upstream's T2I pipeline. Run `runners/extract_refs.py` once to materialize per-entry reference images before evaluating. Pin torch to `2.3.1+cu121`; do NOT let pip auto-upgrade torch when installing `mamba_ssm` (the upgrade breaks lietorch/droid_backends/sam2/pyiqa). The adapter ships a pure-PyTorch `mamba_ssm.selective_scan` shim so VFIMamba runs without recompiling.

### 6.6 T2V-CompBench

**Overview.** 7 compositional dims. 4 MLLM dims (`consistent_attribute`, `action_binding`, `object_interactions`, `dynamic_attribute`) use LLaVA-1.6-34B at temperature 0 with `chatml_direct` conversation template and 3-seed averaging. 3 CV dims: `generative_numeracy` (GroundingDINO + counting), `spatial_relationships` (GD + Depth-Anything V1, 2D+3D combined), `motion_binding` (GD + SAM-H + DOT = cotracker2 + RAFT estimator + RAFT refiner). Aggregator: unweighted mean across the 7 dims.

**Running example A — `action_binding` paper-mode on 3 CogVideoX-5B videos (~5 min after LLaVA load, requires ≥80 GB GPU)**

```bash
# 1. Fetch smoke videos + the t2vcompbench CV ckpts + LLaVA-1.6-34B
videvalkit fetch-smoke-data  --bench t2vcompbench
videvalkit fetch-checkpoints --bench t2vcompbench   # GroundingDINO + SAM-H + Depth-Anything + DOT (~4 GB)
videvalkit fetch-checkpoints --bench hf-models      # LLaVA-1.6-34B + Qwen2.5-7B + LLaVA-Video-7B + CLIP + DepthAnything (~95 GB)

# 2. Clone the T2V-CompBench upstream repo (carries the LLaVA eval script)
videvalkit fetch-upstream --bench t2vcompbench

# 3. Stage 3 action_binding-tagged videos
mkdir -p ~/runs/t2vcomp_paper/videos/CogVideoX-5B
ls ~/.cache/videvalkit/smoke-data/t2vcompbench/videos/CogVideoX-5B/action_binding_*.mp4 \
  | head -3 | xargs -I{} ln -sf {} ~/runs/t2vcomp_paper/videos/CogVideoX-5B/

# 4. Evaluate (LLaVA-1.6-34B at temp 0, 3 seeds × 3 videos = 9 inference calls, ~5 min)
videvalkit eval --bench t2vcompbench \
    --videos ~/runs/t2vcomp_paper/videos \
    --workspace ~/runs/t2vcomp_paper/ws \
    --models CogVideoX-5B \
    --dimensions action_binding \
    --prompts-file ~/.cache/videvalkit/smoke-data/t2vcompbench/prompts.jsonl \
    --extra-kwarg 'mode="upstream"'
```

**Expected:** `per_dimension.action_binding ≈ 7.22` (raw 1-10 scale) → normalized via upstream's `(raw - 1) / 9` formula = **0.691**. Paper Table 2 for CogVideoX-5B on the full 200-video set is 0.533; our 3-video smoke is well within the expected ±0.15 sampling band.

**Checkpoint:** `hf-models/liuhaotian/llava-v1.6-34b/` (68 GB across 15 safetensors files) — equivalent to `allow_patterns=["hf-models/liuhaotian/llava-v1.6-34b/*"]`.

**Running example B — `action_binding` toolkit-mode (no LLaVA-34B, runs on a small GPU)**

If you don't have an 80 GB GPU, swap to `mode="toolkit"` which routes the MLLM dim through your configured VLM judge:

```bash
videvalkit eval --bench t2vcompbench \
    --videos ~/runs/t2vcomp_toolkit/videos \
    --workspace ~/runs/t2vcomp_toolkit/ws \
    --models CogVideoX-5B \
    --dimensions action_binding \
    --judge gemma-4-31b-local \
    --prompts-file ~/.cache/videvalkit/smoke-data/t2vcompbench/prompts.jsonl \
    --extra-kwarg 'mode="toolkit"'
# → per_dimension.action_binding ≈ 0.33  (different absolute number due to Gemma vs LLaVA-34B judge; paper-Δ not asserted)
```

**Gotchas:** `--mode upstream` is the paper-exact path (LLaVA-1.6-34B subprocess shim) and is required for byte-exact paper-Δ. `--mode toolkit` routes the 4 MLLM dims through your configured VLM judge — useful when you don't have the 80 GB GPU but accepts a paper-Δ caveat that's logged into `meta.scorers_used`.

**Validation:** 6/7 dims within ±0.020 against paper Table 5 in upstream mode; `consistent_attribute` shows +0.186 drift attributed to LLaVA HEAD-revision drift (paper's revision is not pinned in the upstream `requirements.txt`).

---

## 7. Configuring scorers (VLM/LLM judges)

Cross-ref DEV_MANUAL section 16. Judges are pluggable: every benchmark that calls into a VLM or LLM does so through the `SUPPORTED_JUDGES` registry plus a precedence chain that lets you override per-host without forking the toolkit.

### Precedence chain (lowest to highest)

1. Built-in `SUPPORTED_JUDGES` defaults in `src/videvalkit/configs/judges.py`
2. Environment variable defaults (e.g. `VIDEVALKIT_JUDGE_DEFAULT`)
3. `~/.config/videvalkit/judges.yaml` - user-wide overrides and new judge names
4. `<workspace>/judges.yaml` - per-project pinning
5. CLI flags - highest precedence: `--judge`, `--judge-endpoint`, `--judge-model`, `--judge-kind`, `--judge-api-key-env`

### Local vLLM Gemma / Qwen on ports 8003 / 8004

```yaml
# ~/.config/videvalkit/judges.yaml
gemma-4-31b-local:
  kind: openai_compatible
  endpoint: http://localhost:8003/v1
  model: google/gemma-4-31b-it
  provider: google
  api_key_env: null
  request_timeout_s: 180

qwen3-32b-local:
  kind: openai_compatible
  endpoint: http://localhost:8004/v1
  model: Qwen/Qwen3-32B
  provider: Qwen
  api_key_env: null
```

Launch the corresponding vLLM servers (one-time, then leave them running):

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31b-it --port 8003 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --served-model-name google/gemma-4-31b-it &

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B --port 8004 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --served-model-name Qwen/Qwen3-32B &
```

The `--served-model-name` must match the YAML's `model` field byte-for-byte; vLLM routes by this name.

### OpenAI GPT-4o

```yaml
# ~/.config/videvalkit/judges.yaml
gpt-4o:
  kind: openai_compatible
  endpoint: https://api.openai.com/v1
  model: gpt-4o-2024-11-20
  provider: openai
  api_key_env: OPENAI_API_KEY
  cost_per_million_input_tokens: 2.50
  cost_per_million_output_tokens: 10.00
  cost_per_image_input: 0.00765

gpt-4o-2024-08-06:           # for Video-Bench paper reproduction
  kind: openai_compatible
  endpoint: https://api.openai.com/v1
  model: gpt-4o-2024-08-06
  provider: openai
  api_key_env: OPENAI_API_KEY
```

```bash
export OPENAI_API_KEY=sk-...
videvalkit eval --bench videobench --judge gpt-4o-2024-08-06 ...
```

### Gemini via Google AI Studio

```yaml
gemini-3-flash:
  kind: gemini
  model: gemini-3-flash-preview
  provider: google
  api_key_env: GEMINI_API_KEY
  cost_per_million_input_tokens: 0.075
  cost_per_million_output_tokens: 0.30
```

```bash
export GEMINI_API_KEY=...
```

### Anthropic Claude

```yaml
claude-sonnet-4-6:
  kind: anthropic
  model: claude-sonnet-4-6
  provider: anthropic
  api_key_env: ANTHROPIC_API_KEY
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### CLI selection

```bash
# Whole-bench judge swap
videvalkit eval --bench vbench2 --videos ... \
  --scorer-vlm gemma-4-31b-local        # instead of the default lmms-lab/LLaVA-Video-7B-Qwen2

# Per-dim override (mix and match)
videvalkit eval --bench vbench2 --videos ... \
  --scorer-vlm gemma-4-31b-local \
  --scorer-vlm-dim complex_plot=claude-sonnet-4-6 \
  --scorer-vlm-dim human_interaction=gpt-4o-2024-11-20

# Ad-hoc endpoint override (no YAML edit)
videvalkit eval --bench videobench --videos ... \
  --judge-kind openai_compatible \
  --judge-endpoint http://10.0.1.7:9001/v1 \
  --judge-model Qwen/Qwen3-32B \
  --judge-api-key-env null
```

When you swap away from a benchmark's paper-default scorer, the toolkit:
1. Writes the actual scorer to `Summary.meta.scorers_used`.
2. Refuses to claim Delta-vs-paper in `compare-leaderboard` output.
3. Widens the tolerance band 3x if you pass `--allow-judge-substitution`.
4. Logs token usage per call to `api_logs/`.

Inspect cost after a run:

```bash
videvalkit api-usage --workspace runs/first
```

---

## 8. Configuring GPUs

Cross-ref DEV_MANUAL section 17. Three dispatch modes:

| Mode | Trigger | Behavior |
|---|---|---|
| `single` | `--gpu N` | Whole bench runs on one device |
| `dim_parallel` | `--gpus 0,1,2` (or `auto`) | Dims sharded across the pool, each as its own subprocess with `CUDA_VISIBLE_DEVICES=N` |
| `affinity` | `--gpu-affinity Dim=N,...` | User pins specific dims; remaining dims fall back to the pool |

The mode is inferred from the flags; there is no explicit `--mode` switch.

### Single-GPU example

```bash
videvalkit eval --bench worldjen --gpu 2 --videos ... --workspace ...
```

### Dim-parallel example (18 VBench-2.0 dims across 3 GPUs)

```bash
videvalkit eval --bench vbench2 --videos ... --gpus 0,1,2 --gpu-strategy most_free_memory
```

### Affinity example (pin LLaVA-34B dims to dedicated GPUs)

```bash
videvalkit eval --bench t2vcompbench --videos ... \
  --gpu-affinity consistent_attribute=0,action_binding=1,object_interactions=2 \
  --gpus 3                                # other dims fall back to GPU 3
```

### Auto-selection strategies

| Strategy | Picks GPU with... |
|---|---|
| `most_free_memory` (default) | `argmax(free_mem_gb)` at acquire time |
| `round_robin` | next index in the pool |
| `least_utilization` | `argmin(utilization_percent)` at acquire time |

Pass via `--gpu-strategy {most_free_memory, round_robin, least_utilization}` or in the YAML.

### Persistent default via `compute.yaml`

```yaml
# ~/.config/videvalkit/compute.yaml
compute:
  gpus: auto                              # default for all benches
  auto_strategy: most_free_memory
  reserve_mem_gb: 6                       # do not pick a GPU with less than 6 GB free
```

Per-workspace overrides go in `<workspace>/config.yaml` with the same schema, plus a `compute.affinity` block for benchmark.dim pinning.

The router writes a `compute_log.jsonl` to the workspace, one line per dim launch, paired with `api_log.jsonl` to answer "what did this benchmark run cost me?" - GPU minutes plus token spend.

Min-memory hints are baked into each bench's manifest (e.g. T2V-CompBench `consistent_attribute` requires `min_mem_gb=75, exclusive=True`); the router refuses to schedule a dim onto a GPU that does not meet the hint.

---

## 9. Cross-benchmark aggregation

Once a workspace contains summary files for one or more benchmarks, aggregate them into a unified leaderboard:

```bash
videvalkit aggregate --workspace runs/first
# Output:
#   runs/first/results/leaderboard/cross_benchmark.json
# Console:
#   #1  seedance20         z=+0.521
#   #2  pangu_model3_141   z=+0.183
#   #3  wan-14B-pe-141     z=-0.704
```

Five aggregators ship out of the box:

| Aggregator | What it does |
|---|---|
| `weighted_sum` | User-specified weights per dim or per bench |
| `vbench_weighted` | `Total = 0.54 * Quality + 0.46 * Semantic`, the VBench v1 headline |
| `vbench2_category` | 5 category means averaged into `Overall`, the VBench-2.0 headline |
| `phas` | PHAS = `Sum(w_i * mu_i) - lambda * sigma^2`, the WorldJen headline |
| `bt` | Bradley-Terry pairwise ranking across models for `videvalkit aggregate` cross-bench |

Select via `--aggregator <name>` on `aggregate`. The default is `weighted_sum` with equal weights across benches.

Cross-bench output schema:

```json
{
  "models": ["seedance20", "pangu_model3_141", "wan-14B-pe-141"],
  "ranked": [
    {"model": "seedance20", "z_score": 0.521, "per_bench": {"worldjen": 4.12, "vbench": 0.815, "..."}}
  ],
  "bt_rating": {"seedance20": 1.42, "...": "..."},
  "meta": {"aggregator": "weighted_sum", "n_benches": 3, "...": "..."}
}
```

---

## 10. Standalone metrics

Cross-ref DEV_MANUAL section 14. Standalone metrics compute a single algorithm on two sets of videos - no prompts, no dims, no aggregator. Useful when you want one number (FID, FVD, CLIP-Score, PSNR, SSIM, LPIPS) without the benchmark scaffolding.

```bash
videvalkit metric \
  --name fvd \
  --gen-videos path/to/generated/ \
  --ref-videos path/to/reference/ \
  --device cuda:0 \
  --out fvd_result.json
```

Supported metrics:

| Metric | Needs reference | Notes |
|---|---|---|
| `fid` | yes | clean-fid / pytorch-fid (Inception-V3 pool3) |
| `fvd` | yes | canonical I3D Kinetics-400 |
| `clipscore` | no | OpenAI CLIP ViT-B/32 |
| `psnr` | yes | paired video PSNR |
| `ssim` | yes | paired video SSIM |
| `lpips` | yes | AlexNet or VGG backbone |

Note: the standalone metrics module is on the planned-2026-05-18 channel; the registry and CLI are wired, and FVD/FID/CLIP-Score are usable now. PSNR/SSIM/LPIPS are scheduled per the priority list in DEV_MANUAL section 14.

Returns a flat JSON: `{"fvd": 132.4, "n_gen": 200, "n_ref": 200, "backbone": "i3d-k400", ...}`.

---

## 11. Generating your own videos

For users with their own T2V model, organize generations under one root in the per-bench expected layout:

```
<videos_root>/
  <model_name>/
    <prompt_id>.mp4
    ...
```

Per-bench expected layout (see DEV_MANUAL section 4 for the full spec):

| Bench | Layout |
|---|---|
| `vbench` | `<root>/<model>/<prompt_id>-<sample_idx>.mp4` (5 samples per prompt) |
| `vbench2` | `<root>/<model>/<dim>/<prompt_id>-<sample_idx>.mp4` (dim-organized, 1-3 samples per prompt depending on dim) |
| `videobench` | `<root>/<model>/<dim>/<prompt_id>.mp4` (dim-organized; the upstream zips also follow this) |
| `worldjen` | `<root>/<model>/<prompt_id>.mp4` (1 sample per prompt) |
| `worldscore` | `<root>/<model>/{static,dynamic}/<prompt_id>.mp4` (split-organized) |
| `t2vcompbench` | `<root>/<model>/<dim>/<prompt_id>.mp4` (dim-organized, 200 prompts per dim for a full run) |

Once laid out, pass the `<videos_root>` to `videvalkit eval --videos <root>` and the adapter walks the expected layout. Per-bench prompt files live under `~/.cache/videvalkit/smoke-data/<bench>/prompts/` (for paper reproduction) or your own `--prompts-file <path>` (for custom prompts).

Resume is automatic: re-running the same command skips `(model, dim, prompt)` triples whose `results/raw/*.json` already exists. To force a re-run, delete the JSONs.

---

## 12. Custom prompts (auto-labeler)

VBench v1 and VBench-2.0 prompt-dependent dims (`object_class`, `color`, `spatial_relationship`, `scene`, `human_action`, `multiple_objects`, plus most of VBench-2.0) require per-prompt `auxiliary_info` tags - the upstream code reads these to compose detection / matching prompts. Your custom prompts do not have them.

The auto-labeler fills the gap using a local LLM:

```bash
python scripts/auto_label_prompts.py \
  --prompts /data/my_prompts/prompts.jsonl \
  --out-dir runs/my_ws/prompts/auto_labels \
  --benchmarks vbench,vbench2 \
  --judge qwen3-32b-local
```

Produces `vbench_full_info.json` and `vbench2_full_info.json` in the output directory, each with one entry per prompt containing the auto-extracted `auxiliary_info` block. Per-dim schemas the LLM follows are bundled in `videvalkit/prompt_labelers/`.

Use the auto-labeled file:

```bash
videvalkit eval --bench vbench \
  --prompts-file runs/my_ws/prompts/auto_labels/vbench_full_info.json \
  --videos ... --workspace ...
```

Caveats: the auto-labeler is good but not perfect. Limitations are documented in TEST_MANUAL section 2.4. For paper-Delta claims always use the upstream prompts verbatim.

---

## 13. Troubleshooting and FAQ

Run `videvalkit doctor` first; it surfaces most issues at a glance.

| Symptom | Likely cause | Fix |
|---|---|---|
| HF auth fails on `fetch-checkpoints` | Token not set, or your network is using a HF mirror that drops some files | `hf auth login` with a read-scope token; if behind a mirror, `export HF_ENDPOINT=https://hf-mirror.com`. The WorldScore parquet must be fetched from `huggingface.co` directly (mirrors drop files). |
| Out-of-disk during fetch | `~/.cache/videvalkit` on a small partition | `export VIDEVALKIT_CACHE_ROOT=/data/videvalkit_cache` before fetching; full smoke + ckpt fetch is about 130 GB |
| OOM with LLaVA-1.6-34B on < 80 GB GPU | T2V-CompBench paper-mode requires 80 GB | switch to `--mode toolkit --judge gemma-4-31b-local` (paper-Delta widens) |
| Judge endpoint not reachable | vLLM not running, or `--served-model-name` does not match the registry's `model` field | `curl http://host:port/v1/models | jq` to see the actual name; align the registry or the launch flag |
| `cv2` reads wrong frames on H.264 paper videos | Sparse-keyframe seek bug | fixed in `utils/video.py:extract_frames` (sequential `cap.read()` with a `wanted_set` instead of `cap.set(CAP_PROP_POS_FRAMES)`). If you see duplicate frames, ensure you are on toolkit version >= 0.0.1 post-2026-05-18 |
| WorldJen phase A never runs | A prebuilt `vqa_questions_50prompts.jsonl` is in the workspace | Phase A is silently skipped if the file exists. To force phase A, delete the file or run with `--no-prebuilt-vqa` |
| `Gemma broken pipe / RemoteProtocolError` under concurrency | bursty concurrency against vLLM | set `--max-concurrency 2` for any Gemma-targeted bench; stagger benchmarks that share Gemma |
| `detectron2` build fails on install | `nvcc --version` does not match `torch.version.cuda` | install the prebuilt detectron2 wheel for your CUDA |
| `flash-attn` import ABI error | wheel does not match torch | `pip install flash-attn==2.5.8 --no-build-isolation` |
| VBench v1 `dynamic_degree` flickers between runs | RAFT GPU non-determinism | expected; widen tolerance to +/-0.025 for that dim |
| `api_logs/` reaches several GB | high call volume | `scripts/clear_cache.py --api-logs --older-than 30d` |
| Eval exits immediately with "no videos found" | layout mismatch | confirm the per-bench layout in section 11; `videobench` and `t2vcompbench` use dim-organized layouts |

For deeper triage when scores look wrong, follow the debug ladder in TEST_MANUAL section 5.

---

## 14. License and citation

The toolkit (this codebase) is **Apache-2.0**. Each upstream we adapt has its own license, surfaced under `LICENSES/` at the repo root.

Notable upstream licenses:

| Upstream | License |
|---|---|
| VBench, VBench-2.0 | Apache-2.0 |
| Video-Bench, WorldJen | respective papers' licenses |
| T2V-CompBench (LLaVA-1.6-34B subprocess path) | research-only |
| DROID-SLAM, SEA-RAFT, GroundingDINO, SAM | respective open licenses |
| WorldScore checkpoints | derivative use under each backbone's terms |

### Citing the toolkit

```bibtex
@software{videogenevalkit2026,
  title  = {videogenevalkit: A unified evaluation toolkit for text-to-video generation},
  author = {Liu, Ning and contributors},
  year   = {2026},
  url    = {https://github.com/videogenevalkit/videogenevalkit}
}
```

### Citing the underlying benchmarks

When you publish numbers from a specific adapter, **cite the original benchmark paper** in addition to the toolkit. BibTeX entries for all six anchored benchmarks live in `docs/citations.bib`. Specifically:

- VBench v1: Huang et al., CVPR 2024
- VBench-2.0: Zheng et al., CVPR 2025
- Video-Bench: Han et al., CVPR 2025 (arXiv:2504.04907)
- WorldJen: WorldJen team, project page on github.com/moonmath-ai
- WorldScore: Duan et al.
- T2V-CompBench: Sun et al., ECCV 2024

---

> End of user manual. For paper-Delta validation results see [TEST_MANUAL.md](TEST_MANUAL.md); for the toolkit's internal architecture see [DEV_MANUAL.md](DEV_MANUAL.md).
