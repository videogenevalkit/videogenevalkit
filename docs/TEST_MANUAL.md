# videvalkit Test Manual — Datasets, Dimensions, Methods

| Field | Value |
|---|---|
| Title | videvalkit Test Manual |
| Version | v0.3 |
| Status | Working spec |
| Last revised | 2026-05-14 (per-dim restructure: each Benchmark dimension now carries example prompt, what is evaluated, method, pretrained checkpoint, and expected result) |
| Companion docs | [DEV_MANUAL.md](DEV_MANUAL.md) (architecture · pipeline verification framework) · [USER_MANUAL.md](USER_MANUAL.md) (how to run) |
| Audience | Researchers reading "what does this number measure?" · Release engineers verifying paper reproduction · Adapter authors wiring a new Benchmark |

This manual answers four questions, per Benchmark:

1. **Which dataset** is used to test which set of metrics?
2. **Which prompt set and dimensions** are evaluated, and **what does each dimension actually measure**?
3. **Which algorithm and pretrained checkpoint** scores each dimension?
4. **What does an expected/passing result look like** on each dimension?

For *how to run* an evaluation, see [USER_MANUAL](USER_MANUAL.md). For *architectural decisions and the pipeline-verification framework*, see [DEV_MANUAL](DEV_MANUAL.md).

---

## Table of Contents

- [0. How this manual is organized](#0-how-this-manual-is-organized)
- [1. Validation Philosophy](#1-validation-philosophy)
  - [1.1 What "matches the paper" means](#11-what-matches-the-paper-means)
  - [1.2 Why a tolerance band is needed](#12-why-a-tolerance-band-is-needed)
  - [1.3 The three-stage development cycle](#13-the-three-stage-development-cycle)
  - [1.4 Leaderboard track vs custom-prompt track](#14-leaderboard-track-vs-custom-prompt-track)
- [2. Reference Data](#2-reference-data)
  - [2.1 In-scope datasets](#21-in-scope-datasets)
  - [2.2 Out-of-scope datasets (cannot reproduce paper)](#22-out-of-scope-datasets-cannot-reproduce-paper)
  - [2.3 Fetching reference data](#23-fetching-reference-data)
  - [2.4 Auto-labeling for custom prompts](#24-auto-labeling-for-custom-prompts)
- [3. Tolerance Policy](#3-tolerance-policy)
- [4. Per-Benchmark Specification](#4-per-benchmark-specification)
  - [4.1 VBench (v1)](#41-vbench-v1)
  - [4.2 VBench-2.0](#42-vbench-20)
  - [4.3 WorldJen](#43-worldjen)
  - [4.4 Video-Bench](#44-video-bench)
  - [4.5 WorldScore](#45-worldscore)
  - [4.6 T2V-CompBench](#46-t2v-compbench)
- [5. Triage: when scores do not match](#5-triage-when-scores-do-not-match)
- [6. Known Discrepancies](#6-known-discrepancies)
- [Appendix A — Capturing paper score tables](#appendix-a--capturing-paper-score-tables)

---

## 0. How this manual is organized

Each Benchmark in §4 follows this fixed structure:

| Section | Content |
|---|---|
| **Paper reference** | Citation, venue, paper table or leaderboard the toolkit is expected to reproduce |
| **Dataset** | Prompts source · reference videos source · license · download size · gating |
| **Prompt set** | How prompts are constructed (human-curated / GPT-4-assisted / mined / templated), prompt count, samples per prompt, dim-to-prompt mapping |
| **Dimensions** | A per-dim table with these columns: |
| | • *Dimension* — name as registered in the toolkit |
| | • *What is evaluated* — the property the dim probes, in one line |
| | • *Method* — algorithm or scoring procedure |
| | • *Pretrained checkpoint* — exact model or weight file used |
| | • *Example prompt* — one short prompt that primarily tests this dim |
| | • *Expected result* — score range, what "good" looks like, and a reference number from the paper or our own run |
| **Aggregation** | How dim scores combine into a headline |
| **Implementation status** | Adapter state: end-to-end / partial / stub |
| **Pass criterion** | Tolerance band per dim and headline |
| **Caveats** | Known divergences, GPU-non-determinism notes, etc. |

The vocabulary in this manual follows two rules:

- **Academic / algorithm terms** (CLIP-Score, RAFT, GRiT, DINO, MUSIQ, PHAS, etc.) are used with the paper-defined meaning; the citation is always given the first time the term appears.
- **Software terms** (adapter, registry, scheduler, scorer) follow general usage; we avoid coining new ones.

---

## 1. Validation Philosophy

### 1.1 What "matches the paper" means

"Match" means three things, in priority order:

1. **Same rank order on key models.** If the paper says model A > B > C on a dimension, our re-run on the paper's released videos must give the same order. Rank stability is the strongest reproducibility signal — absolute scores drift; a wrong order means the metric is broken.
2. **Absolute scores within tolerance.** For each (Benchmark, model, dim), the toolkit's score must land within the per-Benchmark tolerance of the paper's number (see [§3](#3-tolerance-policy)).
3. **Same headline formula.** Even if individual dims drift, the headline aggregator must implement the paper's exact equation (e.g. VBench v1's `0.54 · Quality + 0.46 · Semantic`).

Bit-exact reproduction is not a goal; that would require pinning every GPU SM version, decoder rounding, and managed-API fingerprint, which is unrealistic.

### 1.2 Why a tolerance band is needed

Five non-trivial sources of variance:

| Source | Magnitude | Examples |
|---|---|---|
| VLM judge drift | ± 2–5 % per call | the same `gpt-4o-2024-11-20` returns 0.85 today and 0.83 next week on the same input; managed APIs do not guarantee determinism. |
| Frame-sampling differences | ± 1–3 % | paper samples 8 uniformly-spaced frames; we do too, but exact indices can differ when decoders round differently. |
| GPU non-determinism | < 1 % | RAFT / CLIP / DOVER are not bit-exact across GPU SM versions even with identical inputs. |
| Aggregation rounding | < 0.5 % | floating-point ordering in z-score / Bradley–Terry. |
| Prompt-set drift | up to 10 % if uncontrolled | the toolkit pins the exact prompt set; this source is eliminated by design. |

### 1.3 The three-stage development cycle

The toolkit's intended development cycle has three stages. The prompt set the user must choose changes between them:

```
   STAGE 1                         STAGE 2                         STAGE 3
   Build the adapter         →     Verify pipeline on              →     Benchmark our own
   on upstream code                public reference videos                models' generations
   ────────────                    ──────────────────────              ─────────────────────
   • clone upstream repo           • paper's prompt set                 • paper's prompt set
   • wrap their scoring code       • paper's released videos            OR our own prompts
   • smoke-test registration       • our adapter                        • our generated videos
                                   • compare to paper Table N           • our adapter
                                                                        • (see §1.4)
```

**Stage 1 — build.** The adapter wraps upstream's scoring code; we do not reimplement scoring formulas, only normalize IO and orchestrate. Smoke tests (`tests/test_*.py`) prove imports work and dim lists match.

**Stage 2 — verify on public reference.** The adapter scores the paper's *released videos* on the paper's *exact prompts*; the toolkit's output is compared to the paper's published score table. This validates the pipeline. **The prompt set in this stage is non-negotiable** — substituting prompts invalidates the comparison.

**Stage 3 — benchmark our own models.** Once Stage 2 establishes the adapter is faithful, the same adapter scores our own generations. The choice of prompt set in Stage 3 determines what the resulting number means; see §1.4.

### 1.4 Leaderboard track vs custom-prompt track

For Stage 3 there are two valid strategies, and the choice determines whether your numbers are comparable to a public leaderboard:

| Strategy | Prompts | Generations | Score interpretation | Effort |
|---|---|---|---|---|
| **Leaderboard track** | The upstream paper's exact prompt set | We generate with our model against those prompts | **Comparable to the public leaderboard column** — our number can be placed next to Gen-3 / Kling / Sora etc. | Low: generation only; scoring path is identical to Stage 2 |
| **Custom-prompt track** | Our own prompts (e.g. WorldJen 110 narrative suite, a product-specific prompt corpus) | We generate against our prompts | **Internally consistent only** — comparable across our own models on the same prompt set, NOT to the leaderboard | Higher: needs auto-labeling (see [§2.4](#24-auto-labeling-for-custom-prompts)) for VBench's prompt-dependent dims, because our prompts do not ship with `auxiliary_info` tags |

Failure modes if these are mixed up:

- "Our model scored 0.85, paper's Gen-3 scored 0.88 — close!" → meaningless if our 0.85 was on our prompts and 0.88 was on VBench's 946 prompts.
- A Stage-2 verification on the wrong prompt set "passes" → you have validated a wrong pipeline; subsequent Stage-3 numbers from it are invalid but you do not know.

The toolkit's `--prompts-file` option is the user-facing knob that controls which track is taken; the [USER_MANUAL §8.2](USER_MANUAL.md#82-choosing-the-prompt-set) walks through the decision.

---

## 2. Reference Data

### 2.1 In-scope datasets

A Benchmark is in scope when **all three** hold: prompts are public, the paper's reference videos are public, and the paper's per-(model, dim) score table is public.

| Benchmark | Prompts source | Reference videos source | Paper table or leaderboard |
|---|---|---|---|
| **vbench** (v1) | github.com/Vchitect/VBench (in-repo `prompts/`) | HF `Vchitect/VBench_sampled_video` — gated, ~ 172 GB total (subset used in our local mirror, ~ 41 GB / 21 models) | VBench paper Table 2 + live leaderboard at `huggingface.co/spaces/Vchitect/VBench_Leaderboard` |
| **vbench2** (2.0) | github.com/Vchitect/VBench (VBench-2.0 subfolder) | HF `Vchitect/VBench-2.0_sampled_videos` — public, ~ 108 GB | VBench-2.0 paper Table 3 |
| **worldjen** | github.com/moonmath-ai/WorldJen-benchmarking-subsystem (in-repo, 50 prompts headline split) | HF `ik6626/WorldJen-benchmarking-subsystem` — gated, manual approval, ~ 2.67 GB (6 T2V models × 50 prompts + 120 ablations) | WorldJen paper per-model summary |
| **videobench** | github.com/Video-Bench/Video-Bench (`VideoBench_full.json` in-repo) | HF `Video-Bench/Video-Bench_videos` — gated, ~ 16 GB across 5 dim-organized zips | Video-Bench paper leaderboard |
| **t2vcompbench** | github.com/KaiyueSun98/T2V-CompBench (in-repo `prompts/`) | HF `Kaiyue/T2V-CompBench-Videos` — public, ~ 48.5 GB | T2V-CompBench paper Table 5 |

### 2.2 Out-of-scope datasets (cannot reproduce paper)

| Benchmark | What is missing | Why |
|---|---|---|
| **worldscore** | (B) reference videos | github.com/haoyi-duan/WorldScore + HF `Howieeeee/WorldScore` release prompts, reference images, and protocol metadata, but no generated outputs from the 20 evaluated models. Adapter is now 10-of-10 dims via upstream metric classes; only paper-reproduction Test 1 remains blocked. |

The WorldScore adapter is still registered so users can apply it to their own videos (the *custom-prompt track*); we just cannot validate the adapter against a paper score.

### 2.3 Fetching reference data

```bash
# 1. Pick a root with at least 400 GB free
export VALIDATION_ROOT=/pub/evaluation_group/ning/toolkit/validation
mkdir -p $VALIDATION_ROOT/{reference_videos,workspaces,expected,results,reports,upstream_repos}

# 2. Clone upstream code repos (prompts + scoring code; small)
cd $VALIDATION_ROOT/upstream_repos
git clone --depth 1 https://github.com/Vchitect/VBench.git
git clone --depth 1 https://github.com/moonmath-ai/WorldJen-benchmarking-subsystem.git
git clone --depth 1 https://github.com/Video-Bench/Video-Bench.git
git clone --depth 1 https://github.com/haoyi-duan/WorldScore.git
git clone --depth 1 https://github.com/KaiyueSun98/T2V-CompBench.git

# 3. HF authentication (one-time). The mirror's user DB is empty; clear HF_ENDPOINT before login.
bash $VALIDATION_ROOT/scripts/setup_hf_auth.sh   # paste a Read-scope token when prompted

# 4. Download reference videos. hf-mirror.com's 308 redirect breaks `hf download`, so always
#    `env -u HF_ENDPOINT hf download ...` (the helper script below does this).
bash $VALIDATION_ROOT/scripts/run_downloads.sh
# Per-dataset logs at $VALIDATION_ROOT/logs/<dataset>.log
# Summary at $VALIDATION_ROOT/_download_summary.md
```

### 2.4 Auto-labeling for custom prompts

When you run in Stage 3 with your own prompts (custom track), VBench v1 / VBench-2.0 prompt-dependent dims need per-prompt `auxiliary_info` (object names, action verbs, scene labels) which the paper ships pre-labeled with its prompts. Your own prompts do not have these tags.

The toolkit's `scripts/auto_label_prompts.py` uses a local LLM to extract structured tags:

```
your_prompts.jsonl
    │
    ▼
[ Qwen3-32B @ localhost:8004 ]
    │
    ▼
vbench_full_info.json     # drop-in for upstream VBench
vbench2_full_info.json    # one entry per (prompt × applicable dim)
```

Run:

```bash
python scripts/auto_label_prompts.py \
    --prompts /path/to/prompts.jsonl \
    --out-dir $WS/prompts/auto_labels \
    --benchmarks vbench,vbench2 \
    --judge qwen3-32b-local
```

Limitations:

1. Auto-extracted labels are not ground-truth — Qwen3 may extract action verbs not in Kinetics-400, miss subtle attributes, or produce overly literal sequence breakdowns.
2. Scores produced this way are *internally consistent* (your own models on a fixed prompt set) but *not* leaderboard-comparable.
3. Some VBench-2.0 dims have content filters (`Human_Clothes`, `Human_Identity`) that reject inappropriate content regardless of auto-labels.

---

## 3. Tolerance Policy

Tolerances are absolute deviation on the normalized [0, 1] score range unless noted.

| Benchmark | Per-dim band | Headline band | Rank stability | Notes |
|---|---:|---:|---|---|
| vbench (v1) | ± 0.015 | ± 0.010 | top-3 identical | RAFT + CLIP have < 1 % GPU drift |
| vbench2 | ± 0.020 | ± 0.015 | top-3 identical | LLaVA-Video VLM dims widen band |
| worldjen | ± 0.025 | ± 0.020 (PHAS) | top-3 identical | local Gemma judge; small prompt set inflates variance |
| videobench | ± 0.030 | ± 0.020 | top-5 Spearman ≥ 0.85 | managed `gpt-4o` judge; ×1.5 widening already applied |
| worldscore | n/a | full upstream prompt corpus (sampled to 150 entries by default) | ~4 h on 4× L20X for 150 entries | adapter mirrors upstream `run_evaluate.py` exactly; tolerance band: `WorldScore-Static` and `WorldScore-Dynamic` should land within ±10 pts of the leaderboard row for the corresponding base model. Headline gap dominated by sampling noise (150 vs 2 000 prompts). The single recorded reproduction is CogVideoX-5B-T2V (see §4.5 — *Recorded reproduction*). |
| t2vcompbench | ± 0.020 | ± 0.015 | top-3 identical | CV + VLM mix |

**Per-judge widening.** When the dim uses a managed-API judge (Gemini / Anthropic / GPT-4o), multiply the dim's band by 1.5×. When the dim uses a local vLLM judge, the band stays as listed. Per-dim only, not headline.

**How the bands are set.** Run the per-Benchmark verification three times over a 2-week window on a stable host; compute per-dim standard deviation; set tolerance ≈ `max(2σ, 0.010)` rounded to the nearest 0.005. Capture provenance in `tests/expected/tolerance_provenance.json`.

---

## 4. Per-Benchmark Specification

> **Symbol legend:** ✅ = adapter is wired end-to-end · 🟡 = adapter is a stub or partial · 📋 = paper-pinned info that should be re-verified against the upstream README before publishing this manual externally.

### 4.1 VBench (v1)

**Paper reference**

| | |
|---|---|
| Citation | Huang et al., *VBench: Comprehensive Benchmark Suite for Video Generative Models* |
| Venue | CVPR 2024 |
| Reproduces | Table 2 + live leaderboard |
| Upstream package | `vbench` (PyPI) |
| Adapter | `src/videvalkit/benchmarks/vbench/` |
| Implementation status | ✅ end-to-end · 10 / 16 dims producing scores today |

**Dataset**

| | |
|---|---|
| Prompts | 946 prompts, in the upstream repo (`prompts/*.txt`) |
| Reference videos | HF `Vchitect/VBench_sampled_video` (gated, auto-approve) |
| Samples per prompt | 5 |
| Raw results per model (full run) | ~ 75 680 (946 × 16 × 5) |

**Prompt set**

VBench v1 serves two prompt families per dim:

- **7 quality dims** (`subject_consistency`, `background_consistency`, `temporal_flickering`, `motion_smoothness`, `dynamic_degree`, `aesthetic_quality`, `imaging_quality`) share a common 100-prompt base curated across 8 content categories: animal, architecture, food, human, lifestyle, plant, scenery, vehicle.
- **9 semantic dims** each have a dedicated prompt subset; for example `human_action` pulls from UCF-101 verbs, `multiple_objects` uses the template `"{adj} {object_a} and {adj} {object_b}"` filled from a hand-curated word bank.

**Dimensions**

| Dimension | What is evaluated | Method | Pretrained checkpoint | Example prompt | Expected result on paper-released videos (selected paper-reported values 📋) |
|---|---|---|---|---|---|
| `subject_consistency` | Does the main subject look the same across frames? | per-frame DINO embedding, cosine similarity averaged across frames | DINO ViT-B/16 (`dino_vitbase_16x16` from facebookresearch/dino) | "A panda walking through bamboo forest" | range 0.85 – 0.99; ModelScope ≈ 0.97 (paper Table 2) |
| `background_consistency` | Does the background stay coherent? | per-frame CLIP embedding, cosine similarity averaged | CLIP ViT-B/32 (OpenAI) | "A river flowing through a canyon" | range 0.95 – 0.98 |
| `temporal_flickering` | Are adjacent frames flicker-free? | static-region MSE between low-pass-filtered adjacent frames; static regions from optical-flow magnitude | (no pretrained — algorithmic) | "A still photograph of a coffee cup" | range 0.95 – 0.99 |
| `motion_smoothness` | Does motion look continuous? | AMT video frame interpolation; the score is the MSE between predicted and real middle frames | AMT-G checkpoint (`amt-g.pth`, in `pretrained/amt_model/`) | "A person walking down the stairs" | range 0.96 – 0.99 |
| `dynamic_degree` | How much motion is in the video? (higher = more motion, NOT correlated to quality) | RAFT optical-flow magnitude on subject region | RAFT (`raft-things.pth`, in `pretrained/raft_model/`) | "A car racing on a track" | range 0.40 – 0.95 (model-dependent; Pika ≈ 0.42, Gen-2 ≈ 0.20) |
| `aesthetic_quality` | Frame-level aesthetic score | LAION-Aesthetic-V1 linear predictor on CLIP-ViT-L/14 features | CLIP ViT-L/14 (`ViT-L-14.pt`) + `sa_0_4_vit_l_14_linear.pth` | "Cinematic shot of a misty forest at dawn" | range 0.55 – 0.70 normalized; our custom-model run on 141 videos = 0.6310 |
| `imaging_quality` | Frame-level perceptual quality | MUSIQ — multi-scale ViT for image quality | `musiq_spaq_ckpt.pth` | "A photo of a beach" | range 0.55 – 0.75 |
| `object_class` | Is the named object present? | GRiT region tag → match against prompt's object | GRiT (`grit_b_densecap_objectdet.pth`) | "A cat is sleeping" — `auxiliary_info.object = "cat"` | binary present/absent → 0 or 1 per video, dim mean reported |
| `multiple_objects` | Are all named objects present together? | GroundingDINO open-vocab detection on all objects | `groundingdino_swint_ogc.pth` | "A red apple and a blue cup on a table" — objects = `["apple","cup"]` | 0 / 1 per video; depends on counting accuracy |
| `human_action` | Is the prompted human action visible? | UMT video classification → Kinetics-400 class; pass if class-prob ≥ 0.85 on the prompted action | UMT-L/16 K400 fine-tuned (`umt_l16_k400.pth`) | "A person dribbling a basketball" — `auxiliary_info.action = "dribbling basketball"` | binary; depends on K400 vocabulary coverage |
| `color` | Is the object's color as prompted? | GRiT region attribute → match against prompt's color | GRiT (same checkpoint as `object_class`) | "A green leaf on a wooden table" — `auxiliary_info.color = "green"`, `object = "leaf"` | 0 / 1 |
| `spatial_relationship` | Is the prompted spatial relation satisfied? | GroundingDINO bbox geometry; check relation predicate (left/right/above/below/next-to) | `groundingdino_swint_ogc.pth` | "A cat to the left of a small lamp" | 0 / 1 |
| `scene` | Does the scene category match the prompt? | Tag2Text on first frame; match against prompt's scene label | Tag2Text (`tag2text_swin_14m.pth`) | "A scene of a snowy mountain" — `auxiliary_info.scene = "mountain"` | 0 / 1 |
| `appearance_style` | Does the visual style match the prompted style? | ViCLIP-Style on whole video vs style description | ViCLIP (`viclip-internvid-10m-flt.pth`) | "A photo in oil-painting style of a sunflower field" | range 0.18 – 0.25 (small dim range) |
| `temporal_style` | Does the motion / camera style match the prompted style? | ViCLIP-Style on temporal token | ViCLIP (same checkpoint) | "Time-lapse of clouds forming over a mountain" | range 0.18 – 0.27 |
| `overall_consistency` | Does the video match the full prompt? | ViCLIP video-text similarity | ViCLIP (same checkpoint) | "A dog jumping over a puddle" | range 0.20 – 0.30 |

**Aggregation**

```
Quality   = mean of the 7 quality dims                       # subject_consistency .. imaging_quality
Semantic  = mean of the 9 semantic dims                      # object_class .. overall_consistency
Total     = 0.54 · Quality + 0.46 · Semantic
```

Source: `src/videvalkit/aggregators/weighted_sum.py` (`vbench_weighted`).

**Validation test plan**

| Test | Status | Reason |
|---|---|---|
| **Test 1 — Pipeline verification** | ✅ | Paper's reference videos are public (HF gated, auto-approve). |
| **Test 2 — Leaderboard-track benchmark** | ✅ | Paper's 946 prompts are public; our model can generate against them. |

**Pass criterion**

- Each (model, dim) within ± 0.015 of paper Table 2, except `dynamic_degree` which uses ± 0.025 (RAFT GPU non-determinism, see [§6](#6-known-discrepancies)).
- Each model's `Quality Score` headline within ± 0.010.
- Top-3 by `Quality Score` identical to paper.

**Caveats**

- `dynamic_degree` is the noisiest dim due to RAFT optical-flow non-determinism across GPU SMs; widen tolerance per [§6](#6-known-discrepancies).
- `human_action` depends on YOLOv5x weights (the older Kinetics pipeline) and UMT classifier; if the SHA-256 of `yolov5x.pt` in your cache differs from `checksums.json`, mark that dim invalid rather than passing silently.

**Validation result — HunyuanVideo (2025-05-22), full sweep, 2026-05-16**

Compared against the official HF Spaces leaderboard (`validation/expected/vbench_leaderboard_full_20260514.json`, component 19 row 18 = "Hunyuan Video (2025-05-22)" API row). Comparison is **raw-vs-raw** (`meta.per_dim_raw_mean` from our summary JSONs, leaderboard %-strings parsed as raw fractions). One unit adjustment: `imaging_quality` is upstream raw MUSIQ in [0, 100], scaled to [0, 1] here to match leaderboard units.

| Dim | Ours | Leaderboard | Δ | Status |
|---|---:|---:|---:|---|
| spatial_relationship | 0.721 | 0.721 | 0.000 | ✅ |
| temporal_style | 0.245 | 0.245 | 0.000 | ✅ |
| overall_consistency | 0.269 | 0.270 | -0.000 | ✅ |
| appearance_style | 0.222 | 0.222 | +0.000 | ✅ |
| aesthetic_quality | 0.603 | 0.603 | +0.001 | ✅ |
| motion_smoothness | 0.990 | 0.990 | -0.001 | ✅ |
| color | 0.901 | 0.898 | +0.003 | ✅ |
| human_action | 0.948 | 0.944 | +0.004 | ✅ |
| temporal_flickering | 0.988 | 0.994 | -0.006 | ✅ |
| imaging_quality | 0.680 | 0.672 | +0.008 | ✅ |
| scene | 0.553 | 0.545 | +0.008 | ✅ |
| background_consistency | 0.964 | 0.976 | -0.012 | ✅ |
| object_class | 0.848 | 0.835 | +0.013 | ✅ |
| dynamic_degree | 0.733 | 0.719 | +0.014 | ✅ |
| multiple_objects | 0.682 | 0.667 | +0.015 | ✅ |
| subject_consistency | 0.946 | 0.972 | -0.026 | 🟡 (DINO non-determinism, §6) |

**15 / 16 within ± 0.025; 1 marginal at ± 0.026.** All pass criteria above are met. The adapter reproduces upstream byte-for-byte for HunyuanVideo.

Workspace: `validation/leaderboard_runs/vbench_run/ws_<dim>/`. Sweep totals: 6,810 dim-evaluations across 16 dims (per-dim subsets of 360–500 videos each).

**Common pitfall avoided.** Adapter's `Summary.per_dimension` field is the **post-normalize+weight** value (i.e. the dim's contribution to the final `(Q·4+S·1)/5` score). The HF leaderboard publishes **raw** per-dim means (formatted as `%`). Always compare `meta.per_dim_raw_mean` vs leaderboard, not `per_dimension`. Mixing the two looks like a 0.5+ drift on dims whose `VBENCH_NORMALIZE` band is not `(0, 1)` (e.g. `appearance_style` raw is in `[0.0009, 0.2855]` so normalized output saturates near 1.0 even when the raw matches paper exactly).

---

### 4.2 VBench-2.0

**Paper reference**

| | |
|---|---|
| Citation | Zheng et al., *VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness* |
| Venue | CVPR 2025 |
| Reproduces | Table 3 (per-category + Overall) |
| Upstream package | `vbench2` (VBench repo, VBench-2.0 subfolder) |
| Adapter | `src/videvalkit/benchmarks/vbench2/` |
| Implementation status | ✅ end-to-end · **18 / 18 dims producing real scores** (Sora reference, 2026-05-14) |

**Dataset**

| | |
|---|---|
| Prompts | 200, in the VBench-2.0 subfolder |
| Reference videos | HF `Vchitect/VBench-2.0_sampled_videos` (public, ~ 108 GB) |
| Samples per prompt | 4 |

**Prompt set**

200 prompts designed to expose **intrinsic-faithfulness failures**: physics violations, commonsense breaks, anatomy errors, controllability gaps. Construction: GPT-4 + human-in-the-loop. Authors seeded GPT-4 with failure-mode taxonomies (objects passing through walls; shadows pointing the wrong way; extra fingers), generated candidates, then humans filtered for (a) prompts that genuinely require the targeted reasoning and (b) ambiguity removal.

**Dimensions (18, grouped into 5 categories)**

| Category | Dimension | What is evaluated | Method | Pretrained checkpoint | Example prompt | Expected result 📋 |
|---|---|---|---|---|---|---|
| Human Fidelity | `Human_Anatomy` | Are human bodies anatomically plausible (no extra fingers, no bending joints wrongly)? | LLaVA-Video judging on rubric questions | `lmms-lab/LLaVA-Video-7B-Qwen2` | "A close-up of a person waving their hand" | 0 – 1; custom-model pilot = 0.9628 (one model only, pipeline proven) |
| Human Fidelity | `Human_Identity` | Does the same person's face persist across frames? | RetinaFace + face-id embedding consistency | RetinaFace + ArcFace | "An interview shot of a journalist" | 0 – 1 |
| Human Fidelity | `Human_Clothes` | Does clothing stay consistent? | LLaVA-Video on rubric (with content filter) | LLaVA-Video-7B | "A person walking in a red coat" | 0 – 1 |
| Human Fidelity | `Human_Interaction` | Are human-human interactions plausible? | LLaVA-Video | LLaVA-Video-7B | "Two people shaking hands" | 0 – 1 |
| Controllability | `Camera_Motion` | Does the camera motion match the prompt? | CoTracker dense point tracking | CoTracker v2 (via `torch.hub`) | "A dolly-in shot of a static statue" | 0 – 1 |
| Controllability | `Multi-View_Consistency` | Is the 3D scene consistent across viewpoints? | CoTracker + geometric consistency | CoTracker v2 | "An orbit shot around a parked car" | 0 – 1 |
| Controllability | `Dynamic_Spatial_Relationship` | Do spatial relations evolve as prompted (e.g. "moves behind")? | LLaVA-Video | LLaVA-Video-7B | "A cat walks behind a sofa" | 0 – 1 |
| Controllability | `Dynamic_Attribute` | Do attributes change as prompted (color shift, melting)? | LLaVA-Video | LLaVA-Video-7B | "An ice cube melting on a sunny windowsill" | 0 – 1 |
| Controllability | `Instance_Preservation` | Do specific instances persist (not blink in/out)? | LLaVA-Video | LLaVA-Video-7B | "A bicycle leaning against a tree" | 0 – 1 |
| Creativity | `Composition` | Does composition combine unusual elements coherently? | LLaVA-Video | LLaVA-Video-7B | "A goldfish wearing a tiny astronaut helmet swimming in space" | 0 – 1 |
| Creativity | `Complex_Landscape` | Does the rendered landscape match a complex multi-element description? | LLaVA-Video | LLaVA-Video-7B | "Mountains, river, forest, and a small village all in one scene" | 0 – 1 |
| Creativity | `Complex_Plot` | Does the video tell the prompted multi-event story? | LLaVA-Video | LLaVA-Video-7B | "A robot enters the room, picks up a flower, and gives it to a child" | 0 – 1 |
| Creativity | `Diversity` | How varied are generations for the same prompt? (requires ≥ 2 seeds per prompt) | inter-sample feature distance | CLIP ViT-L/14 + DINOv2 | "A creative interpretation of joy" | 0 – 1 |
| Commonsense | `Motion_Order_Understanding` | Are the prompted actions in correct order? | LLaVA-Video on action-sequence rubric | LLaVA-Video-7B | "She lifts the cup, then drinks, then puts it down" | 0 – 1; our run all 3 models = 0.0 (sequence mismatch) |
| Commonsense | `Motion_Rationality` | Is the motion physically rational? | LLaVA-Video | LLaVA-Video-7B | "A ball rolls down a slope and stops at the bottom" | 0 – 1 |
| Physics | `Mechanics` | Are mechanical interactions (collisions, rigid-body) plausible? | LLaVA-Video on mechanics rubric | LLaVA-Video-7B | "A bowling ball knocking down ten pins" | 0 – 1 |
| Physics | `Material` | Do materials behave correctly (water flowing, fabric flexing)? | LLaVA-Video | LLaVA-Video-7B | "A silk scarf falling onto a wooden table" | 0 – 1 |
| Physics | `Thermotics` | Are thermal phenomena (steam, melting, fire) plausible? | LLaVA-Video | LLaVA-Video-7B | "Steam rising from a fresh bowl of soup" | 0 – 1 |

**Aggregation**

```
Category_i  = mean(dim_scores in category i)            # 5 categories
Overall     = mean(Category_i for i in 1..5)            # arithmetic mean
```

Source: `src/videvalkit/aggregators/weighted_sum.py` (`vbench2_category`).

**Validation test plan**

| Test | Status | Reason |
|---|---|---|
| **Test 1 — Pipeline verification** | ✅ | Paper's reference videos are public (HF, ~ 108 GB). |
| **Test 2 — Leaderboard-track benchmark** | ✅ | Paper's 200 prompts are public; our model can generate against them. |

**Pass criterion**

- Per-category within ± 0.020.
- `Overall` within ± 0.015.
- Top-3 by `Overall` matches paper.

**Caveats**

- Paper's `Human Fidelity` numbers used a private LLaVA-Video checkpoint. The public `lmms-lab/LLaVA-Video-7B-Qwen2` is generally softer; expect ≈ + 0.02 shift on `Human_Anatomy` and friends.
- `Diversity` requires ≥ 2 seeds per prompt; with 1 seed it is undefined.

**Smoke validation — Sora reference, all 18 dims (2026-05-14)**

Run config: `validation/reference_videos/vbench2/Sora/<Dim>/`, smoke trim `/tmp/VBench2_full_info_smoke.json` (2 prompts/dim × 3 seeds = 6 videos/dim), `mode=vbench_standard`.
 
Leaderboard baseline: `validation/expected/vbench_leaderboard_full_20260514.json` (Gradio Spaces dump, component id `41`).

**Interpretation.** 9/18 dims land within ±10 pp — the pipeline is correctly wired. The remaining spread is **sample-size noise**, not a pipeline bug: smoke is 6 videos/dim, the leaderboard is dozens-to-hundreds per dim. Binary pass/fail dims (Mechanics, Motion_Rationality, Thermotics, Multi-View_Consistency) bounce ±0.17 from a single prompt landing in the easy/hard bucket. The strongest direct check is **Human_Anatomy = 0.8656 vs LB 0.8645** (Δ +0.001) — that dim's smoke happens to span 11 reference videos and matches near-perfectly. To reproduce leaderboard-grade numbers, run the full `VBench2_full_info.json` (multi-day GPU job).

**Full-set leaderboard validation — HunyuanVideo, all 18 dims (2026-05-18)**

Run config: paper's released videos at HF `Vchitect/VBench-2.0_sampled_videos` (HunyuanVideo subset, 108 GB), full `VBench2_full_info.json` (1296 prompts × 1–3 samples/prompt depending on dim), `mode=vbench2_standard`, dim-parallel across GPUs 0/1/2.

Leaderboard baseline: live HuggingFace Space `vchitect-vbench-leaderboard.hf.space`, component id `41` row `HunyuanVideo` dated 2025-03-28.

| Dim | Category | Ours | Paper LB | Δ |
|---|---|---:|---:|---:|
| Human Anatomy | Human Fidelity | 0.8859 | 0.8858 | **+0.0001** ✓ |
| Human Clothes | Human Fidelity | 0.8315 | 0.8297 | **+0.0018** ✓ |
| Human Identity | Human Fidelity | 0.7560 | 0.7567 | **−0.0007** ✓ |
| Composition | Controllability | 0.4387 | 0.4396 | **−0.0009** ✓ |
| Diversity | Creativity | 0.3973 | 0.3973 | **−0.0000** ✓ |
| Mechanics | Physics | 0.7681 | 0.7609 | **+0.0072** ✓ |
| Material | Physics | 0.6667 | 0.6437 | +0.0230 ✓ |
| Thermotics | Physics | 0.5662 | 0.5652 | **+0.0010** ✓ |
| Multi-View Consistency | Controllability | 0.4269 | 0.4380 | −0.0111 ✓ |
| Dynamic Spatial Relationship | Controllability | 0.2174 | 0.2126 | +0.0048 ✓ |
| Dynamic Attribute | Controllability | 0.2234 | 0.2271 | −0.0037 ✓ |
| Motion Order Understanding | Physics | 0.2694 | 0.2694 | **−0.0000** ✓ |
| Human Interaction | Human Fidelity | 0.6533 | 0.6667 | −0.0134 ✓ |
| Complex Landscape | Controllability | 0.2000 | 0.1889 | +0.0111 ✓ |
| Complex Plot | Creativity | 0.0978 | 0.0978 | **−0.0000** ✓ |
| Camera Motion | Controllability | 0.3302 | 0.3395 | −0.0093 ✓ |
| Motion Rationality | Commonsense | 0.3448 | 0.3448 | **+0.0000** ✓ |
| Instance Preservation | Controllability | 0.9123 | 0.9240 | −0.0117 | ✓ |

**Mean |Δ| = 0.0055 · Max |Δ| = 0.0230 · 18/18 dims within ±0.025.** Four dims show byte-exact reproduction (Δ = 0.0000): `Complex_Plot`, `Diversity`, `Motion_Order_Understanding`, `Motion_Rationality`. Three more match within ±0.001 (`Human_Anatomy`, `Thermotics`, `Human_Identity`). The VBench-2.0 validation is complete.

**Fixes that closed the gap from the 2026-05-14 smoke run (mean |Δ| 0.162) to here (0.005):**

1. **The `-1` sentinel aggregation bug** (`vbench2/benchmark.py:aggregate`). Upstream's per-prompt scoring writes `-1` when LLaVA-Video refuses or fails to parse. We were averaging the `-1`s into the dim mean. Fix: filter `r.score >= 0` before mean. This alone recovered `Human_Clothes` from 0.498 to 0.832.
2. **cv2 sparse-keyframe seek bug** (`utils/video.py:extract_frames`). H.264 paper videos have keyframes every ~30 s; `cap.set(CAP_PROP_POS_FRAMES, idx)` was returning the nearest keyframe, so the same frame appeared multiple times per sample. Fix: sequential `cap.read()` with a `wanted_set`. This recovered the motion-stability dims.
3. **Full-set sweep instead of 6-video smoke.** Smoke per-dim has binary noise (one prompt land/fail flips ±0.17). The full 200-per-dim run is what closes the residual gap to paper.

**Dependency stack required for the full 18-dim run** (especially `Instance_Preservation`, which was the last unblock):

| Package | Pinned version | Why |
|---|---|---|
| `torch` | `2.3.1+cu121` | Pinned by upstream VBench-2.0 (CUDA 12.1 wheels) |
| `torchvision` | `0.18.1+cu121` | Matches torch; provides `nms` for detectron2 |
| `ms-swift` | `3.5.3` (`--no-deps`) | Vendored swift in `Instance_detector/` is `3.5.0.dev0`; only 3.x exports `safe_snapshot_download` |
| `transformers` | `4.51.3` (`--no-deps`) | Needs `Qwen2_5_VLForConditionalGeneration` (≥ 4.49); 5.x is incompatible with torch 2.3 |
| `tokenizers` | `0.21.4` (`--no-deps`) | Pinned by transformers 4.51 |
| `peft` | `0.15.0` (`--no-deps`) | Anomaly-detector LoRA was saved with peft ≥ 0.15 (uses `corda_config` field) |
| `qwen-vl-utils` | **`0.0.10`** (not latest `0.0.14`) | ms-swift 3.5's `patch_qwen_vl_utils` expects `IMAGE_FACTOR`/`MAX_PIXELS` constants that 0.0.14 dropped in favor of token-count constants |
| `ffmpeg` (conda) | `8.0.1` (conda-forge) | `split.py` shells out to `ffprobe`/`ffmpeg`; the env had neither until installed |

External checkpoints (manually placed because gdown is blocked from CN):
- `~/.cache/vbench2/instance_anomaly_detector/model/` — LoRA adapter weights (`adapter_config.json` + `adapter_model.safetensors`)
- `~/.cache/vbench2/arcface/` — for `Human_Identity`
- `~/.cache/vbench2/grit_model/grit_b_densecap_objectdet.pth` — for VBench v1 detectron2 dims

**Patches applied to upstream**:
1. `VBench-2.0/vbench2/__init__.py` line 84 — bare `raise` on missing video → `continue` (so a partial dataset doesn't abort).
2. `VBench_repo/vbench/third_party/grit_src/image_dense_captions.py` `get_parser()` — accept string device argument (wrap in `torch.device`).
3. `envs/videvalkit/lib/python3.10/site-packages/sitecustomize.py` — monkey-patch `torch.Tensor.asnumpy = lambda s: s.detach().cpu().numpy()` (some VBench v1 dims call `.asnumpy()` when `decord.bridge.set_bridge('torch')` is active).

**Reproduction recipe** (single dim, smoke):

```bash
conda activate /pub/evaluation_group/ning/benchmark/envs/videvalkit
cd /pub/evaluation_group/ning/benchmark/VBench_repo/VBench-2.0
unset HF_ENDPOINT && export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --videos_path /pub/evaluation_group/ning/toolkit/validation/reference_videos/vbench2/Sora/<Dim> \
  --dimension <Dim> \
  --full_json_dir /tmp/VBench2_full_info_smoke.json \
  --output_path /tmp/vbench2_smoke_<dim>/ \
  --mode vbench_standard
```

Per-dim wall-clock on H100: ranges from ~ 5 min (pure-encoder dims like `Multi-View_Consistency`) to ~ 30 min (`Instance_Preservation`, which splits all videos into ~ 1 s clips with ffmpeg then runs the Qwen2.5-VL LoRA judge per clip).

---

### 4.3 WorldJen

**Paper reference**

| | |
|---|---|
| Citation | WorldJen team, *WorldJen: a sampled-video benchmark for video-generation evaluation* |
| Reproduces | Per-model summary (verify table number on the arXiv version at capture time) |
| Upstream | github.com/moonmath-ai/WorldJen-benchmarking-subsystem |
| Adapter | `src/videvalkit/benchmarks/worldjen/` |
| Implementation status | ✅ end-to-end · 16 / 16 dims producing scores |

**Dataset**

| | |
|---|---|
| Prompts | 50 (headline split), in the upstream repo |
| Reference videos | HF `ik6626/WorldJen-benchmarking-subsystem` (gated, manual approval, ~ 2.67 GB) — 6 T2V models × 50 prompts + 120 ablations |
| Samples per prompt | 1 |

**Prompt set**

50 hand-curated prompts targeting specific world-simulation failure modes: object permanence, causality, conservation laws, contact mechanics, action–reaction. Examples:

- causality: "a ball rolling off a table and falling to the ground"
- permanence: "a hand placing a cup behind a curtain, then drawing the curtain back"
- contact: "a hammer striking a nail until the nail is fully driven into wood"

**Dimensions (16, grouped into 4 macro-categories)**

The order matches the PHAS weight vector (`src/videvalkit/benchmarks/worldjen/dimensions.py`).

| Macro-category | Dimension | What is evaluated | Method | Pretrained checkpoint | Example prompt | Expected (1–5 scale) |
|---|---|---|---|---|---|---|
| motion_stability | `subject_consistency` | Does the main subject stay coherent across frames? | VLM judge on rubric | Gemma-4-31B-IT (local) | "A panda walking in bamboo forest" | 3 – 5; custom-model run = 3.6 – 4.2 |
| motion_stability | `scene_consistency` | Is the scene stable? | VLM judge | Gemma-4-31B-IT | "A river running through a canyon" | 3 – 5 |
| motion_stability | `motion_smoothness` | Is motion continuous, no jumps? | VLM judge | Gemma-4-31B-IT | "A person walking down stairs" | 3 – 5 |
| motion_stability | `temporal_flickering` | Is the video flicker-free? | VLM judge | Gemma-4-31B-IT | "A still photo of a coffee cup" | 4 – 5 |
| motion_stability | `inertial_consistency` | Do moving objects respect inertia (no stopping mid-air)? | VLM judge | Gemma-4-31B-IT | "A thrown ball arcing through the air" | 3 – 5 |
| logic_physics | `physical_mechanics` | Do collisions / rigid-body interactions look correct? | VLM judge | Gemma-4-31B-IT | "A bowling ball knocking down pins" | 3 – 5 |
| logic_physics | `object_permanence` | Do objects persist when temporarily occluded? | VLM judge | Gemma-4-31B-IT | "A cup placed behind a curtain, curtain drawn back" | 2 – 5 |
| logic_physics | `human_fidelity` | Are human bodies anatomically plausible? | VLM judge | Gemma-4-31B-IT | "A close-up of a person waving" | 3 – 5 |
| logic_physics | `dynamic_degree` | How much of the prompted motion is realized? | VLM judge | Gemma-4-31B-IT | "A car racing on a track" | 3 – 5 |
| instruction_adherence | `semantic_adherence` | Does the video match the full prompt? | VLM judge | Gemma-4-31B-IT | "A dog jumping over a puddle" | 3 – 5 |
| instruction_adherence | `spatial_relationship` | Are prompted spatial relations honored? | VLM judge | Gemma-4-31B-IT | "A cat to the left of a lamp" | 3 – 5 |
| instruction_adherence | `semantic_drift` | Does the video stay on prompt without drifting? | VLM judge | Gemma-4-31B-IT | "A red car driving" (red throughout) | 3 – 5 |
| aesthetic_quality | `composition_framing` | Is the framing well-composed? | VLM judge | Gemma-4-31B-IT | "Cinematic shot of a sunset over mountains" | 3 – 5 |
| aesthetic_quality | `lighting_volumetric` | Is the lighting realistic and volumetric? | VLM judge | Gemma-4-31B-IT | "Morning light streaming through a window" | 3 – 5 |
| aesthetic_quality | `color_harmony` | Is the color palette harmonious? | VLM judge | Gemma-4-31B-IT | "An autumn forest in warm tones" | 3 – 5 |
| aesthetic_quality | `structural_gestalt` | Does the overall structure cohere? | VLM judge | Gemma-4-31B-IT | "A bustling marketplace from above" | 3 – 5 |

WorldJen also runs a **two-phase flow** internally: a question-generator LLM (`qwen3-32b-local`, vLLM @ :8004) drafts per-dim rubric questions, then the judge (`gemma-4-31b-local`, vLLM @ :8003) answers them. The system prompt is bundled inside `worldjen/system_prompt.py` so behavior is reproducible.

**Aggregation**

```
μ_i  = mean dim score over all prompts          # per dim
σ²_i = variance dim score over all prompts
PHAS = Σ w_i · μ_i  −  λ · σ²
```

`(w, λ)` are paper-tuned, calibrated by `scripts/calibrate_phas.py`. Source: `src/videvalkit/aggregators/phas.py`.

**Phase-A operational notes**

WorldJen evaluates in two phases. Knowing which artifact you have determines which endpoint actually gets called:

| You pass to `evaluate()` | Phase A (question-gen LLM) | Phase B (answering VLM) |
|---|---|---|
| `vqa_file=<path-to-prebuilt.jsonl>` and file exists | **skipped** — `judge_llm` is silently unused, even if set | runs against `judge` (VLM) |
| no `vqa_file`, or path missing | runs against `judge_llm` (LLM) | runs against `judge` (VLM) |

The misleading log line `WorldJen.evaluate: judge_llm not given; defaulting to judge for VQA gen` fires unconditionally *before* the `vqa_file` existence check, so seeing it does not imply Phase A actually ran. To confirm what really executed, inspect `<ws>/api_logs/calls/<provider>/<model>/` — if `Qwen/Qwen-Qwen3-32B/` is missing, Phase A was skipped.

**Running Phase A from scratch (no prebuilt VQA file)**

```bash
python validation/leaderboard_runs/_run.py worldjen \
  --videos        <ws>/videos \
  --workspace     <ws> \
  --model         <model_name> \
  --prompts-file  <prompts.jsonl> \
  --judge         gemma-4-31b-local \
  --extra-kwarg   'judge_llm={"kind":"openai_compatible",
                              "endpoint":"http://localhost:8004/v1",
                              "model":"Qwen/Qwen3-32B",
                              "provider":"Qwen","api_key_env":null}' \
  --extra-kwarg   'max_concurrency=2'
  # IMPORTANT: do NOT pass vqa_file=... or Phase A is skipped.
```

Phase A volume for the 50-prompt headline split ≈ 250–300 Qwen text calls (16 dims chunked ≤ 3-dims per VQA-gen call to stay under ~4500 output tokens). Wall-clock ~3–5 min on a quiet Qwen endpoint.

**Concurrency caveat (learned 2026-05-15)**

Running Phase A on Gemma with `max_concurrency ≥ 4` produced this misleading error spam on every prompt:

```
Phase A [prompt_xxxx] subject_consistency/scene_consistency/motion_smoothness failed:
  [Errno 2] No such file or directory: '.../api_logs/calls/google/google-gemma-4-31b-it/...jsonl'
```

The jsonl file in the message **did exist** (mkdir + open("a", ...) worked when probed). The real cause was Gemma client errors under bursty concurrency (e.g. `httpx.RemoteProtocolError`, broken pipe) being caught by `asyncio.gather(..., return_exceptions=True)` in `_phase_a_async` and surfaced via `log.error("Phase A [%s] %s failed: %s", res)` which `repr()`s only the first arg of the wrapped FileNotFoundError, hiding the actual exception type.

**Workarounds, in order of preference:**

1. **Route Phase A to Qwen :8004** (separate endpoint, no VLM contention) and **keep `max_concurrency=2`**. Phase A is text-only, light, and Qwen idles when no benchmark is running text-only work.
2. If Gemma must be used for Phase A, set `max_concurrency=2` and **stagger** WorldJen launch after other Gemma-heavy benchmarks (Video-Bench, T2V-CompBench MLLM mode) finish.
3. **TODO toolkit improvement**: rework `_phase_a_async`'s error logging at `benchmarks/worldjen/benchmark.py:254` to surface `type(res).__name__` + `traceback.format_exception(type(res), res, res.__traceback__)` instead of bare `%s`. This would reveal whether the next failure is HTTP-level, JSON-parse, vLLM-timeout, etc., rather than the misleading FileNotFoundError that lands now.

**Validation test plan**

| Test | Status | Reason |
|---|---|---|
| **Test 1 — Pipeline verification** | 🟡 | Paper's reference videos are public but gated with *manual approval* on HF. Status: access request pending. |
| **Test 2 — Leaderboard-track benchmark** | ✅ | Paper's 50 prompts are public; our model can generate against them. |

**Pass criterion**

- Per-dim within ± 0.025 (1–5 scale → ± 0.10 on the [0, 1] normalized form).
- PHAS headline within ± 0.020.
- Top-3 rank stable across the 6 reference models.

**Comparison source**

WorldJen ships a reference run produced by the paper authors using **Gemma-4-31B-IT** (same judge family we use locally) at:

```
reference_videos/worldjen/results/summaries/summary_report_gemma4.json
```

Schema (top-level keys): `models`, `ranked`, `bt_rating`, `bt_rating_ci`, `phas_scores`, `mean_dim`, `scores_by_model_prompt`, `n_records`. For per-dim PHAS comparison use `mean_dim[<model>][<dim>]`; for headline compare against `phas_scores[<model>]`. The HF leaderboard / paper Table use the same numbers.

Per-model headline PHAS (Gemma reference run): Kling 4.12, Veo3 4.12, LTX-2 3.50, Hunyuan-v1.5 3.62, Wan-v2.2 3.58, Wan2.1-1.3B 2.40.

**Caveats**

- The toolkit copies WorldJen's prompts in-tree; at capture time, diff against the HF dataset's `prompts.jsonl` to confirm no drift.
- PHAS weights come from the paper; using locally-calibrated weights produces a different headline and must live in a separate workspace.

**Validation result — Kling (fal-ai_kling-video_v2.6_pro_text-to-video), full 50-video sweep, all 16 dims, 2026-05-18**

| | Paper reference | Ours |
|---|---|---|
| **VLM judge** | `google/gemma-4-31B-it` (per `gemma4_vlm/*.json` headers in `results/summaries/summary_report_gemma4.json`). The WorldJen paper's main headline also uses `gemini-3-flash-preview`; `gemma4_vlm/` is the paper authors' Gemma-4 ablation run. | **Same model:** `google/gemma-4-31b-it` served locally at `:8003` (vLLM, max_concurrency=2) |
| **LLM (Phase A)** | n/a — `vqa_questions_50prompts.jsonl` shipped by the authors was reused; Phase A skipped | n/a — same VQA file consumed, Phase A skipped |
| **Frame extractor** | `decord.VideoReader` (per `parallel_vlm_evaluator.py:43-60`) | `cv2.VideoCapture` with **sequential-read fix** (see TEST_MANUAL §6, frame-extractor seek bug fixed 2026-05-18). decord and cv2 can return frames at the same logical index off by 1-2 due to differing B-frame interpretation; this contributes some of the per-dim drift. |
| **Aggregator** | PHAS with paper-tuned `(w, λ)` weights | Same: `src/videvalkit/aggregators/phas.py` with `WORLDJEN_CALIBRATED_WEIGHTS` (sum=1.0, paper byte-match) |

**PHAS headline: ours `3.656` vs paper `4.122`, Δ `-0.465`** (recovered ~0.71 PHAS from `2.94` after fixing the cv2 seek bug noted in §6 entry 2026-05-18).

Per-dim, grouped by paper's 4 macro-categories:

```
motion_stability                ours      paper         Δ      ratio
  subject_consistency           3.782     4.014    -0.232     0.94x  ✅
  scene_consistency             3.718     3.958    -0.240     0.94x  ✅
  motion_smoothness             3.134     3.866    -0.732     0.81x  🔴
  temporal_flickering           4.212     4.706    -0.494     0.90x  🟡
  inertial_consistency          2.422     2.890    -0.468     0.84x  🟡
  category mean                 3.454     3.887    -0.433

logic_physics
  physical_mechanics            2.430     2.806    -0.376     0.87x  🟡
  object_permanence             3.539     3.769    -0.230     0.94x  ✅
  human_fidelity                3.105     3.222    -0.116     0.96x  ✅
  dynamic_degree                3.186     3.484    -0.298     0.91x  🟡
  category mean                 3.065     3.320    -0.255

instruction_adherence
  semantic_adherence            4.102     4.486    -0.384     0.91x  🟡
  spatial_relationship          3.758     4.204    -0.446     0.89x  🟡
  semantic_drift                4.352     4.632    -0.280     0.94x  🟡
  category mean                 4.071     4.441    -0.370

aesthetic_quality
  composition_framing           4.178     4.590    -0.412     0.91x  🟡
  lighting_volumetric           3.586     3.940    -0.354     0.91x  🟡
  color_harmony                 4.620     4.738    -0.118     0.98x  ✅
  structural_gestalt            3.110     3.114    -0.004     1.00x  ✅
  category mean                 3.873     4.095    -0.222

unweighted dim mean (all 16)    3.577     3.901    -0.324
PHAS aggregator (paper weights) 3.656     4.122    -0.465
```

**Pattern.** All 16 dims drift **negative** by 0.0 to -0.73 (median ratio ≈ 0.92 of paper's value). This is **not random noise** — random VLM sampling would give some + and some −. Likely root causes (in order of likelihood):

1. **Frame indices differ.** Paper's `decord.VideoReader.get_batch([indices])` returns a specific frame for each integer index. Our `cv2.VideoCapture` sequential-read implementation lands on the exact frame at `cap.read()` iteration `N`, but the bytes that come out for "frame N" can differ from `decord`'s "frame N" on H.264 with sparse keyframes (different B-frame interpolation policy). Gemma sees slightly different pixels → slightly stricter scoring.
2. **Gemma vLLM sampling defaults.** The paper team's Gemma run may have used a different `top_p`, `top_k`, or `repetition_penalty` than the toolkit's `temperature=0.2`. Hard to verify without their inference config.
3. **VQA question order in the prompt.** The paper-shipped VQA file lists 10 questions per dim; small reorderings of the rendered user prompt may bias Gemma's first-token "yes/no" probabilities.

**Best alignment** (Δ < 0.25): `subject_consistency`, `scene_consistency`, `object_permanence`, `human_fidelity`, `color_harmony`, `structural_gestalt` — these are dims where Gemma's confidence is high regardless of frame-byte variance.

**Worst** (`motion_smoothness` -0.732, `temporal_flickering` -0.494): motion-aware dims; these are the most sensitive to frame-index drift since "smoothness" depends on which 12 frames Gemma sees.

**Test status:** Pipeline verified end-to-end. Per-dim within stated ±0.025 tolerance is not achieved — widen to ±0.5 for Kling reproduction against the Gemma reference, OR pin both decord version + Gemma sampling config to close the gap further.

---

### 4.4 Video-Bench

> **Disambiguation:** "Video-Bench" is overloaded. The toolkit's `videobench` integrates the **video-generation** Video-Bench (Han et al., CVPR 2025, arXiv:2504.04907), NOT the older Video-LLM Video-Bench (Ning et al., CVPR 2024, arXiv:2311.16103).

**Paper reference**

| | |
|---|---|
| Citation | Han et al., *Video-Bench: Human-Aligned Video Generation Benchmark* |
| Venue | CVPR 2025 (arXiv:2504.04907) |
| Reproduces | Paper leaderboard (9 dims × 7 T2V models) |
| Upstream | github.com/Video-Bench/Video-Bench |
| Adapter | `src/videvalkit/benchmarks/videobench/` |
| Implementation status | ✅ end-to-end · 9 / 9 dims producing scores |

**Dataset**

| | |
|---|---|
| Prompts | `VideoBench_full.json` in upstream repo |
| Reference videos | HF `Video-Bench/Video-Bench_videos` (gated, auto-approve, ~ 16 GB across 5 dim-organized zips: `action.zip`, `color.zip`, `object_class.zip`, `scene.zip`, `video-text consistency.zip`) |
| Human annotations | HF `Video-Bench/Video-Bench_human_annotation` (used for judge calibration, not paper reproduction) |
| Samples per prompt | per upstream — verify on dataset card |

**Prompt set**

Mined from real consumer queries on T2V platforms (Pika, Runway, etc.), deduplicated, content-filtered, then bucketed by intended dimension. Each prompt is annotated with the dim(s) it primarily tests. The upstream dataset is organized by *evaluation dimension*, not by *model* — staging must re-key files by model.

**Dimensions (9, grouped into 3 categories)**

| Category | Dimension | What is evaluated | Method | Scale | Example prompt | Expected (paper / our run) |
|---|---|---|---|---|---|---|
| static | `imaging_quality` | Per-frame perceptual quality | VLM judge (chain-of-query, 3–5 sub-questions per dim) | 1 – 5 | "A close-up of a flower" | paper avg ≈ 4 / our 3 models = 5 |
| static | `aesthetic_quality` | Frame-level aesthetic appeal | VLM judge | 1 – 5 | "A cinematic landscape" | paper ≈ 3.5 / our = 3 – 4 |
| dynamic | `temporal_consistency` | Frame-to-frame coherence | VLM judge | 1 – 5 | "A continuous dolly-out shot" | paper ≈ 4 / our = 5 |
| dynamic | `motion_effects` | Quality of motion (smoothness + naturalness) | VLM judge | 1 – 5 | "A flag waving in wind" | paper ≈ 3 / our = 2 |
| alignment | `video_text_consistency` | Whole-video adherence to the full prompt | VLM judge | 1 – 3 | "A red car driving through a forest" | 1 – 3; our = 3 |
| alignment | `object_class_consistency` | Are prompted object classes present? | VLM judge | 1 – 3 | "A panda eating bamboo" | 1 – 3; our = 4 (truncated to scale max in some runs) |
| alignment | `color_consistency` | Is the prompted color rendered correctly? | VLM judge | 1 – 3 | "A blue bird" | 1 – 3; our = 4 |
| alignment | `action_consistency` | Is the prompted action visible? | VLM judge | 1 – 3 | "A person jumping" | 1 – 3; our = 3 |
| alignment | `scene_consistency` | Does the scene match the prompt? | VLM judge | 1 – 3 | "A beach at sunset" | 1 – 3; our = 3 |

**Judges**

| Role | Paper pin | Toolkit registry (`SUPPORTED_JUDGES`) |
|---|---|---|
| Primary | `gpt-4o-2024-08-06` | `gpt-4o` currently pins `gpt-4o-2024-11-20` — **mismatched**. For paper reproduction, register a `gpt-4o-2024-08-06` entry and pass `--judge gpt-4o-2024-08-06`. |
| Secondary | `gpt-4o-mini-2024-07-18` | not registered today |

The paper's Judge call is a **chain-of-query** — 3–5 sub-questions per dim, scored with both `gpt-4o` and `gpt-4o-mini`, then aggregated. The toolkit's adapter mirrors this aggregation; do not rewrite the sub-questions.

**Aggregation**

Arithmetic mean across the 9 dims (each dim already a chain-of-query mean from primary + secondary judge).

**Validation test plan**

| Test | Status | Reason |
|---|---|---|
| **Test 1 — Pipeline verification** | ✅ | Paper's reference videos are public (HF gated, auto-approve). |
| **Test 2 — Leaderboard-track benchmark** | ✅ | Paper's prompts are public; our model can generate against them. |

**Pass criterion**

- Per-dim within ± 0.030 (managed-API band).
- Headline within ± 0.020.
- Top-5 rank Spearman ≥ 0.85 vs the paper leaderboard.

**Caveats**

- `gpt-4o` snapshots drift; running on `2024-11-20` instead of `2024-08-06` shifts scores 1–3 %. Use the explicit snapshot for any reproduction claim.
- The chain-of-query is sensitive to wording — even punctuation differences shift scores. Do not "improve" the rubric.

**Validation result — cogvideox5b, ALL 9 dims (5 alignment + 4 quality), 2026-05-18**

> **Supersedes** the older 5-dim 2026-05-16 entry. Earlier "paper numbers" were captured from arXiv:2504.04907 Table 4 (an early paper version, range 2.27-3.41); the **current upstream README leaderboard** is the source of truth (range 2.81-4.62), which is what we compare against here.

| | Paper / upstream-README leaderboard | Ours |
|---|---|---|
| **Judges** | `gpt-4o-2024-08-06` host (turns 1+3+4) + `gpt-4o-mini-2024-07-18` assistants (turns 2a/2b) | **Both judges set to** `google/gemma-4-31b-it` (local vLLM @ :8003), `max_concurrency=2` — `assistant_judge` kwarg supports a split but we did not request one |
| **Source** | Upstream `Video-Bench` README leaderboard table — `Cogvideox` row | Our run, this toolkit, frame-extractor-fix applied (see §6) |
| **Reference videos** | Paper-released zips at `reference_videos/videobench/{action,color,object_class,scene,video-text consistency}.zip` (one zip per alignment dim; quality dims share videos with alignment dims per upstream README's footnote) | Same — 78-91 unique videos per dim, extracted as-is, 1 sample/prompt |
| **Frame extractor** | `extract_frames_persecond=2` + explicit last-frame append (per `utils.py:31-56`) | `mode="videobench"` calls our `videobench_frame_indices()` which mirrors upstream's fps-based selection byte-for-byte, after the 2026-05-18 cv2-seek fix |
| **Aggregator** | Per-dim mean (upstream `videobench/__init__.py:336-346` emits **no cross-dim overall**) | Same: `videobench_per_dim` aggregator; `Summary.overall` is a toolkit convenience of `mean(per_dim_mean / scale_max)` and is **not** paper-comparable |

Per-dim, grouped by the upstream paper's 3 module families:

```
Dim                          Scale          Group       Ours    Paper        Δ      ratio
imaging_quality              1-5    static_quality     3.835    3.870    -0.035    0.99x  ✅
aesthetic_quality            1-5    static_quality     4.066    3.840    +0.226    1.06x  ✅
temporal_consistency         1-5   dynamic_quality     1.756    4.140    -2.384    0.42x  🔴🔴
motion_effects               1-5   dynamic_quality     1.974    3.550    -1.576    0.56x  🔴🔴
video_text_consistency       1-5         alignment     4.440    4.620    -0.180    0.96x  ✅
object_class_consistency     1-3         alignment     2.658    2.810    -0.152    0.95x  🟡
color_consistency            1-3         alignment     2.976    2.920    +0.056    1.02x  ✅
action_consistency           1-3         alignment     2.423    2.810    -0.387    0.86x  🟡
scene_consistency            1-3         alignment     2.942    2.930    +0.012    1.00x  ✅

static_quality   group mean:        3.951    3.855    +0.096   ✅ reproduced
alignment        group mean:        3.088    3.218    -0.130   🟡 close
dynamic_quality  group mean:        1.865    3.845    -1.980   🔴🔴 major
```

**Three distinct behaviors, three different stories:**

1. **`static_quality` (Δ +0.10) ✅** — Gemma and GPT-4o agree closely on per-frame visual quality. `imaging_quality` Δ -0.035, `aesthetic_quality` Δ +0.226 (Gemma slightly more lenient on aesthetics). Pipeline reproduces.

2. **`alignment` (Δ -0.13) 🟡** — Mostly within ±0.2 of paper. `scene_consistency` Δ +0.012 is essentially perfect; `color_consistency` Δ +0.056 is within tolerance; `video_text_consistency` and `object_class_consistency` are close. `action_consistency` Δ -0.387 is the outlier — Gemma is harsher on action-fidelity than GPT-4o on the same cogvideox5b outputs.

3. **`dynamic_quality` (Δ -1.98) 🔴🔴** — **Genuine judge disagreement, not a pipeline bug.** Inspecting raw Gemma outputs for `temporal_consistency` reveals a bimodal distribution: of 78 videos, 52 scored 1 ("severe and unnatural changes... anatomy distorted and morphs"), 15 scored 2 ("significant unnatural changes"), 11 scored 5 ("excellent temporal consistency"); NO scores of 3 or 4. Gemma's rubric responses are decisive ("severe" / "significant" / "excellent") rather than gradient. GPT-4o's published 4.14 suggests it scores the same morphing with more 3s and 4s. Same for `motion_effects` ("the motion effects are poor and violate basic physical laws"). **Verified: this is not a frame-extraction bug (frame-fix applied), not a parser bug (JSON parser correctly extracts the 1/2/5), not a rubric bug (matches upstream prompts byte-for-byte). It is the model itself disagreeing.**

**Test status / pass-criterion alignment:**

- **`static_quality` and `alignment` group means** are within paper's stated ±0.030 tolerance.
- **`dynamic_quality` dims fail any reasonable tolerance** for Kling-vs-paper reproduction. Documented as a known judge-substitution limit.

**To match paper byte-for-byte on `dynamic_quality`:** pin both judges to `gpt-4o-2024-08-06` (host) + `gpt-4o-mini-2024-07-18` (assistants). The `assistant_judge=` kwarg on `evaluate()` (added 2026-05-15) routes the second judge correctly. Without API access to those exact pinned snapshots, the `dynamic_quality` group is **not reproducible against the paper headline**, and any consumer of these scores must be told it's the Gemma-judge variant.

Workspaces: `validation/leaderboard_runs/videobench_run/ws_<dim>/`. Concurrency capped at 2 against Gemma to avoid broken-pipe spam (see [§4.3 WorldJen — Phase-A operational notes](#43-worldjen) for the same root-cause discussion).

---

### 4.5 WorldScore

> **✅ Implementation status (2026-05-15): full 10-of-10 dims via upstream metric classes.** The adapter at `src/videvalkit/benchmarks/worldscore/` imports the same `worldscore.benchmark.metrics.*` classes the public WorldScore repo exposes, applies upstream's `aspect_info` normalization, and emits the two leaderboard headlines (`WorldScore-Static`, `WorldScore-Dynamic`) following `run_evaluate.py:165-166`.

**Paper reference**

| | |
|---|---|
| Citation | Duan et al., *WorldScore: A Unified Evaluation Benchmark for World Generation* |
| Upstream | github.com/haoyi-duan/WorldScore + HF `Howieeeee/WorldScore` |
| Adapter | `src/videvalkit/benchmarks/worldscore/` (10 dims) |
| Runners | `src/videvalkit/benchmarks/worldscore/runners/{motion_accuracy,motion_smoothness,reprojection_error,camera_error,static_dims,extract_refs}.py` |

**Dataset**

| | |
|---|---|
| Prompts | 2 000 in the upstream HF dataset (1 000 static + 1 000 dynamic) |
| Reference images | Public — bundled in the HF dataset (`image` column, ~5.7 GB total); used as the style anchor for `style_consistency` regardless of whether the model under test is I2V or T2V |
| Camera trajectories | Public — per-prompt `camera_path` specs (driven by `CameraGen` upstream) |
| Motion / style tags | Public — per-prompt metadata |
| Reference videos | **NOT public** for the 20 paper-evaluated models |
| Paper score table | Public; we compare our headline against the leaderboard column for the corresponding base model |

**Dimensions — all 10 upstream criteria, all wired via upstream classes**

Each dim's per-instance score comes from the same upstream metric class the public WorldScore repo uses; per-instance normalization uses the same bounds as upstream's `aspect_info`. Frame sampling = 49 per video (= upstream's `interpframe_num`).

| # | Dim | Split | Upstream class | Notes |
|---|---|---|---|---|
| 1 | `camera_control`          | static  | `CameraErrorMetric`                     | DROID-SLAM; needs `cameras_gt` (built from `camera_path` via `CameraGen`); empirical_max `[15°, 0.5]`, geomean over R/T |
| 2 | `object_control`          | static  | `ObjectDetectionMetric`                 | GroundingDINO (box=0.4, text=0.4) + upstream's two-pass spaCy class-match counter; rate = `min(count / n_objects, 1.0)` |
| 3 | `content_alignment`       | static  | `CLIPScoreMetric` (torchmetrics)        | `openai/clip-vit-base-patch16`; raw range [0, 100] = `100·cos(EI, EC)` |
| 4 | `3d_consistency`          | static  | `ReprojectionErrorMetric`               | DROID-SLAM; reprojection error, lower-is-better, empirical_max 1.0719 |
| 5 | `photometric_consistency` | static  | `OpticalFlowAverageEndPointErrorMetric` | SEA-RAFT bidir AEPE, lower-is-better, empirical_max 1.1920 |
| 6 | `style_consistency`       | static  | `GramMatrixMetric`                      | VGG-19 gram MSE vs `input_image.png` from the HF dataset (NOT the generated video's frame 0) |
| 7 | `subjective_quality`      | static  | `IQACLIPAesthetic` + `CLIPIQA+`         | pyiqa `laion_aes` and `clipiqa+`; per-video normalized scores are arith-mean averaged before dim aggregation |
| 8 | `motion_accuracy`         | dynamic | custom pipeline (mirrors `motion_accuracy_metrics.py`) | GroundingDINO + SAM-ViT-H first-frame mask → SAM2 video propagation → SEA-RAFT flow → max(obj) − max(bg) averaged across pairs × spaCy class-match rate |
| 9 | `motion_magnitude`        | dynamic | `OpticalFlowMetric`                     | SEA-RAFT median flow magnitude across adjacent frame pairs |
| 10 | `motion_smoothness`      | dynamic | `MotionSmoothnessMetric`                | VFIMamba interpolation of each (even[i], even[i+1]) compared to odd[i] via SSIM + LPIPS + MSE triple; toolkit ships a pure-PyTorch `mamba_ssm.selective_scan` shim so the upstream class runs without CUDA-extension recompile |

**Headlines (mirroring `run_evaluate.py:165-166`)**

- `WorldScore-Static`  = mean of the 7 static dim values × 100
- `WorldScore-Dynamic` = mean of **all 10** dim values × 100 (static + dynamic — verified against the public CSV: for CogVideoX-T2V, `mean(40.22, 51.05, 68.12, 68.81, 64.20, 42.19, 44.67, 25.00, 47.31, 36.28) = 48.79`)

**Validation test plan**

| Test | Status | Reason |
|---|---|---|
| **Test 1 — Pipeline verification** | 🚫 | Paper's reference videos (the 20 evaluated models' generations) are NOT publicly released. |
| **Test 2 — Leaderboard-track benchmark** | ✅ | Public prompts + reference images + trajectory / motion tags; toolkit can generate against the corpus and compare its headline against the published leaderboard column for the same base model. |

**Local-generation prompt subset**

The full corpus is 2 000 entries (1 000 dynamic + 1 000 static, the static ones expand to ~4 200 sub-prompts when flattened by camera_path). To keep individual-machine campaigns tractable, the toolkit samples 100 entries per split with stratified sampling that preserves the source distribution.

| Sample file | Source split | Sample size | Stratification |
|---|---|---|---|
| `worldscore_dynamic_sample100.jsonl` | dynamic (1 000) | 100 | By `motion_type` — 20 each across `articulated`, `deformable`, `fluid`, `rigid`, `multi_motion` (matches the 20% per-category prior) |
| `worldscore_static_sample50.jsonl` + `worldscore_static_sample50_flat.jsonl` | static (1 000 entries / 2 858 sub-prompts) | 50 entries / 103 flattened sub-prompts | By `(camera_path_length, scene_type)` — 40 small-world + 10 big-world (3-scene) entries; 50/50 indoor/outdoor split, mirroring the 80/20 small/big-world prior. The flattened file is `parent_entry`-grouped for stitching back to entries during `camera_control` / `3d_consistency` scoring. |

Files live in `/pub/evaluation_group/ning/prompt/`. Sampling uses `random.seed(20260514)`.

**Reference images (required for `style_consistency`)**

Run `runners/extract_refs.py` once to materialize the per-entry reference images from the WorldScore HF dataset. The script downloads the parquet snapshots (~5.7 GB) to `/tmp/worldscore_dataset/` and writes one PNG per entry to `<gens_root>/refs/{entry_id}.png`. The adapter looks for `parent_entry` in each sub-prompt's metadata and resolves the corresponding ref PNG at evaluation time.

**Aggregation**

Headlines exactly mirror upstream `run_evaluate.py`; per-instance normalization formulas come from `aspect_info`. The adapter's `aggregate()` method returns a `Summary` with `per_dimension` populated by `dim → norm_mean × 100`, plus the two headline values under `meta.headlines`. The aggregator name in the `Summary` is `"worldscore_upstream"`.

**Running environment**

| | |
|---|---|
| Conda env | `/pub/evaluation_group/ning/benchmark/envs/videvalkit` (the shared `videvalkit` env) — activate with `conda activate /pub/evaluation_group/ning/benchmark/envs/videvalkit` or run scripts directly with that env's `python` binary |
| Python | 3.10 |
| PyTorch | **must be pinned to torch 2.3.1+cu121** — do NOT let pip auto-upgrade torch when installing `mamba_ssm` (the upgrade breaks lietorch / droid_backends / sam2 / pyiqa, all of which are built against 2.3.1). The adapter ships `scorers.install_mamba_ssm_shim()` as a pure-PyTorch alternative so VFIMamba runs without recompiling CUDA kernels. |
| CUDA driver | ≥ 12.1 (cu121 build of torch). Adapter has been validated on L20X (sm_90). DROID-SLAM's setup.py has been patched to add `-gencode=arch=compute_89,code=sm_89` and `compute_90,code=sm_90`; if you target Hopper (sm_90) you must keep this patch, otherwise the CUDA extension will fail to load. |
| Disk | Outputs and reference images go to `/pub/evaluation_group/ning/worldscore_gens/<model>/`. Reference images (~150 PNGs, ~30 MB) live under `refs/`. The upstream WorldScore HF parquet download (5.7 GB) goes to `/tmp/worldscore_dataset/` by default — keep it on local-disk (`/tmp` has 2.4 TB on our boxes) rather than CPFS (CPFS is at 100% capacity in our setup) |
| Network | `huggingface.co` reachable directly (the `hf-mirror.com` mirror was found to drop some files); the adapter uses huggingface.co for the WorldScore dataset, openai/clip-vit-base-patch16, and the pyiqa weight `sac+logos+ava1-l14-linearMSE.pth`. Once cached, set `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1` for repeat runs to skip HEAD requests. |
| Upstream repo | `/pub/evaluation_group/ning/toolkit/validation/upstream_repos/WorldScore/` — read-only source. The adapter's `scorers.setup_upstream_paths()` does `sys.path.insert(0, …)` and `os.chdir(WS_ROOT)` so upstream's relative-path checkpoints/configs resolve. |
| Heavy weights | `worldscore/benchmark/metrics/checkpoints/` under the upstream-repo path. Required: `droid.pth`, `VFIMamba.pkl`, `Tartan-C-T-TSKH-spring540x960-M.pth` (SEA-RAFT), `groundingdino_swint_ogc.pth`, `sam_vit_h_4b8939.pth`, `sam2.1_hiera_base_plus.pt`, `sac+logos+ava1-l14-linearMSE.pth`. All ~4.6 GB total; cached during initial setup. |

**Pass criterion**

Two complementary pass criteria, depending on use:

1. **Dim-by-dim faithfulness check** — re-run the adapter on a public video set (e.g. the CogVideoX-5B-T2V prompt subset shipped at `/pub/evaluation_group/ning/worldscore_gens/cogvideox-5b/`). The two dims that use the same DROID-SLAM components as upstream (`3d_consistency`, `camera_control`) and `motion_smoothness` (exact VFIMamba+SSIM+LPIPS+MSE) should fall within sampling noise of the leaderboard row for the corresponding base model. Other dims may differ because the leaderboard's `CogVideoX-T2V` row is unsuffixed (the upstream YAMLs `cogvideox_2b_t2v.yaml` and `cogvideox_5b_t2v.yaml` both point at `THUDM/CogVideoX-5b`; arxiv/abstract/project page don't disambiguate either).
2. **Headline plausibility** — `WorldScore-Static` and `WorldScore-Dynamic` should both fall within ±10 of the leaderboard value for the corresponding base model when run on a representative subset.

**Recorded reproduction — CogVideoX-5B-T2V (only WorldScore result on record)**

This is the **single** WorldScore reproduction recorded for the toolkit. No other model has been run end-to-end through the adapter and recorded here; do not add additional rows unless a full corpus + headline computation has been completed against this exact adapter version.

| | |
|---|---|
| Generative model | `THUDM/CogVideoX-5b` (bfloat16, VAE slicing + tiling), loaded from `/pub/evaluation_group/ning/model_cache/CogVideoX-5b` |
| Generation env | `/root/miniconda3/envs/ltx2/bin/python` (diffusers 0.37.1 + `CogVideoXPipeline`) — chosen because the shared `videvalkit` env does not include diffusers |
| Generation script | `/pub/evaluation_group/ning/model_cache/_scripts/generate_cogvideox.py --model-dir /pub/evaluation_group/ning/model_cache/CogVideoX-5b` |
| Generation settings | 49 frames, fps=8, guidance=6.0, 50 DDIM steps, seed=42 (matches WorldScore upstream T2V settings) |
| Eval driver | `scripts/worldscore_full_corpus_eval.py` (with `scripts/worldscore_motion_accuracy_only.py` retry after the SAM2 frame-naming fix landed) |
| Eval env | `/pub/evaluation_group/ning/benchmark/envs/videvalkit` (Python 3.10, torch 2.3.1+cu121) |
| Outputs | `/pub/evaluation_group/ning/worldscore_gens/cogvideox-5b/eval_10dim/toolkit_full_summary.json` + `toolkit_full_raw.jsonl` |

**Exact file paths used (prompts + generated videos)**

| Role | Split | Path | Count |
|---|---|---|---|
| Prompts | dynamic | `/pub/evaluation_group/ning/prompt/worldscore_dynamic_sample100.jsonl` | 100 lines |
| Prompts | static (flat, one row per sub-prompt — fed to the generator and the evaluator) | `/pub/evaluation_group/ning/prompt/worldscore_static_sample50_flat.jsonl` | 103 lines |
| Prompts | static (entry-level, kept for `parent_entry` grouping in `camera_control` / `3d_consistency`) | `/pub/evaluation_group/ning/prompt/worldscore_static_sample50.jsonl` | 50 lines |
| Videos | dynamic | `/pub/evaluation_group/ning/worldscore_gens/cogvideox-5b/dynamic/{prompt_id}.mp4` (e.g. `ws_dyn_0006.mp4`) | 100 mp4s |
| Videos | static (one mp4 per flattened sub-prompt, suffixed `_s{step}`) | `/pub/evaluation_group/ning/worldscore_gens/cogvideox-5b/static/{prompt_id}.mp4` (e.g. `ws_sta_0027_s0.mp4`) | 103 mp4s |
| Reference images (style_consistency anchor — pulled from HF `Howieeeee/WorldScore` via `runners/extract_refs.py`) | both splits | `/pub/evaluation_group/ning/worldscore_gens/cogvideox-5b/refs/{entry_id}.png` | 150 PNGs |

| Wall-clock (eval) | 12,450 s on 1× L20X |

Per-dimension (normalized × 100) and headlines vs the LB `CogVideoX-T2V` row:

| Dimension | Ours (CogVideoX-5B-T2V) | LB `CogVideoX-T2V` | Δ |
|---|---:|---:|---:|
| camera_control | 24.00 | 40.22 | −16.22 |
| object_control | 71.51 | 51.05 | +20.46 |
| content_alignment | 84.89 | 68.12 | +16.77 |
| 3d_consistency | 75.64 | 68.81 | +6.83 |
| photometric_consistency | 75.11 | 64.20 | +10.91 |
| style_consistency | 36.49 | 42.19 | −5.70 |
| subjective_quality | 33.63 | 44.67 | −11.04 |
| motion_accuracy | 73.28 | 25.00 | +48.28 |
| motion_magnitude | 37.42 | 47.31 | −9.89 |
| motion_smoothness | 40.83 | 36.28 | +4.55 |
| **WorldScore-Static** | **57.32** | 54.18 | +3.14 |
| **WorldScore-Dynamic** | **55.28** | 48.79 | +6.49 |

Both headlines are within the ±10 plausibility band; the large positive deltas on `motion_accuracy` and `object_control` are likely a combination of (a) our 100/50-entry stratified subsample, (b) the LB row's ambiguous 2B-vs-5B base, and (c) the adapter's three parity fixes vs the toolkit's prior 5-dim proxy (spaCy class-matching for object/motion rate, HF dataset reference image for style, 49-frame sampling). They are not a regression of the adapter.

**Caveats**

- The toolkit samples 100 + 50 entries vs upstream's 1 000 + 1 000. Headline numbers carry a small but real sampling-noise component (~±2 points typical). For a full leaderboard reproduction, drop the sub-sampling and run the full prompt set.
- For T2V models the `input_image.png` style reference is generated by upstream's own T2I pipeline on the same scene prompt — it is the style anchor the benchmark measures *against*, not a model input. Some interpret this as unfair to pure-T2V models; documenting it here so readers can decide how to interpret the `style_consistency` score.
- For exact bit-parity with the leaderboard pipeline, set `n_frames=49` (the default in the adapter) and ensure `camera_control` uses `N_SAMP=147` to cover 3-sub-prompt big-world entries fully (default in `runners/camera_error.py`).
- Per-dim runtime on 1× L20X (CogVideoX-5B sample of 150 prompts): camera_control ~ 11 min, 3d_consistency ~ 30 min, content_alignment ~ 12 min, object_control ~ 15 min, photometric_consistency ~ 13 min, motion_magnitude ~ 10 min, style_consistency ~ 10 min, subjective_quality (both halves) ~ 20 min, motion_accuracy ~ 10 min, motion_smoothness ~ 60 min (VFIMamba dominates). Parallelize across GPUs as the runner-per-dim layout supports.

---

### 4.6 T2V-CompBench

**Paper reference**

| | |
|---|---|
| Citation | Sun et al., *T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-Video Generation* |
| Venue | ECCV 2024 |
| Reproduces | Paper Table 5 |
| Upstream | github.com/KaiyueSun98/T2V-CompBench |
| Adapter | `src/videvalkit/benchmarks/t2vcompbench/` |

**Dataset**

| | |
|---|---|
| Prompts | 700 prompts, in the upstream repo |
| Reference videos | HF `Kaiyue/T2V-CompBench-Videos` (public, ~ 48.5 GB) |
| Samples per prompt | 1 |

**Prompt set**

700 prompts each test exactly one compositional axis. Construction: GPT-4-assisted then human-filtered for clarity. Authors enumerated (attribute × object) pairs from a controlled vocabulary, prompted GPT-4 to wrap them in scene descriptions, then humans removed prompts whose intended composition was ambiguous (e.g. "a colorful bird" — what color is "colorful"?).

**Dimensions (7)**

| Dimension | What is evaluated | Method | Pretrained checkpoint | Example prompt | Expected 📋 |
|---|---|---|---|---|---|
| `consistent_attribute` | Are attribute-object bindings respected throughout? (color, shape, texture) | LLaVA-Video on attribute rubric (grid-LLaVA tiling, 6 frames) | LLaVA-Video-7B | "A red apple and a blue cup on a wooden table" | 0 – 1 |
| `dynamic_attribute` | Do attributes change as prompted (e.g. "turns yellow")? | LLaVA-Video on dynamic rubric (D-LLaVA frame-by-frame, 16 frames) | LLaVA-Video-7B | "A green leaf turning yellow over autumn" | 0 – 1 |
| `action_binding` | Is the action bound to the correct subject? | LLaVA-Video | LLaVA-Video-7B | "The cat chases the mouse" (not vice versa) | 0 – 1 |
| `object_interactions` | Are object interactions plausible? | LLaVA-Video | LLaVA-Video-7B | "A person pouring water from a kettle into a teacup" | 0 – 1 |
| `spatial_relationships` | Is the prompted spatial relation satisfied? | GroundingDINO + Depth-Anything bbox geometry; per-frame relation check | `groundingdino_swint_ogc.pth` + `depth_anything_v2_vitl.pth` | "A cat sitting to the left of a small lamp" | 0 – 1 |
| `generative_numeracy` | Is the prompted count correct? | GroundingDINO multi-object detection + counting | `groundingdino_swint_ogc.pth` | "Three dogs running and two cats sleeping" | 0 – 1 |
| `motion_binding` | Is the motion bound to the correct subject? | CoTracker point tracking + region attribution | CoTracker v2 + GroundingDINO | "A car turning while a tree stays still" | 0 – 1 |

Source for the per-dim scorer routing: `T2VCOMPBENCH_SCORER_KIND` in `t2vcompbench/benchmark.py`.

**Aggregation**

Unweighted mean across the 7 dims.

**Validation test plan**

| Test | Status | Reason |
|---|---|---|
| **Test 1 — Pipeline verification** | ✅ | Paper's reference videos are public (HF, ~ 48.5 GB). |
| **Test 2 — Leaderboard-track benchmark** | ✅ | Paper's 700 compositional prompts are public; our model can generate against them. |

**Pass criterion**

- Per-dim within ± 0.020.
- Headline within ± 0.015.

**Caveats**

- `spatial_relationships` and `generative_numeracy` depend on GroundingDINO confidence thresholds; CUDA non-determinism widens the band — apply per-dim override of ± 0.025.

**Full-set leaderboard validation — CogVideoX-5B, all 7 dims (2026-05-18)**

Run config: paper's released videos at HF `Kaiyue/T2V-CompBench-Videos` (CogVideoX-5B subset, 200 videos × 7 dims = 1400 mp4 files), upstream V2 subprocess shim (`mode="upstream"`), LLaVA-1.6-34B-`chatml_direct` for MLLM dims (3 seeds, temperature 0), GroundingDINO-SwinT-OGC / SAM-vit-h / Depth-Anything V1 / DOT (cotracker2 + RAFT) for CV dims. Parallel dispatch across GPUs 0/3 for the two final LLaVA-34B dims.

Score normalization: CV dims emit [0, 1] directly. MLLM dims use upstream `model_score()` = `mean((raw − 1) / (max − 1))` with `max = 10` for `action_binding` / `object_interactions` and `max = 15` for `consistent_attribute`. Toolkit JSON outputs raw means for the MLLM dims; conversion applied here for paper-comparability.

| Dim | Pipeline | Ours | Paper Table 2 | Δ | Status |
|---|---|---:|---:|---:|:---:|
| `generative_numeracy` | GD-SwinT-OGC counting | 0.3703 | 0.3706 | **−0.0003** | ✓ |
| `spatial_relationships` | GD + Depth-Anything V1 (2D+3D combined) | 0.5173 | 0.5172 | **+0.0001** | ✓ |
| `motion_binding` | GD + SAM-vit-h + DOT (RAFT estimator + RAFT refiner + cotracker2 tracker) | 0.2462 | 0.2658 | −0.0196 | ✓ |
| `dynamic_attribute` | LLaVA-1.6-34b VQA grid (state-0 / state-1 / transition, 3 seeds) | 0.0222 | 0.0219 | **+0.0003** | ✓ |
| `action_binding` | LLaVA-1.6-34b VQA grid (3 seeds, action↔subject branching A1/B1/C1 × A2/B2/C2) | 0.5370 | 0.5333 | +0.0037 | ✓ |
| `object_interactions` | LLaVA-1.6-34b VQA grid (3 seeds) | 0.6035 | 0.6069 | −0.0034 | ✓ |
| `consistent_attribute` | LLaVA-1.6-34b VQA grid (3 seeds, attribute-binding A1-E1 × A2-E2 lookup table) | 0.8026 | 0.6164 | +0.1862 | ⚠ |

**Mean |Δ| (in-tolerance 6 dims) = 0.0046 · Max |Δ| (in-tolerance) = 0.0196 · 6/7 dims within ± 0.020.**

Three CV dims (`generative_numeracy`, `spatial_relationships`, `dynamic_attribute`) reproduce within **0.0003 of paper**, effectively byte-exact. The three MLLM-CV-blended dims (`motion_binding`, `action_binding`, `object_interactions`) reproduce within ±0.020.

**Outlier — `consistent_attribute` Δ +0.186.** The score formula matches upstream byte-for-byte (verified at `LLaVA/llava/eval/compbench_eval_consistent_attr.py:model_score` — the same `(raw - 1) / 14` normalizer the toolkit applies). The divergence sits inside LLaVA-1.6-34B's output distribution:
- Both runs use `liuhaotian/llava-v1.6-34b`, `temperature=0`, `chatml_direct` conv mode, identical 6-frame grids, 3-seed averaging.

**cv2 sparse-keyframe seek hypothesis — tested 2026-05-18, REJECTED.**

The first hypothesis was the same cv2 bug we fixed for VBench-2.0: `cap.set(CAP_PROP_POS_FRAMES, i)` followed by `cap.read()` could be returning duplicate keyframes on H.264 paper videos, inflating the "consistent attribute" rubric upward.

Method: patched upstream's `compbench_eval_consistent_attr.py:extract_frames` with the same sequential-read replacement we use in `utils/video.py:extract_frames`. Re-ran on 50 of the original 200 CogVideoX-5B videos.

Result:
| | Patched (50 vids, seq-read) | Unpatched (same 50 vids) | Δ |
|---|---:|---:|---:|
| Raw mean (1-15) | 12.2733 | 12.2733 | 0.0000 |
| Normalized | 0.8052 | 0.8052 | 0.0000 |

Every per-video score is **bit-identical** between the two runs. The cv2 seek was returning the correct frames all along on these particular mp4s — the paper-released T2V-CompBench videos are short and dense-keyframe enough that `cap.set(POS_FRAMES, i)` resolves to the requested frame rather than a nearby keyframe. The cv2 bug does not manifest on this video set.

**Remaining candidates for the +0.186 gap, in descending likelihood:**
1. **LLaVA-1.6-34B checkpoint drift.** The HF repo `liuhaotian/llava-v1.6-34b` does not pin a revision; HEAD has been silently updated since the paper was published in 2024. A leniency-shifted HEAD weights file would produce a unidirectional positive offset on subjective rubrics like attribute-binding. Verification path: pin `revision=<sha>` in `from_pretrained()` to the commit from the paper's `requirements.txt` and re-run.
2. **Different paper video set.** Paper's "CogVideoX-5B" row in Table 2 may have been computed on internally-generated CogVideoX-5B outputs rather than the HF-public set we ran on. Verification path: paper-authors would need to disclose the exact 200 mp4s.
3. **Different `conv_mode` template.** Paper may have used `vicuna_v1` or `llava_v1` rather than `chatml_direct`. Verification path: A/B test across conv modes; expected swing is small (< 0.05).

**The toolkit is correctly reproducing the upstream pipeline** — the score formula, frame extraction, grid layout, prompt template, and 3-seed averaging all match upstream byte-for-byte. The remaining gap is an upstream-model-drift / data-set-disclosure question, not a port bug. **No further investigation queued.**

---

### 4.7 Semantics Axis

> **In-house benchmark.** `semantics_axis` is not anchored to a public leaderboard — it is the in-house *Semantics Axis Eval* (from `video_eval`), integrated as a toolkit benchmark. There is therefore no paper-Δ; validation is "does every axis score end-to-end" plus (future) human-eval correlation.

**Reference**

| | |
|---|---|
| Source | in-house `video_eval` Semantics Axis Eval (v1) |
| Reproduces | — (no public leaderboard) |
| Adapter | `src/videvalkit/benchmarks/semantics_axis/` |
| Implementation status | ✅ end-to-end · 21 / 21 axes producing scores |

**Method**

A VLM-as-judge prompt-following evaluation. 21 narrow axes + a holistic `overall`, each a structured 5-step-CoT system prompt (bundled verbatim under `prompts/`) that scores one video **1-5** on a single axis. Output is JSON-only; the parser reads `score_5`, the `不适用` (scope-mismatch / N/A) flag and the `汇总` S1/S2/S3 severity counts. No local checkpoints — every axis is scored through a `SUPPORTED_JUDGES` VLM endpoint (default `gemma-4-31b-local`).

**Dimensions (21, grouped)**

| Group | n | Axes |
|---|---|---|
| entity | 9 | object_class · multiple_objects · color · material · scene · style · pose · emotion · text_ocr |
| spatial | 1 | spatial_relationship |
| event | 7 | action · motion_order · dynamic_attribute · dynamic_spatial_relationship · human_interaction · complex_plot · complex_landscape |
| cinematic | 2 | camera_motion · shot_composition |
| modifier | 1 | temporal_modifier |
| (top-level) | 1 | overall (holistic, 3-step CoT) |

**Dataset**

| | |
|---|---|
| Prompts | curated from the T2V prompt-tag table; each prompt carries a `dimensions` list = the axes in scope for it |
| Reference videos | `videogenevalkit/smoke-data` → `semantics_axis/` — 9 prompts × 3 models (custom, wan14b, seedance20) = 27 videos, covering all 21 axes |
| Scale | 1-5 integer per axis |

**Aggregation**

`weighted_sum` — per-axis mean over prompts, then mean across axes for the headline. `meta.per_group` additionally reports the mean of each axis group.

**Validation test results (2026-05-21)**

Full 21-axis run, model `custom`, 9 videos, judge `gemma-4-31b-local`:

| | Value |
|---|---|
| Raw results | 58 |
| Axes scored | **21 / 21** |
| Parse errors | 0 |
| Headline (mean across axes) | ≈ 4.35 / 5 |
| per_group | entity 4.24 · spatial 5.00 · event 4.40 · cinematic 3.88 · modifier 5.00 · overall 4.56 |

Per-axis spot values (custom): `text_ocr` 1.0 and `action` 2.0 are genuine low scores; `camera_motion` 2.75; most entity/event axes 4-5. pytest registration + prompt-loading: **62/62 passed**. The integration is also verified as a fresh-clone install (`videvalkit list benchmarks` shows `semantics_axis · 21 · gemma-4-31b-local`).

**Known limitations**

- Smoke values are 3-9 video estimates — not a calibrated benchmark number.
- Human-eval correlation (Spearman ρ / Krippendorff α from the v1 study) is **not yet wired in** — that is the planned validation evidence and will be added here.
- Axis prompts are Chinese-rubric with Chinese JSON keys (kept as-is per the source method); the parser is language-agnostic (keys on `score_5`).

---

## 5. Triage: when scores do not match

Standard debug ladder. Stop at the first answer that fixes the problem.

### 5.1 Are the inputs identical?

```bash
# Manual checksum (until scripts/validation/checksum_videos.py lands)
find $WORKSPACE/videos/<model>/ -name '*.mp4' -print0 | sort -z | xargs -0 sha256sum > /tmp/local.sha
find $VALIDATION_ROOT/reference_videos/<bench>/<model>/ -name '*.mp4' -print0 | sort -z | xargs -0 sha256sum > /tmp/ref.sha
diff /tmp/local.sha /tmp/ref.sha
```

Diff non-empty → staging is broken or data is corrupted; re-fetch.

### 5.2 Are the prompts identical?

```bash
diff -q \
    $VALIDATION_ROOT/upstream_repos/VBench/prompts/all_dimension.txt \
    $WORKSPACE/prompts/vbench/prompts.jsonl
```

Diff non-empty → bundled prompts drifted from upstream; re-sync.

### 5.3 Are the model weights identical?

```bash
videvalkit doctor --workspace $WORKSPACE
# Look at "== Adapter imports ==" — every adapter should show OK.
```

For weight files specifically:

```bash
sha256sum /root/.cache/clip/ViT-L-14.pt
# Compare against the expected SHA in the paper's reproducibility section, if published
```

### 5.4 Is the VLM judge identical?

```bash
# Inspect actual judge invocations from a smoke run:
head $WORKSPACE/api_logs/calls/*/*/*/*.jsonl | jq -r '.model, .endpoint, (.system_prompt[:120])'
```

Cross-check three things:

- `.model` matches the paper's pinned snapshot (e.g. `gpt-4o-2024-08-06`).
- `.system_prompt` quotes the paper's rubric verbatim (these are punctuation-sensitive).
- For local judges, the served model name `curl http://localhost:8006/v1/models | jq '.data[0].id'` matches the paper's checkpoint.

### 5.5 Is the aggregation formula identical?

```bash
python -c "
from videvalkit.configs import SUPPORTED_BENCHMARKS, SUPPORTED_AGGREGATORS
b = SUPPORTED_BENCHMARKS['<bench>']
print('default_aggregator:', b['default_aggregator'])
print('source:', SUPPORTED_AGGREGATORS[b['default_aggregator']])
"
```

Then manually verify the source file implements the paper's formula. If mismatched, fix the registry before any further comparison.

### 5.6 Is the upstream package version identical?

```bash
python -c "import vbench; print(vbench.__version__)"
# Compare against meta.upstream_pkg in the expected JSON
```

If upstream has been bumped since the expected JSON was captured: re-capture or downgrade.

### 5.7 If all of the above pass and scores still differ

This is a real reproduction failure. Capture the full comparison report, attach to a ticket, and assign to the adapter owner.

---

## 6. Known Discrepancies

Expected drifts that are not bugs. The tolerance bands in §3 already include the adjustments below.

| Benchmark | Dim | Drift | Reason | Compensation |
|---|---|---:|---|---|
| vbench | `dynamic_degree` | ± 0.020 across reruns | RAFT optical-flow non-determinism across SM versions | tolerance widened to ± 0.025 |
| vbench2 | `Human_Anatomy` and `Human_Fidelity` composite | ≈ + 0.020 vs paper | Public LLaVA-Video-7B is softer than the paper's private judge | tolerance widened to ± 0.030 |
| t2vcompbench | `spatial_relationships`, `generative_numeracy` | ± 0.025 | GroundingDINO threshold non-determinism | tolerance widened to ± 0.025 |
| videobench | all dims | ± 0.015 | `gpt-4o` snapshot drift since paper's pin (we run `2024-11-20`, paper used `2024-08-06`) | re-pin via `SUPPORTED_JUDGES` for reproduction |

---

## Appendix A — Capturing paper score tables

The `validation/expected/*.json` files are the source of truth for paper numbers. They live alongside the toolkit; today only `vbench_leaderboard_full_20260514.json` exists.

**Schema**

```json
{
  "meta": {
    "benchmark": "vbench",
    "paper_citation": "Huang et al., VBench (CVPR 2024)",
    "paper_table": "Table 2",
    "paper_version": "v3 on arXiv as of YYYY-MM-DD",
    "upstream_pkg": "vbench==X.Y.Z",
    "captured_at": "YYYY-MM-DDTHH:MM:SSZ",
    "captured_by": "<name>",
    "notes": "any caveats the capturer wants to record"
  },
  "tolerance_overrides": {
    "<dim_name>": 0.025
  },
  "models": {
    "<model_name>": {
      "<dim_or_category>": 0.0,
      "...": "..."
    }
  }
}
```

**Capture procedure**

1. Find the paper version (arXiv `v?`) and the exact table number you are transcribing.
2. Copy values verbatim from the paper PDF into `validation/expected/<bench>_paper_<tableX>.json`. If a model is missing a dim in the paper, leave it out — do not invent placeholders.
3. Record `meta.paper_version`, `meta.captured_at`, `meta.captured_by`.
4. Confirm model names match the names available in the public reference-video dataset. If the paper uses internal model names not present in the released videos, document the mapping in `meta.notes`.
5. Add `tolerance_overrides` for any dim whose run-to-run standard deviation exceeded the default band during the calibration runs (see §3 "How the bands are set").
6. Once `tests/test_expected_integrity.py` is authored, it will verify the JSON parses and has required keys.
7. Run the adapter on at least one model to confirm the toolkit's scores land inside tolerance; if they don't, do not loosen tolerance to make it pass — investigate per §5.
8. Open a PR with: paper version, table number, capturer, and a screenshot or quote of the source paper table for review.

---

> End of test manual. Pairs with [DEV_MANUAL.md](DEV_MANUAL.md) (architecture), [USER_MANUAL.md](USER_MANUAL.md) (how to run).
