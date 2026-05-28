# videvalkit Development Manual — Multi-Benchmark / Multi-VLM Unified Video Evaluation Toolkit

| Field | Content |
| ---- | ------- |
| Title | videvalkit Development Manual |
| Version | **v0.2** (aligned with v0.0.1 code) |
| Status | Working version; updates with the code |
| Created | 2026-05-11 |
| Last revised | **2026-05-13** (M0.5 adds 5 benchmarks · `BaseScorer` abstraction · shared env · quickstart / sequence / repro / cost sections · full English translation with consistent terminology) |
| Codename | videvalkit |
| Scope | Engineering reference for development, integration, and external distribution; shared context for the per-module detail docs |
| Audience | Toolkit developers · Benchmark integrators · VLM backend integrators · Downstream deployment/ops teams · End users |

> This is a standalone development manual. All developer-facing interface contracts, distribution flow, remote VLM port conventions, and end-user steps are written inline; per-module specifics (e.g. a benchmark's official score reproduction details or a VLM backend's tuning) live in sub-docs and are not duplicated here.

---

## Implementation Snapshot (v0.0.1, 2026-05-13)

| Dimension | Count | Registered |
|---|---:|---|
| Benchmarks (`SUPPORTED_BENCHMARKS`) | **9** | 6 anchored: vbench · vbench2 · videobench · worldjen · t2vcompbench · worldscore  ·  3 supplementary stubs: physics_iq · vbench_pp · v_reasonbench |
| VLM Judges (`SUPPORTED_JUDGES`) | **8** | gemma-4-31b-local · qwen3-32b-local · qwen3-vl-32b-local · local-llava-video-7b · gemini-3-flash · gemini-2.5-pro · claude-sonnet-4-6 · gpt-4o |
| Aggregators (`SUPPORTED_AGGREGATORS`) | **5** | weighted_sum · vbench_weighted · vbench2_category · phas · bt |
| Core abstractions (`core/`) | **3 + types** | `BaseBenchmark` · `BaseScorer` · `BaseAggregator` + `WorkspaceLayout` + pydantic types |
| CLI subcommands (`videvalkit`) | **5** | `list` · `doctor` · `eval` · `prepare-workspace` · `aggregate` (all production-ready as of 2026-05-13) |
| Test suite | **66+** | 17 core unit + 3 HTTP integration (stdlib fake server) + 46+ adapter integration (all 6 anchored benchmarks have smoke + e2e tests) |
| Conda env | **1 shared** | `envs/videvalkit.yaml` (used by all 6 anchored benchmarks); per-benchmark mode reachable via `CondaEnvDispatcher` |

> Use this table as "what v0.0.1 actually supports at a glance." Refresh before each release.

## 60-Second Quickstart

```bash
# 1) Use the existing env (or recreate from envs/videvalkit.yaml)
conda activate /pub/evaluation_group/ning/benchmark/envs/videvalkit

# 2) Bootstrap a workspace (symlinks the videos dir into ws/videos/{model}/)
videvalkit prepare-workspace \
  --workspace /data/ws/quickstart \
  --videos /data/my_videos        # contains {model_a}/, {model_b}/ ...

# 3) Run a single benchmark (VBench v1, no VLM needed)
videvalkit eval --bench vbench --workspace /data/ws/quickstart

# 4) Inspect results
cat /data/ws/quickstart/results/summary/vbench/<model>.json | jq

# 5) Cross-benchmark aggregation (after running >1 benchmark)
videvalkit aggregate --workspace /data/ws/quickstart
```

> **Status note:** v0.0.1 ships the full 5-command CLI (`list`, `doctor`, `eval`, `prepare-workspace`, `aggregate`) plus the Python entry `videvalkit.runner.run(...)`. Python entry-point example in [§9.5](#95-running-a-single-benchmark).

---

## Table of Contents

- [0. Reading Guide](#0-reading-guide)
- [1. Product Overview](#1-product-overview)
  - [1.1 Vision](#11-vision)
  - [1.2 Positioning](#12-positioning)
  - [1.3 Problems Solved](#13-problems-solved)
  - [1.4 Design Principles](#14-design-principles)
- [2. Users and Scenarios](#2-users-and-scenarios)
  - [2.1 Target Users](#21-target-users)
  - [2.2 Roles](#22-roles)
  - [2.3 Scenarios](#23-scenarios)
- [3. Core Object Model](#3-core-object-model)
  - [3.1 Object Chains](#31-object-chains)
  - [3.2 Object Definitions](#32-object-definitions)
  - [3.3 The Three Registries](#33-the-three-registries)
  - [3.4 Call Sequence](#34-call-sequence-what-videvalkit-eval-actually-does)
  - [3.5 Terminology Alignment Table](#35-terminology-alignment-table-read-first)
- [4. Module Overview (A–H)](#4-module-overview-ah-eight-modules)
- [5. Module Detail](#5-module-detail)
  - [5.1 Module A — Core Abstraction Layer](#51-module-a--core-abstraction-layer)
  - [5.2 Module B — Benchmark Adapters](#52-module-b--benchmark-adapters)
  - [5.3 Module C — VLM Judge Abstraction & API Integration](#53-module-c--vlm-judge-abstraction--api-integration)
  - [5.4 Module D — Scheduler and Env Isolation](#54-module-d--scheduler-and-env-isolation)
  - [5.5 Module E — Storage / Workspace / Logging](#55-module-e--storage--workspace--logging)
  - [5.6 Module F — Aggregation & Cross-Benchmark Reporting](#56-module-f--aggregation--cross-benchmark-reporting)
  - [5.7 Module G — CLI / Runner / Config](#57-module-g--cli--runner--config)
  - [5.8 Module H — Packaging, Distribution & Remote VLM Onboarding](#58-module-h--packaging-distribution--remote-vlm-onboarding)
- [6. Cross-Module Collaboration](#6-cross-module-collaboration)
- [7. Deployment Topologies and Port Plan](#7-deployment-topologies-and-port-plan)
  - [7.1 Three Topologies](#71-three-topologies)
  - [7.2 Remote VLM Port Convention (8001–8009)](#72-remote-vlm-port-convention-80018009)
  - [7.3 Health Checks, Concurrency, Rate Limits](#73-health-checks-concurrency-rate-limits)
  - [7.4 Failover and Replication](#74-failover-and-replication)
- [8. Runtime Environment & Dependency Management](#8-runtime-environment--dependency-management)
  - [8.1 Single Env vs Multi Env](#81-single-env-vs-multi-env)
  - [8.2 CUDA / torch Compatibility Matrix](#82-cuda--torch-compatibility-matrix)
  - [8.3 VLM Backend Version Matrix](#83-vlm-backend-version-matrix)
  - [8.4 Model Weights and Local Loading (Offline)](#84-model-weights-and-local-loading-offline)
- [9. User Manual](#9-user-manual)
  - [9.1 Installation](#91-installation)
  - [9.2 Pre-downloading Model Weights (Offline / Restricted Network)](#92-pre-downloading-model-weights-offline--restricted-network)
  - [9.3 Starting the VLM Backends](#93-starting-the-vlm-backends)
  - [9.4 Preparing a Workspace](#94-preparing-a-workspace)
  - [9.5 Running a Single Benchmark](#95-running-a-single-benchmark)
  - [9.6 Cross-Benchmark Aggregation](#96-cross-benchmark-aggregation)
  - [9.7 Resume and Retry](#97-resume-and-retry)
  - [9.8 Troubleshooting and `doctor`](#98-troubleshooting-and-doctor)
- [10. Non-Functional Requirements](#10-non-functional-requirements)
  - [10.7 Reproducibility](#107-reproducibility)
  - [10.8 Scale and Compute / Token Estimates](#108-scale-and-compute--token-estimates-v001-reference)
- [11. Milestones](#11-milestones)
  - [11.1.5 M0.5 — Five Supplementary Benchmarks](#1115-milestone-m05--five-supplementary-benchmarks-landed)
- [12. Open Questions](#12-open-questions)
- [13. Prompt Auto-Labeler](#13-prompt-auto-labeler-added-2026-05-13)
- [14. Standalone Metrics Module — `videvalkit.metrics`](#14-standalone-metrics-module--videvalkitmetrics-planned-2026-05-18) — planned
- [15. Open-Source Distribution](#15-open-source-distribution-planned-2026-05-18) — planned
- [16. User-Configurable VLM Endpoints + API Token Logging](#16-user-configurable-vlm-endpoints--api-token-logging-planned-2026-05-18) — planned

---

## 0. Reading Guide

The manual is organized as: positioning → core objects → module boundaries → module conventions → cross-module collaboration → deployment & distribution → user manual → milestones & open questions. The intent is to let you quickly understand:

- what videvalkit is and is not;
- what the core abstractions are (`Benchmark` / `Scorer` / `Aggregator` / `Workspace` / `Scheduler`) and how the three registries thread them together;
- what each of the eight modules (A–H) owns, and how to extend (add a Benchmark, a VLM backend, an Aggregator);
- when the toolkit is delivered to another team as "package + remote VLM service," what that team has to do;
- because of the wide version drift across upstream Benchmarks and VLMs, what developers and end users each need to know.

Suggested reading order: **§3 → §4 → §5** to understand the object model and extension points (developers); **§1 → §7 → §8 → §9** to understand deployment and usage (integrators and end users).

---

## 1. Product Overview

### 1.1 Vision

Generative-video quality evaluation today lives in many upstream repositories: VBench (v1) needs detectron2 + CUDA ≤ 12.1; VBench-2.0 needs CUDA 11.8 + flash-attn 2.7.2 + mmcv 2.2.0; Video-Bench has no packaged release and its metrics are scattered across the README; WorldJen has no official Python package and its two-phase flow demands a hand-written VLM Judge; T2VCompBench and WorldScore each carry their own input conventions. Each Benchmark has its own prompt format, its own video naming convention, its own output JSON shape, its own weighted aggregation formula. A researcher who integrates a new Benchmark has to re-learn a new workflow and re-write a new glue script.

videvalkit's goal is to provide **one unified entry point and one unified workspace layout** for video-generation model evaluation: one CLI, one workspace layout, one api_logs schema, one Python entry. You pick a Benchmark, pick a VLM Judge, pick an Aggregator, and you get back upstream-equivalent scores plus cross-Benchmark comparison.

**The 6 Benchmarks integrated in v0.0.1:** VBench (v1) / VBench-2.0 / Video-Bench / WorldJen are the 4 anchored at M0; T2VCompBench and WorldScore are the 2 added in M0.5 — adapter skeletons landed (`src/videvalkit/benchmarks/<name>/benchmark.py`), with upstream-score-reproduction fidelity annotated in each sub-README. This manual covers the toolkit-side contract; it does not replace each adapter's own implementation notes.

### 1.2 Positioning

videvalkit is positioned as a **unified evaluation toolkit for generative-video research teams**, summarized in three lines:

- **One config dict, one entry point.** Pick a Benchmark, point at a videos directory, get scores.
- **Adapter, not reimplementation.** Upstream pipelines are integrated through thin adapters; score reproduction defers to upstream repos. The toolkit owns orchestration, IO, logging, scheduling, and aggregation only.
- **Environment isolation.** Benchmarks historically have wildly different version pins (VBench v1 wants CUDA ≤ 12.1 + detectron2; VBench-2.0 wants CUDA 11.8 + flash-attn 2.7.2 + mmcv 2.2.0). The toolkit unifies the **default** path onto a single shared env `envs/videvalkit.yaml` (v0.0.1 default); when an upstream upgrade breaks the shared env, fall back to per-Benchmark env mode (`CondaEnvDispatcher` is ready; activation is just changing `SUPPORTED_BENCHMARKS[name]["env"]` to the per-env name). VLM backends sit behind HTTP/SDK calls and are fully decoupled from any Benchmark env — see [§7. Deployment](#7-deployment-topologies-and-port-plan).

Things the toolkit explicitly **does not** do:

- **No video generation.** `BaseT2VModel` is reserved in the abstraction layer but never implemented; videos are produced elsewhere and land in `videos/{model}/...` before they enter the toolkit.
- **No re-implementation of official metrics.** For example we do not re-compute VBench's RAFT-based motion smoothness — we wrap the upstream result.
- **No distributed compute beyond a single machine.** `EnvDispatcher` is local-only today; the interface is reserved so a future Ray/SSH layer can replace it without touching adapter code.

### 1.3 Problems Solved

**Fragmented evaluation entry points.** Every Benchmark ships its own CLI, its own prompt index, its own video naming convention. A team running 4 Benchmarks in parallel ends up maintaining 4 glue scripts and re-debugging each upstream upgrade. videvalkit normalizes video input to `videos/{model}/{prompt_id}-{idx}.mp4`, prompts to `prompts/{benchmark}/prompts.jsonl`, every (model, dim, prompt) output to a single JSON file, and all evaluations to a single CLI.

**VLM Judges that cannot be compared across Benchmarks.** Multiple Benchmarks need a VLM Judge (VBench-2.0, Video-Bench, WorldJen), but each upstream calls its own local script or SDK with its own prompt template and scoring semantics. videvalkit abstracts the VLM Judge into a single interface (sync + async) with three backends (OpenAI-compatible / Gemini SDK / Anthropic SDK) sharing one signature; the same local vLLM endpoint can serve three Benchmarks.

**High version-drift risk, poor reproducibility.** Upstream pinnings for torch/transformers are mutually incompatible; forcing them into one env almost certainly breaks. videvalkit's strategy is **per-Benchmark conda env + a lightweight core env dispatcher** (currently the per-bench mode is opt-in and the shared env is the default; both paths preserved). All API calls are persisted as **frozen-schema** jsonl files so every run can be replayed and audited offline against `api_logs/`.

**Distribution and onboarding cost.** The toolkit ultimately ships to other teams and runs on machines outside our control. Their GPU resources, network isolation, available VLM models, and ability to call external APIs (Gemini / Anthropic) all differ. videvalkit's "core package + selectable VLM endpoints" design lets integrators choose between local VLM deployment (ports 8001–8009 by convention), a corporate intranet proxy, or external APIs, without modifying toolkit code.

### 1.4 Design Principles

**Registry-driven.** Three Python dicts — `SUPPORTED_BENCHMARKS` / `SUPPORTED_JUDGES` / `SUPPORTED_AGGREGATORS` — are the toolkit's source of truth. Adding a Benchmark / VLM / Aggregator = one dict row + one class. No YAML plugin layer, no plugin loader, no dynamic discovery. This principle runs through the whole manual.

**Adapter-first.** Every Benchmark is a `BaseBenchmark` subclass that calls the upstream package's Python API; we do not re-derive score formulas. The toolkit owns only (a) input-layout normalization (toolkit video paths ↔ upstream-expected paths via symlink staging); (b) output normalization (each upstream JSON shape → `RawResult`); (c) cross-Benchmark orchestration (scheduler, api_logs, frame cache); (d) aggregation (PHAS, BT, cross-Benchmark z-score).

**Environment isolation.** Today all 6 Benchmarks share one `videvalkit` conda env (see `envs/videvalkit.yaml`), at the cost of having toolkit-side compatibility adjustments to the upstream pins (see [§8.2](#82-cuda--torch-compatibility-matrix)). When the pin conflicts become irreconcilable, we fall back to per-Benchmark env (`envs/vbench.yaml` etc. are retained), dispatched via `CondaEnvDispatcher`.

**Pluggable VLMs.** A Judge can be a local vLLM/SGLang/Ollama OpenAI-compatible endpoint (typically `localhost:800x/v1`) or an external OpenAI/Gemini/Anthropic API; the Benchmark adapter code is agnostic to this difference. Each endpoint is one row in `SUPPORTED_JUDGES` (kind / endpoint / model / provider / api_key_env).

**Workspace as single truth.** All input and output of one evaluation task lives under `$WORKSPACE_ROOT/`: raw videos, prompts, frame cache, model-weight cache, results/raw, results/summary, leaderboard export, api_logs. `results/raw/` is the resume primitive — each (model, dim, prompt) is one JSON file; if it already exists, skip.

**Minimal trusted surface.** `BaseBenchmark` has 4 methods: `list_prompts / list_required_videos / evaluate / aggregate`. A Scorer (the abstraction Judges and metric scorers share) exposes one method: `score(ctx) -> ScoreResult` (sync + async). `BaseAggregator` has one method: `aggregate(raw_results) -> Summary`. New integrators have a very narrow target.

---

## 2. Users and Scenarios

### 2.1 Target Users

videvalkit has four primary user classes:

- **Toolkit core developers** — maintain the core abstractions, Scheduler, Storage, Aggregator; integrate new Benchmarks / VLM backends; own packaging and release.
- **Benchmark integrators** — usually domain experts for a specific Benchmark, integrating a new evaluation suite into the toolkit; only need to care about `BaseBenchmark` subclassing.
- **VLM integrators** — either developers adding a new managed API (e.g. next-generation Gemini), or operators deploying a fleet of local vLLM endpoints.
- **End evaluators** — researchers who pick a Benchmark, model(s), and VLM Judge, run an evaluation, and need official-equivalent scores plus cross-Benchmark comparison.

An implicit fifth class: **downstream deployment teams.** The toolkit ships as a package; downstream teams run it on their own servers — their VLM endpoint deployment, network topology, and available API keys differ from ours. The manual must cover their path.

### 2.2 Roles

| Role | Responsibility | Typical activity | Interfaces of interest |
| ---- | -------------- | ---------------- | ---------------------- |
| **Core developer** | Maintain toolkit skeleton and abstractions | Scheduler, Storage, Aggregator, CLI, packaging | `core/`, `scheduler/`, `storage/`, `pyproject.toml` |
| **Benchmark integrator** | Add a new evaluation suite | Write a `BaseBenchmark` subclass; register in `SUPPORTED_BENCHMARKS` | `benchmarks/<name>/benchmark.py`, `configs/benchmarks.py` |
| **VLM integrator** | Add a new VLM backend or local endpoint | Register in `SUPPORTED_JUDGES`; write a new backend if a new protocol | `configs/judges.py`, `scorers/vlm_judge/` |
| **End evaluator** | Pick Benchmark + model + Judge; run scores | Produce leaderboard, cross-Benchmark report, per-sample analysis | CLI, Python API |
| **Downstream ops** | Deploy the toolkit on a remote machine | Start local VLM, configure ports, run doctor, keep endpoints alive | `envs/videvalkit.yaml`, [§7](#7-deployment-topologies-and-port-plan), [§9.2](#92-pre-downloading-model-weights-offline--restricted-network) |
| **AI agent** | Invoke the toolkit indirectly via CLI | Run evaluations on behalf of a user; summarize results | CLI (`list / doctor / eval / aggregate / prepare-workspace`) |

### 2.3 Scenarios

**Scenario 1 — Run a Benchmark's official scores; reproduce upstream.** User prepares `videos/{model}/...`, picks `vbench` and a few dimensions, expects scores equivalent to the official leaderboard. Modules involved: A, B, D, E, G.

**Scenario 2 — Run a Benchmark that requires a VLM Judge (VBench-2.0 / Video-Bench / WorldJen).** User starts a local vLLM endpoint (e.g. port 8003 for Gemma-3-31B), or fills `GEMINI_API_KEY` in their environment, then passes `--judge gemma-4-31b-local` to the CLI. The toolkit transparently routes via OpenAI-compatible HTTP or the google-genai SDK; every call lands in `api_logs/`. Modules: B, C, D, E, G, H.

**Scenario 3 — Compare multiple models across Benchmarks.** User has run vbench / vbench2 / videobench / worldjen in one workspace; running `videvalkit aggregate --workspace ...` z-score-normalizes the four Summaries and runs Bradley-Terry for a unified ranking. Modules: F, G.

**Scenario 4 — Ship the toolkit to a downstream team that runs it themselves.** The downstream gets a wheel + `envs/videvalkit.yaml` + a README. They create a conda env, start local VLM endpoints per [§9.2](#92-pre-downloading-model-weights-offline--restricted-network), run `videvalkit doctor` to verify health, then `videvalkit eval` on their videos. Modules: C, H, the user manual.

**Scenario 5 — Resume from a crashed long run.** A long evaluation crashed; restart the `eval` command; the toolkit skips already-completed (model, dim, prompt) triples based on the JSON files already on disk. Modules: B, E.

**Scenario 6 — Extend with a new VLM or Benchmark.** A core developer adds a new VLM backend (e.g. a new Doubao-VL release) or wires in a new Benchmark (e.g. MotionBench-Plus). Following [§5.2](#52-module-b--benchmark-adapters) / [§5.3](#53-module-c--vlm-judge-abstraction--api-integration), the whole task fits in one day.

---

## 3. Core Object Model

### 3.1 Object Chains

videvalkit's domain logic sits on two chains.

**Evaluation output chain: Video → Benchmark → RawResult → Summary → CrossReport.** A set of Videos (named `{model}/{prompt_id}-{idx}.mp4`) feeds into a Benchmark adapter's `evaluate()`, which iterates (model, dim, prompt) and calls one or more Scorers (a GPU metric Scorer or a VLM Judge), producing one RawResult per triple (one JSON file). The Benchmark's `aggregate()` collapses all RawResults for the same (Benchmark, model) into one Summary, written to `results/summary/`. Cross-Benchmark `combine_summaries()` then z-score-normalizes several Summaries and produces a CrossReport.

**Orchestration chain: CLI → Runner → BenchmarkRegistry → Scheduler → Workspace.** The CLI parses args and hands off to `runner.run()`; the runner looks up the Benchmark class from `SUPPORTED_BENCHMARKS`, decides whether GPU/VLM is needed, constructs a Scheduler and a Workspace, and hands the whole bundle to the Benchmark adapter. The adapter accesses all videos, prompts, API calls, and caches through the path API provided by `Workspace.layout`.

### 3.2 Object Definitions

**Benchmark.** The integration entity for an evaluation suite — a `BaseBenchmark` subclass. Responsible for: listing this Benchmark's prompts (`list_prompts`), listing which videos each model must produce (`list_required_videos`), running scoring (`evaluate`), and aggregating this Benchmark's score (`aggregate`). A Benchmark does not care whether a VLM is local or external, nor about scheduling — those concerns belong to the Judge and the Scheduler.

**Method / Model.** A video-generation model under evaluation, identified by a string name. The toolkit does not generate videos; a model is just a directory name — `videos/{model}/...`. Model is the lowest grouping and ranking unit in RawResult / Summary / CrossReport.

**Dimension.** One evaluation axis within a Benchmark, e.g. VBench's `subject_consistency / motion_smoothness / ...`, VBench-2.0's 5 categories, WorldJen's PHAS sub-items. Dimension is the second-level key on RawResult; when the user passes `--dimensions`, the toolkit runs only that subset.

**Prompt.** One test input in an evaluation suite. Each Benchmark keeps its own list in `prompts/{benchmark}/prompts.jsonl`. Each row has at least `prompt_id` and the prompt text; other fields are Benchmark-specific. `prompt_id` is the authoritative source for the prefix in `videos/{model}/{prompt_id}-{idx}.mp4`.

**Video.** One mp4 file at path `videos/{model}/{prompt_id}-{idx}.mp4`. The same prompt can be sampled multiple times (`{idx}`); whether to aggregate samples is up to the Benchmark.

**RawResult.** One scoring atom for (Benchmark, model, dim, prompt), serialized to one JSON file: `results/raw/{benchmark}/{model}/{dim}/{prompt_id}.json`. This is the resume primitive — already-present means skip.

**Summary.** The (Benchmark, model)-level aggregated result (a pydantic model), written by the Benchmark's `aggregate()` to `results/summary/{benchmark}/{model}.json`. Contains per-dim mean and variance, the Benchmark's own official top-line score, and any Benchmark-defined metadata.

**Scorer.** The minimum atom of evaluation. A `BaseScorer` subclass (`src/videvalkit/core/scorer.py`): takes a `ScoreContext(video, prompt, ...)`, returns a `ScoreResult(score, raw, meta)`. A Scorer is one level below a Benchmark — a Benchmark typically *composes* several Scorers (one per dimension). A Scorer declares its `kind: WorkloadKind` (`gpu_metric` / `vlm_judge_http` / `vlm_judge_api` / `cpu`), which the Scheduler uses to route the work to the right worker pool.

**Judge.** A special class of Scorer: the **VLM Judge**. Code lives under `src/videvalkit/scorers/vlm_judge/`, with three concrete backends: `OpenAICompatibleVLMJudge` (vLLM / Ollama / SGLang / OpenAI / DeepSeek / Together), `GeminiVLMJudge` (google-genai SDK), `AnthropicVLMJudge` (anthropic SDK). All Judges share the same sync + async signatures and are automatically wired with an `ApiCallLogger` and `FrameCache`. The word *Judge* is a documentation convention for "VLM-class Scorer"; there is no `BaseJudge` class in the code — a VLM Judge is a subclass of `BaseScorer`. The folder `scorers/metric/` (placeholder in v0.0.1) will host pure-CV Scorers like CLIP-Score / RAFT-Flow / aesthetic predictors.

**Aggregator.** A cross-model or cross-Benchmark aggregation algorithm. Registered today: `weighted_sum`, `vbench_weighted`, `vbench2_category`, `phas`, `bt` (Bradley-Terry, cross-model), `cross` (z-score + BT, cross-Benchmark). All in `SUPPORTED_AGGREGATORS`.

**Workspace / Layout.** Every directory under `$WORKSPACE_ROOT/` is managed by `Workspace.layout`, the single source of truth for IO. Adapters are not allowed to assemble paths on their own.

**Scheduler.** Execution dispatch for three workload kinds. `gpu_metric` → `GPUWorkerPool` (one subprocess per GPU); `vlm_judge_http` → `HTTPDispatcher` (asyncio + Semaphore + per-provider TokenBucket + backoff retry); `vlm_judge_api` → SDK-internal concurrency. Cross-env subprocess dispatch is `CondaEnvDispatcher`'s job (used only when falling back to per-Benchmark env mode).

### 3.3 The Three Registries

videvalkit's extensibility lives in three Python dicts. Modifying one row is how you declare a new capability.

**`SUPPORTED_BENCHMARKS`** (`src/videvalkit/configs/benchmarks.py`, 9 rows in v0.0.1):

```python
_SHARED_ENV = "videvalkit"   # shared env; see §1.2
SUPPORTED_BENCHMARKS = {
    "vbench":        dict(cls=VBenchBenchmark,        env=_SHARED_ENV,
                          needs_gpu=True,  needs_judge=False,
                          default_aggregator="vbench_weighted"),
    "vbench2":       dict(cls=VBench2Benchmark,       env=_SHARED_ENV,
                          needs_gpu=True,  needs_judge=True,
                          default_judge="local-llava-video-7b",
                          default_aggregator="vbench2_category"),
    "videobench":    dict(cls=VideoBenchBenchmark,    env=_SHARED_ENV,
                          needs_gpu=False, needs_judge=True,
                          default_judge="gpt-4o",
                          default_aggregator="weighted_sum"),
    "worldjen":      dict(cls=WorldJenBenchmark,      env=_SHARED_ENV,
                          needs_gpu=False, needs_judge=True,
                          default_judge="gemma-4-31b-local",
                          default_aggregator="phas"),
    "t2vcompbench":  dict(cls=T2VCompBenchBenchmark,  env=_SHARED_ENV,
                          needs_gpu=True,  needs_judge=True,    # GD/Depth/SAM + LLaVA
                          default_judge="local-llava-video-7b",
                          default_aggregator="weighted_sum"),
    "worldscore":    dict(cls=WorldScoreBenchmark,    env=_SHARED_ENV,
                          needs_gpu=True,  needs_judge=False,   # VGG16 + pyiqa CLIP
                          default_aggregator="weighted_sum"),
    # ─── Supplementary stubs (M0.5+); not all dims wired ───────────────
    "physics_iq":    dict(cls=PhysicsIQBenchmark,     env=_SHARED_ENV,
                          needs_gpu=True,  needs_judge=False,
                          default_aggregator="weighted_sum"),
    "vbench_pp":     dict(cls=VBenchPPBenchmark,      env=_SHARED_ENV,
                          needs_gpu=True,  needs_judge=True,
                          default_judge="gpt-4o",
                          default_aggregator="weighted_sum"),
    "v_reasonbench": dict(cls=VReasonBenchBenchmark,  env=_SHARED_ENV,
                          needs_gpu=False, needs_judge=True,
                          default_judge="gpt-4o",
                          default_aggregator="weighted_sum"),
}
```

Each row answers four questions: which adapter class to use, which conda env it runs in, whether GPU is needed, whether a VLM Judge is needed (with a default Judge). The runner assembles the execution plan from this.

**`SUPPORTED_JUDGES`** (`src/videvalkit/configs/judges.py`):

```python
SUPPORTED_JUDGES = {
    "gemma-4-31b-local": dict(kind="openai_compatible",
                              endpoint="http://localhost:8003/v1",
                              model="google/gemma-4-31b-it",
                              provider="google", api_key_env=None),
    "claude-sonnet-4-6": dict(kind="anthropic",
                              model="claude-sonnet-4-6",
                              provider="anthropic",
                              api_key_env="ANTHROPIC_API_KEY"),
    ...
}
```

Each row is a kwargs dict: `kind` decides which class to instantiate (openai_compatible / gemini / anthropic); the other fields are passed straight to its constructor. A local endpoint and a managed API are distinguished by the presence of `endpoint` and `api_key_env`; from the CLI's perspective they are equivalent. This is the foundation for [§7. Deployment](#7-deployment-topologies-and-port-plan).

**`SUPPORTED_AGGREGATORS`** (`src/videvalkit/configs/aggregators.py`): each row registers a `BaseAggregator` subclass plus its default parameters (e.g. PHAS weight table, BT bootstrap count). Benchmark adapters reference the registered implementations inside `aggregate()`.

> These three tables are the most-referenced objects in the developer docs. Every PR that modifies them must include unit tests in the corresponding module (B / C / F) and a one-line note on which Benchmark / VLM / upstream version the new row corresponds to.

### 3.4 Call Sequence — what `videvalkit eval` actually does

The diagram below anchors the static §3 objects to runtime; **each letter in brackets is the §4 module that owns the step**:

```
CLI: videvalkit eval --bench vbench2 --judge local-llava-video-7b ...
  │
  └─ runner.run(benchmark, videos, workspace, models, judge, ...)        [G]
      │
      ├─ SUPPORTED_BENCHMARKS["vbench2"]    ─── look up adapter class    [A,G]
      ├─ SUPPORTED_JUDGES["local-llava-video-7b"]                        [A,G]
      ├─ SUPPORTED_AGGREGATORS["vbench2_category"]                       [A,G]
      │
      ├─ Workspace(workspace_root)          ─── layout owns all paths    [E]
      ├─ build_judge(cfg, layout)           ─── factory wires logger+cache [C]
      │     └─ HTTPDispatcher prepares TokenBucket                        [D]
      │
      ├─ adapter = VBench2Benchmark(config)                              [B]
      │
      ├─ prompts = adapter.list_prompts(dimensions)                      [B]
      ├─ videos  = adapter.list_required_videos(prompts, models, layout) [B]
      │
      ├─ raw = adapter.evaluate(videos, layout, dimensions, judge, ...)  [B → C/D]
      │     │
      │     │  for each (model, dim, prompt):
      │     ├─ FrameCache.get(video) ─── decode + reuse                   [C/E]
      │     ├─ judge.score_async(prompt, frames)
      │     │     └─ HTTPDispatcher.submit → vLLM endpoint :8006         [D]
      │     │            └─ ApiCallLogger.write(jsonl)                   [E]
      │     ├─ RawResult → results/raw/{bench}/{m}/{d}/{pid}.json        [E]
      │     └─ if file already exists, skip (resume primitive)           [E]
      │
      ├─ summary = adapter.aggregate(raw)                                [B → F]
      │     └─ Summary → results/summary/{bench}/{model}.json            [E]
      │
      └─ return {"summary": ..., "raw_paths": [...], "workspace": ...}

# Cross-Benchmark aggregation is a separate short path:
# CLI: videvalkit aggregate --workspace ...
#   └─ combine_summaries(summaries) → CrossReport                        [F]
#       └─ leaderboard JSON → results/leaderboard/cross_benchmark.json   [E]
```

Keeping this diagram visible while reading §5 makes module dependencies (downward, rightward) obvious at a glance. **No module ever calls upward into G CLI/Runner** — that is the key invariant that lets the whole toolkit stay swappable.

### 3.5 Terminology Alignment Table — read first

| Doc term | Code symbol | One line | Example |
|---|---|---|---|
| Benchmark | `BaseBenchmark` (`core/benchmark.py`) | One adapter to an evaluation suite (VBench / WorldJen / ...) | `VBench2Benchmark` |
| Scorer | `BaseScorer` (`core/scorer.py`) | Smallest evaluation atom: (video, prompt) → score | `OpenAICompatibleVLMJudge` |
| Judge | (a subset of Scorer) `scorers/vlm_judge/*` | The "VLM as judge" kind of Scorer | same as above |
| Metric Scorer | (a subset of Scorer) `scorers/metric/*` | The "pure-CV model" kind of Scorer (CLIP / RAFT / aesthetic) | landing in M1 |
| Aggregator | `BaseAggregator` (`core/aggregator.py`) | Aggregation algorithm across prompts / models / Benchmarks | `BradleyTerryAggregator` |
| Workspace / Layout | `Workspace` + `WorkspaceLayout` (`storage/`) | Single source of truth for `$WORKSPACE_ROOT/` paths | `layout.raw_json(bench, model, dim, pid)` |
| Scheduler | `Scheduler` + `*Pool` (`scheduler/`) | Facade for dispatching the three workload kinds | `GPUWorkerPool` / `HTTPDispatcher` |
| RawResult | pydantic model (`core/types.py`) | (bench, model, dim, prompt) score atom → JSON | `results/raw/.../{pid}.json` |
| Summary | pydantic model (`core/types.py`) | (bench, model) aggregated scores | `results/summary/{bench}/{model}.json` |
| CrossReport | (inlined in aggregators in v0.0.1) | Cross-Benchmark report (z-score + BT) | output of `aggregate` CLI |
| WorkloadKind | `Literal["gpu_metric","vlm_judge_http","vlm_judge_api","cpu"]` | A Scorer self-declares its kind for routing | `scorer.kind = "vlm_judge_http"` |

> Key convention: **"Judge" is a documentation habit** (inherited from VLMEvalKit). The code has no `BaseJudge` class — it is a subclass of `BaseScorer`. Saying "add a judge" means "add a `vlm_judge_*` Scorer + add a row in `SUPPORTED_JUDGES`."

---

## 4. Module Overview (A–H, eight modules)

| ID  | Module | Core responsibility | Priority |
| --- | ------ | ------------------- | -------- |
| **A** | **Core Abstraction Layer** | `BaseBenchmark / BaseScorer / BaseAggregator / WorkspaceLayout / pydantic types` — the toolkit's de facto interface ("Judge" is a subclass of Scorer, not an independent base, see §3.5) | P0 |
| **B** | **Benchmark Adapters** | 4 anchored at M0 (VBench / VBench-2.0 / Video-Bench / WorldJen) + 2 added in M0.5 (T2VCompBench / WorldScore) = **6 total**; input-layout normalization, output-JSON normalization, upstream version coupling | P0 |
| **C** | **VLM Judge Abstraction & API Integration** | Three backends; FrameCache; ApiCallLogger; unified CLI surface for `--judge` whether it's a local endpoint or a managed API | P0 |
| **D** | **Scheduler and Env Isolation** | `GPUWorkerPool / HTTPDispatcher / CondaEnvDispatcher`, TokenBucket, backoff retry, cross-env subprocess dispatch; default in v0.0.1 is a *shared env*, per-Benchmark mode is opt-in | P0 |
| **E** | **Storage / Workspace / Logging** | `$WORKSPACE_ROOT` layout, results/raw resume primitive, frozen `api_logs` schema, frame_cache, model-weight cache | P0 |
| **F** | **Aggregation & Cross-Benchmark Reporting** | Per-Benchmark `aggregate()` defaults, Bradley-Terry, PHAS, cross-Benchmark z-score + BT, PHAS weight calibration | P1 |
| **G** | **CLI / Runner / Config** | `videvalkit list / doctor / eval / aggregate / prepare-workspace`, `runner.run()`, registry entry-points | P0 |
| **H** | **Packaging, Distribution & Remote VLM Onboarding** | wheel/sdist build, downstream install paths, remote VLM port convention, secrets handling, `doctor` output for external readers | P0 |

> Priority reflects "is this module load-bearing for the toolkit foundation"; it is **not** the same as the iteration order. See [§11. Milestones](#11-milestones) for ordering.

Module relationships at a glance:

```
                +----------------------+
                |    G CLI / Runner    |
                +-----+----------+-----+
                      |          |
                      v          v
   +------ A Core Abstractions ------+   D Scheduler
   |   Benchmark/Scorer/Aggregator  |   GPU + HTTP + EnvDispatcher
   +-----+------+------+-------------+        |
         |      |      |                      v
         v      v      v     E Storage / Workspace / api_logs
       B Adapters  C VLMs   F Aggregation
         |      |
         |      +--> Remote VLM endpoints (8001-8009)
         |             or Gemini / Anthropic managed APIs
         v
   Upstream Benchmark packages
   (VBench / VBench2 / VideoBench / WorldJen / ...)

H Packaging / Distribution / secrets    cuts across all modules; downstream-facing
```

---

## 5. Module Detail

> Each module follows: positioning → tier-1 requirements → tier-2 requirements → key objects & boundaries → cross-module collaboration → milestone scope → open questions. Tier-1 requirements are necessary for the module to exist; tier-2 are quality-of-life or extension.

### 5.1 Module A — Core Abstraction Layer

#### 5.1.1 Positioning

Module A is the toolkit's de facto interface layer. It defines the three base classes (`BaseBenchmark / BaseScorer / BaseAggregator`), the pydantic types (`RawResult / Summary / Score / VideoSpec / PromptSpec`), and the `WorkspaceLayout` path API. Every other module depends on A; A depends on nothing else. A's stability directly determines the v1.0 compatibility surface.

#### 5.1.2 Tier-1 Requirements

**A1. `BaseBenchmark` interface**

- Only 4 required methods: `list_prompts(self) -> Iterable[PromptItem]`, `list_required_videos(self, prompts, models, layout) -> list[VideoSpec]`, `evaluate(self, videos, layout, dimensions, judge, ...) -> list[RawResult]`, `aggregate(self, raw_results) -> Summary`.
- Adapters cannot assemble paths themselves — they must go through `workspace.layout`. Adapters cannot make HTTP calls directly — they must go through `scheduler.submit(...)` or a Judge.
- `__init__` accepts only an immutable config dict (from `SUPPORTED_BENCHMARKS[name]`); no toolkit-external objects.

**A2. `BaseScorer` interface**

- Sync and async pair: `score(self, ctx: ScoreContext) -> ScoreResult` and `score_async(...)`, identical signature.
- `ScoreResult` is a pydantic model with at minimum `score: float | dict`, `raw: dict`, `meta: dict`.
- A Scorer is unaware of whether its endpoint is local or managed and is unaware of frame cache — those are injected by the factory at construction time.
- VLM Judges (`scorers/vlm_judge/*`) are the canonical Scorer subclass.

**A3. `BaseAggregator` interface**

- `aggregate(self, raw_results: list[RawResult]) -> Summary` or for cross-Benchmark `combine(summaries: list[Summary]) -> CrossReport`.
- An Aggregator does not read or write the filesystem — it is a pure function over numbers. The runner does the persistence.

**A4. `WorkspaceLayout`**

- Provides `videos_dir / prompts_dir(benchmark) / frames_cache_dir / model_cache_dir / raw_dir(benchmark, model, dim) / summary_path(benchmark, model) / leaderboard_dir / api_logs_root` etc.
- The `api_logs` directory naming rules (`calls/{provider}/{model}/{YYYY-MM}/{date}.{ts}.jsonl` etc.) are locked by A; no other module is allowed to modify them — they are the contract for any external analysis tooling reading those jsonls.

#### 5.1.3 Tier-2 Requirements

- **A5.** Coexistence of `RawResult` v1 and v2 schemas — protects old workspaces from breaking on upstream field changes.
- **A6.** Lightweight concurrency primitives on Workspace: `Workspace.lock()` and `Workspace.in_progress_set()` for future multi-process use on the same workspace.

#### 5.1.4 Key Objects & Boundaries

- **Owned by A:** base classes, pydantic types, `WorkspaceLayout`, all enums and constants.
- **Not owned:** any specific Benchmark / Judge / Aggregator implementation; scheduling and concurrency; CLI.

#### 5.1.5 Cross-Module Collaboration

- A is depended on by every other module. Any base-class signature change must announce a compat impact to B/C/F/G.
- A ↔ H: the pydantic types A exposes are also the toolkit's externally-stable API; version-bump rules are coordinated with H.

#### 5.1.6 Milestone Scope

- P0: A1–A4 in full.
- P1: A5 (v1/v2 schema coexistence).
- P2: A6 (lock / in_progress).

#### 5.1.7 Open Questions

- Should `Score.value` be constrained to `[0, 1]`, or left to each Benchmark? Today it is unconstrained, but cross-Benchmark z-score implicitly assumes comparability within a Benchmark.

### 5.2 Module B — Benchmark Adapters

#### 5.2.1 Positioning

Module B owns the integration of every concrete Benchmark. The first principle is **adapter, not reimplementation** — score formulas and metric implementations defer to upstream. B does three things only: map the toolkit's video/prompt layout to what each upstream expects (usually via symlink staging); normalize each upstream's wildly different output JSON into `RawResult`; aggregate to `Summary` in `aggregate()` according to the official upstream weighting.

#### 5.2.2 Tier-1 Requirements

**B1. Six adapters implementing `BaseBenchmark`** (4 anchored + 2 supplementary):

- `VBenchBenchmark` — call upstream `vbench` Python API; videos must be staged into upstream-expected `videos/{prompt}_idx.mp4` via symlinks; parse outputs from upstream `VBench_evaluation_results/...`.
- `VBench2Benchmark` — call upstream `vbench2` Python API; VLM-requiring dims go through `judge.score_async()` rather than the upstream VLM CLI.
- `VideoBenchBenchmark` — no standalone upstream package; the adapter reuses the toolkit's VLM Judge plus its own prompt templates; scoring semantics follow the paper's Table 3.
- `WorldJenBenchmark` — no upstream package; in-tree fork. The adapter implements WorldJen's two-stage PHAS flow (candidate scoring → preference aggregation); system prompts and weights ship inside the toolkit at `benchmarks/worldjen/prompts/`.
- `T2VCompBenchBenchmark` — compositional 7-dim suite (GroundingDINO + Depth + SAM + LLaVA-Video).
- `WorldScoreBenchmark` — **10-of-10** dims via the upstream metric classes (the same `worldscore.benchmark.metrics.*` import path). Per-instance normalization uses upstream's `aspect_info`; the two headlines mirror `run_evaluate.py:165-166` (Static = mean of the 7 static dim values; Dynamic = mean of all 10). `style_consistency` requires the upstream reference images (`Howieeeee/WorldScore` HF dataset → `runners/extract_refs.py`). `motion_accuracy` uses spaCy class-matching for the rate factor. 49 frames per video (= upstream's `interpframe_num`).

**B2. Input-layout normalization**

- Upstream-expected video paths usually differ from the toolkit's `videos/{model}/{prompt_id}-{idx}.mp4`; at the start of `evaluate()` the adapter uses symlinks under `$WORKSPACE_ROOT/staging/{benchmark}/{model}/` to construct the upstream view, without copying videos.
- The adapter must either clean up `staging/` at the end, or retain it for upstream tooling (per Benchmark convention).

**B3. Output normalization**

- For each piece of JSON the upstream emits, the adapter immediately translates it to a `RawResult` and writes `results/raw/{benchmark}/{model}/{dim}/{prompt_id}.json`.
- If the upstream emits a large batch JSON, the adapter splits it per prompt before persisting — this is required for resume to work.

**B4. Resume support**

- At the start of `evaluate()`, enumerate `list_required_videos(model)`; for each (model, dim, prompt), check whether `results/raw/...` already exists; skip if it does.
- The skip decision is based on file existence + basic schema validation (pydantic parse succeeds), never on file size or timestamps.

#### 5.2.3 Tier-2 Requirements

- **B5.** One README per Benchmark — each `benchmarks/<name>/README.md` documents at minimum the upstream repo URL, commit/tag pin, upstream-expected video naming, the official score-reproduction command, and any diffs versus toolkit defaults.
- **B6.** One smoke test per Benchmark — under `tests/`, runs in under 30 seconds using mock videos or a minimal subset; runs in CI.
- **B7.** Upstream upgrade helper: `scripts/upgrade_<bench>.py` that detects upstream commit drift and schema changes.

#### 5.2.4 Key Objects & Boundaries

- **Owned:** `benchmarks/<name>/benchmark.py`, `benchmarks/entry.py` (cross-env subprocess entry), each Benchmark's prompt files, the Benchmark's README.
- **Not owned:** VLM Judge implementations (C); GPU/HTTP scheduling (D); CrossReport (F).

#### 5.2.5 Cross-Module Collaboration

- B ← A: implements A's `BaseBenchmark`.
- B ↔ C: calls VLMs via a `judge: BaseScorer` instance; adapter sees only the interface, not local-vs-managed distinctions.
- B ↔ D: submits GPU/HTTP tasks via `scheduler: Scheduler`; does not write raw asyncio nor fork subprocesses.
- B ↔ E: every IO goes through `workspace.layout`; no string-path assembly.
- B → F: hands RawResults to the Aggregator; Aggregator does not write to disk.

#### 5.2.6 Milestone Scope

- P0: B1–B4 (nine adapters minimally functional + resume).
- P1: B5 (README) × 9, B6 (smoke) × 9.
- P2: B7 (upstream-upgrade helper).

#### 5.2.7 Open Questions

- VBench v1 wants `transformers==4.33.2`; VBench-2.0 wants `>=4.45`. The compromise today is to install the newer version, then override with `pip install transformers==4.33.2 --no-deps`; that policy must be enshrined in the `envs/videvalkit.yaml` release notes.
- Does WorldJen's in-tree fork migrate back to an external package once upstream cuts a release? Or do we keep the in-tree copy long-term for reproducibility?

### 5.3 Module C — VLM Judge Abstraction & API Integration

#### 5.3.1 Positioning

Module C abstracts "VLM as a judge" into a capability decoupled from any Benchmark. Three backends are exposed through one signature: `OpenAICompatibleVLMJudge` covers vLLM, Ollama, SGLang, OpenAI, DeepSeek, Together — any OpenAI-protocol endpoint; `GeminiVLMJudge` uses the google-genai SDK; `AnthropicVLMJudge` uses the anthropic SDK.

This is where the "pluggable VLM" promise lands, and where downstream teams hook their own VLM deployments (see [§7](#7-deployment-topologies-and-port-plan)).

#### 5.3.2 Tier-1 Requirements

**C1. Three backends, all implementing `BaseScorer` (the VLM Judge variant)**

- Identical sync + async signatures; return a `ScoreResult` model.
- Failure semantics: network errors (5xx, timeouts, connection reset) auto-retry with TokenBucket + exponential backoff; 4xx is not retried and is raised verbatim (401 / 403 / 422 are configuration errors).
- Every call goes through `ApiCallLogger` and is persisted to `api_logs/calls/...jsonl`, with schema frozen against the existing `video_eval/results/api_logs/`.
- **Offline-mode constraint:** when `$VIDEVALKIT_OFFLINE=1`, a Judge backend resolves its local snapshot dir via `resolve_hf_model_path()` before startup; if the toolkit launches a vLLM endpoint itself (a rare dev case), the command line uses a local path rather than a repo id (see [§8.4.4](#844-loading-from-local-no-torchhub--hf-network)). No backend in this section is allowed to call `from_pretrained(repo_id)` directly.

**C2. `SUPPORTED_JUDGES` registry**

- Each row is a kwargs dict (`kind / endpoint / model / provider / api_key_env`); adding a Judge = adding one row.
- Local endpoints and managed APIs are equivalent at the CLI layer; `--judge gemma-4-31b-local` and `--judge gemini-2.5-pro` look the same to the user.
- `provider` decides the partitioning of api_logs; users cross-referencing one VLM's usage across Benchmarks rely on this.

**C3. Factory + injection**

- `scorers/vlm_judge/factory.py:build_judge(name, layout)` is the sole instantiation entry point; the factory wires in `ApiCallLogger` and `FrameCache` and resolves API keys.
- Adapters never `import OpenAICompatibleVLMJudge` directly — always go through the factory.

**C4. FrameCache**

- The same video, sampled by the same (mode, n_frames), is decoded once; subsequent reads hit `frames_cache/{video_hash}/{mode}_{n}/`.
- The directory is managed by `Workspace.layout.frames_cache_dir` and is shared across Benchmarks.

#### 5.3.3 Tier-2 Requirements

- **C5.** Token usage and cost summarization: every jsonl entry records `input_tokens / output_tokens`; a standalone `scripts/api_cost.py` aggregates by provider / model / day.
- **C6.** Call replay: replay a Judge call from jsonl (no actual request) for post-mortem analysis or prompt-debug.
- **C7.** Batch API: when a Benchmark has bulk, non-real-time prompts, route through OpenAI / Anthropic batch APIs.

#### 5.3.4 Key Objects & Boundaries

- **Owned:** `scorers/vlm_judge/{base,openai_compat,gemini,anthropic,factory}.py`, `SUPPORTED_JUDGES`, FrameCache, ApiCallLogger.
- **Not owned:** the VLM server processes themselves (vLLM / SGLang are not the toolkit's responsibility — they are deployed by the downstream per [§7.2](#72-remote-vlm-port-convention-80018009) / [§9.3](#93-starting-the-vlm-backends)); the per-Benchmark scoring prompt templates (those live with each Benchmark).

#### 5.3.5 Cross-Module Collaboration

- C ← A: implements the Scorer interface.
- C ↔ B: adapters obtain a Judge from the factory; adapters are unaware of backend differences.
- C ↔ D: HTTP Judges go through `HTTPDispatcher` for rate limiting; SDK Judges use SDK-internal concurrency.
- C ↔ E: through `layout`, writes api_logs and reads frame cache.
- C ↔ H: API keys come from environment variables; H provides `.env.example` and secrets documentation.

#### 5.3.6 Milestone Scope

- P0: C1–C4 in full.
- P1: C5 (cost summary).
- P2: C6 (replay), C7 (batch API).

#### 5.3.7 Open Questions

- When managed APIs (Gemini / Anthropic) are unreachable in a downstream's network, do we offer a `--judge-fallback` to a local VLM? Fallback breaks score reproducibility — needs an explicit policy: forbid or allow.
- Should every Judge do a minimal real call (real-probe) during doctor? Today doctor only checks endpoint reachability + key presence, with zero token cost.

### 5.4 Module D — Scheduler and Env Isolation

#### 5.4.1 Positioning

Module D abstracts the toolkit's three workload kinds behind a single `Scheduler` facade, hiding concurrency and cross-env subprocess complexity from adapters. Three workload kinds: `gpu_metric` (VBench v1's RAFT/CLIP and similar local metric models), `vlm_judge_http` (OpenAI-compatible endpoints), `vlm_judge_api` (Gemini / Anthropic SDKs). When a Benchmark's env differs from the core env (per-Benchmark mode), `CondaEnvDispatcher` dispatches a subprocess across envs.

#### 5.4.2 Tier-1 Requirements

**D1. `GPUWorkerPool`**

- One worker subprocess per available GPU, isolated by `CUDA_VISIBLE_DEVICES`; weights are lazy-loaded once per worker lifetime.
- Workers reuse weights from `Workspace.layout.model_cache_dir`; downloads outside the toolkit's directory tree are not allowed.
- Pool size from CLI `--gpus N` or env `VIDEVALKIT_GPUS=...`; defaults to all available.

**D2. `HTTPDispatcher`**

- asyncio + `asyncio.Semaphore` for concurrency; one `TokenBucket` per provider for RPM + TPM control.
- Backoff: 429 / 5xx / network errors back off `min(2^n, 60)` seconds, up to 5 retries; 4xx and parse errors raise immediately.
- All traffic flows through `ApiCallLogger` to jsonl, including request + response + token usage; image fields are elided to keep files small.

**D3. `CondaEnvDispatcher`**

- **Bypassed by default in v0.0.1:** all 6 Benchmarks share `envs/videvalkit.yaml` (see §1.2 / §8.1) and `SUPPORTED_BENCHMARKS[name]["env"]` is uniformly `_SHARED_ENV = "videvalkit"`, so adapters are called in-process and CondaEnvDispatcher does not actually fork subprocesses.
- **Fallback path:** when an upstream upgrade breaks the shared env (e.g. VBench-2.0 forces mmcv 3.0 + a CUDA bump), change `SUPPORTED_BENCHMARKS["vbench2"]["env"]` to `"videvalkit-vbench2"` and add `envs/videvalkit-vbench2.yaml`; the dispatcher activates at runtime.
- Cross-env dispatch convention: `conda run -n <env> python -m videvalkit.benchmarks.entry --benchmark <name> --method <model>`; payload via JSON over stdin, result via JSON over stdout; the parent and child do not bridge their async loops — the child manages its own.
- Both parent and child install toolkit-core so `entry.py` and the pydantic types are available; only the upstream Benchmark package version differs.

**D4. Facade**

- Adapters see only `Scheduler.submit(workload, item) -> Future`; `workload` is one of `"gpu_metric" / "vlm_judge_http" / "vlm_judge_api"`.
- Adapters never write raw asyncio, never fork subprocesses, never `import aiohttp` themselves.

#### 5.4.3 Tier-2 Requirements

- **D5.** Multi-machine scheduling (SSH/Ray): replace `EnvDispatcher` without touching adapter code. Post-P3.
- **D6.** Pool autoscaling: long-idle GPU workers exit to free memory; new tasks revive them.

#### 5.4.4 Key Objects & Boundaries

- **Owned:** `scheduler/{base,gpu_pool,http_pool,rate_limit,env_dispatcher}.py`, TokenBucket, concurrency, retries.
- **Not owned:** specific VLM call details (C); specific metric model weights (B).

#### 5.4.5 Cross-Module Collaboration

- D ↔ B: B is D's primary consumer.
- D ↔ C: HTTP Judges go through D's rate limiter; SDK Judges manage concurrency internally but the logger is still injected by D.
- D ↔ E: workers and dispatchers read model_cache and write api_logs via `layout`.

#### 5.4.6 Milestone Scope

- P0: D1–D4.
- P2: D5, D6.

#### 5.4.7 Open Questions

- When a downstream's VLM endpoint is rate-limited more aggressively than the toolkit's default TokenBucket (e.g. they put an Nginx `limit_req` in front), do we expose a per-endpoint override? Today only `SUPPORTED_JUDGES[name]['rate_limit']` works; there is no CLI flag.

### 5.5 Module E — Storage / Workspace / Logging

#### 5.5.1 Positioning

Module E is the toolkit's IO single-truth. Every input and output of one evaluation task lives under `$WORKSPACE_ROOT/`: raw videos, prompts, frame cache, model-weight cache, results/raw, results/summary, leaderboard export, api_logs. E centralizes the naming rules and the read/write API; no other module is allowed to bypass `workspace.layout` and assemble paths directly.

#### 5.5.2 Tier-1 Requirements

**E1. Workspace layout**

```text
$WORKSPACE_ROOT/
├── videos/{model}/{prompt_id}-{idx}.mp4
├── prompts/{benchmark}/prompts.jsonl
├── frames_cache/{video_hash}/{mode}_{n}/
├── model_cache/
├── results/
│   ├── raw/{benchmark}/{model}/{dim}/{prompt_id}.json
│   ├── summary/{benchmark}/{model}.json
│   └── leaderboard/{benchmark}_export.{json,zip}
├── api_logs/
│   ├── calls/{provider}/{model}/{YYYY-MM}/{date}.{ts}.jsonl
│   ├── stats/{provider}/{model}/{user}_{YYYY-MM}.{ts}.jsonl
│   └── zips/
└── staging/{benchmark}/{model}/        # symlink view for upstream layout
```

Each directory has a dedicated `Layout` method returning a `Path`; adapters cannot do `Path(workspace) / "results" / ...` string assembly.

**E2. `results/raw/` is the resume primitive**

- One JSON per (Benchmark, model, dim, prompt); if present, skip. Adapters never read it for internal state — only existence matters.
- A JSON that fails pydantic parsing is treated as nonexistent; it triggers a re-run that overwrites.

**E3. `api_logs/` schema is frozen**

- File naming, directory hierarchy, and jsonl fields all match the existing `video_eval/results/api_logs/` so existing analyzers keep working unchanged. **Any change requires a schema migration process.**
- Each jsonl line carries at least: `ts / provider / model / endpoint / request / response / tokens / latency_ms`; image fields are elided (size + hash retained, no base64).

**E4. `frames_cache/`**

- Videos are keyed by `sha256(file_contents)`; the same video sampled by the same (mode, n_frames) is decoded once.
- The cache key is `(mode, n_frames)`; different sampling strategies occupy distinct subdirs.
- Cache cleanup lives in `scripts/clear_cache.py`; the toolkit does not GC automatically.

#### 5.5.3 Tier-2 Requirements

- **E5.** A `workspace.json` metadata file recording creation time, toolkit version, Benchmark version pins — for post-hoc audit.
- **E6.** Leaderboard export zip: package `results/summary/` + a visualization HTML for non-developer review.

#### 5.5.4 Key Objects & Boundaries

- **Owned:** `storage/{workspace,layout,api_log}.py`, `utils/frame_cache.py`, workspace metadata.
- **Not owned:** the field definitions of RawResult / Summary (A); scoring content (B/C).

#### 5.5.5 Cross-Module Collaboration

- E ↔ everyone: every IO routes through E.
- E ↔ H: workspace is the downstream's runtime state directory; H's install package does not ship a pre-populated workspace but does ship `videvalkit prepare-workspace`.

#### 5.5.6 Milestone Scope

- P0: E1–E4.
- P1: E5, E6.

#### 5.5.7 Open Questions

- At large evaluation scale, `api_logs/` can balloon to many GB. Do we offer a gzipped jsonl write path? Today logs are plain jsonl.

### 5.6 Module F — Aggregation & Cross-Benchmark Reporting

#### 5.6.1 Positioning

Module F covers three layers of aggregation: (1) within-Benchmark official-weight aggregation; (2) cross-model Bradley-Terry preference aggregation within one Benchmark; (3) cross-Benchmark z-score normalization and unified ranking.

#### 5.6.2 Tier-1 Requirements

**F1. Per-Benchmark `aggregate()` defaults**

- Each Benchmark adapter's `aggregate()` reuses `SUPPORTED_AGGREGATORS[default_aggregator]`. VBench uses `vbench_weighted` (Quality + Semantic split); VBench-2.0 uses `vbench2_category` (5-category mean); WorldJen uses `phas` (PHAS-weighted with variance penalty).
- Aggregators are pure functions returning a `Summary`; they do not write disk — the runner does.

**F2. Bradley-Terry (cross-model, within Benchmark)**

- Derive pairwise preferences from per-(model, prompt) means; iterative MLE fit; bootstrap prompt-level CIs.
- Implementation in `aggregators/bt.py`. CLI: `videvalkit aggregate --workspace ... --kind bt --bench vbench`.

**F3. Cross-Benchmark `combine_summaries()`**

- Within each Benchmark, z-score normalize; take the cross-Benchmark mean as the unified score; re-run BT on the implied preferences.
- Output `results/leaderboard/cross_benchmark.json`, ranked, with z-scores and confidence intervals.

**F4. PHAS weight calibration**

- WorldJen's PHAS defaults come from the paper; if the team has human-evaluation CSVs, `scripts/calibrate_phas.py` refits the weights. Calibration is decoupled from the main evaluation path.

#### 5.6.3 Tier-2 Requirements

- **F5.** Every Aggregator accepts a `dim_subset` parameter — aggregate only a user-specified subset of dimensions, useful for ablations.
- **F6.** Cross-Benchmark markdown export: render `cross_benchmark.json` to a table-with-prose md for pasting into reports.

#### 5.6.4 Key Objects & Boundaries

- **Owned:** `aggregators/{weighted_sum,vbench_weighted,vbench2_category,phas,bt,cross}.py`, `SUPPORTED_AGGREGATORS`, `scripts/calibrate_phas.py`, `scripts/cross_report.py`.
- **Not owned:** RawResult / Summary field definitions (A); per-Benchmark official scoring (B).

#### 5.6.5 Cross-Module Collaboration

- F ← B: consumes RawResults.
- F → G: the runner calls F and persists Summary / CrossReport.

#### 5.6.6 Milestone Scope

- P0: F1.
- P1: F2, F3, F4.
- P2: F5, F6.

#### 5.6.7 Open Questions

- When a Benchmark has very few samples (< 10 prompts), z-score and BT are statistically weak — should the cross-report annotate a confidence tier?

### 5.7 Module G — CLI / Runner / Config

#### 5.7.1 Positioning

Module G is the entry point for users and agents. The CLI uses click; subcommands are `list / doctor / eval / aggregate / prepare-workspace`. `runner.run()` is the Python API; the CLI is a thin shell over it. The config module (`configs/`) maintains the three registries.

#### 5.7.2 Tier-1 Requirements

**G1. CLI subcommands**

- `videvalkit list (benchmarks|judges|aggregators)` — list the contents of the three registries; agent-friendly.
- `videvalkit doctor [--workspace PATH] [--json]` — check conda envs, adapter imports, Judge endpoint reachability, API key presence, optional workspace health.
- `videvalkit prepare-workspace --workspace PATH --videos PATH` — bootstrap a workspace and symlink the external videos directory.
- `videvalkit eval --bench X --videos PATH --workspace PATH [--models ...] [--dimensions ...] [--judge ...] [--aggregator ...]` — run one Benchmark.
- `videvalkit aggregate --workspace PATH [--output PATH]` — combine all `summary/*/*.json` in a workspace into a cross-Benchmark report.

**G2. `runner.run()`**

- Single Python entry point, parameters one-to-one with the CLI eval subcommand; returns a dict with `summary` and a few paths.
- Internal logic: look up registries → construct Workspace → construct Scheduler (decide pool composition by needs_gpu / needs_judge) → construct Judge (if needed) → instantiate Benchmark → `evaluate()` → `aggregate()` → persist.

**G3. Config registry entry**

- `configs/__init__.py` re-exports `SUPPORTED_BENCHMARKS / SUPPORTED_JUDGES / SUPPORTED_AGGREGATORS`; other modules import from here, not from submodules.
- Adding a row: one registry row + one implementation file + one unit test, all in one PR.

#### 5.7.3 Tier-2 Requirements

- **G4.** `videvalkit explain --bench X` — print one Benchmark's dimension list, official weights, required VLM, recommended GPU footprint.
- **G5.** `videvalkit replay --workspace PATH --benchmark X --model Y` — replay one evaluate based on api_logs (no actual VLM call); validates adapter parsing robustness.
- **G6.** `--config FILE` to read a toml/yaml config, avoiding long command lines.

#### 5.7.4 Key Objects & Boundaries

- **Owned:** `cli.py`, `runner.py`, `configs/`, `diagnostics.py` (doctor implementation).
- **Not owned:** execution details delegated to B/C/D/F.

#### 5.7.5 Cross-Module Collaboration

- G ↔ every module.
- G ↔ H: the CLI is the toolkit's externally-stable surface; G's compat is guarded by H's release notes.

#### 5.7.6 Milestone Scope

- P0: G1–G3.
- P1: G4, G5, G6.

#### 5.7.7 Open Questions

- Should `--gpus`, `--max-concurrency`, etc. be exposed on the CLI? Today they are environment variables; exposing them makes agent invocation more explicit at the cost of CLI surface area.

### 5.8 Module H — Packaging, Distribution & Remote VLM Onboarding

#### 5.8.1 Positioning

Module H is the bridge from internal development to downstream use. Covers: building wheel/sdist; declaring Python / CUDA compatibility; giving downstream a zero-to-running deployment doc; specifying the remote VLM port and config; managing API keys and secrets; shaping `doctor`'s output for self-service triage.

#### 5.8.2 Tier-1 Requirements

**H1. Build and distribution**

- Build a wheel (`videvalkit-X.Y.Z-py3-none-any.whl`) and sdist; upload to an internal PyPI or hand the package directly to the downstream.
- `[project.dependencies]` in `pyproject.toml` declares only pure-Python deps (pydantic / aiohttp / click / pyyaml / tqdm / numpy). Heavy deps (torch / transformers / mmcv / flash-attn / upstream Benchmark packages) are installed via the `envs/videvalkit.yaml` conda env and **must not** enter wheel dependencies — they will not install otherwise.
- Ship `envs/videvalkit.yaml` plus `envs/<bench>.yaml` fallbacks; the release tarball contains both.

**H2. Remote VLM onboarding convention**

- The default `endpoint` for each `SUPPORTED_JUDGES` entry is `http://localhost:800x/v1`; downstream teams either reuse the defaults or front a reverse proxy to consolidate multiple VLMs onto one host.
- Key reference: [§7](#7-deployment-topologies-and-port-plan); downstream teams start their own vLLM / SGLang per that section.
- Downstream extension path: **don't** ask them to edit `configs/judges.py` (fork-risk); provide `$VIDEVALKIT_JUDGES_FILE` pointing at an extra yaml/json that the runner merges into `SUPPORTED_JUDGES` at start.

**H3. Secrets management**

- API keys via environment variables (`GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY`); the toolkit does not read `~/.config` or dotfiles to avoid accidental capture.
- `.env.example` enumerates all possible variable names with brief comments; downstream copies it to `.env` as needed.
- `doctor` checks key presence (boolean) only; never prints key contents; jsonl logs never store keys.

**H3b. Offline model-weight delivery**

- Keep the [§8.4.1 manifest](#841-the-complete-model-weight-manifest), `scripts/download_weights.py`, and `checksums.json` in sync: one new row in the manifest = the script must parse it = `checksums.json` has the entry = doctor will validate. Any PR that swaps a metric model or adds a VLM must touch all three.
- The release team prepares two offline tarball tiers per release: "minimal 80 GB" (VBench v1 + WorldJen local Judge) and "recommended 200 GB" (full offline for the 4 anchored Benchmarks); size + sha256 in the CHANGELOG.
- The doc explicitly says: "if downstream can reach HF / GitHub directly, skip this section" — avoid dragging everyone into the weight-packaging chain.

**H4. `doctor` readable to external users**

- Downstream-friendly output: every line should be copy-pasteable as a search query. JSON mode is stable for agents.
- At minimum covers: conda env presence; the 9 adapter imports; per-Judge endpoint reachability; per-Judge api_key presence; workspace subdir existence and entry counts.

#### 5.8.3 Tier-2 Requirements

- **H5.** Online install script `bash install_videvalkit.sh` — one command to create the conda env + install the wheel + initialize a default workspace.
- **H6.** Offline bundle: wheel + upstream Benchmark source + locked conda env + into one tarball, for fully-offline machines.
- **H7.** Compatibility declaration: every release lists compatible CUDA / torch / transformers ranges in the CHANGELOG; breaking changes require a minor version bump.

#### 5.8.4 Key Objects & Boundaries

- **Owned:** `pyproject.toml`, `envs/*.yaml`, `scripts/install_*.sh`, `CHANGELOG.md`, `.env.example`, downstream-facing docs.
- **Not owned:** the downstream's VLM server processes; their GPU / network resources.

#### 5.8.5 Cross-Module Collaboration

- H ↔ G: the CLI and doctor outputs are externally stable; any CLI change must leave a trace in H's CHANGELOG.
- H ↔ C: remote VLM endpoint convention is led by H; C only implements Judge backends.
- H ↔ E: workspace bootstrap is triggered by H's install script.

#### 5.8.6 Milestone Scope

- P0: H1–H4.
- P1: H5.
- P2: H6, H7.

#### 5.8.7 Open Questions

- Do we publish an official Docker image (core env + toolkit baked in, with VLM services in separate images)? Dockerization saves CUDA pain for downstream but adds maintenance.
- Do we ship a docker-compose minimal kit (toolkit core + one local VLM service)?

---

## 6. Cross-Module Collaboration

### 6.1 The registries are the contract line

The three registries are the most important contract lines in the toolkit. Any PR touching them must satisfy:

- New Benchmark: B implements the adapter → add a row in `SUPPORTED_BENCHMARKS` → add a row in `benchmarks/entry.py` → verify `envs/videvalkit.yaml` covers deps → one smoke test → README.
- New Judge: C implements a backend (if a new protocol) → add a row in `SUPPORTED_JUDGES` → doctor auto-covers it → add the API key variable to `.env.example` if needed.
- New Aggregator: F implements → add a row in `SUPPORTED_AGGREGATORS` → if applicable update the relevant adapter's `default_aggregator`.

### 6.2 Single IO path

Every code path that reads or writes disk must go through `Workspace.layout`. This rule protects:

- resume capability (`results/raw/` naming consistency is the prerequisite for skipping);
- api_logs schema stability (external analyzers depend on the directory structure);
- cache reuse (the same video shares its frame cache across Benchmarks).

Code review rejects any `Path("/some/path") / "results"` string assembly; use `layout.raw_dir(benchmark, model, dim)` etc.

### 6.3 Adapters depend on Judge and Scheduler interfaces only

- Adapters obtain a Judge from the factory and submit work via the scheduler. The allowed dependency direction is: B → A's interfaces; B uses instances of C/D but does not import their concrete subclasses.
- This is what lets the toolkit switch between per-Benchmark env and unified env modes — when subprocesses cross env boundaries, the import boundary must stay tight.

### 6.4 Doctor is the downstream's first mirror

`videvalkit doctor` serves three audiences at once: internal devs (verify a PR did not break the env), downstream ops (triage), agents (consume `--json`). Therefore doctor must:

- be read-only with no side effects;
- not consume tokens during endpoint probes;
- on failure, emit a **searchable error string**, not just `MISS`.

### 6.5 Remote VLM ↔ toolkit decoupling

VLM server processes are not the toolkit's responsibility. Boundary:

- Toolkit owns: calling VLMs via HTTP/SDK, persisting api_logs, token-bucket rate limiting.
- Downstream owns: starting vLLM/SGLang/Ollama, keep-alive, Nginx reverse proxy, machine resource monitoring.
- The contract between them is the `endpoint` field in `SUPPORTED_JUDGES`.

### 6.6 Secrets and api_logs boundary

API keys **always** come from environment variables; the toolkit reads no dotfiles, prints no keys, persists no keys in jsonl. Doctor reports presence, not content. Downstream sets environment variables per `.env.example`.

---

## 7. Deployment Topologies and Port Plan

### 7.1 Three Topologies

**Topology 1 — Developer workstation.** One 8×GPU machine; all VLMs are local vLLM endpoints (8001–8009); all Benchmarks run on this machine; managed APIs (Gemini / Anthropic) optional as baseline comparisons. This is the default for daily development and small-scale evaluation.

**Topology 2 — Evaluation-dedicated host (no or restricted external network).** One GPU server, no managed-API access. The toolkit is uploaded onto the host; local vLLM endpoints handle all Judges; all managed Judges in `SUPPORTED_JUDGES` show `MISS` in doctor but do not block. This is the typical downstream topology.

**Topology 3 — Managed-API first.** No GPU resources; pure-managed Judges (Gemini / Anthropic). Only suitable for Benchmarks that do not require GPU metrics (Video-Bench, parts of WorldJen). Local entries in `SUPPORTED_JUDGES` are unreachable; doctor marks them MISS but does not block.

### 7.2 Remote VLM Port Convention (8001–8009)

The toolkit assumes the downstream runs several OpenAI-compatible VLM servers on the same machine (or the same intranet). We propose a **conventional port plan**; the downstream enables what it needs:

| Port | Default model | Judge name | VRAM budget | Note |
| ---- | --------------- | ----------- | ------------ | ---- |
| 8001 | reserved (small VLM, e.g. Qwen2.5-VL-7B) | `qwen25-vl-7b-local` | ~ 20 GB | entry-level |
| 8002 | reserved (mid VLM) | — | ~ 40 GB | |
| 8003 | `google/gemma-3-31b-it` (video frames + text) | `gemma-4-31b-local` | 1 × 80 GB | registered |
| 8004 | `Qwen/Qwen3-32B` (text reasoning / second-stage judge) | `qwen3-32b-local` | 1 × 80 GB | registered |
| 8005 | `Qwen/Qwen3-VL-32B-Instruct` | `qwen3-vl-32b-local` | 1 × 80 GB | registered |
| 8006 | `lmms-lab/LLaVA-Video-7B-Qwen2` | `local-llava-video-7b` | ~ 20 GB | VBench-2.0 default |
| 8007 | reserved (backup 32B VLM) | — | 1 × 80 GB | |
| 8008 | reserved (backup 70B text, for text aggregation) | — | 2 × 80 GB | |
| 8009 | reserved (experimental) | — | — | |

Ports 8001 and 8007–8009 are reserved; downstream teams may reuse or redirect them in their own docs. **Port numbers are not enforced** — downstream can override the endpoint URL via `$VIDEVALKIT_JUDGES_FILE` ([§5.8.2](#582-tier-1-requirements)), but keeping the convention helps triage.

**Recommended startup template** (vLLM example):

```bash
# 8003: Gemma-3-31B  (image frames + text input)
CUDA_VISIBLE_DEVICES=0 \
  python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3-31b-it \
    --port 8003 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --served-model-name google/gemma-3-31b-it

# 8005: Qwen3-VL-32B
CUDA_VISIBLE_DEVICES=1 \
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-32B-Instruct \
    --port 8005 \
    --max-model-len 32768 \
    --limit-mm-per-prompt image=16,video=1 \
    --served-model-name Qwen/Qwen3-VL-32B-Instruct
```

> Reference launch scripts live under `scripts/launch_vlm/<model>.sh`; downstream teams adjust `--gpu-memory-utilization` and `--max-model-len` for their VRAM and concurrency budgets.

### 7.3 Health Checks, Concurrency, Rate Limits

**Probe.** During doctor, each endpoint is probed via GET `/v1/models` (OpenAI-compatible) or the SDK's minimal call — only verifies reachability and that the model is loaded; consumes no tokens.

**Concurrency.** Controlled by `SUPPORTED_JUDGES[name]['rate_limit']` (`rpm` + `tpm` + `max_concurrency`); defaults are conservative (local: `max_concurrency=16, rpm=1200, tpm=300_000`; managed APIs: per provider documentation). `HTTPDispatcher` keeps one TokenBucket per provider.

**Backoff.** 429 / 5xx / network errors back off `min(2^n, 60)` seconds, up to 5 retries; any 4xx raises immediately — they are configuration errors.

**Call logging.** Every Judge call writes one `api_logs/calls/{provider}/{model}/{YYYY-MM}/{date}.{ts}.jsonl` line, schema frozen. Downstream audit consumes this directory directly.

### 7.4 Failover and Replication

The toolkit does **not** ship replication / load balancing — it is too close to the ops layer and downstream preferences vary. Recommended approaches:

- The downstream fronts the endpoints with Nginx round-robin; the toolkit sees one endpoint.
- Or: the downstream registers multiple Judge names (`gemma-4-31b-local-a` / `-b`) in `$VIDEVALKIT_JUDGES_FILE`, each pointing at a different backend endpoint; the user switches explicitly.

Failover is out of scope for v1; if an endpoint is unavailable, the toolkit retries per the backoff policy until the cap, then raises `JudgeUnavailable`; the runner exits with already-completed `results/raw/` intact, ready to resume after the endpoint is restored.

---

## 8. Runtime Environment & Dependency Management

### 8.1 Single Env vs Multi Env

**Default today: single env (`envs/videvalkit.yaml`).** All 9 Benchmarks + toolkit core install into one conda env. The cost is that the toolkit adjusts some upstream pins for compatibility:

- `torch==2.3.1 + cu121` covers both VBench v1 and VBench-2.0 (VBench-2.0 README says torch 2.5.1, but 2.3.x works in practice, and flash-attn prebuilt wheels must match torch).
- `transformers` is not pinned to VBench v1's 4.33.2; we leave it ≥ 4.45 so VBench-2.0 works; `pip install transformers==4.33.2 --no-deps` is the temporary override when needed.
- `mmcv==2.2.0` per VBench-2.0 README.

**Fallback: per-Benchmark env (`envs/vbench.yaml` / `envs/vbench2.yaml` / ...).** When pin conflicts become irreconcilable (e.g. detectron2's CUDA ≤ 12.1 hard requirement collides with VBench-2.0's 11.8), revert to one env per Benchmark. `CondaEnvDispatcher` activates; the core dispatches subprocesses via `conda run -n videvalkit-<bench>`. The per-Benchmark yamls are marked `DEPRECATED` today but retained as the fallback path.

### 8.2 CUDA / torch Compatibility Matrix

| Benchmark | Upstream expectation | Unified env config | Risk |
| --------- | -------------------- | ------------------ | ---- |
| VBench v1 | CUDA ≤ 12.1, torch 2.1.x, detectron2 from source | torch 2.3.1 + cu121; detectron2 reinstalled | detectron2 needs the host CUDA toolkit to compile; prebuilt wheels are not always available |
| VBench-2.0 | CUDA 11.8, torch 2.5.1, flash-attn 2.7.2, mmcv 2.2.0 | torch 2.3.1 + cu121; flash-attn 2.5.x (torch-matched); mmcv 2.2.0 | flash-attn must match the torch ABI — a mismatch breaks at import |
| Video-Bench | no hard GPU dep (VLM Judge only) | served by local VLM endpoints | — |
| WorldJen | none (in-tree fork) | served by local VLM endpoints | — |
| T2VCompBench | torch + GroundingDINO + SAM | unified env | GroundingDINO build can fail on newer CUDA |
| WorldScore | DROID-SLAM + SEA-RAFT + GroundingDINO + SAM-ViT-H + SAM2 + VFIMamba + VGG-19 + pyiqa (clipiqa+, laion_aes) + torchmetrics CLIPScore + spaCy en_core_web_sm | unified env (`videvalkit`) | mamba_ssm CUDA extension would upgrade torch beyond 2.3.1+cu121 → toolkit ships a pure-PyTorch `selective_scan` shim in `scorers.install_mamba_ssm_shim()`; tests prefer the shim. DROID-SLAM patched at `validation/upstream_repos/WorldScore/worldscore/benchmark/metrics/third_party/droid_slam/` (added sm_89/sm_90 gencode + modified `terminate()`) |

**Downstream upgrade path:** when the downstream's hardware forces CUDA 11.8 (typical for some cloud-vendor images), fork `envs/videvalkit.yaml` into `envs/videvalkit-cu118.yaml`, swap `cu121` → `cu118`, bump `torch` to 2.5.1, refit flash-attn, then `conda env create`. No toolkit code change required.

**WorldScore-specific environment notes:**

- Conda env: `/pub/evaluation_group/ning/benchmark/envs/videvalkit` (Python 3.10, torch 2.3.1+cu121).
- All WorldScore runners and the adapter's `scorers.setup_upstream_paths()` add the upstream repo at `validation/upstream_repos/WorldScore/` to `sys.path` and `os.chdir(WS_ROOT)`. The toolkit ships a pure-PyTorch `mamba_ssm.selective_scan` shim (`scorers.install_mamba_ssm_shim()`) so VFIMamba (`motion_smoothness`) can import without recompiling CUDA kernels — avoid `pip install mamba_ssm` since it forces a torch upgrade that breaks lietorch / droid_backends / sam2 / pyiqa.
- The DROID-SLAM CUDA extension has been patched to compile for sm_89 and sm_90 (Hopper). The patch lives in `validation/upstream_repos/WorldScore/thirdparty/DROID-SLAM/setup.py` and is preserved through normal upstream sync.
- All heavy weights (~4.6 GB) live under `validation/upstream_repos/WorldScore/worldscore/benchmark/metrics/checkpoints/`. The WorldScore HF dataset (5.7 GB of reference images + metadata) is downloaded to `/tmp/worldscore_dataset/` by `runners/extract_refs.py` — keep it on `/tmp` (local disk) rather than CPFS.

### 8.3 VLM Backend Version Matrix

VLM backend versioning is where downstream onboarding most often stumbles. Below is a recommended matrix as of 2026-05; new integrators start here and downgrade only if they hit issues.

| VLM family | Recommended version | Inference framework | Note |
| ---------- | ------------------- | -------------------- | ---- |
| Gemma-3 | `google/gemma-3-31b-it` | vLLM ≥ 0.6.x | video frames (image list) + text; WorldJen default |
| Qwen3-VL | `Qwen/Qwen3-VL-32B-Instruct` | vLLM ≥ 0.6.x | requires `--limit-mm-per-prompt video=1` |
| Qwen3 | `Qwen/Qwen3-32B` | vLLM | text-only; second-stage judging |
| LLaVA-Video | `lmms-lab/LLaVA-Video-7B-Qwen2` | vLLM or SGLang | VBench-2.0 default for some dims |
| InternVL3 | `OpenGVLab/InternVL3-38B` | SGLang | alternative video VLM |
| GPT-4o | `gpt-4o-2024-11-20` | OpenAI API | baseline comparison |
| Gemini-2.5 | `gemini-2.5-pro` / `gemini-3-flash-preview` | google-genai SDK | baseline comparison |
| Claude | `claude-sonnet-4-6` | anthropic SDK | baseline comparison |

> Upstream model versions move; this table is a starting point only. Every release refreshes it in the CHANGELOG.

**Onboarding a new VLM backend** (OpenAI-compatible case):

1. Start vLLM / SGLang / Ollama on your machine; note that `--served-model-name` must match the `model` field in the registry.
2. Add a row to `$VIDEVALKIT_JUDGES_FILE` (or to `configs/judges.py` and reinstall the wheel):

   ```yaml
   my-new-vlm-local:
     kind: openai_compatible
     endpoint: http://localhost:8007/v1
     model: org/model-name
     provider: org
     api_key_env: null
   ```

3. `videvalkit doctor` and confirm `reach=OK`.
4. Run a smoke: `videvalkit eval --bench videobench --judge my-new-vlm-local --videos ... --workspace ...`.

**Onboarding a non-OpenAI-protocol VLM:** add a new backend class under `scorers/vlm_judge/` (implementing the `BaseScorer` sync + async interface) and register the new `kind` in the factory. This is internal toolkit work and is not expected of downstream integrators.

### 8.4 Model Weights and Local Loading (Offline)

Downstream environments often cannot run `torch.hub.download_url_to_file` / `huggingface_hub.snapshot_download` / vLLM's "fetch on startup" — typical causes are no external network, an intranet proxy that does not reach HuggingFace, or security review requiring all model weights to be pre-vetted. This section lists every model weight videvalkit touches, where it comes from by default, where it lives on disk, and how the toolkit loads it from local in offline scenarios.

#### 8.4.1 The Complete Model-Weight Manifest

The toolkit does **not** generate videos; the weights below are all consumed during evaluation, grouped by source.

**A. Metric models bundled by upstream Benchmarks** (the most common stumbling block — these are auto-downloaded the first time a dim runs, and not always to the HF cache):

| Model | Use | Upstream source | Approx size | Used by |
| ----- | --- | ---------------- | ----------- | ------- |
| ViCLIP-L-14 | subject_consistency | VBench `model_zoo` (HF mirror) | 1.5 GB | VBench v1 |
| CLIP ViT-L/14 | overall_consistency / some semantic dims | OpenAI release (HF) | 890 MB | VBench v1, v2 |
| RAFT-Things | motion_smoothness | princeton-vl/RAFT (GitHub release) | 21 MB | VBench v1 |
| AMT | dynamic_degree | MCG-NKU/AMT (GitHub release) | 178 MB | VBench v1 |
| LAION-Aesthetic-Predictor | aesthetic_quality | LAION repo (HF) | 4 MB (head) + CLIP (above) | VBench v1 |
| MUSIQ | imaging_quality | google-research/google-research (TF→torch conversion) | 84 MB | VBench v1 |
| RAM++ | object_class / multiple_objects | xinyu1205/recognize-anything (HF) | 5.6 GB | VBench v1 |
| GroundingDINO-T | object_class (bbox) | IDEA-Research/GroundingDINO (HF) | 660 MB | VBench v1 |
| YOLOv5x | human_action | ultralytics (torch hub) | 174 MB | VBench v1 |
| UMT | scene / some v2 dims | OpenGVLab/unmasked_teacher (HF) | 1.8 GB | VBench-2.0 |
| Q-Align | imaging_quality_v2 | Q-Future/Q-Align (HF) | 8.5 GB | VBench-2.0 |
| InternVideo2-S2 | video_text_consistency_v2 | OpenGVLab/InternVideo2 (HF) | 1.6 GB | VBench-2.0 |
| (remaining VBench-2.0 models) | see VBench-2.0 README | HF | ~ 10 GB total | VBench-2.0 |

Total roughly **30–35 GB** (VBench v1 + VBench-2.0 full). If the downstream only runs a subset of VBench v1 dims, they can fetch on demand.

**B. Open-source VLM weights** (loaded by local vLLM/SGLang; downstream must download):

| Model | Use | HF repo | Approx size | Default port |
| ----- | --- | ------- | ------------ | ------------- |
| `google/gemma-3-31b-it` | WorldJen default Judge | google/gemma-3-31b-it | 62 GB | 8003 |
| `Qwen/Qwen3-32B` | second-stage text judge | Qwen/Qwen3-32B | 65 GB | 8004 |
| `Qwen/Qwen3-VL-32B-Instruct` | alternative video VLM | Qwen/Qwen3-VL-32B-Instruct | 65 GB | 8005 |
| `lmms-lab/LLaVA-Video-7B-Qwen2` | VBench-2.0 default | lmms-lab/LLaVA-Video-7B-Qwen2 | 15 GB | 8006 |
| `OpenGVLab/InternVL3-38B` (optional) | alternative video VLM | OpenGVLab/InternVL3-38B | 78 GB | 8007 |

Total ~ 200–280 GB; pick endpoints per the Benchmarks you enable. Minimum viable combo (WorldJen + Video-Bench via managed API + VBench v1) ~ 80 GB; recommended combo (all four anchored Benchmarks fully offline) ~ 200 GB.

**C. Managed API backends:** Gemini / Anthropic / OpenAI require no weight download — only API keys. When the downstream has no external network, all of these are disabled and doctor shows `key=MISS`.

#### 8.4.2 Filesystem Convention

The toolkit standardizes on a root cache directory `$VIDEVALKIT_MODEL_ROOT`, defaulting to `$WORKSPACE_ROOT/model_cache/`. Downstream teams can export it to any large-disk location (e.g. `/data/model_cache`). Expected layout:

```text
$VIDEVALKIT_MODEL_ROOT/
├── hf/                                # HuggingFace-style cache (used as HF_HOME)
│   ├── hub/
│   │   ├── models--google--gemma-3-31b-it/
│   │   ├── models--Qwen--Qwen3-32B/
│   │   ├── models--Qwen--Qwen3-VL-32B-Instruct/
│   │   ├── models--lmms-lab--LLaVA-Video-7B-Qwen2/
│   │   └── ...
│   └── transformers/                  # legacy TRANSFORMERS_CACHE
├── vbench/                            # VBench v1's expected model_zoo layout
│   ├── ViCLIP-L_InternVid-FLT-10M.pth
│   ├── raft-things.pth
│   ├── amt/amt-s.pth
│   ├── musiq_ava_ckpt.pth
│   ├── ram_plus_swin_large_14m.pth
│   ├── groundingdino_swint_ogc.pth
│   ├── yolov5x.pt
│   └── ...
├── vbench2/                           # VBench-2.0 expected layout
│   ├── umt_l16_25m.pth
│   ├── q-align/
│   ├── internvideo2/
│   └── ...
└── checksums.json                     # sha256 for every .pth/.bin, for verification
```

The toolkit glues this layout via three environment variables:

```bash
export VIDEVALKIT_MODEL_ROOT=/data/model_cache
export HF_HOME=$VIDEVALKIT_MODEL_ROOT/hf
export HF_HUB_OFFLINE=1                 # offline: forbid any HF network call
# legacy compatibility:
export TRANSFORMERS_CACHE=$HF_HOME/transformers
```

The VBench v1 adapter injects `$VIDEVALKIT_MODEL_ROOT/vbench/` into the upstream `MODEL_ZOO_PATH` global at import time; VBench-2.0 does the same with `vbench2/`. The injection point is `src/videvalkit/benchmarks/vbench/benchmark.py:_inject_model_root()`.

#### 8.4.3 Download and Verification Script

The toolkit ships an **offline pre-download script**: `scripts/download_weights.py`. It does two things only: mirror weights into `$VIDEVALKIT_MODEL_ROOT` per the manifest, and verify SHA-256 against `checksums.json`.

```bash
# On a machine with external network access (or the offline-bundle preparation team)
export VIDEVALKIT_MODEL_ROOT=/data/model_cache
python scripts/download_weights.py --all                  # all of A + B
python scripts/download_weights.py --benchmark vbench     # VBench v1 metrics only
python scripts/download_weights.py --judge gemma-4-31b-local
python scripts/download_weights.py --verify-only          # verify an existing dir
```

After downloading, the entire `/data/model_cache` can be tarred and shipped to the target host. The target host extracts and `export VIDEVALKIT_MODEL_ROOT=...`.

> Note: `download_weights.py` calls `huggingface_hub.snapshot_download(local_dir=..., local_dir_use_symlinks=False)` (not `cached_download`) so files land directly under `--local-dir`; offline loading does not need the hash-tree dir layout. GitHub release assets use `urllib`.

#### 8.4.4 Loading from Local (no torch.hub / HF network)

The toolkit enforces three rules so offline mode never triggers a download:

1. **All transformers / huggingface calls use local paths or repo id + offline mode.** Adapters uniformly use `AutoModel.from_pretrained(model_path, local_files_only=True)` or `snapshot_download(repo_id, local_dir=..., local_files_only=True)`. Bare `from_pretrained(repo_id)` without `local_files_only` is not allowed.
2. **Every `torch.hub.load(...)` / `torch.hub.download_url_to_file(...)` either uses `source="local"` against a pre-staged dir, or is patched at adapter init to read `$VIDEVALKIT_MODEL_ROOT/<benchmark>/...`.** For the VBench v1 use of YOLOv5 (`torch.hub.load("ultralytics/yolov5", "yolov5x")`), the adapter rewrites it to `torch.hub.load(local_path, "custom", path=...,source="local")`.
3. **vLLM / SGLang launch commands use `--model /local/path/...`** (never an HF repo id) and `export HF_HUB_OFFLINE=1`. The local path looks like `$VIDEVALKIT_MODEL_ROOT/hf/hub/models--google--gemma-3-31b-it/snapshots/<rev>/`.

The code-side entry is `src/videvalkit/utils/model_loader.py`:

```python
from videvalkit.utils.model_loader import (
    resolve_hf_model_path,        # repo_id -> local snapshot dir
    resolve_metric_weight_path,   # ("vbench", "raft-things.pth") -> Path
    ensure_offline_env,           # set HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / etc.
)
```

Every adapter and every VLM Judge backend obtains paths from this module; new Benchmarks / VLMs must follow the same API rather than assembling paths or calling `from_pretrained(repo_id)`.

`ensure_offline_env()` runs once at toolkit `__init__`: when `$VIDEVALKIT_OFFLINE=1` is detected, it auto-exports `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` / `HF_DATASETS_OFFLINE=1`, and sets `HF_ENDPOINT` to an unreachable string so accidental `from_pretrained(repo_id)` calls fail fast rather than quietly timing out.

#### 8.4.5 Doctor's model-weight checks

In offline mode `videvalkit doctor` additionally checks:

- whether `$VIDEVALKIT_MODEL_ROOT` exists and is readable;
- whether each manifest entry is present (existence check against `checksums.json` keys; no full hash, too slow);
- when `$VIDEVALKIT_OFFLINE=1`, whether every Judge resolved to a local path (any Judge that still carries `repo_id` is flagged yellow).

Example output:

```
== Model weights ==
  VIDEVALKIT_MODEL_ROOT=/data/model_cache  exists=OK
  vbench  metric weights: 9/9 present
  vbench2 metric weights: 13/13 present
  hf hub:
    google/gemma-3-31b-it           snapshot=OK at /data/model_cache/hf/hub/...
    Qwen/Qwen3-VL-32B-Instruct      MISS — run scripts/download_weights.py --judge qwen3-vl-32b-local
```

#### 8.4.6 When the downstream cannot pre-download anything

In a few downstream scenarios HF and GitHub are both unreachable; the toolkit preparation team must ship weights offline. Flow:

1. The preparation team (with external network) runs `scripts/download_weights.py --all`.
2. `tar -czf videvalkit-weights-2026-05.tar.gz -C /data/model_cache .`.
3. Hand over the tarball + wheel + `envs/videvalkit.yaml` + DEV_MANUAL.md together.
4. On the target host: `tar -xzf videvalkit-weights-2026-05.tar.gz -C /data/model_cache && export VIDEVALKIT_MODEL_ROOT=/data/model_cache && videvalkit doctor`.

The weight tarball is large (~ 200 GB full); we recommend splitting into "minimum 80 GB" and "recommended 200 GB" tiers so downstream picks one.

---

## 9. User Manual

> **The user-facing how-to-run content has moved to [USER_MANUAL.md](USER_MANUAL_en.md).** That document covers installation, starting VLM backends, workspace preparation, running evaluations, customizing the inputs (videos / prompts / dimensions / judge), and troubleshooting.
>
> The remainder of this DEV_MANUAL focuses on architecture, module decisions, and the engineering rigor that backs releases. The pipeline-verification framework that used to live in TEST_MANUAL §3.2 is now in [§9b below](#9b-pipeline-verification-framework).

### 9a. Quick cross-reference

| You want to … | Doc to read |
|---|---|
| Install the toolkit, run a benchmark, customize inputs | [USER_MANUAL.md](USER_MANUAL_en.md) |
| Know what each metric measures and which paper it reproduces | [TEST_MANUAL.md](TEST_MANUAL.md) |
| Know what scores we got and how they align with the leaderboard | [TEST_MANUAL.md](TEST_MANUAL.md) |
| Understand the architecture, module boundaries, design tradeoffs | this document (§1–§8) |
| Verify a release candidate before tagging | [§9b](#9b-pipeline-verification-framework) below |

### 9b. Pipeline verification framework

A score landing inside the tolerance band is the *output* of a correctly-wired pipeline — not, by itself, evidence that the pipeline is correct. A wrong adapter can coincidentally land in tolerance on one model and drift wildly on the next. Conversely, a slightly-failing score with a perfect-integrity pipeline is more reassuring than a passing score with no integrity check, because the former is debuggable.

The workflow below runs **7 structural checks** before any score-comparison verdict is considered authoritative. Each check emits a fingerprint; a Benchmark is considered "verified for release" only when all 7 fingerprints land in the per-Benchmark `validation/expected/<bench>_paper_*.json:meta.verification` block.

| Stage | Question it answers | Failure mode it catches |
|---|---|---|
| 1. Adapter integrity | Does the toolkit call the upstream library at the right entrypoint? | Upstream patch silently renames a function; our adapter imports a stub that returns 0.0. |
| 2. Dimension coverage | Are all paper dims implemented? | We forgot a dim; the headline mean is computed over n-1 dims. |
| 3. Aggregator fidelity | Does the headline formula match the paper's? | Paper says `0.54·Q + 0.46·S`; we implement `0.5·Q + 0.5·S`; scores all drift by the same constant. |
| 4. Prompt-set integrity | Are the prompts served to the model byte-identical to the paper's? | Upstream hot-fixes a typo; we still ship the pre-typo prompt. |
| 5. Reference-video integrity | Are we scoring the *same generations* the paper scored? | HF dataset maintainer silently re-uploads with different seed; we score the new ones and disagree with the paper. |
| 6. Judge configuration | Is the judge model, snapshot, and rubric identical to the paper's? | `gpt-4o-2024-05-13` was deprecated; calls now land on `gpt-4o-2024-11-20` which is softer by 2 %. |
| 7. Score equivalence | Are the numbers in tolerance? | The bands in TEST_MANUAL §3 do their job. |

> **Implementation status (2026-05-14):** the toolkit-side scripts that automate stages 1–7 (`scripts/validation/compare.py`, `verify_prompts.py`, `checksum_videos.py`, `build_manifest.py`, and per-Benchmark `tests/integration/test_adapter_<bench>.py`) are not yet authored. The 7 stages below describe the target framework. Today, the closest thing is `pytest tests/` (which covers stages 1–3 at the registration level).

#### Stage 1 — Adapter integrity

`videvalkit info <bench>` is not implemented; inspect the registry directly:

```bash
python -c "
from videvalkit.configs import SUPPORTED_BENCHMARKS
cfg = SUPPORTED_BENCHMARKS['<bench>']
cls = cfg['cls']
print('class:', cls.__module__ + '.' + cls.__name__)
print('dims:', cls.dimensions)
print('needs_gpu:', cfg.get('needs_gpu'))
print('needs_judge:', cfg.get('needs_judge'))
print('default_judge:', cfg.get('default_judge'))
print('default_aggregator:', cfg.get('default_aggregator'))
print('env:', cfg.get('env'))
"
```

Then run the in-tree integration test:

```bash
pytest tests/test_integration.py -v
```

Per-Benchmark adapter integration tests (`tests/integration/test_adapter_<bench>.py`) are target-state.

#### Stage 2 — Dimension coverage

```bash
python -c "
from videvalkit.configs import SUPPORTED_BENCHMARKS
for d in SUPPORTED_BENCHMARKS['<bench>']['cls'].dimensions: print(d)
" | sort > /tmp/got_dims.txt

jq -r '.models | values[0] | keys[]' \
    /pub/evaluation_group/ning/toolkit/validation/expected/<bench>_paper_*.json | \
    sort > /tmp/expected_dims.txt

diff /tmp/expected_dims.txt /tmp/got_dims.txt
# Expected: empty diff (modulo composite dims, handled in Stage 3).
```

#### Stage 3 — Aggregator fidelity

```bash
python -c "
from videvalkit.configs import SUPPORTED_BENCHMARKS, SUPPORTED_AGGREGATORS
b = SUPPORTED_BENCHMARKS['<bench>']
print('default_aggregator:', b['default_aggregator'])
print('source:', SUPPORTED_AGGREGATORS[b['default_aggregator']])
"
```

Then manually verify the source file implements the paper's formula. Per-benchmark formulas are listed in [TEST_MANUAL §4](TEST_MANUAL.md#4-per-benchmark-specification) under each benchmark's "Aggregation" subsection.

If the formula mismatches, **stop**. Score comparisons are meaningless until the aggregator is fixed.

#### Stage 4 — Prompt-set integrity

```bash
# Target script (not yet authored):
python scripts/validation/verify_prompts.py \
    --bench <bench> \
    --upstream-prompts /pub/evaluation_group/ning/toolkit/validation/upstream_repos/<repo>/prompts/ \
    --toolkit-prompts <toolkit's bundled prompts dir>

# Manual until then:
diff -q \
    /pub/evaluation_group/ning/toolkit/validation/upstream_repos/VBench/prompts/ \
    /pub/evaluation_group/ning/toolkit/src/videvalkit/benchmarks/vbench/prompts/ 2>/dev/null
sha256sum /pub/evaluation_group/ning/toolkit/validation/upstream_repos/VBench/prompts/all_dimension.txt
```

#### Stage 5 — Reference-video / sample-set integrity

```bash
# Target script (not yet authored):
python scripts/validation/checksum_videos.py \
    --videos /pub/evaluation_group/ning/toolkit/validation/reference_videos/<bench>/<model>/ \
    --manifest /pub/evaluation_group/ning/toolkit/validation/expected/<bench>_video_manifest.json

# Manual until then:
find /pub/evaluation_group/ning/toolkit/validation/reference_videos/<bench>/ -name '*.mp4' -print0 \
    | sort -z | xargs -0 sha256sum > /tmp/local.sha256
```

#### Stage 6 — Judge configuration

For VLM-judged dims, confirm the judge model, snapshot, and rubric all match the paper's.

```bash
head $VALIDATION_ROOT/workspaces/<bench>-T1/api_logs/calls/*/*/*/*.jsonl | \
    jq -r '"\(.endpoint)\t\(.model)\t\(.system_prompt[:80])"'
```

Cross-check three things:

- **Model + snapshot** matches the paper's pinned identifier (`gpt-4o-2024-08-06` ≠ `gpt-4o-2024-11-20`).
- **System prompt / rubric** quotes match the paper's text verbatim — punctuation-sensitive.
- **Chain-of-query structure** (Video-Bench, T2V-CompBench Action/Interaction): number, order, and wording of sub-questions match the paper's algorithm box.

For local judges, verify the served name:

```bash
curl -s http://localhost:8006/v1/models | jq '.data[0].id'
```

#### Stage 7 — Score equivalence within tolerance

Only after stages 1–6 pass is a numeric comparison meaningful.

```bash
# Target script (not yet authored):
python scripts/validation/compare.py \
    --bench <bench> \
    --result $VALIDATION_ROOT/workspaces/<bench>-T1/results/summary/ \
    --expected $VALIDATION_ROOT/expected/<bench>_paper_*.json \
    --report $VALIDATION_ROOT/reports/<bench>-T1-$(date +%F).md
```

Until `compare.py` is authored, comparisons happen via ad-hoc Python loading the per-benchmark `results/summary/{bench}/{model}.json` and diffing against expected values.

#### Verification fingerprint

After all 7 stages pass, emit a single per-Benchmark fingerprint to `reports/verification-<bench>-<date>.json`:

```json
{
  "benchmark": "<bench>",
  "verified_at": "YYYY-MM-DDTHH:MM:SSZ",
  "toolkit_version": "v0.0.1",
  "stages": {
    "adapter_integrity":     {"status": "ok", "upstream_pkg": "vbench==X.Y.Z"},
    "dimension_coverage":    {"status": "ok", "n_dims": 16},
    "aggregator_fidelity":   {"status": "ok", "formula_hash": "<sha>"},
    "prompt_set_integrity":  {"status": "ok", "prompt_sha": "<sha>"},
    "video_manifest":        {"status": "ok", "manifest_sha": "<sha>"},
    "judge_pin":             {"status": "ok", "judge_id": "gpt-4o-2024-08-06"},
    "score_equivalence":     {"status": "ok", "pass_count": "112/112"}
  },
  "fingerprint": "<sha256 of the above object>"
}
```

A Benchmark whose fingerprint has not been re-emitted in 30 days is considered stale and blocks the release acceptance checklist.

#### What this workflow does NOT catch

- **Generator-side bugs.** If a generation model was used incorrectly upstream (wrong seed, wrong cfg), our adapter has no way to detect it; we just score whatever was uploaded.
- **Paper errata.** If the paper's published Table 2 has a typo, our scores will mismatch on the typo'd cell forever. Document in `meta.notes` once confirmed with upstream.
- **VLM provider drift mid-run.** If `gpt-4o-2024-08-06` is silently re-tuned by OpenAI between our calibration date and today, only re-running the full calibration surfaces it. Re-calibrate every 3 months.

---

> The legacy §9 user-manual subsections below are kept temporarily for backward-compatibility with bookmarks. New readers should use [USER_MANUAL.md](USER_MANUAL_en.md).

### 9.1 Installation

**Prerequisites:**

- Linux x86_64 (macOS supports CPU-only flows; cannot run VBench v1 GPU metrics);
- conda / miniforge;
- 8 × GPU recommended (minimum 1 × 80 GB GPU for topology 3);
- If you plan to use managed VLM APIs: network access to that provider.

**One-line install:**

```bash
# 1. Obtain the toolkit (wheel + envs dir + scripts dir)
tar -xzf videvalkit-0.0.1.tar.gz
cd videvalkit-0.0.1

# 2. Create the conda env (toolkit core + 4 anchored Benchmark upstreams + their deps)
conda env create -f envs/videvalkit.yaml
conda activate videvalkit

# 3. Verify
videvalkit --help
videvalkit list benchmarks
```

**Common install failure modes:**

- `detectron2` build fails → check that `nvcc --version` matches `torch.version.cuda`; if not, install the detectron2 prebuilt wheel.
- `flash-attn` raises an ABI error on import → confirm flash-attn matches the torch version; typically `pip install flash-attn==2.5.8 --no-build-isolation`.
- `mmcv` installs but import is missing a dynamic library → use `mim install mmcv==2.2.0` instead of `pip install`.

### 9.2 Pre-downloading Model Weights (Offline / Restricted Network)

> If the target host has direct access to HuggingFace and GitHub, this section is optional; weights download lazily during the first evaluate. **But we recommend at least running `scripts/download_weights.py --verify-only` to surface availability up front** — otherwise you discover a missing ckpt only after evaluate has reached prompt 50 and crashed.

The full model list is in [§8.4.1](#841-the-complete-model-weight-manifest). Steps:

```bash
# 1. Pick a root (≥ 300 GB recommended)
export VIDEVALKIT_MODEL_ROOT=/data/model_cache
export HF_HOME=$VIDEVALKIT_MODEL_ROOT/hf
mkdir -p $VIDEVALKIT_MODEL_ROOT/{hf,vbench,vbench2}

# 2. On an internet-connected host, fetch everything
python scripts/download_weights.py --all
# Or selectively:
python scripts/download_weights.py --benchmark vbench --benchmark vbench2
python scripts/download_weights.py --judge gemma-4-31b-local --judge local-llava-video-7b

# 3. Verify
python scripts/download_weights.py --verify-only
# Prints whether each file's sha256 matches checksums.json

# 4. Offline delivery: pack and ship
tar -czf videvalkit-weights-2026-05.tar.gz -C /data/model_cache .
# On the target host:
tar -xzf videvalkit-weights-2026-05.tar.gz -C /data/model_cache
export VIDEVALKIT_MODEL_ROOT=/data/model_cache
export HF_HOME=$VIDEVALKIT_MODEL_ROOT/hf
export HF_HUB_OFFLINE=1
export VIDEVALKIT_OFFLINE=1     # enable the toolkit's offline guardrail
```

Start vLLM **with a local path, not a repo id**:

```bash
# ❌ Wrong — triggers an HF download attempt
python -m vllm.entrypoints.openai.api_server --model google/gemma-3-31b-it ...

# ✅ Right
GEMMA_DIR=$(ls -d $VIDEVALKIT_MODEL_ROOT/hf/hub/models--google--gemma-3-31b-it/snapshots/*/ | head -1)
python -m vllm.entrypoints.openai.api_server \
    --model "$GEMMA_DIR" \
    --served-model-name google/gemma-3-31b-it \
    --port 8003 ...
```

**Note on `--served-model-name`**: it must match `SUPPORTED_JUDGES[name]['model']` because the toolkit places this string in the `model` field of OpenAI-compatible requests; vLLM routes by that. The local path does not affect this value.

Verify weights are in place:

```bash
videvalkit doctor --workspace /data/ws/eval_2026_05
# Look at the == Model weights == section:
#   vbench  metric weights: 9/9 present
#   hf hub: google/gemma-3-31b-it  snapshot=OK at /data/model_cache/...
```

### 9.3 Starting the VLM Backends

Per the [§7.2 port plan](#72-remote-vlm-port-convention-80018009), start the VLMs you need. Minimum one; typical three (Gemma-3-31B + Qwen3-32B + LLaVA-Video-7B).

```bash
# On the GPU server (same host as the toolkit or different)
conda activate videvalkit  # or a dedicated vLLM env

# 8003: Gemma-3-31B
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-31b-it --port 8003 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --served-model-name google/gemma-3-31b-it \
  > logs/vllm_8003.log 2>&1 &

# 8004: Qwen3-32B
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B --port 8004 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --served-model-name Qwen/Qwen3-32B \
  > logs/vllm_8004.log 2>&1 &

# 8006: LLaVA-Video-7B
CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server \
  --model lmms-lab/LLaVA-Video-7B-Qwen2 --port 8006 \
  --max-model-len 32768 \
  --served-model-name lmms-lab/LLaVA-Video-7B-Qwen2 \
  > logs/vllm_8006.log 2>&1 &
```

**Wait until the model has loaded** (30 s – 10 min, depending on size and disk IO):

```bash
curl -s http://localhost:8003/v1/models | jq
# When the expected model name appears, you are good
```

**If you also use managed-API Judges**, set the keys in `.env` (or `export`):

```bash
export GEMINI_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

> The toolkit does not read `.env` files; either `export` directly or `set -a; source .env; set +a`.

### 9.4 Preparing a Workspace

```bash
# Layout: videos by model in subdirs
ls /data/my_videos/
# cogvideox-5b/  wan-2.1/  hunyuanvideo/  ...

# One-shot workspace bootstrap
videvalkit prepare-workspace \
  --workspace /data/ws/eval_2026_05 \
  --videos /data/my_videos
```

The toolkit creates the standard directories under the workspace and symlinks `/data/my_videos/{model}/` into `videos/{model}/`. Videos are not copied.

**Video naming requirement:** `videos/{model}/{prompt_id}-{idx}.mp4`. `prompt_id` must resolve in the relevant Benchmark's `prompts/{benchmark}/prompts.jsonl`; `idx` is the 0-based sample index (distinguishes multiple samples per prompt).

### 9.5 Running a Single Benchmark

**VBench v1 (pure metric, no VLM):**

```bash
videvalkit eval \
  --bench vbench \
  --videos /data/ws/eval_2026_05/videos \
  --workspace /data/ws/eval_2026_05 \
  --models cogvideox-5b --models wan-2.1 \
  --dimensions subject_consistency --dimensions motion_smoothness
```

**VBench-2.0 (requires VLM):**

```bash
videvalkit eval \
  --bench vbench2 \
  --videos /data/ws/eval_2026_05/videos \
  --workspace /data/ws/eval_2026_05 \
  --judge local-llava-video-7b
```

**Video-Bench (VLM-only):**

```bash
videvalkit eval \
  --bench videobench \
  --videos /data/ws/eval_2026_05/videos \
  --workspace /data/ws/eval_2026_05 \
  --judge gemini-2.5-pro
```

**WorldJen (VLM + PHAS aggregation):**

```bash
videvalkit eval \
  --bench worldjen \
  --videos /data/ws/eval_2026_05/videos \
  --workspace /data/ws/eval_2026_05 \
  --judge gemma-4-31b-local
```

After each command:

- `results/raw/{bench}/{model}/{dim}/{prompt_id}.json` — one JSON per (model, dim, prompt)
- `results/summary/{bench}/{model}.json` — official aggregated scores for that (Benchmark, model)
- `api_logs/calls/{provider}/...` — every VLM call

### 9.6 Cross-Benchmark Aggregation

After running multiple Benchmarks:

```bash
videvalkit aggregate --workspace /data/ws/eval_2026_05
# Output: results/leaderboard/cross_benchmark.json
# Console also prints the unified z-score ranking
```

### 9.7 Resume and Retry

A restart is a resume. The toolkit skips (model, dim, prompt) triples whose `results/raw/*.json` already exists. **There is no `--resume` flag** — it is the default behavior.

To force a re-run for a specific triple, delete the JSON. To clear a Benchmark entirely:

```bash
rm -rf /data/ws/eval_2026_05/results/raw/vbench2
rm -rf /data/ws/eval_2026_05/results/summary/vbench2
videvalkit eval --bench vbench2 ...
```

### 9.8 Troubleshooting and `doctor`

Run first:

```bash
videvalkit doctor --workspace /data/ws/eval_2026_05
```

Output resembles:

```
== Conda envs ==
  videvalkit    OK
== Adapter imports ==
  vbench        OK
  vbench2       OK
  videobench    OK
  worldjen      OK
== Judges ==
  gemma-4-31b-local        reach=OK    key=--    (openai_compatible)
  qwen3-32b-local          reach=OK    key=--    (openai_compatible)
  qwen3-vl-32b-local       reach=MISS  key=--    (openai_compatible)
  local-llava-video-7b     reach=OK    key=--    (openai_compatible)
  gemini-3-flash           reach=--    key=OK    (gemini)
  gemini-2.5-pro           reach=--    key=OK    (gemini)
  claude-sonnet-4-6        reach=--    key=MISS  (anthropic)
  gpt-4o                   reach=OK    key=OK    (openai_compatible)
== Workspace ==
  videos        exists=OK    n=4
  prompts       exists=OK    n=4
  results/raw   exists=OK    n=1247
```

`reach=MISS` usually means the corresponding vLLM has not been started or has crashed; `key=MISS` means the managed-Judge API key has not been exported. Neither blocks unless you actually use that Judge; doctor just lists which Judges are currently unavailable.

`--json` mode emits a stable structure for agents:

```bash
videvalkit doctor --workspace /data/ws/eval_2026_05 --json | jq
```

**Common errors:**

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `reach=MISS` while vLLM is running | `--served-model-name` doesn't match the registry's `model` | `curl /v1/models` to see the actual name; align |
| evaluate hangs on a specific prompt | VLM endpoint stuck (OOM, handle leak) | restart that endpoint; resume skips completed work |
| `results/summary/...` is missing some dims | VBench v1 leaves some dims off by default | specify `--dimensions` explicitly |
| `api_logs/...` reaches several GB | high call volume | `scripts/clear_cache.py --api-logs --older-than 30d` |
| `frames_cache/` reaches tens of GB | many videos | same as above, `--frames-cache` |

---

## 10. Non-Functional Requirements

### 10.1 Scale and Extensibility

- A single workspace supports ≥ 100k RawResult JSONs, ≥ 10 models, ≥ 4 Benchmarks; aggregation stays acceptable beyond that (< 60 s aggregation).
- Per-endpoint VLM concurrency stays ≤ 64; beyond that, scale horizontally with replicas.
- Distributed multi-machine is out of scope for v1; the interface is reserved ([§5.4 D5](#543-tier-2-requirements)).

### 10.2 Performance Expectations

- `videvalkit list` / `doctor` / `aggregate` should return in seconds (doctor ≤ 30 s because it probes endpoints).
- `eval`'s GPU portion is bounded by the worker pool (full single-card utilization); HTTP portion is bounded by the token bucket (we will not flood managed APIs).
- Resume decisions should complete within 30 s of restart (scan of `results/raw/`).

### 10.3 Security and Permissions

- API keys via environment variables only — not in source, not in jsonl logs.
- Managed-API call records are stored in jsonl in plaintext; downstream teams must control access to `api_logs/`.
- The toolkit has no remote-execution capability (never opens a listener); all calls are outbound HTTP / SDK.

### 10.4 Observability

- `api_logs/calls/...jsonl` is the source of truth: token usage, latency, error codes.
- We recommend that downstream teams run a daily `scripts/api_cost.py` summary by provider / model / day.

### 10.5 Portability

- Toolkit core is OS-agnostic; the GPU path is validated on Linux only. macOS / Windows support only the CLI + managed-Judge flow.
- We do not pin Python to a minor version: `>=3.10`; 3.10 recommended.

### 10.6 Maintainability

- Every module has tests under `tests/`; CI runs them. Today: 17 core unit tests + 3 HTTP integration tests (stdlib fake server) + 7 adapter integration tests = 27 tests, no external services or GPUs required.
- Every registry change requires a test; CI blocks untested registry rows.

### 10.7 Reproducibility

The toolkit forces every `eval` to emit a `meta.repro` field in the persisted `summary.json` for *exact reproduction*. Required fields:

| Field | Source | Description |
| ----- | ------ | ----------- |
| `toolkit_commit` | `git rev-parse HEAD` (toolkit) | videvalkit code version |
| `adapter_version` | each Benchmark adapter's `__version__` | upstream wrapper version |
| `upstream_pkg` | `vbench==X.Y.Z` / `vbench2==X.Y.Z` | upstream PyPI / git ref |
| `judge` | `f"{provider}:{model}"` + `system_fingerprint` (when OpenAI exposes it) | VLM identifier used |
| `ckpt_checksums` | `scripts/download_weights.py --verify` output | SHA-256s of local model weights used |
| `seed` | the adapter passes one if it does any random sampling | otherwise `null` |
| `runtime` | `python` + `torch` + `transformers` + `cuda` versions | also printed by doctor |

> When the Judge is a managed API (Gemini / Anthropic / GPT), the same `model` field can behave differently on different days; when publishing leaderboards externally, archive `api_logs/calls/{provider}/{model}/...` alongside the summary (zip).

Enforcement: `tests/test_summary_repro.py` checks every required field; CI blocks anything missing.

### 10.8 Scale and Compute / Token Estimates (v0.0.1 reference)

| Benchmark | dims | prompts | samples/prompt | RawResults/model | Wall-clock per model (1 × A100) | Judge cost estimate (gpt-4o) |
| --------- | ---: | ------: | -------------: | ----------------: | ------------------------------- | ----------------------------- |
| vbench (v1) | 16 | 946 | 5 | ~75 680 | 6–8 h (includes RAFT/CLIP/SubjectConsistency) | n/a |
| vbench2 | 18 | 200 | 4 | ~14 400 | 5–7 h (includes LLaVA-Video-7B inference) | n/a local / ~$45 managed |
| videobench | 14 | 800 | 1 | ~11 200 | Judge time only, no GPU | ~$25 |
| worldjen | 5 | 110 | 1 | 550 | < 1 h judge + < 30 min CV | < $5 |
| t2vcompbench | 7 | 700 | 1 | ~4 900 | 3 h (GD + Depth + SAM + LLaVA) | ~$15 |
| worldscore | 10 | 2 000 (1 000 static + 1 000 dynamic) | 1 | 20 000 | ~4 h for 150 prompts on 4× L20 (motion_smoothness with VFIMamba is dominant; static dims ~ 12 min each, motion_smoothness ~ 35–60 min, motion_accuracy ~ 10 min, 3d_consistency ~ 30 min, camera_control ~ 11 min) | n/a |
| **All 6 for one model** | — | ~5 756 | — | **~121 730** | **~22 h** (serial, single GPU) | **~$85** (worst-case managed judge) |

> Use this table to plan "N models, how long, how much money." Real numbers swing ±30% with concurrency, resolution, and Judge choice; treat the table as the median. If all Judges run on local vLLM, managed cost goes to zero but GPU-hours rise ~1.5×.

---

## 11. Milestones

### 11.1 Phase 1 — Single-Benchmark Eval Runs + VLM Integration Works (M0)

- Module A: all tier-1 requirements
- Module B: 4 anchored adapters minimally usable (B1–B4)
- Module C: 3 backends + factory + ApiCallLogger + FrameCache (C1–C4)
- Module D: 3 pools + facade (D1–D4)
- Module E: single-workspace + results/raw resume + api_logs schema (E1–E4)
- Module G: 5 CLI subcommands + runner (G1–G3)
- Module H: wheel build + envs/videvalkit.yaml + doctor for external readers (H1–H4)

**M0 exit criteria:** internally we run 1 model × 4 Benchmarks end-to-end; a downstream team can self-deploy in 1 day after receiving the package.

**v0.0.1 completeness against M0 (as of 2026-05-13):**

| M0 item | Status | Note |
| ------- | ------ | ---- |
| Module A tier-1 | ✅ | `BaseBenchmark / BaseScorer / BaseAggregator` plus types + layout (~390 LOC) |
| Module B × 4 (M0 anchored) | 🟡 | vbench / vbench2 / videobench / worldjen adapter skeletons in place; per-Benchmark score-reproduction fidelity per sub-README |
| Module B × 2 (M0.5 added) | 🟡 / ✅ | t2vcompbench skeleton in place; **worldscore upgraded to full 10-of-10 via upstream metric classes** (`run_evaluate.py`-aligned headlines, `aspect_info` normalization, `interpframe_num=49`, WorldScore HF reference images, spaCy class-matching). |
| Module C × 3 backends | ✅ | OpenAI-compatible / Gemini / Anthropic + factory + 8 Judges registered |
| Module D × 4 pools/facade | ✅ | GPU pool / HTTP pool / RateLimit / EnvDispatcher (bypassed under the shared env) |
| Module E workspace + api_logs | ✅ | layout + api_log writer landed |
| Module G CLI | 🟡 | 3 subcommands (`list / doctor / aggregate`); `eval` via Python `runner.run()`; CLI subcommand lands in M1 |
| Module H packaging | 🟡 | `pyproject.toml` + 6 conda env yamls; wheel build not run |

**Still owed at M0:** the `videvalkit eval` and `videvalkit prepare-workspace` CLI subcommands; downstream wheel build + smoke test.

### 11.1.5 Milestone M0.5 — Two Supplementary Benchmarks (landed)

Added after the M0 anchor and shipped in v0.0.1:

- **t2vcompbench** — compositional 7-dim (GroundingDINO + Depth + SAM + LLaVA-Video)
- **worldscore** — **10-of-10 dims** via upstream metric classes. Each dim's per-instance score is produced by the same class the public WorldScore code imports (`from worldscore.benchmark.metrics import CLIPScoreMetric, ObjectDetectionMetric, ...`); per-instance normalization uses upstream's `aspect_info`; the two headlines `WorldScore-Static` (mean of 7 static dims) and `WorldScore-Dynamic` (mean of all 10) match `run_evaluate.py:165-166`. Reference images for `style_consistency` are extracted from the public WorldScore HF dataset (`Howieeeee/WorldScore`) by `runners/extract_refs.py`. `motion_accuracy` uses upstream's spaCy class-match rate (not a simple n_masks cap). Frame sampling = 49 (= upstream's `interpframe_num`). The adapter ships a pure-PyTorch `mamba_ssm.selective_scan` shim so VFIMamba (`motion_smoothness`) runs without recompiling CUDA kernels.

Score-reproduction notes live in `src/videvalkit/benchmarks/<name>/README.md` (pending).

### 11.2 Phase 2 — Aggregation + Cross-Benchmark Report + Upstream-Upgrade Channel (M1)

- Module F: BT + cross combine + PHAS calibration (F1–F4)
- Module B: a README and a smoke test per Benchmark (B5, B6) — all 6
- Module C: token-usage statistics (C5)
- Module G: `eval` / `prepare-workspace` CLI subcommands land (M0 debt)
- Module H: install script (H5)

**M1 exit criteria:** at least one downstream has run ≥ 2 Benchmarks + `aggregate` + filed ≥ 1 issue / PR; all 6 CLI subcommands documented; all 6 Benchmarks have a README and a smoke test.

### 11.3 Phase 3 — Usability and Extension (M2)

- Module C: call replay + batch API (C6, C7)
- Module D: multi-machine scheduling (D5)
- Module E: large-log archiving + leaderboard zip (E5, E6)
- Module F: CLI markdown report export (F5, F6)
- Module G: explain / replay / config files (G4–G6)
- Module H: offline bundle + CHANGELOG automation (H6, H7)

**M2 exit criteria:** the multi-machine scheduler runs a single long job (≥ 24 h) with zero data loss; the offline install bundle runs end-to-end with no network; CHANGELOG automation + Docker image (if §12.5 lands "yes").

### 11.4 Phasing Note

Module priority and milestones do not align one-to-one — Module F is P1 by module priority but the cross-Benchmark report has high external-demo value and may move into M1 early. Adjust per actual demand each milestone.

---

## 12. Open Questions

### 12.1 External extensibility of the registries

Allowing the downstream to inject extra Judges via `$VIDEVALKIT_JUDGES_FILE` is the recommended path today (avoid forking source). Should we also support `$VIDEVALKIT_BENCHMARKS_FILE` / `$VIDEVALKIT_AGGREGATORS_FILE`? Currently not, because those extensions usually need new code — but a small parameter override (e.g. Aggregator weights) could go through a file.

### 12.2 Remote VLM port convention

8001–8009 is a suggestion, not enforcement; downstream teams are free to change ports. Do we document "use the convention" vs "free choice" as two recommendations? Lean toward not enforcing; the manual's examples all use the convention.

### 12.3 Long-term feasibility of VBench-2.0 + VBench v1 sharing one env

Today they coexist with reasonable compromises, but any upstream upgrade can break this. When it breaks, do we fall back to per-Benchmark env (`CondaEnvDispatcher`), or fork into `envs/videvalkit-vbench2.yaml`? Lean toward the former — the code path is ready.

### 12.4 Reproducibility of managed-API Judges

Gemini / Anthropic / GPT all iterate continuously; the same `model` field can behave differently on different days. Do we also record `system_fingerprint` (OpenAI field) or similar? Today we record only the `model` string and `provider`.

### 12.5 Dockerization

Do we ship an official Docker image (toolkit-core + envs/videvalkit) plus a docker-compose (with an example vLLM service)? Depends on downstream demand. Defer through phase 1; decide in phase 2.

### 12.6 Sensitive data handling

`api_logs/calls/.../jsonl` writes the prompt text and (elided) response. If downstream content is sensitive, do we provide a "redact-on-write" mode? Today none — downstream controls directory permissions.

### 12.7 Model-manifest version alignment

The manifest in [§8.4.1](#841-the-complete-model-weight-manifest), `scripts/download_weights.py`, and `checksums.json` must stay in sync. Today this is manual PR discipline. Do we add `tests/test_weights_manifest.py` to verify alignment in CI? Lean toward yes; not blocking v1.

### 12.8 Fallback for upstream Benchmarks' own download behavior

VBench and VBench-2.0 spread their metric-model download calls across multiple files; the toolkit injects `MODEL_ZOO_PATH` via `_inject_model_root()`. **An upstream upgrade can introduce a new download site** (e.g. `torch.hub.load("ultralytics/yolov5", "yolov5x")`) that the toolkit hasn't patched, breaking the offline host. Do we run an `--offline` smoke in CI (with `unshare -n`)? Lean toward yes; needs a GPU runner.

### 12.9 PHAS default weights

WorldJen's PHAS defaults come from the paper; downstream scenarios may differ. Do we ship an "industry default weights" preset alongside the "paper weights" in the release? Coordinates with F4.

---

## 13. Prompt Auto-Labeler (added 2026-05-13)

### Why this exists

VBench v1 and VBench-2.0 split their dimensions into:

* **prompt-agnostic** — algorithm needs only the video (e.g. CLIP / DINO /
  RAFT). Works on any video out of the box.
* **prompt-dependent** — algorithm reads per-prompt `auxiliary_info` tags
  from upstream's `VBench_full_info.json` / `VBench2_full_info.json`. Use
  custom prompts without those tags and the algorithm either refuses
  (`custom_input` mode) or hits `ZeroDivisionError` (content filter).

### Affected dims

| Benchmark | Prompt-dependent dim | Why |
|---|---|---|
| VBench v1 | `color`, `object_class`, `multiple_objects`, `spatial_relationship`, `scene`, `appearance_style`, `human_action` | Needs ground-truth tags |
| VBench-2.0 | `Composition`, `Dynamic_Attribute`, `Dynamic_Spatial_Relationship`, `Instance_Preservation`, `Complex_Plot`, `Complex_Landscape`, `Motion_Rationality`, `Motion_Order_Understanding`, `Mechanics`, `Thermotics`, `Material`, `Camera_Motion`, `Human_Interaction` | Same |

### The pipeline

`scripts/auto_label_prompts.py` calls a local LLM (default Qwen3-32B at
:8004 via the `qwen3-32b-local` judge in `SUPPORTED_JUDGES`) to read each
prompt and emit per-dim auxiliary_info in upstream's **exact format**.

Two output files (drop-in replacements for upstream's full_info JSON):

* `vbench_full_info.json` — VBench v1 format
  (one entry per prompt; `auxiliary_info` is a dict keyed by dim)
* `vbench2_full_info.json` — VBench-2.0 format
  (one entry per `(prompt, dim)` pair; `auxiliary_info` is the bare
  value — list / string / object — matching upstream's per-dim conventions)

### Usage

```bash
PY=/pub/evaluation_group/ning/benchmark/envs/videvalkit/bin/python

$PY scripts/auto_label_prompts.py \
    --prompts /pub/evaluation_group/ning/video_eval/worldjen_local/data/prompts/prompts_110.jsonl \
    --out-dir $WORKSPACE_ROOT/prompts/auto_labels \
    --benchmarks vbench,vbench2 \
    --judge qwen3-32b-local
```

Output is incremental (resumable): each prompt × dim is committed to disk
right after LLM completion; killing and re-running picks up where it left
off.

### Sample output

Input prompt (WorldJen 110-prompt set, id `000`):

> *"In the school hallway, a girl with a ponytail suddenly slaps the boy
> next to her. The boy covers his cheek, eyes widened, staring at her with
> a bewildered expression."*

**vbench_full_info.json**:
```json
{
  "prompt_id": "000",
  "prompt_en": "...",
  "dimension": ["object_class", "multiple_objects",
                "spatial_relationship", "scene", "human_action"],
  "auxiliary_info": {
    "object_class":         {"object": "girl"},
    "multiple_objects":     {"object": "girl and boy"},
    "spatial_relationship": {"spatial_relationship":
                              {"object_a": "boy",
                               "object_b": "girl",
                               "relationship": "next to"}},
    "scene":                {"scene": {"scene": "school hallway"}}
  }
}
```

**vbench2_full_info.json** (one entry per dim):
```json
[
  {"prompt_id": "000", "prompt_en": "...",
   "dimension": ["Motion_Order_Understanding"],
   "auxiliary_info": ["slaps", "covers cheek", "eyes widen",
                      "stares bewildered"]},
  {"prompt_id": "000", "prompt_en": "...",
   "dimension": ["Motion_Rationality"],
   "auxiliary_info": [
     "Does the boy's reaction appear to be a natural response to being slapped? (yes or no)",
     "Does the motion of the slap look physically plausible and not exaggerated? (yes or no)",
     "..."]},
  {"prompt_id": "000", "prompt_en": "...", "dimension": ["Human_Interaction"]},
  {"prompt_id": "000", "prompt_en": "...", "dimension": ["Human_Anatomy"]}
]
```

### How to use the auto-labels with the benchmark adapter

After generation, point the adapter at the auto-labeled JSON instead of
upstream's bundled file. Two ways:

1. **Per-call override** — pass `full_info_dir=` kwarg through the
   evaluate() chain:
   ```python
   bench.evaluate(
       layout=layout, models=[...], dimensions=[...],
       full_info_dir=str(workspace / "prompts/auto_labels/vbench_full_info.json"),
   )
   ```
   (adapter already forwards `**kwargs` to upstream's `VBench(...)`)

2. **Persistent override** — symlink your auto-labeled file over the
   upstream default so all toolkit calls use it:
   ```bash
   ln -sf $WORKSPACE_ROOT/prompts/auto_labels/vbench_full_info.json \
       /pub/evaluation_group/ning/benchmark/VBench_repo/vbench/VBench_full_info.json
   ```

For VBench-2.0, the adapter passes `full_info_dir=<pkg_dir>/VBench2_full_info.json`
— same swap pattern.

### Per-dim schemas the LLM follows

Defined in `videvalkit/utils/prompt_labeler.py`:

* `VBENCH_V1_DIM_SCHEMAS` — 7 dims
* `VBENCH2_DIM_SCHEMAS` — 5 dims (including the 2 motion-related ones the
  user cares about: `Motion_Order_Understanding` and `Motion_Rationality`)

Each schema entry has:
* `instruction` — what to ask the LLM
* `format` — JSON shape the LLM should return
* `to_aux` — function mapping parsed LLM output → upstream's exact
  auxiliary_info schema (returns `None` to skip dim, returns `{}` to
  tag-only without aux info)

To add a new dim:
1. Add to one of the `*_DIM_SCHEMAS` dicts
2. Verify a small test run produces parseable JSON in the right shape
3. (Optional) Add a smoke check to `tests/test_skeleton.py`

### Caveats / honest limitations

* **LLM extraction is not perfect**. Qwen3-32B will occasionally:
  - say "no color mentioned" when there is one
  - extract the wrong primary object when multiple are mentioned
  - mis-order a sequence
* Run quality control on the output before trusting it for leaderboard-grade scoring.
* **Auto-labeled scores are not directly comparable** to VBench / VBench-2.0
  public leaderboards. Leaderboards use upstream's exact curated prompt
  suite. Auto-labels are useful for:
  - Comparing your own models on a fixed in-house prompt set
  - Coverage analysis ("does my model handle these dims at all?")
  - Sanity-checking VBench's metric outputs against your prompts
* **Some dims can't be cleanly extracted** from narrative prompts.
  E.g. `appearance_style` rarely matches anything in narrative prompts
  unless the prompt explicitly says "in Van Gogh style". Expect coverage
  to be lower than the prompt-agnostic dim set.

### Files added

* `src/videvalkit/utils/prompt_labeler.py` — core labeler + schemas
* `scripts/auto_label_prompts.py` — CLI driver
* `DEV_MANUAL.md` §13 — this section

---

## 14. Standalone Metrics Module — `videvalkit.metrics` (planned 2026-05-18)

### Why this exists

The current 6-benchmark layout (`vbench`, `vbench2`, `videobench`, `worldjen`, `worldscore`, `t2vcompbench`) covers **prompt-suite-based** evaluation: each adapter ships with a curated prompt list, per-dim scorers, and an aggregator. Some users want something more atomic: **a single algorithm computed on two sets of videos**, e.g.

- **FID** between generated frames and real frames (Inception-V3 features)
- **FVD** between generated videos and real videos (I3D / VideoMAE features)
- **CLIP-Score** between video frames and prompts (already inside `scorers/`, but exposed as a benchmark-internal scorer only)
- **PSNR / SSIM** for paired video evaluation
- **LPIPS** for perceptual similarity

These don't fit the `Benchmark` interface — they don't have prompts, dims, or aggregators. They're **standalone metrics**.

### Module shape

```
src/videvalkit/metrics/
    __init__.py              # SUPPORTED_METRICS registry + factory
    base.py                  # Metric ABC: score(gen, ref, **kwargs) -> dict
    fid.py                   # FID via clean-fid / pytorch-fid (Inception-V3 pool3)
    fvd.py                   # FVD via the canonical I3D Kinetics-400 implementation
    psnr.py                  # paired video PSNR (simple wrapper)
    ssim.py                  # paired video SSIM
    lpips.py                 # LPIPS (AlexNet / VGG backbones)
    clip_score.py            # CLIP-Score lifted out of scorers/ for standalone use
```

### `Metric` interface

```python
class Metric(ABC):
    name: str                      # registry key
    requires_ref: bool             # True for FID/FVD/PSNR; False for CLIP-Score
    accepts_paired: bool           # True if the metric is "for each (gen, ref) pair"
                                   #   vs "two distributions"
    output_keys: tuple[str, ...]   # what `score()` returns

    @abstractmethod
    def score(
        self,
        gen_videos: list[Path],
        ref_videos: list[Path] | None = None,
        *,
        prompts: list[str] | None = None,   # only used by prompt-aware metrics
        device: str = "cuda",
        **kwargs,
    ) -> dict[str, float]:
        """Single-shot metric on the two sets. Returns a flat dict."""
```

### Registry

A new top-level registry `SUPPORTED_METRICS` (matching `SUPPORTED_BENCHMARKS`, `SUPPORTED_JUDGES`, `SUPPORTED_AGGREGATORS`).

```python
SUPPORTED_METRICS = {
    "fid":      {"cls": FIDMetric,    "needs_gpu": True,  "requires_ref": True},
    "fvd":      {"cls": FVDMetric,    "needs_gpu": True,  "requires_ref": True},
    "clip_score": {"cls": CLIPScoreMetric, "needs_gpu": True, "requires_ref": False},
    "psnr":     {"cls": PSNRMetric,   "needs_gpu": False, "requires_ref": True, "accepts_paired": True},
    "ssim":     {"cls": SSIMMetric,   "needs_gpu": False, "requires_ref": True, "accepts_paired": True},
    "lpips":    {"cls": LPIPSMetric,  "needs_gpu": True,  "requires_ref": True, "accepts_paired": True},
}
```

This is the **fourth registry** alongside benchmarks / judges / aggregators (see [§3.3](#33-the-three-registries) which becomes "Four Registries").

### CLI surface

```bash
videvalkit metric \
    --name fvd \
    --gen-videos path/to/generated/ \
    --ref-videos path/to/reference/ \
    --device cuda:0 \
    --out fvd_result.json
```

Returns a flat JSON: `{"fvd": 132.4, "n_gen": 200, "n_ref": 200, "backbone": "i3d-k400", ...}`. No prompts, no dims, no aggregator. Compatible with `videvalkit aggregate` if the user wants to combine multiple metric outputs into one report.

### Object-model implications

Add `Metric` to [§3.2 Object Definitions](#32-object-definitions) between `Scorer` and `Aggregator`:

```
Benchmark → PromptItem → VideoSpec → Scorer → RawResult → Aggregator → Summary
                                              │
                                              └── (separate path) ──> Metric → MetricResult
```

`MetricResult` is a new core type (analogous to `Summary` but flat — no per_dim):

```python
@dataclass
class MetricResult:
    name: str                      # e.g. "fid"
    value: float
    n_gen: int
    n_ref: int | None
    backbone: str | None           # e.g. "inception-v3", "i3d-k400"
    extras: dict[str, Any]         # per-metric details
    timestamp: str
```

### Initial implementation priority (suggested order)

1. **FVD** — most-requested, no toolkit equivalent today
2. **FID** — straightforward, pin to `clean-fid` library for reproducibility
3. **CLIP-Score (standalone)** — lift from existing `scorers/clip_score.py`, no new deps
4. **PSNR / SSIM** — trivial, paired-video case useful for video restoration evals
5. **LPIPS** — moderate, requires AlexNet / VGG checkpoints

### What this is NOT

- Not a benchmark — no curated prompts, no per-dim aggregation, no leaderboard alignment
- Not a replacement for the 6 benchmark adapters — it's the **atomic primitive** users can call when they want one number
- Not part of `videvalkit aggregate` cross-benchmark reports unless the user explicitly opts in

### Files to add

* `src/videvalkit/metrics/__init__.py` + `base.py` + one file per metric
* `src/videvalkit/configs/metrics.py` (registry, parallel to `judges.py`, `benchmarks.py`)
* `src/videvalkit/cli.py` — add `metric` subcommand
* `tests/test_metrics_smoke.py` — one smoke per metric
* `DEV_MANUAL.md` §14 — this section
* `USER_MANUAL.md` — add "Running a single metric" subsection

---

## 15. Open-Source Distribution (planned 2026-05-18)

The toolkit currently lives at `/pub/evaluation_group/ning/toolkit/` and assumes the user has access to `/pub/evaluation_group/ning/benchmark/envs/videvalkit/`, the reference videos under `/pub/evaluation_group/ning/toolkit/validation/reference_videos/`, and the upstream checkpoints at `/pub/evaluation_group/ning/toolkit/validation/upstream_ckpts/`. None of this is true for an external user who clones the GitHub repo and lands on an empty box.

This section is the **distribution plan** for those external users.

### 15.1 Distribution surfaces

**Five artifact channels** (updated 2026-05-19 — `env-tarball` and the Docker image are the canonical install paths; `conda env create` from yaml is retained but flagged experimental):

| Channel | Hosts | Why |
|---|---|---|
| **GitHub** (`videogenevalkit/videogenevalkit`) | code, configs, scripts, manuals, tests, Dockerfile | source of truth for the toolkit |
| **HF dataset** (`videogenevalkit/env-tarball`) | conda-pack snapshot of the working env, ~7.6 GB compressed | **canonical install** — bypasses pip resolution entirely; users `tar xzf` and run |
| **HF dataset** (`videogenevalkit/checkpoints`) | model weights, ~ 125 GB total | per-bench fetch via `videvalkit fetch-checkpoints` |
| **HF dataset** (`videogenevalkit/smoke-data`) | small (~ 3 GB) sample videos + prompt JSONs across all 6 benches | "can I run anything?" before downloading the heavy stuff |
| **GHCR** (`ghcr.io/videogenevalkit/videogenevalkit`) | Docker image built from `env-tarball` + repo source, ~15-25 GB | one-command install for users with NVIDIA Container Toolkit |

User journey (zero-to-first-score), **tarball path** (recommended):

```
1. clone the GitHub repo
2. hf download videogenevalkit/env-tarball videvalkit-env.tar.gz     (~ 7.6 GB, ~ 5 min)
3. tar xzf videvalkit-env.tar.gz -C /opt/videvalkit-env               (~ 1 min)
4. source /opt/videvalkit-env/bin/activate && conda-unpack            (~ 30 s)
5. pip install --no-deps -e .                                         (~ 10 s)
6. bash scripts/post_install.sh                                       (~ 15 min — git installs)
7. videvalkit doctor                                                  (sanity check)
8. videvalkit fetch-smoke-data + fetch-checkpoints + eval             (per-bench)
```

Tarball path total: **~ 30 minutes**, zero install-time dependency resolution, zero failure modes from PyPI/sdist conflicts.

**Docker path** (alternative — same outcome, different packaging):

```
1. docker pull ghcr.io/videogenevalkit/videogenevalkit:0.1.0          (~ 10-25 min image pull)
2. docker run --rm --gpus all videogenevalkit/videogenevalkit:0.1.0 doctor
3. docker run --rm --gpus all -v ...:/cache videogenevalkit/videogenevalkit:0.1.0 \
       eval --bench worldjen ...
```

**Why `conda env create -f envs/videvalkit.yaml` is now experimental:** the lock has ~350 transitive version pins. The original env at `/pub/.../envs/videvalkit/` was built incrementally with many `pip install --no-deps` patches over a multi-week validation campaign. Pip's resolver cannot reconstruct this from scratch — testing during 2026-05-19 showed 10+ consecutive failures iterating on triton, fsspec, pillow, mmcv, detectron2, aliyun-*, droid_backends conflicts. The yaml is retained for contributors who want to debug or evolve the env; not recommended for end users.

The conda-pack tarball at `videogenevalkit/env-tarball` is regenerated from the working env whenever the maintainer runs:

```bash
conda-pack \
    -p /path/to/working/env \
    --ignore-editable-packages \
    --ignore-missing-files \
    --exclude "lib/python3.10/site-packages/SAM_2-*" \
    --exclude "lib/python3.10/site-packages/groundingdino*" \
    --exclude "lib/python3.10/site-packages/vbench*" \
    --exclude "lib/python3.10/site-packages/videvalkit*" \
    --output videvalkit-env.tar.gz \
    --n-threads 16
```

then uploads via `huggingface_hub.HfApi.upload_file(repo_id="videogenevalkit/env-tarball", ...)`. End-to-end ~5 minutes on a fast link.

### 15.2 Conda env packaging

Currently two envs are referenced in `DEV_MANUAL` §8.1:

| Env file | Contents | Used by |
|---|---|---|
| `envs/videvalkit.yaml` | Python 3.10, torch 2.3, transformers, cv2, decord, clean-fid, pyiqa, sam2, GroundingDINO (HF variant), Depth-Anything V2, Co-Tracker3, spacy, pydantic, click, httpx, … | Default — covers all benchmark adapters in toolkit-mode + standalone metrics |
| `envs/videvalkit-vbench.yaml` | base env + installed `vbench`, `vbench2` packages | VBench v1 / VBench-2.0 only |

**Packaging tasks** (in order):

1. **Pin every dependency**. Replace `>=` ranges with `==` in both YAMLs after a clean install. Capture the resolved environment via `conda env export --from-history` (best for portability) and `conda env export` (full lock, for reproducibility CI).
2. **Test the YAMLs on a clean machine.** Spin up a fresh container, run `conda env create -f envs/videvalkit.yaml`, run `videvalkit doctor`, run smoke tests.
3. **Add `envs/videvalkit-cpu.yaml`** — drops torch CUDA pin so users on CPU-only / Mac M1 can at least exercise the *managed-API* benchmarks (Video-Bench with GPT-4o, WorldJen with Gemini).
4. **Document the env-name → adapter mapping.** Each benchmark's `BaseBenchmark.env_name` is the key — the user knows from `videvalkit list benchmarks` which env they need.

`envs/videvalkit.yaml` skeleton (to be locked):

```yaml
name: videvalkit
channels: [conda-forge, pytorch, nvidia, defaults]
dependencies:
  - python=3.10
  - pip
  - pytorch=2.3.1
  - pytorch-cuda=12.1
  - torchvision
  - opencv
  - decord
  - ffmpeg
  - pip:
    - transformers==4.45.2
    - accelerate>=0.34
    - clean-fid==0.1.35
    - pyiqa
    - sam-2
    - GroundingDINO @ git+https://github.com/IDEA-Research/GroundingDINO.git
    - depth-anything-v2
    - cotracker @ git+https://github.com/facebookresearch/co-tracker.git
    - spacy
    - pydantic
    - click
    - httpx[http2]
    - huggingface_hub[hf_xet]
    - mamba_ssm     # required by VFIMamba (WorldScore motion_smoothness)
    - "videvalkit[all]"  # the toolkit itself, installed in editable mode after clone
```

### 15.2.1 Why Docker + `conda env create` from yaml were dropped from the user-facing install

Both were ruled out as the canonical install path after testing on 2026-05-19. The user manuals now ship only the tarball path. This subsection records *why*, so contributors who try to revive either path know what they're getting into.

#### Docker build constraints

Docker is the natural distribution format for ML toolkits, but the build itself has constraints we cannot meet inside our development environment:

1. **The dev host is itself a Kubernetes pod**. `/proc/1/cgroup` shows `kubepods.slice/.../cri-containerd-...` — meaning the shell where the env was built is already inside a container. Building a Docker image from inside a container requires Docker-in-Docker (DinD), which needs either a privileged container or user namespaces enabled.
2. **User namespaces are disabled at kernel level**: `cat /proc/sys/user/max_user_namespaces = 0`, and `/proc/sys` is read-only from inside the pod. Every container build tool we tested (docker, podman, buildah, kaniko-standalone) failed at this step.
3. **The pod is not privileged**: no `CAP_SYS_ADMIN`, no `/dev/fuse`, no nested-virt support.

The image is still buildable on any host that *isn't* a restricted K8s pod — a laptop with Docker Desktop, an EC2 VM, a GitHub Actions runner, etc. The repo ships:
- `Dockerfile` at repo root — copies the env tarball + repo source into a `nvidia/cuda:12.1.0-runtime-ubuntu22.04` base.
- `scripts/build_image.sh` — single-command build helper.
- `.github/workflows/build-and-push.yml` — auto-build + push to `ghcr.io/videogenevalkit/videogenevalkit` on every `v*.*.*` git tag.

We *recommend* using the CI workflow for the canonical image build; manual builds on a contributor laptop work but are not reproducible across hosts.

#### `conda env create -f envs/videvalkit.yaml` failure modes

The yaml is retained for contributors but is not the user-facing path because pip's resolver cannot reconstruct the working env from scratch. Testing during 2026-05-19 hit 10+ consecutive failures:

| Iteration | Failure | Root cause |
|---|---|---|
| v1 | `mmcv` build error | mmcv 2.1.0 has no PyPI wheel; sdist needs torch present at build time |
| v2 | `aliyun-python-sdk-core` build error | sdist-only on PyPI; build env fails to find `pkg_resources` |
| v3 | `--no-build-isolation` invalid in requirements.txt | conda-env-create's pip layer doesn't support per-pkg `--no-build-isolation` |
| v4 | `detectron2` build error | Build needs torch already installed — same as mmcv |
| v5 | mmcv wheel not found for `torch 2.3.0` | Openmmlab wheel index has `torch2.3.0` not `torch2.3` |
| v6 | `triton 3.7.0` vs `torch 2.3.1` | torch 2.3.1 expects `triton<3`, but the working env has 3.7.0 via `--no-deps` |
| v7 | `aliyun-*` reappeared after triton fix | transitive of `modelscope`; sdist-only |
| v8 | `droid_backends==0.0.0` not on PyPI | DROID-SLAM's C++ extension, built from source only |
| v9 | `fsspec==2026.4.0` conflicts with 5 other pinned packages | resolver can't satisfy all |
| v10 | `pillow==12.2.0` conflicts with 5 others | resolver can't satisfy all |

The deeper problem: the source env at `/pub/.../envs/videvalkit/` was built incrementally with ~30 `pip install --no-deps` patches over a multi-week validation campaign. Each patch worked because the user manually verified runtime correctness. The `pip freeze` capture of the END state is internally consistent only if you skip pip's resolver — which is exactly what conda-pack does.

**For contributors**: if you want to evolve the env, do it on top of the existing tarball (download, activate, `pip install --no-deps NEW_PKG==VERSION`, re-pack). Don't try to regenerate the lock from scratch — it will lose information the resolver can't see.

### 15.2.2 Building and uploading the env tarball (maintainer procedure)

The canonical env tarball at `videogenevalkit/env-tarball` is regenerated by the maintainer whenever the source env changes (new dep, fixed CVE, version bump). Procedure:

```bash
# 1. Ensure conda-pack is available (separate from the toolkit env, to avoid contaminating it)
/root/miniconda3/bin/pip install conda-pack

# 2. Snapshot the working env. Exclude editable installs (they reference /pub paths
#    that won't exist at the user's machine; they're re-installed via fetch-upstream
#    + pip install -e .  at user-install time).
/root/miniconda3/bin/conda-pack \
    -p /pub/evaluation_group/ning/benchmark/envs/videvalkit \
    --ignore-editable-packages \
    --ignore-missing-files \
    --exclude "lib/python3.10/site-packages/SAM_2-*" \
    --exclude "lib/python3.10/site-packages/groundingdino*" \
    --exclude "lib/python3.10/site-packages/vbench*" \
    --exclude "lib/python3.10/site-packages/videvalkit*" \
    --output /tmp/videvalkit-env.tar.gz \
    --n-threads 16
# Takes ~3 minutes. Output: ~7.6 GB compressed (~15 GB unpacked).

# 3. Verify the tarball extracts and torch imports
mkdir -p /tmp/env-test
tar xzf /tmp/videvalkit-env.tar.gz -C /tmp/env-test
/tmp/env-test/bin/conda-unpack
/tmp/env-test/bin/python -c "
import torch, transformers, mmcv, decord, cv2, pyiqa
print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'transformers={transformers.__version__}  mmcv={mmcv.__version__}')
print(f'decord+cv2+pyiqa imported OK')
"

# 4. Upload to HuggingFace. Authentication via the videogenevalkit account's
#    write-scope token at ~/.cache/huggingface/token.
env -u HF_ENDPOINT python -c "
import os
os.environ.pop('HF_ENDPOINT', None)
from huggingface_hub import HfApi, create_repo
api = HfApi()
create_repo('videogenevalkit/env-tarball', repo_type='dataset', private=False, exist_ok=True)
api.upload_file(
    path_or_fileobj='/tmp/videvalkit-env.tar.gz',
    path_in_repo='videvalkit-env.tar.gz',
    repo_id='videogenevalkit/env-tarball',
    repo_type='dataset',
    commit_message='Regenerated: conda-pack of working env (torch 2.3.1+cu121, N packages, validated)',
)
print('Done.  https://huggingface.co/datasets/videogenevalkit/env-tarball')
"
# Upload time: ~5-10 min on a fast link (~80 MB/s). 7.6 GB single-file upload.
```

After the upload completes:

1. **Test the install path end-to-end** by downloading + extracting the new tarball on a clean machine, running `videvalkit doctor`, and running at least one bench smoke (`videvalkit eval --bench worldjen ...`).
2. **Bump the version label** in `pyproject.toml` and `src/videvalkit/__init__.py:__version__` if the env change is user-visible.
3. **Push a `v0.X.Y` git tag** to trigger the Docker image rebuild via `.github/workflows/build-and-push.yml`.

The image build pipeline fetches the new tarball from `videogenevalkit/env-tarball` automatically; no separate Docker image push is needed.

**Tarball lifecycle and storage on HF**: each upload creates a new commit on the HF dataset; previous tarball versions remain available via `revision=<commit_sha>` in `hf_hub_download`. Free-tier HF orgs have a soft cap of ~1 TB of dataset storage across all repos; the videogenevalkit org currently uses ~135 GB total (checkpoints + smoke-data + env-tarball), well under quota.

### 15.3 Checkpoint distribution via HuggingFace

**HF dataset:** `videogenevalkit/checkpoints` (LFS-tracked, public)

Canonical layout — bench-namespaced top level so `fetch-checkpoints --bench <name>` resolves to a single `allow_patterns="<bench>/*"` glob. This matches `videogenevalkit/smoke-data`'s `<bench>/videos/<model>/...` convention.

```
t2vcompbench/                                                              (~ 4.0 GB)
  groundingdino_swint_ogc.pth                                ~ 700 MB     (object dims)
  sam_vit_h_4b8939.pth                                       ~ 2.5 GB    (SAM)
  depth_anything_vitl14.pth                                  ~ 1.4 GB    (spatial)
  cvo_raft_patch_8.pth                                       ~ 200 MB    (DOT estimator)
  movi_f_cotracker2_patch_4_wind_8.pth                       ~ 240 MB    (DOT tracker)
  movi_f_raft_patch_4_alpha.pth                              ~ 80 MB     (DOT refiner)
worldscore/                                                                (~ 6.3 GB)
  groundingdino_swint_ogc.pth                                ~ 700 MB    (object_alignment)
  sam_vit_h_4b8939.pth                                       ~ 2.5 GB    (object_alignment SAM)
  sam2.1_hiera_large.pt                                      ~ 900 MB    (motion_accuracy)
  sam2.1_hiera_base_plus.pt                                  ~ 330 MB    (motion_accuracy fallback)
  droid.pth                                                  ~ 1.0 GB    (camera_control + 3d_consistency)
  raft-things.pth                                            ~ 20 MB     (optical flow inside SLAM)
  Tartan-C-T-TSKH-spring540x960-M.pth                        ~ 130 MB    (SEA-RAFT optical_flow)
  VFIMamba.pkl                                               ~ 120 MB    (motion_smoothness)
  sac+logos+ava1-l14-linearMSE.pth                           ~ 4 MB      (subjective_quality predictor)
vbench/                                                                    (< 1 MB)
  VBench_full_info.json                                                   (1746 prompt entries + tag dim mapping)
vbench2/                                                                   (< 1 MB)
  VBench2_full_info.json                                                  (1296 prompt entries + dim metadata)
hf-models/                                                                 (~ 95 GB — HF cache mirror, optional)
  liuhaotian/llava-v1.6-34b/                                ~ 68 GB     (T2V-CompBench MLLM upstream-mode — OPTIONAL)
  lmms-lab/LLaVA-Video-7B-Qwen2/                            ~ 15 GB     (VBench-2.0 LLaVA dims, T2V-CompBench toolkit-mode)
  Qwen/Qwen2.5-7B-Instruct/                                 ~ 15 GB     (VBench-2.0 complex_plot judge LLM)
  openai/clip-vit-base-patch16/                             ~ 600 MB    (WorldScore content_alignment + CLIP-Score metric)
  LiheYoung/depth_anything_vitl14/                          ~ 1.3 GB    (HF-side weights for Depth-Anything)

manifest.json                                                              (toolkit reads this to map dim → required ckpt paths)
checksums.sha256                                                           (per-file SHA256, written at upload time)
```

Notes:
- `t2vcompbench/` and `worldscore/` intentionally each carry their own copy of `groundingdino_swint_ogc.pth` / `sam_vit_h_4b8939.pth`. Upstream code in each repo hardcodes a different relative path, and a single per-bench fetch is simpler than symlink trickery. Total redundancy: ~3.2 GB.
- `hf-models/<org>/<repo>/` flattens the standard HF cache (`models--org--repo/snapshots/<hash>/`) into a direct snapshot. The fetcher writes both forms so `transformers.from_pretrained` resolves either way.
- VBench v1 / VBench-2.0 prompts and `*_full_info.json` are shipped here (not in smoke-data) because they're metadata, not video samples.
- VBench v1 `pretrained/` is bundled here (~7 GB across AMT / ViClip / UMT / GRiT / DINO / CLIP / MUSIQ / tag2text) so that the toolkit is fully offline. Same for VBench-2.0 `third_party/` (~0.8 GB: CoTracker3, ArcFace, RetinaFace, Dense_match).

**Total at full coverage:** ~125 GB. Most users won't download all of it.

### 15.3.1 Per-dim model dependency map

Each adapter's scoring code calls into a specific set of pretrained models. These tables make explicit *which* dim consumes *which* checkpoint, so users can trim the fetch with `fetch-checkpoints --bench <name> --dim <dim>` and know exactly what they're committing to download.

#### VBench v1 (16 dims)

| Dim | Category | Primary scorer | Aux model |
|---|---|---|---|
| `subject_consistency` | Quality | DINO ViT-B/16 (`dino_vitbase16_pretrain.pth`) | — |
| `background_consistency` | Quality | OpenAI CLIP ViT-B/32 | — |
| `motion_smoothness` | Quality | AMT-S (`amt-s.pth`, optical-flow interpolator) | — |
| `dynamic_degree` | Quality | RAFT (`raft-things.pth`) | — |
| `aesthetic_quality` | Quality | LAION aesthetic predictor (`sac+logos+ava1-l14-linearMSE.pth`) + CLIP ViT-L/14 | — |
| `imaging_quality` | Quality | MUSIQ (`musiq_spaq_ckpt-358bb6af.pth`, pyiqa) | — |
| `temporal_flickering` | Quality | RAFT + frame-diff statistics | — |
| `object_class` | Semantic | GRiT (`grit_b_densecap_objectdet.pth`) | — |
| `multiple_objects` | Semantic | GRiT (same) | — |
| `human_action` | Semantic | UMT (`l16_ptk710_ftk710_ftk400_f16_res224.pth`) + CLIP | — |
| `color` | Semantic | GRiT + color-region matcher | — |
| `spatial_relationship` | Semantic | GRiT + relative-coords parser | — |
| `scene` | Semantic | tag2text (`tag2text_swin_14m.pth`) | — |
| `appearance_style` | Semantic | CLIP ViT-L/14 + ViClip | — |
| `temporal_style` | Semantic | CLIP ViT-L/14 + ViClip-InternVid-10M-FLT | — |
| `overall_consistency` | Semantic | ViClip (`ViClip-InternVid-10M-FLT.pth`) | — |

Total VBench v1 model storage: **~ 7.7 GB** (ViClip 1.7 GB + tag2text 4.2 GB + UMT 579 MB + CLIP ViT-L/14 933 MB + GRiT 399 MB + DINO 343 MB + CLIP ViT-B/32 354 MB + MUSIQ 109 MB + AMT 12 MB + CLIP RN50 256 MB). All bundled under `vbench/pretrained/` on HF.

#### VBench-2.0 (18 dims)

| Dim | Category | Primary scorer | Aux model |
|---|---|---|---|
| `dynamic_attribute` | Controllability | LLaVA-Video-7B-Qwen2 | — |
| `dynamic_spatial_relationship` | Controllability | LLaVA-Video-7B-Qwen2 | — |
| `composition` | Controllability | LLaVA-Video-7B-Qwen2 | — |
| `complex_landscape` | Controllability | LLaVA-Video-7B-Qwen2 | Qwen2.5-7B-Instruct (LLM judge) |
| `complex_plot` | Creativity | LLaVA-Video-7B-Qwen2 | Qwen2.5-7B-Instruct (LLM judge) |
| `human_clothes` | Human Fidelity | LLaVA-Video-7B-Qwen2 | — |
| `human_interaction` | Human Fidelity | LLaVA-Video-7B-Qwen2 | Qwen2.5-7B-Instruct (LLM judge) |
| `material` | Physics | LLaVA-Video-7B-Qwen2 | — |
| `mechanics` | Physics | LLaVA-Video-7B-Qwen2 | — |
| `thermotics` | Physics | LLaVA-Video-7B-Qwen2 | — |
| `motion_rationality` | Commonsense | LLaVA-Video-7B-Qwen2 | — |
| `motion_order_understanding` | Physics | LLaVA-Video-7B-Qwen2 | Qwen2.5-7B-Instruct (LLM judge) |
| `camera_motion` | Controllability | CoTracker3 (`cotracker2.pth`, vendored) | — |
| `multi_view_consistency` | Controllability | CoTracker3 + RAFT (`raft-things.pth`) | Dense_match (`scaled_offline/online.pth`) |
| `diversity` | Creativity | VGG-19 (torchvision, auto-fetch via `vgg19(pretrained=True)`) | — |
| `human_anatomy` | Human Fidelity | ViTDetector (vendored anatomical detector) | — |
| `human_identity` | Human Fidelity | ArcFace (`resnet18_110.pth`) + RetinaFace (`retinaface_resnet50_*.pth`) + CoTracker3 | torchvision model_zoo |
| `instance_preservation` | Controllability | Instance_detector = Qwen2.5-VL-3B-Instruct via ms-swift | — |

Total VBench-2.0 model storage: **~ 23 GB** (LLaVA-Video-7B 15 GB + Qwen2.5-7B 15 GB shared with above + Qwen2.5-VL-3B 7 GB + CoTracker 204 MB + ArcFace 103 MB + RetinaFace 110 MB + Dense_match 204 MB). The three LLMs live under `hf-models/`; the CV-specific weights live under `vbench2/third_party/`.

#### Video-Bench (9 dims)

| Dim | Group | Primary scorer | Aux model |
|---|---|---|---|
| `video_text_consistency` | Alignment | VLM judge (GPT-4o paper / Gemma-4-31B-IT local) | — |
| `object_class` | Alignment | VLM judge | — |
| `color` | Alignment | VLM judge | — |
| `scene` | Alignment | VLM judge | — |
| `action` | Alignment | VLM judge | — |
| `aesthetic_quality` | Static-Quality | VLM judge (1–10 rubric) | — |
| `imaging_quality` | Static-Quality | VLM judge (1–10 rubric) | — |
| `overall_consistency` | Static-Quality | VLM judge (1–10 rubric) | — |
| `temporal_consistency` | Dynamic-Quality | VLM judge (1–10 rubric) | — |
| `motion_effects` | Dynamic-Quality | VLM judge (1–10 rubric) | — |

All 9 dims share **one** model — the VLM judge — making Video-Bench the simplest by model count. Paper uses GPT-4o (managed API); we substitute Gemma-4-31B-IT local for the validation sweep, which is why `dynamic_quality` dims drift (~Δ +0.2 documented in TEST_MANUAL.md §4.4). No bench-specific .pth files; the JSON `VideoBench_full.json` carries the prompts + dim mapping.

#### WorldJen (16 dims, PHAS aggregator)

| Dim | Macro-category | Primary scorer | Aux model |
|---|---|---|---|
| `subject_consistency` | motion_stability | VLM (16-frame sampled) | — |
| `scene_consistency` | motion_stability | VLM (16-frame sampled) | — |
| `motion_smoothness` | motion_stability | VLM (12-frame micro) | — |
| `temporal_flickering` | motion_stability | VLM (12-frame micro) | — |
| `inertial_consistency` | motion_stability | VLM (12-frame micro) | — |
| `physical_mechanics` | logic_physics | VLM (12-frame micro) | — |
| `object_permanence` | logic_physics | VLM (16-frame sampled) | — |
| `human_fidelity` | logic_physics | VLM (12-frame micro) | — |
| `dynamic_degree` | logic_physics | VLM (32-frame holistic) | — |
| `semantic_adherence` | instruction_adherence | VLM (32-frame holistic) | — |
| `spatial_relationship` | instruction_adherence | VLM (32-frame holistic) | — |
| `semantic_drift` | instruction_adherence | VLM (32-frame holistic) | — |
| `composition_framing` | aesthetic_quality | VLM (32-frame holistic) | — |
| `lighting_volumetric` | aesthetic_quality | VLM (32-frame holistic) | — |
| `color_harmony` | aesthetic_quality | VLM (32-frame holistic) | — |
| `structural_gestalt` | aesthetic_quality | VLM (32-frame holistic) | — |

Two models drive everything: **Phase A** uses an LLM (Qwen2.5-7B-Instruct local / Gemini-3-Flash managed) to generate per-prompt VQA questions; **Phase B** uses a VLM (Gemma-4-31B-IT local / Gemini-3-Flash managed) to answer them on sampled frames. PHAS aggregator combines the 16 normalized scores with learned per-dim weights. No .pth files needed — both models are bundled under `hf-models/`. The `prompts.jsonl` + `vqa_questions_50prompts.jsonl` live in the smoke-data repo.

#### WorldScore (10 dims)

| Dim | Split | Primary scorer | Aux model |
|---|---|---|---|
| `content_alignment` | static | torchmetrics `CLIPScore` (`openai/clip-vit-base-patch16`) | — |
| `object_control` | static | GroundingDINO (`groundingdino_swint_ogc.pth`) | + SAM-H (`sam_vit_h_4b8939.pth`) |
| `photometric_consistency` | static | SEA-RAFT optical flow (`Tartan-C-T-TSKH-spring540x960-M.pth`) | AEPE |
| `motion_magnitude` | dynamic | SEA-RAFT (same) | median-flow stat |
| `style_consistency` | static | VGG-19 Gram matrix (torchvision built-in) | WorldScore ref images |
| `subjective_quality` | static | pyiqa LAION aesthetic (`sac+logos+ava1-l14-linearMSE.pth`) + CLIP-IQA+ | mean of two |
| `3d_consistency` | static | DROID-SLAM (`droid.pth` + `raft-things.pth`) | reprojection error |
| `camera_control` | static | DROID-SLAM (same) | trajectory-error vs ground truth |
| `motion_smoothness` | dynamic | VFIMamba (`VFIMamba.pkl`) | SSIM + LPIPS |
| `motion_accuracy` | dynamic | SEA-RAFT + SAM2 (`sam2.1_hiera_large.pt`) | flow-on-mask metric |

Total WorldScore model storage: **~ 6.3 GB** (all under `worldscore/` on HF, plus CLIP-ViT-Base 600 MB under `hf-models/openai/`). VGG-19 auto-fetches via torchvision on first run.

#### T2V-CompBench V2 (7 dims)

| Dim | Pipeline | Primary scorer | Aux model |
|---|---|---|---|
| `consistent_attribute` | MLLM | LLaVA-1.6-34B (`liuhaotian/llava-v1.6-34b`) | — |
| `action_binding` | MLLM | LLaVA-1.6-34B | — |
| `object_interactions` | MLLM | LLaVA-1.6-34B | — |
| `dynamic_attribute` | MLLM | LLaVA-1.6-34B | — |
| `numeracy` | CV | GroundingDINO (`groundingdino_swint_ogc.pth`) | SAM-H |
| `spatial_relationships` | CV (3D mode) | GroundingDINO + SAM-H + Depth-Anything V1 (`depth_anything_vitl14.pth`) | — |
| `motion_binding` | CV | GroundingDINO + SAM-H + DOT (`cvo_raft_patch_8.pth` + `movi_f_raft_patch_4_alpha.pth` + `movi_f_cotracker2_patch_4_wind_8.pth`) | — |

Total T2V-CompBench model storage: **~ 72 GB** (LLaVA-1.6-34B 68 GB dominates; the CV stack is 4 GB). LLaVA bundled under `hf-models/liuhaotian/`; CV stack under `t2vcompbench/`. In `mode="upstream"` the 4 MLLM dims use a subprocess shim into the paper's LLaVA-1.6-34B; in `mode="toolkit"` they use the user's configured VLM judge (Gemma local by default).

**Why we bundle CV-only stacks separately for T2V-CompBench and WorldScore.** Both call into GroundingDINO + SAM-H, but each upstream code base hardcodes the .pth path relative to its repo root (e.g. `T2V-CompBench/Grounded-Segment-Anything/groundingdino_swint_ogc.pth` vs `WorldScore/worldscore/benchmark/metrics/checkpoints/groundingdino_swint_ogc.pth`). Symlink trickery across these two paths is brittle; the 3 GB of redundancy is the price of upstream-fidelity.

**`videvalkit fetch-checkpoints` design:**

```bash
videvalkit fetch-checkpoints --bench worldjen        # downloads only what worldjen needs (~ 16 GB)
videvalkit fetch-checkpoints --bench t2vcompbench    # ~ 80 GB (with LLaVA-1.6-34b option to skip)
videvalkit fetch-checkpoints --bench t2vcompbench --skip-mllm-upstream    # ~ 12 GB (CV dims only)
videvalkit fetch-checkpoints --metric fvd            # ~ 100 MB (just I3D)
videvalkit fetch-checkpoints --all                   # ~ 110 GB
videvalkit fetch-checkpoints --dry-run               # shows what would download + sizes, no fetch
```

Internally: `huggingface_hub.snapshot_download(repo_id, allow_patterns=...)` per benchmark's manifest entry. Resumable (HF handles partial downloads). Checksums verified on completion via `manifest.json`.

### 15.3.2 Swappable scorer models — bigger vLLMs and managed APIs (planned 2026-05-18)

**Motivation.** §15.3.1 lists each dim's *default* scorer (the one the upstream paper used). For benchmarks where the scorer is a VLM/MLLM, the user often wants to substitute a different model:

- **Better fidelity** — run VBench-2.0's LLaVA-Video-7B dims against Gemma-4-31B-IT or Qwen3-32B-VL instead. The 7B paper-default is a research artifact, not a quality ceiling.
- **Different judge for ablation** — Video-Bench paper uses GPT-4o; a user wants to re-score with GPT-4o, Gemini, and Gemma side-by-side to see judge variance.
- **No local GPU available** — fall back to a managed API (Gemini-3-Flash, Claude-Sonnet-4.6, GPT-4o) for everything.
- **Latency vs cost tradeoff** — small local model for triage runs, big managed model for final paper-comparable numbers.

The infrastructure for this already exists in §16 (`SUPPORTED_JUDGES` registry + `~/.config/videvalkit/judges.yaml` precedence chain). §15.3.2 documents which dims actually accept a swap, what "swap" means (different base model, vs different host endpoint), and the fidelity contract when you swap.

#### What's swappable, by benchmark

| Benchmark | Dims with VLM/MLLM scorer | Paper-default | Swappable to | Δ vs paper expected |
|---|---|---|---|---|
| **VBench v1** | 0 of 16 | — | (no VLM scorers; all CV-based) | n/a |
| **VBench-2.0** | 12 of 18 (everything except `camera_motion`, `multi_view_consistency`, `diversity`, `human_anatomy`, `human_identity`, `instance_preservation`) | LLaVA-Video-7B-Qwen2 | Any OpenAI-compatible VLM ≥ 7B with frame-list input | up to ±0.10 per dim until calibrated against paper |
| **VBench-2.0** | 5 of 18 (`complex_landscape`, `complex_plot`, `human_interaction`, `motion_order_understanding`, plus internal Complex_Plot judge step) | Qwen2.5-7B-Instruct | Any OpenAI-compatible LLM ≥ 7B (text-only judge) | up to ±0.05 per dim |
| **VBench-2.0** | 1 of 18 (`instance_preservation`) | Qwen2.5-VL-3B-Instruct (via ms-swift) | Any Qwen-VL-compatible (3B/7B/72B) — others may need adapter | not yet measured |
| **Video-Bench** | 9 of 9 — ALL dims | GPT-4o (paper), Gemma-4-31B-IT (toolkit default local) | Any VLM with multi-image input | dynamic-quality dims documented Δ ~ +0.2 swap from GPT-4o → Gemma (TEST_MANUAL §4.4) |
| **WorldJen** | 16 of 16 — Phase B VLM is shared | Gemini-3-Flash (paper) or Gemma-4-31B-IT (toolkit default local) | Any VLM with 12-/16-/32-frame input | PHAS 3.66 vs paper-Gemini 4.12 (Δ −0.47), documented in TEST_MANUAL §4.3 |
| **WorldJen** | 16 of 16 — Phase A LLM | Qwen2.5-7B-Instruct (local) or Gemini-3-Flash (paper) | Any LLM | Phase B downstream-affects all dims; the 50-prompt headline VQA file is checked into smoke-data so users can skip Phase A entirely |
| **WorldScore** | 0 of 10 | — | (no VLM scorers; all CV-based) | n/a |
| **T2V-CompBench** | 4 of 7 (`consistent_attribute`, `action_binding`, `object_interactions`, `dynamic_attribute`) | LLaVA-1.6-34B (paper-mode) | OpenAI-compatible VLM ≥ 7B (toolkit-mode) | toolkit-mode emits comparable-but-not-byte-identical numbers; paper-mode is opt-in for byte-exact |

**The two dims VBench-2.0 explicitly cannot swap:** `human_anatomy` (vendored ViTDetector — anatomical-keypoint regression, not a VLM) and `human_identity` (vendored ArcFace+RetinaFace+CoTracker3 — face-recognition pipeline, not a VLM). These are vendored under `vbench2/third_party/` and are kept as-is.

#### Three swap mechanisms

1. **Endpoint swap (same model, different host).** User has Gemma-4-31B-IT running locally on `:8003`, wants to point at a remote vLLM cluster on `https://vllm.internal:8443`:
   ```yaml
   # ~/.config/videvalkit/judges.yaml
   judges:
     gemma-4-31b-local:
       kind: openai_compatible
       base_url: https://vllm.internal:8443/v1   # was http://localhost:8003/v1
       model: google/gemma-4-31b-it
       api_key_env: VLLM_INTERNAL_KEY
   ```
   No code change; toolkit picks up the new endpoint via §16's precedence chain. Useful for staging environments.

2. **Model swap (same kind, different size/family).** User wants to replace LLaVA-Video-7B with Gemma-4-31B-IT for VBench-2.0's 12 VLM dims:
   ```bash
   videvalkit eval --bench vbench2 --videos ... \
       --scorer-vlm gemma-4-31b-local              # was the default lmms-lab/LLaVA-Video-7B-Qwen2
   ```
   The toolkit routes every dim that has `paper_scorer="LLaVA-Video-7B-Qwen2"` through the named judge instead. The benchmark adapter calls `ctx.vlm_judge.chat_with_frames(prompt, frames)` — the judge name resolves via §16's `resolve_judge()` factory; the resolved client just answers the chat. Any model meeting the API contract works.

3. **Mode swap (paper-mode → toolkit-mode).** Drops down to the same VLM but inside a paper-faithful invocation (specific frame layout, prompt template, seed). Only T2V-CompBench currently has this:
   ```bash
   videvalkit eval --bench t2vcompbench --videos ... --mode upstream    # paper-exact: subprocess shim into LLaVA-1.6-34B paper script
   videvalkit eval --bench t2vcompbench --videos ... --mode toolkit     # toolkit native: configurable VLM, same rubric prompts, different scaffolding
   ```
   Other benchmarks (VBench-2.0, Video-Bench, WorldJen) have only toolkit-mode; the model swap is the only knob.

#### Per-dim VLM override (advanced)

For ablation runs where the user wants to mix-and-match (e.g. "use Gemma for all VBench-2.0 dims *except* `complex_plot` which uses Claude-Sonnet-4.6"):

```yaml
# <workspace>/config.yaml
scorers:
  vbench2:
    default_vlm: gemma-4-31b-local
    default_llm: qwen3-32b-local
    per_dim_override:
      complex_plot:
        vlm: claude-sonnet-4-6     # use Claude for complex reasoning
        llm: claude-sonnet-4-6     # same model handles judge role
      human_interaction:
        vlm: gpt-4o-2024-11-20     # use GPT-4o for HF dim
  videobench:
    default_vlm: gemini-3-flash
  worldjen:
    phase_a_llm: qwen3-32b-local
    phase_b_vlm: gemini-3-flash
  t2vcompbench:
    default_vlm: llava-1.6-34b-local    # keep paper-mode for byte-exact
```

CLI equivalent:
```bash
videvalkit eval --bench vbench2 --videos ... \
    --scorer-vlm gemma-4-31b-local \
    --scorer-vlm-dim complex_plot=claude-sonnet-4-6 \
    --scorer-vlm-dim human_interaction=gpt-4o-2024-11-20
```

#### Fidelity contract — what the user is accepting when they swap

Each dim records the *paper-default* scorer in its manifest. When the user runs with a non-default scorer, the toolkit:

1. **Writes the actual scorer to `Summary.meta.scorers_used`** so downstream tools (and TEST_MANUAL's leaderboard comparison) can flag "this number is not directly comparable to paper Table N".
2. **Refuses to claim Δ-vs-paper** in `compare-leaderboard` output for non-default scorers. The tolerance bands in TEST_MANUAL.md §3 are calibrated for the paper-default scorer; substituting changes the absolute number unpredictably.
3. **Widens tolerance to "judge-substitution band"** (3× the paper band) in `compare-leaderboard` if the user passes `--allow-judge-substitution`. This is documented in Video-Bench §4.4 (`dynamic_quality` Gemma-vs-GPT-4o → ~Δ +0.2 per dim) as the precedent.
4. **Logs token usage** per §16's `api_log.jsonl` schema, including which dim and which (substituted) judge.

The cost of swap is *real, measurable, and bookkept* — not silent. A user who runs VBench-2.0 with Gemma instead of LLaVA-Video-7B gets:
- A complete summary with full per-dim and per-category scores.
- A `meta.scorers_used = {"default": "gemma-4-31b-local"}` annotation.
- A `compare-leaderboard` output that shows numbers but adds `STATUS: judge-substituted, paper-Δ not asserted` per row.
- An `api_log.jsonl` line per VLM call with token counts.

#### Per-judge calibration (future work)

When a user runs the same model+benchmark with two different judges 100+ times, the toolkit can fit a per-dim affine calibration `paper_score ≈ a · our_score + b` from the observed pair distribution. Stored in `~/.cache/videvalkit/judge_calibration/<bench>_<scorer>.json`. After calibration, `compare-leaderboard` can apply the affine map and re-assert Δ-vs-paper with a wider band. This is a v2 capability; not in the initial release.

### Files to add / modify (consolidates §15.3.2 + §16 wiring)

* `src/videvalkit/configs/scorers.py` — new module: `default_scorer_for(bench, dim) -> JudgeRef`, `resolve_scorer(JudgeRef, overrides) -> Judge`
* `src/videvalkit/benchmarks/<bench>/manifest.py` — add `DIM_DEFAULT_SCORER` dict (per the §15.3.1 table)
* `src/videvalkit/benchmarks/<bench>/benchmark.py` — `evaluate()` reads `ctx.scorer_router.for_dim(dim)` instead of hard-coding the judge
* `src/videvalkit/cli.py` — `--scorer-vlm`, `--scorer-llm`, `--scorer-vlm-dim`, `--scorer-llm-dim`, `--mode {paper,toolkit}` flags
* `src/videvalkit/storage/summary.py` — add `meta.scorers_used` field, persisted in every `Summary` output
* `~/.config/videvalkit/judges.yaml` (covered by §16) + `<workspace>/config.yaml` `scorers:` block (new)
* `USER_MANUAL.md` — new subsection "Swapping scorers: when and why"
* `DEV_MANUAL.md` §15.3.2 — this section

### 15.4 Smoke-data via HuggingFace

**HF dataset:** `videvalkit/smoke-data` (~ 1.8 GB total, public, no gating)

What it contains (one minimal-but-real slice per anchored benchmark):

```
videos/
  worldjen-kling-50/                          # 50 Kling videos, one per prompt
                                              # source: HF Vchitect/WorldJen-...; 250 MB
  videobench-cogvideox5b-action/              # 78 action videos for action_consistency
                                              # source: Video-Bench/action.zip; 200 MB
  vbench-hunyuan-subject-consistency/         # 360 vids for 1 dim (72 prompts × 5 samples)
                                              # source: VBench leaderboard; 350 MB
  vbench2-hunyuan-camera-motion/              # 324 vids for 1 dim
                                              # source: Vchitect/VBench-2.0_sampled_videos; 200 MB
  worldscore-cogvideox5b-dynamic/             # 100 videos, one per dynamic prompt
                                              # source: WorldScore-paper-recommended generations; 55 MB
  worldscore-cogvideox5b-static/              # 103 videos, one per flat-static prompt
                                              # source: same; 38 MB
  worldscore-refs/                            # 150 PNG reference images (one per static-prompt entry)
                                              # source: same; 241 MB
  t2vcompbench-cogvideox5b-200per-dim/        # 1400 videos (200 prompts × 7 dims × 1 sample)
                                              # source: Kaiyue/T2V-CompBench-Videos; 570 MB
prompts/
  worldjen-50.jsonl                           # paper's 50-prompt headline split
  worldjen-vqa-questions-50.jsonl             # paper's pre-built VQA file (so Phase A is skippable)
  videobench-action.jsonl                     # 78 action_consistency prompts
  vbench-subject-consistency.jsonl            # 72 prompts tagged subject_consistency
  vbench2-camera-motion.jsonl                 # 108 Camera_Motion prompts + auxiliary_info
  worldscore-dynamic-100.jsonl                # 100 dynamic prompts (camera motion descriptors)
  worldscore-static-flat-103.jsonl            # 103 static prompts (flat / entry-flattened form)
  worldscore-static-entry-50.jsonl            # 50 static prompts (entry-level form; same content, different shape)
  t2vcompbench-1400.jsonl                     # 1400 prompts (200 × 7 dims)
README.md                                     # links source URLs + describes what each dir contains
```

Purpose: a user with the toolkit installed can run end-to-end on a representative slice from each of the 6 anchored benchmarks in ~30–60 minutes and confirm the pipeline works without committing to the full 100+ GB checkpoint download.

**Internal source paths** (when packaging from this workspace into the HF dataset):

| Benchmark | Videos | Prompts |
|---|---|---|
| worldjen | `validation/reference_videos/worldjen/videos/fal-ai_kling-video_v2.6_pro_text-to-video/*.mp4` | `validation/reference_videos/worldjen/prompts/prompts_50.jsonl` + `vqa_questions_50prompts.jsonl` |
| videobench | `validation/reference_videos/videobench/action.zip` (inner: `action/cogvideox5b/`) | (derive from filenames in the zip) |
| vbench | `validation/reference_videos/vbench/Hunyuan Video (2025-05-22)/*.mp4` (filtered to subject_consistency dim via `VBench_full_info.json`) | `VBench_repo/vbench/VBench_full_info.json` (filtered) |
| vbench2 | `validation/reference_videos/vbench2/HunyuanVideo/Camera_Motion/*.mp4` | `VBench_repo/VBench-2.0/vbench2/prompts/meta_info/Camera_Motion.json` |
| **worldscore** | `/pub/evaluation_group/ning/worldscore_gens/cogvideox-5b/{dynamic,static}/{prompt_id}.mp4` + `/refs/{entry_id}.png` | `/pub/evaluation_group/ning/prompt/worldscore_dynamic_sample100.jsonl`, `worldscore_static_sample50_flat.jsonl`, `worldscore_static_sample50.jsonl` |
| **t2vcompbench** | `validation/reference_videos/t2vcompbench/CogVideoX-5B.zip` (570 MB) | derived from `validation/upstream_repos/T2V-CompBench/meta_data/*.json` (one per dim) |

`videvalkit fetch-smoke-data` mirrors `fetch-checkpoints` — same `huggingface_hub.snapshot_download` pattern. CLI accepts per-benchmark filters:

```bash
videvalkit fetch-smoke-data                    # all 6 (~ 1.8 GB)
videvalkit fetch-smoke-data --bench worldscore # just WorldScore (~ 334 MB videos + 200 KB prompts)
videvalkit fetch-smoke-data --bench worldjen   # just WorldJen (~ 250 MB)
videvalkit fetch-smoke-data --dry-run          # list, don't fetch
```

**WorldScore packaging notes (specific to this benchmark's tri-modal input):**

WorldScore is the only adapter that needs **three** simultaneous inputs per evaluation:
1. Generated video (the model output)
2. Reference image (per-instance style anchor; from the `refs/` dir, keyed by `entry_id` not `prompt_id`)
3. Prompts JSONL (carries `entry_id ↔ prompt_id` mapping)

The smoke-data layout above preserves the `entry_id` / `prompt_id` distinction. Make sure the adapter's `prepare-workspace` symlinks the `refs/` directory alongside the videos, not into the videos dir.

**T2V-CompBench packaging notes:**

The CogVideoX-5B zip already contains a flat `{model}/{dim_short}_{idx}/{prompt}_{sample}.mp4` layout. When repacking for the HF dataset, we'll flatten further to `{model}/{prompt_id}-{sample}.mp4` matching the toolkit's standard `video_layout`. The `meta_data/*.json` files (consistent_attribute_binding.json, dynamic_attribute_binding.json, action_binding.json, object_interactions.json, spatial_relationships.json, generative_numeracy.json, motion_binding.json) provide the per-dim ground-truth schema and are bundled into a single `t2vcompbench-1400.jsonl` for the HF release.

### 15.5 Open-source pre-flight checklist

Before pushing to the public GitHub:

- [ ] **License audit.** Each upstream (VBench, VBench-2.0, Video-Bench, WorldJen, WorldScore, T2V-CompBench) ships under its own license — surface those in `LICENSES/` at repo root.
- [ ] **API key handling.** Move from `os.environ.get("GEMINI_API_KEY")` etc. to a `~/.config/videvalkit/credentials.yaml` pattern with a documented schema (see §16).
- [ ] **Clean test suite.** All tests pass on a fresh env install — no path-shadowing assumptions, no GPU assumptions for the smoke-only tests.
- [ ] **No internal paths.** Strip every `/pub/evaluation_group/...` from code and docs; replace with `$VIDEVALKIT_HOME` or `~/.cache/videvalkit/`.
- [ ] **Reproducibility runs.** Re-run the published TEST_MANUAL §4.x validation results in a clean env from the public artifacts and confirm Δ holds.
- [ ] **Onboarding doc.** README.md becomes the 60-second quickstart (currently in DEV_MANUAL); link to USER_MANUAL for full docs.

### Files to add

* `envs/videvalkit.yaml`, `envs/videvalkit-cpu.yaml`, `envs/videvalkit-vbench.yaml` — pinned conda env files
* `src/videvalkit/cli.py` — add `fetch-checkpoints`, `fetch-smoke-data` subcommands
* `src/videvalkit/utils/hf_fetch.py` — checkpoint manifest reader + downloader
* `scripts/build_hf_artifacts.py` — internal tool to publish the two HF datasets from the live workspace
* `LICENSES/` — third-party licenses
* `README.md` — landing page for GitHub
* `DEV_MANUAL.md` §15 — this section

---

## 16. User-Configurable VLM Endpoints + API Token Logging (planned 2026-05-18)

### Current state

`src/videvalkit/configs/judges.py` hardcodes localhost ports:

```python
SUPPORTED_JUDGES = {
    "gemma-4-31b-local":   dict(kind="openai_compatible", endpoint="http://localhost:8003/v1", model="google/gemma-4-31b-it", ...),
    "qwen3-32b-local":     dict(kind="openai_compatible", endpoint="http://localhost:8004/v1", model="Qwen/Qwen3-32B", ...),
    "qwen3-vl-32b-local":  dict(kind="openai_compatible", endpoint="http://localhost:8005/v1", model="Qwen/Qwen3-VL-32B-Instruct", ...),
    "local-llava-video-7b":dict(kind="openai_compatible", endpoint="http://localhost:8006/v1", model="lmms-lab/LLaVA-Video-7B-Qwen2", ...),
    "gemini-3-flash":      dict(kind="gemini", model="gemini-3-flash-preview", api_key_env="GEMINI_API_KEY", ...),
    "gemini-2.5-pro":      dict(kind="gemini", model="gemini-2.5-pro", api_key_env="GEMINI_API_KEY", ...),
    "claude-sonnet-4-6":   dict(kind="anthropic", model="claude-sonnet-4-6", api_key_env="ANTHROPIC_API_KEY", ...),
    "gpt-4o":              dict(kind="openai_compatible", endpoint="https://api.openai.com/v1", model="gpt-4o-2024-11-20", api_key_env="OPENAI_API_KEY", ...),
}
```

Problem: open-source users have **different ports**, **different model names**, **different API keys**, **different cost-tracking needs**. We need three things:

1. A way for users to register new judge configs without forking the toolkit
2. A way for users to override `endpoint` / `model` / `api_key_env` per-run without editing source
3. Token-usage logging compatible with managed-API billing (matching `video_eval/VLM_screenshot20260428` schema)

### 16.1 User-supplied judge configs

Add precedence (lowest → highest):

1. Built-in `SUPPORTED_JUDGES` (codebase) — examples / sane defaults
2. `~/.config/videvalkit/judges.yaml` — user-wide overrides + new judges
3. Workspace-local `<ws>/judges.yaml` — per-project overrides (e.g. an experiment that pins a specific snapshot)
4. CLI flags — `--judge-endpoint`, `--judge-model`, `--judge-kind`, `--judge-api-key-env` — highest precedence, override everything

`~/.config/videvalkit/judges.yaml` schema:

```yaml
# User's local model server
my-local-qwen:
  kind: openai_compatible
  endpoint: http://10.0.1.7:9001/v1
  model: Qwen/Qwen3-32B
  provider: Qwen
  api_key_env: null
  request_timeout_s: 180

# Their OpenAI account, billed model
my-gpt-4o:
  kind: openai_compatible
  endpoint: https://api.openai.com/v1
  model: gpt-4o-2024-11-20
  provider: openai
  api_key_env: OPENAI_API_KEY
  cost_per_million_input_tokens: 2.50
  cost_per_million_output_tokens: 10.00
  cost_per_image_input: 0.00765    # see https://openai.com/api/pricing/

# Their Gemini, via the Generative-Language API
my-gemini-flash:
  kind: gemini
  model: gemini-3-flash-preview
  provider: google
  api_key_env: GEMINI_API_KEY
  cost_per_million_input_tokens: 0.075
  cost_per_million_output_tokens: 0.30
```

CLI usage:

```bash
# Use the user-registered judge by name (toolkit loads from ~/.config/...)
videvalkit eval --bench videobench --judge my-gpt-4o ...

# Ad-hoc override without registering — kwargs go on the CLI
videvalkit eval --bench videobench \
    --judge-kind openai_compatible \
    --judge-endpoint https://api.openai.com/v1 \
    --judge-model gpt-4o-2024-11-20 \
    --judge-api-key-env OPENAI_API_KEY
```

### 16.2 API token logging

Existing toolkit `src/videvalkit/storage/api_log.py` writes `request` + `response` JSONL with a basic `usage` field. The reference schema at `/pub/evaluation_group/ning/video_eval/VLM_screenshot20260428/api_logger.py` adds these guarantees:

- Per-call `input.json` + `output.json` pair (for replay / debugging)
- Normalized `usage` keys: `prompt_tokens` / `completion_tokens` / `thinking_tokens` / `cached_tokens` / `total_tokens` — applied across providers (OpenAI / Anthropic / Gemini all use different keys natively)
- Wall-clock `duration_ms` and `success` / `error` flags
- Compact `stats/` file per (model, month) for cost aggregation

**Plan:** port the reference schema into `videvalkit.storage.api_log`. Specifically:

```
ws/api_logs/
  calls/<provider>/<model>/<YYYY-MM>/<YYYY-MM-DD>.<run_ts>.jsonl    # full request+response+usage
  stats/<provider>/<model>/<user>_<YYYY-MM>.<run_ts>.jsonl          # compact: just timestamps+usage
  input/<provider>/<model>/<YYYY-MM>/<YYYY-MM-DD>.<run_ts>__<seq>.json    # per-call request
  output/<provider>/<model>/<YYYY-MM>/<YYYY-MM-DD>.<run_ts>__<seq>.json   # per-call response, paired by filename
```

Each `stats/...jsonl` line:

```json
{
  "timestamp": "2026-05-18T11:24:33Z",
  "model": "gpt-4o-2024-11-20",
  "user": "ningliu",
  "usage": {
    "prompt_tokens": 1247,
    "completion_tokens": 138,
    "thinking_tokens": 0,
    "cached_tokens": 0,
    "total_tokens": 1385,
    "image_count": 8
  },
  "estimated_cost_usd": 0.0354,
  "duration_ms": 4220,
  "success": true,
  "error": null,
  "metadata": {"benchmark": "videobench", "dim": "action_consistency", "video": "..."}
}
```

`estimated_cost_usd` is computed from the judge config's `cost_per_million_*` fields (none for local; populated for managed).

### 16.3 New CLI subcommand: `videvalkit api-usage`

```bash
# Aggregate all api_logs in a workspace, group by model / day / dim
videvalkit api-usage --workspace ws

# Output:
# google/gemma-4-31b-it     : 12,847 calls,  17.3M tokens,  $0.00  (local)
# openai/gpt-4o-2024-11-20  :    856 calls,   2.1M tokens, $11.42
# google/gemini-3-flash     :     91 calls, 287K tokens,  $0.04
# anthropic/claude-sonnet-4-6:     0 calls
# ---
# total cost: $11.46
```

### 16.4 Migration plan

1. **Keep `SUPPORTED_JUDGES` in `configs/judges.py` for back-compat** but rename the localhost entries from `"…-local"` → `"…-local-default"`, and add the explanatory note "edit `~/.config/videvalkit/judges.yaml` to override the endpoint port".
2. **Add `videvalkit.scorers.vlm_judge.factory.resolve_judge()`** — accepts a name and applies the precedence order above.
3. **Add cost-tracking to `api_log.py`** — drop the existing `_update_stats` helper and replace with the reference-schema-compatible writer.
4. **Add CLI flags** — `--judge-endpoint`, `--judge-model`, `--judge-kind`, `--judge-api-key-env`, `--judge-config` (path to custom YAML).
5. **Add `videvalkit api-usage` subcommand** — aggregator over a workspace's api_logs.
6. **Document in USER_MANUAL** — three subsections: "Using a local vLLM server", "Using OpenAI", "Using Gemini".

### Files to add / modify

* `src/videvalkit/configs/judges.py` — keep + annotate, no removals
* `src/videvalkit/scorers/vlm_judge/factory.py` — new precedence resolver
* `src/videvalkit/storage/api_log.py` — port the reference schema
* `src/videvalkit/cli.py` — `--judge-*` flags + `api-usage` subcommand
* `src/videvalkit/utils/cost.py` — token → dollars math, per-provider
* `~/.config/videvalkit/judges.yaml` (user-installable example, shipped under `docs/sample_configs/judges.yaml`)
* `USER_MANUAL.md` — three new subsections (above)
* `DEV_MANUAL.md` §16 — this section

---

## 17. User-Configurable GPU Selection & Multi-GPU Dispatch (planned 2026-05-18)

**Motivation.** During the validation sweep we manually pinned VBench-2.0 dims to GPUs 0/1/2 in parallel via `CUDA_VISIBLE_DEVICES=N python ...`, and raced a slow `Instance_Preservation` on GPU 3 by launching a backup on GPU 2. Today there is no first-class way for a toolkit user to say *"run worldjen on GPU 1"* or *"shard VBench-2.0's 18 dims across GPUs 0,1,2"* — they must edit subprocess invocations by hand. This section formalizes that into the public surface.

### 17.1 The user-facing surface

Three layers, precedence highest-first:

1. **CLI flag** (per-invocation):
   ```bash
   videvalkit eval --bench vbench2 --videos ... --gpu 2                            # single GPU
   videvalkit eval --bench vbench2 --videos ... --gpus 0,1,2                       # dim-parallel pool
   videvalkit eval --bench vbench2 --videos ... --gpus auto                        # pick least-loaded
   videvalkit eval --bench vbench2 --videos ... --gpu-affinity Mechanics=0,Motion_Rationality=1
   ```
2. **Workspace YAML** (per-run, checked-in for reproducibility):
   ```yaml
   # <workspace>/config.yaml
   compute:
     gpus: [0, 1, 2]
     affinity:
       vbench2.Mechanics:           0
       vbench2.Motion_Rationality:  1
       t2vcompbench.consistent_attribute: 2   # LLaVA-1.6-34B, GPU-pinned
     reserve_mem_gb: 6                         # don't pick a GPU with less than this free
   ```
3. **User YAML** (persistent default — same precedence pattern as `~/.config/videvalkit/judges.yaml` in §16):
   ```yaml
   # ~/.config/videvalkit/compute.yaml
   compute:
     gpus: auto                                # default for all benches
     auto_strategy: most_free_memory           # alternatives: round_robin, least_utilization
     reserve_mem_gb: 6
   ```
4. **Env var fallback** — `CUDA_VISIBLE_DEVICES` is respected as the floor (the toolkit will never reach a GPU masked out by it).

**Precedence:** CLI > workspace YAML > user YAML > `CUDA_VISIBLE_DEVICES` env > (no constraint = visible GPU 0).

### 17.2 Three dispatch modes

| Mode | When to use | Behavior |
|---|---|---|
| `single` | most users, single-GPU box | Whole bench runs on one device; CLI accepts `--gpu N` |
| `dim_parallel` | multi-GPU box, multi-dim bench (vbench2 18 dims, t2vcompbench 7 dims, worldscore 10 dims) | Dim list partitioned across the GPU pool. Each dim launches in its own subprocess with `CUDA_VISIBLE_DEVICES=N`. Output JSONs merged by the aggregator. |
| `affinity` | mixed-weight dims (e.g. T2V-CompBench `consistent_attribute` 68 GB LLaVA on a dedicated GPU, lighter dims sharing) | User pins specific dims to specific GPUs; remaining dims fall back to the pool |

The dispatch mode is **inferred** — `--gpu 2` → single, `--gpus 0,1,2` → dim_parallel, `--gpu-affinity ...` (or YAML `affinity:` block) → affinity. No explicit `--mode` flag.

### 17.3 The `ComputeRouter` abstraction

A new module owns this concern so individual benchmarks don't reinvent it.

```python
# src/videvalkit/runtime/compute_router.py
@dataclass
class GpuSlot:
    index: int                       # CUDA device index (system, before CVD masking)
    free_mem_gb: float
    util_pct: int
    reserved_by: str | None          # set when assigned to a dim

class ComputeRouter:
    def __init__(self, pool: list[int] | Literal["auto"], reserve_mem_gb: float = 6.0,
                 affinity: dict[str, int] | None = None):
        ...
    def discover(self) -> list[GpuSlot]:
        """Run `nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits` and parse."""
    def acquire(self, dim_id: str, min_mem_gb: float = 0) -> int:
        """Block until a GPU in the pool meets `min_mem_gb`; return its index. Honors affinity[dim_id] if set."""
    def release(self, gpu_index: int): ...
    def launch_subprocess(self, dim_id: str, argv: list[str], env: dict | None = None) -> subprocess.Popen:
        """Convenience: acquire → set CUDA_VISIBLE_DEVICES → Popen → register release on exit."""
```

Each benchmark's `evaluate()` consults `ctx.compute_router` for both single-dim runs (no-op when `pool=[0]`) and dim-parallel runs. The router is the single source of truth for "which GPU am I on", replacing the current ad-hoc `os.environ.setdefault("CUDA_VISIBLE_DEVICES", ...)` scattered across benchmark code.

### 17.4 Min-memory hints per dim

Some dims need a known-large GPU; the router should refuse to schedule them on a half-full one. Hints live in each benchmark's manifest:

```python
# src/videvalkit/benchmarks/t2vcompbench/manifest.py
GPU_REQUIREMENTS = {
    "consistent_attribute": {"min_mem_gb": 75, "exclusive": True},   # LLaVA-1.6-34B fp16
    "action_binding":       {"min_mem_gb": 75, "exclusive": True},
    "object_interactions":  {"min_mem_gb": 75, "exclusive": True},
    "numeracy":             {"min_mem_gb": 8},                       # GroundingDINO + SAM
    "spatial":              {"min_mem_gb": 8},
    "motion_binding":       {"min_mem_gb": 12},                      # DOT
    "dynamic_attribute":    {"min_mem_gb": 12},
}
```

`exclusive: True` means the router does not schedule a second dim onto this GPU until the previous one releases — used by the LLaVA-34B dims to prevent thrashing.

VBench-2.0 analogous map: `Composition`, `Diversity`, `Mechanics` etc. typically need ~16 GB; `Instance_Preservation` is fine on 8 GB.

### 17.5 Auto-select strategies

When `pool="auto"`, the router picks at acquisition time:

- `most_free_memory` (default) — argmax of `free_mem_gb`
- `round_robin` — useful when all GPUs are roughly equally loaded
- `least_utilization` — for compute-bound dims where memory isn't the bottleneck

User picks via `auto_strategy:` in YAML or `--gpu-strategy` CLI flag.

### 17.6 Logging & observability

The router writes a `compute_log.jsonl` to the workspace, one line per dim launch:

```json
{"ts": "2026-05-18T16:52:14Z", "dim_id": "vbench2.Mechanics", "gpu": 0,
 "free_mem_gb_at_acquire": 142.8, "free_mem_gb_at_release": 134.1,
 "duration_s": 612, "wait_s_for_gpu": 0,
 "subprocess_pid": 28847, "exit_code": 0}
```

This pairs with `api_log.jsonl` from §16: one captures judge spend, the other captures local GPU spend. Together they answer *"what did this benchmark run cost me?"*

### 17.7 Worked example: the validation sweep we just did

What we ran by hand:

```bash
# Worker A: 4 dims in series on GPU 0
CUDA_VISIBLE_DEVICES=0 videvalkit eval --bench vbench2 --dim Mechanics ...
# Worker B: 4 dims in series on GPU 1
CUDA_VISIBLE_DEVICES=1 videvalkit eval --bench vbench2 --dim Motion_Rationality ...
# Backup: race Instance_Preservation on GPU 2
CUDA_VISIBLE_DEVICES=2 videvalkit eval --bench vbench2 --dim Instance_Preservation ...
```

What this section enables:

```bash
videvalkit eval --bench vbench2 --videos $VIDEOS --gpus 0,1,2 --gpu-strategy most_free_memory
# Router shards the 18 dims across 3 GPUs, honors GPU_REQUIREMENTS, writes compute_log.jsonl,
# falls back to the next-freest GPU if one OOMs, and aggregates the dim JSONs at the end.
```

### 17.8 Defaults & guard-rails

- **No `--gpus` and no YAML** → behaves exactly like today (single GPU, whatever `CUDA_VISIBLE_DEVICES` exposes first). No surprise multi-GPU greedy behavior.
- **`--gpus auto` + single visible GPU** → degrades to single-mode silently.
- **`reserve_mem_gb` defaults to 6 GB** so background processes (your editor's Pyright, jupyter notebooks) survive.
- **Affinity to a GPU not in the pool** → hard error at startup, not at acquire time.
- **All upstream subprocess invocations** (T2V-CompBench MLLM shim, VBench v1 paper-mode) inherit `CUDA_VISIBLE_DEVICES` from the router — no upstream code changes needed.

### Files to add / modify

* `src/videvalkit/runtime/compute_router.py` — new module (router, `GpuSlot`, `nvidia-smi` parser)
* `src/videvalkit/runtime/__init__.py` — export `ComputeRouter`
* `src/videvalkit/benchmarks/<bench>/manifest.py` — add `GPU_REQUIREMENTS` to each of the 6
* `src/videvalkit/benchmarks/base.py` — `evaluate()` signature: add `compute_router` field on the context object
* `src/videvalkit/cli.py` — `--gpu`, `--gpus`, `--gpu-affinity`, `--gpu-strategy` flags; route through router factory
* `src/videvalkit/configs/compute.py` — YAML schema, precedence resolver (mirrors §16's `judges.py`)
* `~/.config/videvalkit/compute.yaml` (user-installable example, shipped under `docs/sample_configs/compute.yaml`)
* `src/videvalkit/storage/compute_log.py` — JSONL writer
* `USER_MANUAL.md` — new subsection "Running on multiple GPUs"
* `DEV_MANUAL.md` §17 — this section

---

> End of manual. New modules / capabilities should fit the numbering in [§4](#4-module-overview-ah-eight-modules); changes to externally-facing interfaces (registries / CLI / api_logs schema) go through the CHANGELOG flow in [Module H](#58-module-h--packaging-distribution--remote-vlm-onboarding).
