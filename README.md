# videogenevalkit

**Unified evaluation toolkit for text-to-video generation.**

*[English](README.md) · [中文](README.zh.md)*

One CLI, one workspace, one schema. Score a T2V model **three ways** — a whole
**benchmark**, a single **metric**, or a whole **capability** — with the **judge
of your choice**. Benchmark scores compare byte-for-byte against official
leaderboards.

> 📖 **Documentation: the [wiki](docs/index.md) is the primary, bilingual
> (English / 中文) reference** — getting started, guides, and CLI / metrics /
> benchmarks / judges reference. Design rationale lives in
> [`docs/design/`](docs/design/PRODUCT_DESIGN.md); long-form install in
> [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md).

<p align="left">
  <a href="#quickstart"><img alt="quickstart" src="https://img.shields.io/badge/quickstart-30%20min-blue"></a>
  <a href="#whats-supported"><img alt="benchmarks" src="https://img.shields.io/badge/benchmarks-10-orange"></a>
  <a href="https://huggingface.co/datasets/videogenevalkit/checkpoints"><img alt="HF checkpoints" src="https://img.shields.io/badge/HF-checkpoints-yellow?logo=huggingface"></a>
  <a href="LICENSES/"><img alt="licenses" src="https://img.shields.io/badge/licenses-multi--upstream-lightgrey"></a>
</p>

---

## Three ways to evaluate

| Entry point | Command | Answers |
|---|---|---|
| **Benchmark** | `videvalkit eval --bench vbench` | "How does my model score on VBench?" |
| **Metric** | `videvalkit metric run --name fvd ...` | "What's the FVD of these videos?" |
| **Capability** | `videvalkit capabilities eval motion ...` | "How good is motion across all metrics?" |

A metric lifted out of a benchmark is **bit-exact** whether you reach it via
`--bench` or `metric run` — one implementation, no drift. Three design promises:
*adapter, not reimplementation* · *pluggable backends* · *plugin-first (extend
via YAML / pip / local dir, no fork)*.

---

## What's supported

**10 benchmark adapters** — 6 anchored to public leaderboards + Semantics-Axis
(in-house) + 3 supplementary:

| Adapter | Upstream | What it scores |
|---|---|---|
| `vbench` | [Vchitect/VBench](https://github.com/Vchitect/VBench) (CVPR 2024) | 16 dims, quality + semantic; weighted-sum + min-max norm |
| `vbench2` | [Vchitect/VBench-2.0](https://github.com/Vchitect/VBench) | 18 dims across 5 categories incl. Physics |
| `videobench` | [Video-Bench](https://github.com/Video-Bench/Video-Bench) (CVPR 2025) | 9 dims: alignment + static/dynamic quality |
| `worldjen` | [WorldJen](https://github.com/moonmath-ai/WorldJen-benchmarking-subsystem) | 16 dims; PHAS aggregator |
| `worldscore` | [WorldScore](https://github.com/yhw-yhw/WorldScore) | 10 dims: SLAM + RAFT + SAM + IQA stack |
| `t2vcompbench` | [T2V-CompBench V2](https://github.com/KaiyueSun98/T2V-CompBench/tree/V2) | 7 compositional dims; LLaVA-1.6-34B MLLM + CV |
| `semantics_axis` | in-house | 21 prompt-following axes; VLM-judge, 1–5 |

Plus 3 supplementary adapters: Physics-IQ, VBench++, V-ReasonBench.

**Also in v0.2:**

| Capability | What |
|---|---|
| **20 standalone metrics** | FVD · VFID · KVD · CLIP-FVD · CLIP-Score · ViCLIP-Score + 8 bench lift-outs + specialized dims. **16 run judge-free today**; `artifact-diagnostic` runs with a `--judge`. Run any with `videvalkit metric run --name X`. |
| **Judge selection** | `--judge paper/default/<name>` · user `judges.yaml` · ad-hoc `--judge-endpoint` · `--no-judge` for fully offline runs |
| **Eval profiles** | `--profile quick/standard/full` + `videvalkit estimate` cost preview |
| **Training monitor** | `videvalkit watch` + `videvalkit.training.monitor` Python API |
| **Capability tags** | 44-tag vocab; `videvalkit capabilities list/show/eval` |

**8 VLM/LLM judges** out of the box: local vLLM (Gemma-4-31B, Qwen3-32B,
Qwen3-VL-32B, LLaVA-Video-7B) + managed APIs (Gemini, GPT-4o, Claude); add your
own in `~/.config/videvalkit/judges.yaml`. **Aggregators**: weighted_sum ·
vbench_weighted · vbench2_category · phas · bt.

---

## Reproducibility — what you actually get

We re-ran the official leaderboards for the 6 anchored benchmarks. **mean |Δ| vs
published numbers:**

| Benchmark | Model | Dims | Mean \|Δ\| | Notes |
|---|---|---:|---:|---|
| **VBench v1** | HunyuanVideo | 16/16 | **0.012** | matches HF leaderboard |
| **VBench-2.0** | HunyuanVideo | 18/18 | **0.0055** | 4 dims byte-exact |
| **T2V-CompBench** | CogVideoX-5B | 6/7 | **0.0046** | paper-exact LLaVA-1.6-34B; one outlier documented |
| **Video-Bench** | CogVideoX-5B | 9/9 | offset | Gemma stand-in for GPT-4o; static+alignment match |
| **WorldJen** | Kling-v2.6 | 16/16 | Δ −0.47 | decord-vs-cv2 frame variance |
| **WorldScore** | CogVideoX-5B | 10/10 | wired | DROID-SLAM + SEA-RAFT + VFIMamba + SAM2 |

Full per-dim tables: [`docs/TEST_MANUAL.md`](docs/TEST_MANUAL.md).

---

## Quickstart

### 1. Clone + install env (~10 min)

The toolkit ships as a pre-packed `conda-pack` env tarball — the byte-for-byte
environment that produced every result in `docs/TEST_MANUAL.md` (Python 3.10 +
torch 2.3.1+cu121 + ~350 pinned deps). Linux + CUDA required.

```bash
git clone https://github.com/videogenevalkit/videogenevalkit.git
cd videogenevalkit

# Download + unpack the env tarball (~7.6 GB download, ~15 GB on disk)
hf download videogenevalkit/env-tarball videvalkit-env.tar.gz --local-dir /tmp
sudo mkdir -p /opt/videvalkit-env
sudo tar xzf /tmp/videvalkit-env.tar.gz -C /opt/videvalkit-env
sudo chown -R $USER /opt/videvalkit-env
source /opt/videvalkit-env/bin/activate
conda-unpack                       # rewrites absolute paths inside the env

# Install the toolkit + build-from-source extras
pip install --no-deps -e .
bash scripts/post_install.sh       # detectron2, SAM-2, GroundingDINO, ...
videvalkit doctor                  # verify: env, GPU, caches, judges
```

> macOS / no GPU: `videvalkit list / metric list / capabilities list` and
> development work, but benchmark/metric *execution* needs Linux + CUDA.

### 2. Fetch smoke data + checkpoints (~15 min, ~8 GB)

```bash
videvalkit fetch-smoke-data                                    # videos + prompts
videvalkit fetch-checkpoints --bench worldscore --bench t2vcompbench
```

LLaVA-1.6-34B (68 GB, T2V-CompBench paper-mode MLLM) is **not** bundled — it
resolves from upstream HF on first paper-mode run.

### 3. Run your first benchmark (~5 min)

```bash
videvalkit fetch-smoke-data --bench worldjen
mkdir -p ~/runs/worldjen/videos/Kling
ls ~/.cache/videvalkit/smoke-data/worldjen/videos/*/*.mp4 | head -3 \
  | xargs -I{} ln -sf {} ~/runs/worldjen/videos/Kling/

videvalkit eval --bench worldjen \
    --videos ~/runs/worldjen/videos --workspace ~/runs/worldjen/ws \
    --models Kling --judge gemma-4-31b-local --aggregator phas
# → ~/runs/worldjen/ws/results/summary/worldjen/Kling.json
```

No judge endpoint? Run the judge-free path:

```bash
videvalkit list benchmarks --no-judge        # vbench · worldscore · physics_iq · v_reasonbench
videvalkit eval --bench vbench --no-judge --profile quick --videos gen/ --workspace ws/
```

### 4. A single metric / a capability

```bash
# distribution metric (needs a reference set; FVD auto-downloads its backbone)
videvalkit metric run --name fvd --gen-videos gen/ --refs ucf101-fvd --allow-tiny-sample

# cross-metric capability score
videvalkit capabilities eval visual_quality --videos gen/
```

See the [wiki Getting Started](docs/index.md) for the full walkthrough and
[`docs/USER_MANUAL.md`](docs/USER_MANUAL.md) for per-benchmark recipes.

---

## What you DON'T have to do

- **No re-implementing scorers.** Adapters delegate to upstream code byte-for-byte;
  we add IO, scheduling, a registry, and a unified output format.
- **No paper API credentials.** Judges are pluggable; default to local vLLM.
  Managed-API support is a one-line config swap. `--no-judge` skips them entirely.
- **No per-benchmark conda envs.** One shared env covers all adapters.

---

## Documentation

| Doc | What's in it |
|---|---|
| **[wiki](docs/index.md)** (en/zh) | primary reference — getting started, guides, CLI/metrics/benchmarks/judges, architecture, roadmap |
| [design archive](docs/design/PRODUCT_DESIGN.md) (en/zh) | *why* each subsystem is built the way it is |
| [`USER_MANUAL.md`](docs/USER_MANUAL.md) (en/zh) | long-form install + per-benchmark run recipes |
| [`TEST_MANUAL.md`](docs/TEST_MANUAL.md) | per-benchmark validation: Δ vs leaderboard, tolerances, known discrepancies |
| [`DEV_MANUAL.md`](docs/DEV_MANUAL.md) | deep architecture (v0.0.1-era; cross-check the wiki) |

The docs build into a searchable site with a language switcher:
`pip install -r requirements-docs.txt && mkdocs serve`.

---

## License & citation

The toolkit is **Apache-2.0**. Each upstream adapter keeps its own license,
surfaced under [`LICENSES/`](LICENSES/) (VBench/VBench-2.0 Apache-2.0;
T2V-CompBench research-only on the LLaVA path; DROID-SLAM / SEA-RAFT /
GroundingDINO / SAM under their respective terms). Cite the upstream papers when
reporting numbers from a specific adapter.

```bibtex
@software{videogenevalkit2026,
  title  = {videogenevalkit: A unified evaluation toolkit for text-to-video generation},
  year   = {2026},
  url    = {https://github.com/videogenevalkit/videogenevalkit},
}
```
