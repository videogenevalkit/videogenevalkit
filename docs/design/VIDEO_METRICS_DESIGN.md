# Video Metrics — Design

> Rationale for the standalone-metrics layer. The live per-metric catalog (names,
> backbones, status) is the [Metrics Reference](../wiki/reference/Metrics.md); this
> doc explains the *structure* behind it.

---

## 1. Design principles

- **Two tiers.** General T2V quality (applies to any video) vs. specialized
  dimensions (compositional / artifact / identity). The split keeps the common,
  judge-free metrics easy to reach and quarantines the heavier, often
  judge/prompt-conditioned ones.
- **Dual entry, one implementation.** A metric is reachable two ways —
  `eval --bench X --dimensions Y` and `metric run --name Y` — and both call the
  *same* code. No second implementation to drift.
- **Judge-free by default.** Most metrics need no VLM/LLM; `needs_judge` marks the
  exceptions so `--no-judge` can filter them out.
- **Don't ship inapplicable metrics.** PSNR/SSIM/LPIPS/FID-image are
  reference-based per-frame metrics needing a ground-truth video; T2V has none, so
  they are excluded (they belong to I2V/V2V/reconstruction).

---

## 2. Two tiers

**Tier 1 — general T2V quality** (judge-free unless noted):

- *Distribution* (need a reference set): `fvd`, `vfid`, `kvd`, `clip-fvd`.
- *Text-video alignment* (need prompts): `clip-score`, `viclip-score`.
- *Frame perceptual* (lift from vbench): `aesthetic-quality`, `imaging-quality`.
- *Temporal* (lift from vbench/worldscore): `motion-smoothness`,
  `temporal-flickering`, `subject-consistency`, `background-consistency`,
  `dynamic-degree`, `motion-magnitude`.

**Tier 2 — specialized dimensions**: `numeracy`, `spatial-relationship` (CV,
judge-free), `artifact-diagnostic` (MLLM judge), plus `object-binding` and
`motion-accuracy` which are prompt/judge-conditioned and therefore **bench-only**
(run them via their benchmark, not as standalone `--videos` metrics), and
`identity-preservation` (deferred to the i2v phase).

---

## 3. The dual-entry contract (bit-exact lift)

A *lift* exposes a benchmark dimension as a standalone metric by wrapping the
**identical upstream call** the bench adapter makes. Concretely:

- `motion-smoothness` (standalone) wraps the same
  `VBench.evaluate(dimension_list=["motion_smoothness"], mode="custom_input")`
  call as `eval --bench vbench --dimensions motion_smoothness`.
- `motion-magnitude` wraps the worldscore bench's `OpticalFlowMetric` (SEA-RAFT)
  via the same `OpticalFlowScorer`.

The contract: results must be **bit-exact (≤ 1e-6)** between the two entry points.
This is enforced as a `lift-out` PR gate (see [Review Protocol](REVIEW_PROTOCOL.md)).
The payoff: the capability layer and standalone CLI reuse bench machinery with
zero formula divergence.

Distribution metrics are the exception — they have no bench origin and subclass
`BaseDistributionMetric` directly, sharing the Fréchet (`frechet.py`, float64) and
polynomial-MMD (`mmd.py`) utilities and the backbone loaders.

---

## 4. Backbone policy

| Backbone | Used by | Notes |
|---|---|---|
| S3D-K400 | FVD/KVD default | auto-downloads; a valid Kinetics-400 trend metric — ideal for monitoring |
| I3D-K400 | FVD (paper) | torchscript; paper-canonical when the weight is placed (`--backbone i3d-k400`) |
| InceptionV3 | VFID | torchvision, auto |
| CLIP-ViT-L/14 | clip-score, clip-fvd | openai-clip, in-env |
| ViCLIP-L/14 | viclip-score | vendored; weight auto-fetched from OpenGVLab |

FVD auto-falls-back i3d→s3d when paper weights are absent, so monitoring works
out of the box; paper numbers require the I3D weight explicitly.

---

## 5. Registry schema

Every entry in `SUPPORTED_METRICS` carries the required fields `kind`, `source`,
`cls`, `needs_judge`, `compute_kind`, `tags` (the consistency check enforces
these). `kind` drives runner dispatch:

| `kind` | Inputs |
|---|---|
| `distribution_reference` | gen videos + reference set |
| `per_prompt_reference_free` | videos + prompts |
| `per_video_reference_free` | videos |
| `per_video_with_vlm_judge` | videos + judge |

`tags` reference the [capability vocabulary](CAPABILITY_TAGS_DESIGN.md); `source`
records the origin (`canonical/...` or `<bench>/<dim>` for a lift); lifts also
set `also_used_by` for dedup in the capability resolver.

---

## 6. Artifact-diagnostic (Artifact-Bench port)

The v0.2 slice of [Artifact-Bench](https://github.com/FrankYang-17/Artifact-Bench)
(arXiv 2605.18984): an MLLM multi-label detector over a 30-type taxonomy
(3 categories → 11 families → 30 leaves) in `metrics/artifact_taxonomy.py`. It is
a `per_video_with_vlm_judge` metric, research-only license, and overlaps some
lifts by design (`flickering`~temporal-flickering, `identity_drift`~
subject-consistency). The full judge-eval benchmark is a v0.3 deliverable; the
taxonomy leaves are a working port to be reconciled against the paper.
