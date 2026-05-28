# Roadmap & Status

[← Home](Home.md)

> Snapshot: `v0.2-dev` · 24 PRs merged · 363 tests green · ~97% of v0.2 scope.

---

## Version timeline

| Version | Date | Theme | Status |
|---|---|---|---|
| v0.0.1 | 2026-05-13 | Anchored adapters | ✅ |
| v0.1.0 | 2026-05-19 | Validation (mean \|Δ\| published) | ✅ |
| **v0.2.0** | ~2026-07 | Extensibility + metrics + capabilities + training monitor | 🔵 ~97% |
| v0.3.0 | later | Stub benches made real · domestic VLMs · scaffolding CLI | ⏳ |
| v0.4.0 | later | macOS/CPU subset · leaderboard site · NPU | ⏳ |
| v1.0 | 2026-Q4 | Paper · colab · judge-eval subproject | ⏳ |

---

## v0.2 — done

| Area | Delivered |
|---|---|
| Judge selection | `--judge paper/default/<name>` · user `judges.yaml` · ad-hoc endpoint · `--no-judge` |
| Plugin loader | 3-layer discovery (builtin / entry_points / local dirs) |
| Manifest benchmark | YAML Track-A adapter |
| Metrics (14 functional) | FVD · VFID · KVD · CLIP-FVD · CLIP-Score · 7 vbench lifts · numeracy · spatial-relationship |
| Metric/refs CLI | `metric list/show/run` · `refs list/show/register` |
| Quick eval | 3 profiles · subset · `estimate` · `eval-suite` |
| Training monitor | `watch` · `videvalkit.training.monitor` Python API |
| Capability tags | 44-tag vocab · resolver · `capabilities list/show/eval` |
| Quality | Review protocol + CI (3 check scripts) · enhanced `doctor` |
| Docs | this wiki |

---

## v0.2 — remaining

| Item | Status / blocker |
|---|---|
| `artifact-diagnostic` metric | ✅ implemented + mock-tested; runs once a `--judge` endpoint is configured. Taxonomy leaves to reconcile vs the paper; full judge-eval bench is v0.3. |
| `object-binding` | **available via the bench** (`eval --bench t2vcompbench --dimensions consistent_attribute`, MLLM-judged). Not a standalone `--videos` metric — re-wraps the same judge call. |
| `motion-accuracy` | **available via the bench** (`eval --bench worldscore --dimensions motion_accuracy`). Prompt-conditioned (intended motion direction) → inherently bench-only, not a standalone metric. |
| `identity-preservation` | **deferred to the i2v phase** — its prime use (reference-image identity match) is i2v-adjacent, out of v0.2 T2V scope. |
| paper-canonical I3D-FVD run | `i3d_torchscript.pt` hosting (loader ready; S3D fallback covers monitoring) |
| `fetch-refs` cross-machine | `videogenevalkit/reference-videos` HF dataset upload (command shipped; `refs register` is the zero-download local path) |

Unblocking levers: **start a judge endpoint** (→ artifact-diagnostic), **host the
I3D torchscript weight** (→ paper-canonical FVD).

---

## Metric scoreboard

| | Count | Which |
|---|---|---|
| ✅ functional | 16 | fvd, vfid, kvd, clip-fvd, clip-score, viclip-score, 7 vbench lifts, motion-magnitude, numeracy, spatial-relationship |
| ✅ registered (needs judge) | 1 | artifact-diagnostic (Artifact-Bench port; runs with `--judge`) |
| ↪ bench-only / ⏳ deferred | 3 | object-binding, motion-accuracy (via their bench); identity-preservation (deferred to i2v) |

See [Metrics Reference](reference/Metrics.md) for detail.
