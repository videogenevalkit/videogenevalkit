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

## v0.2 — remaining (~3%)

All blocked on an external prerequisite:

| Item | Blocker |
|---|---|
| `object-binding` metric | a running VLM judge endpoint |
| `artifact-diagnostic` metric | judge endpoint + Artifact-Bench taxonomy port |
| `identity-preservation` metric | `insightface` install decision (shared env) |
| `motion-accuracy` metric | worldscore SEA-RAFT runner wiring + GPU |
| `viclip-score` functional | ViCLIP weights hosting |
| paper-canonical I3D-FVD run | `i3d_torchscript.pt` hosting (loader ready) |
| `fetch-refs` | `videogenevalkit/reference-videos` HF dataset |

Unblocking levers: **start a judge endpoint** (→ object-binding + artifact-diagnostic),
**install insightface** (→ identity-preservation), **host I3D + ViCLIP weights**
(→ paper FVD + viclip-score).

---

## Metric scoreboard

| | Count | Which |
|---|---|---|
| ✅ functional | 14 | fvd, vfid, kvd, clip-fvd, clip-score, 7 vbench lifts, numeracy, spatial-relationship |
| ⚪ shell | 2 | viclip-score, motion-magnitude |
| ⛔ not registered | 4 | object-binding, motion-accuracy, identity-preservation, artifact-diagnostic |

See [Metrics Reference](reference/Metrics.md) for detail.
