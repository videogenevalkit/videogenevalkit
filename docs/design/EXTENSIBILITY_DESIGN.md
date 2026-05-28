# Extensibility — Judges & Integration

> Merges the former JUDGE_SELECTION and INTEGRATION_FRAMEWORK design docs: both
> answer "how do you add something without forking?" Operational how-to lives in
> [Judge Selection](../wiki/guides/Judge-Selection.md) and
> [Extending](../wiki/guides/Extending.md); this is the rationale.

---

## Part A — Judge switching

### A.1 The problem

A benchmark's published numbers depend on *which* VLM judged them. v0.0.1 hard-wired
one judge per bench, which made (a) faithful reproduction and (b) cheap iteration
mutually exclusive, and silently swapping the model would corrupt comparisons.

### A.2 Three declared tiers

Every judge-using benchmark declares two slots in its registry entry, and the CLI
exposes a third path:

| Selector | Resolves to | For |
|---|---|---|
| `--judge paper` | the bench's `paper_judge` | faithful reproduction |
| `--judge default` | the bench's `default_judge` | cheap / validated iteration |
| `--judge <name>` | any `SUPPORTED_JUDGES` entry | a specific model |

`resolve_judge(benchmark, judge_name, judge_override)` maps the semantic keyword to
a concrete config. `--judge paper` on a judge-free bench fails fast — the bench
never silently downgrades.

### A.3 Bring your own judge

Two paths, no fork:

- **Persistent** — `~/.config/videvalkit/judges.yaml` adds named entries
  (`kind: openai_compatible | gemini | anthropic`, endpoint, model, `api_key_env`).
- **Ad-hoc** — `--judge-endpoint / --judge-model / --judge-kind / --judge-api-key-env`
  for a one-off, mutually exclusive with `--judge`.

Config precedence (later overrides earlier, top-level key replacement):
`builtin → ~/.config → $CWD/.videvalkit → env → CLI`.

### A.4 The judge-free path

`needs_judge` on benches and metrics powers `--no-judge`, which filters to work
runnable with no VLM/LLM at all (4 benchmarks, 17 metrics). A judge backend is
built once by `build_judge(cfg)` and reused across a run; every call is mirrored to
the workspace API log for offline replay.

---

## Part B — Integration framework

### B.1 Two tracks (80/20)

| Track | When | How |
|---|---|---|
| **A — Manifest** | "prompt → scorer → score" (~80%) | one `manifest.yaml` (`schema_version: 1`, ≤ 12 top-level fields) realized by `ManifestBenchmark` |
| **B — Python adapter** | staging / multi-stage / subprocess (~20%) | a `BaseBenchmark` subclass (4 methods) |

Both register the same way and converge on one runner / workspace / scheduler — the
runner cannot tell them apart. This is the deliberate split: forcing everyone onto
YAML makes complex cases painful; forcing everyone into Python raises the bar for
simple ones.

### B.2 Three-layer plugin discovery

Lowest precedence first:

1. **builtin** — `src/videvalkit/{benchmarks,metrics}/`
2. **pip entry_points** — `[project.entry-points."videvalkit.benchmarks"]`
3. **local dirs** — `~/.videvalkit/<group>/` then `$CWD/.videvalkit/<group>/`

Local plugins use the `__videvalkit_register__()` convention returning
`{"benchmarks": {...}}` / `{"metrics": {...}}`. Same-name conflicts are logged at
INFO (later source wins); `VIDEVALKIT_DISABLE_PLUGINS=1` ignores all third-party
sources. `doctor` prints every entry's resolved source for provenance.

### B.3 Adding a metric

Subclass `BaseScorer` (per-video/per-prompt) or `BaseDistributionMetric`
(FVD-family), or — for a benchmark dimension — write a **bit-exact lift** that
wraps the same upstream call (see [Metrics](VIDEO_METRICS_DESIGN.md)). Register in
`SUPPORTED_METRICS` with the required fields (`kind`, `source`, `needs_judge`,
`compute_kind`, `tags`, `cls`).

### B.4 What this is *not*

No plugin manifest DSL, no dynamic-discovery magic — just three lazy-merge sources
feeding one registry. Scaffolding (`videvalkit new`) and a contract validator
(`videvalkit validate`) were considered and deferred to v0.3; the registry +
manifest already make integration a sub-hour task.
