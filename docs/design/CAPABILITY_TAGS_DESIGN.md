# Capability Tags — Design

> Rationale for the by-ability evaluation axis. The tag list and CLI are in the
> [Capability Tags reference](../wiki/reference/Capability-Tags.md).

---

## 1. Problem

Benchmarks slice quality differently, and the same ability (say, motion) is
measured by several metrics scattered across VBench, WorldScore, and standalone
code. A user who asks "how good is motion?" shouldn't have to know which benchmark
owns which metric. The capability axis is the third entry point — alongside
`--bench` (paper-comparable) and `--name` (single scalar) — letting you evaluate
*by ability*.

## 2. Controlled vocabulary

A fixed **44-tag** vocabulary: 10 top-level capabilities, each with sub-tags
(34 total). Canonical form is `<prefix>.<leaf>` (e.g. `motion.smoothness`,
`comp.spatial`, `real.distribution`). The 10 top-level: motion, visual_quality,
text_alignment, object_fidelity, subject_consistency, physical_plausibility,
temporal_coherence, realism, compositional, style.

**Why controlled, not free-form**: free-form tags drift into synonyms
(`motion`/`movement`/`dynamics`) and break cross-benchmark grouping. Tags are
rejected at load time if not in the vocabulary, and the consistency check verifies
every metric/dim tag is in-vocab. The vocabulary is **versioned**
(`tag_schema_version = 1`); any change bumps the version.

## 3. Resolution

`capabilities eval <tag>`:

1. **Resolve** — a top-level tag expands to all its sub-tags; collect every metric
   and bench-dimension carrying any of them.
2. **Dedup** — a lifted metric and its origin bench-dim share a canonical source,
   so they are counted once (the metric is preferred). This is why lifts record
   `also_used_by`.
3. **Run** — each runnable contributor (per-video, judge-free) computes on the
   videos; metrics needing refs/prompts/judge are skipped *with a reason* (the
   capability axis is a quick per-video read).
4. **Normalize** — min-max each metric to [0, 1].
5. **Aggregate** — mean (or max/min) across contributors → one capability score.

## 4. Scope

v0.2 ships the vocabulary, the resolver, and `capabilities list/show/eval`.
Plugins reuse the existing vocabulary; user-defined custom tags are a later
candidate. Tagging is additive metadata on metrics/dims — it never changes how a
metric computes, only how it can be grouped.
