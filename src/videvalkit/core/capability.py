"""Capability resolver — map a capability tag to the list of metric/bench-dim
contributors, with dedup by canonical source.

Per docs/CAPABILITY_TAGS_DESIGN.md §7-§9 (user 2026-05-20 confirmed):
  * Resolve: tag → list of (source, name, registry_entry) triples
  * Dedup: same canonical_source::name across multiple registry entries → keep
    one. (Used when a lift metric appears both standalone and as bench dim.)
  * Aggregation logic (min-max + dedup + mean) is layered on top of this in
    runner / cli_capability — kept separate so resolution stays pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from videvalkit.configs.capability_taxonomy import (
    ALL_TAGS,
    SUB_TAG_TO_TOP,
    SUB_TAGS_BY_TOP,
    TAG_SCHEMA_VERSION,
    expand_capability,
    is_valid_tag,
)


@dataclass(frozen=True)
class Contributor:
    """One metric or bench-dim that contributes to a capability tag."""

    source_kind: str          # "metric" | "bench_dim"
    source_name: str          # e.g. "fvd" or "vbench/motion_smoothness"
    canonical_source: str | None  # for dedup, e.g. "canonical/vbench-port::motion-smoothness"
    tags: list[str]           # the tags this entry declares
    cfg: dict[str, Any]       # the registry entry (kept by reference)


def _collect_metric_contributors(
    metrics_registry: dict[str, dict[str, Any]] | None = None,
) -> list[Contributor]:
    """Walk SUPPORTED_METRICS (if present) and yield contributors."""
    if metrics_registry is None:
        try:
            from videvalkit.metrics import SUPPORTED_METRICS as _m
            metrics_registry = _m
        except ImportError:
            # metrics module not landed yet (v0.2 M3 in flight)
            return []
    out = []
    for name, cfg in metrics_registry.items():
        tags = cfg.get("tags", [])
        if not tags:
            continue
        canonical = cfg.get("source")
        dedup_key = f"{canonical}::{name}" if canonical else None
        out.append(Contributor(
            source_kind="metric",
            source_name=name,
            canonical_source=dedup_key,
            tags=list(tags),
            cfg=cfg,
        ))
    return out


def _collect_bench_dim_contributors(
    benchmarks_registry: dict[str, dict[str, Any]] | None = None,
) -> list[Contributor]:
    """Walk SUPPORTED_BENCHMARKS and yield bench-dim contributors via
    the ``dim_tags`` field."""
    if benchmarks_registry is None:
        from videvalkit.configs import SUPPORTED_BENCHMARKS
        benchmarks_registry = SUPPORTED_BENCHMARKS
    out = []
    for bench_name, cfg in benchmarks_registry.items():
        dim_tags = cfg.get("dim_tags", {})
        for dim_name, tags in dim_tags.items():
            if not tags:
                continue
            # Bench-dim canonical: "bench/dim" — same as metric source convention
            canonical = f"{bench_name}/{dim_name}"
            out.append(Contributor(
                source_kind="bench_dim",
                source_name=f"{bench_name}/{dim_name}",
                canonical_source=canonical,
                tags=list(tags),
                cfg={"bench": bench_name, "dim": dim_name},
            ))
    return out


def all_contributors() -> list[Contributor]:
    """Return all metric + bench-dim contributors with tags declared."""
    return _collect_metric_contributors() + _collect_bench_dim_contributors()


def resolve_capability(capability: str) -> list[Contributor]:
    """Return all contributors covering the given capability tag.

    Accepts both top-level tags (e.g. ``"motion"``) and sub-tags
    (e.g. ``"motion.smoothness"``). Top-level matches expand to all subs.

    Dedup applied: if a metric and a bench-dim share the same canonical_source,
    keep the metric (preferred — short name, direct call path).
    """
    if not is_valid_tag(capability):
        raise ValueError(
            f"unknown capability {capability!r}; "
            f"see docs/CAPABILITY_TAGS_DESIGN.md §3 for the 44-tag vocab"
        )
    wanted = set(expand_capability(capability))

    matched = [
        c for c in all_contributors()
        if any(t in wanted for t in c.tags)
    ]
    return _dedup_by_canonical(matched)


def _dedup_by_canonical(contributors: list[Contributor]) -> list[Contributor]:
    """Drop duplicates sharing canonical_source. Prefer ``metric`` over
    ``bench_dim`` since the metric is the canonical lift-out and bench-dim
    routes through the same code anyway."""
    seen: dict[str, Contributor] = {}
    for c in contributors:
        key = c.canonical_source
        if key is None:
            # No canonical declared → unique by (kind, name)
            key = f"{c.source_kind}::{c.source_name}"
        existing = seen.get(key)
        if existing is None:
            seen[key] = c
        elif existing.source_kind == "bench_dim" and c.source_kind == "metric":
            # metric beats bench_dim for the same canonical source
            seen[key] = c
        # else: keep existing
    return list(seen.values())


def coverage_report() -> dict[str, list[str]]:
    """For diagnostics: each tag → list of contributor names that cover it.

    Useful for ``videvalkit capabilities list`` to show counts, and for
    REVIEW_PROTOCOL hygiene checks (e.g. "every top-level tag has at least
    one contributor").
    """
    contributors = all_contributors()
    out: dict[str, list[str]] = {t: [] for t in ALL_TAGS}
    for c in contributors:
        for t in c.tags:
            if t in out:
                out[t].append(c.source_name)
    return out


def schema_version() -> int:
    """Return the v1 controlled-vocab schema version."""
    return TAG_SCHEMA_VERSION
