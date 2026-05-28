#!/usr/bin/env python3
"""check_design_doc_consistency.py — cross-doc data consistency.

Per docs/REVIEW_PROTOCOL.md §2: catch drift across the 7 design docs
[PRODUCT / JUDGE / INTEGRATION / VIDEO_METRICS / QUICK_EVAL / CAPABILITY_TAGS
/ REVIEW_PROTOCOL / NPU] where numbers / field names / decisions diverge.

v0.2 checks [conservative — only the highest-payoff invariants]:
  1. Capability taxonomy: 10 + 34 = 44 tags in CAPABILITY_TAGS_DESIGN.md
     matches src/videvalkit/configs/capability_taxonomy.py
  2. Every metric in SUPPORTED_METRICS declares a `tags` field whose values
     are in the controlled vocab [also covered by unit test, doubled here so
     CI catches schema drift even if devs forget to register the metric]
  3. Every bench `dim_tags` entry uses controlled vocab
  4. Every metric registry entry has needs_judge + compute_kind + source fields
     [required per VIDEO_METRICS_DESIGN.md §6]
  5. Every judge-required bench declares paper_judge

Add new invariants as they emerge. Keep each check O(N) and side-effect-free.
"""

from __future__ import annotations

import sys
from pathlib import Path


def check_capability_taxonomy_sizes() -> list[str]:
    """Invariant 1: 10 top + 34 sub = 44 tags."""
    try:
        from videvalkit.configs.capability_taxonomy import (
            ALL_TAGS, SUB_TAGS_BY_TOP, TOP_LEVEL_TAGS,
        )
    except ImportError as e:
        return [f"capability_taxonomy module not importable: {e}"]
    errors: list[str] = []
    if len(TOP_LEVEL_TAGS) != 10:
        errors.append(
            f"TOP_LEVEL_TAGS has {len(TOP_LEVEL_TAGS)} entries; expected 10"
        )
    sub_count = sum(len(v) for v in SUB_TAGS_BY_TOP.values())
    if sub_count != 34:
        errors.append(f"SUB_TAGS_BY_TOP total {sub_count}; expected 34")
    if len(ALL_TAGS) != 44:
        errors.append(f"ALL_TAGS has {len(ALL_TAGS)}; expected 44")
    return errors


def check_metric_tags_in_vocab() -> list[str]:
    """Invariant 2 + 3: every metric/dim tag is in controlled vocab."""
    errors: list[str] = []
    try:
        from videvalkit.configs.capability_taxonomy import ALL_TAGS
        from videvalkit.metrics import SUPPORTED_METRICS
    except ImportError as e:
        return [f"metrics module not importable: {e}"]

    for name, cfg in SUPPORTED_METRICS.items():
        tags = cfg.get("tags", [])
        for t in tags:
            if t not in ALL_TAGS:
                errors.append(
                    f"metric {name!r} has free-form tag {t!r} not in controlled vocab"
                )

    # Bench dim_tags
    try:
        from videvalkit.configs import SUPPORTED_BENCHMARKS
        for bench, bcfg in SUPPORTED_BENCHMARKS.items():
            for dim, tags in bcfg.get("dim_tags", {}).items():
                for t in tags:
                    if t not in ALL_TAGS:
                        errors.append(
                            f"bench {bench!r} dim {dim!r} has free-form tag {t!r}"
                        )
    except ImportError as e:
        errors.append(f"benchmarks registry not importable: {e}")
    return errors


def check_metric_required_fields() -> list[str]:
    """Invariant 4: every metric entry has required fields."""
    errors: list[str] = []
    try:
        from videvalkit.metrics import SUPPORTED_METRICS
    except ImportError as e:
        return [f"metrics module not importable: {e}"]

    REQUIRED = ("kind", "source", "needs_judge", "compute_kind", "tags", "cls")
    for name, cfg in SUPPORTED_METRICS.items():
        missing = [f for f in REQUIRED if f not in cfg]
        if missing:
            errors.append(f"metric {name!r} missing fields: {missing}")
    return errors


def check_paper_judge_declared() -> list[str]:
    """Invariant 5: every needs_judge=True bench declares paper_judge."""
    errors: list[str] = []
    try:
        from videvalkit.configs import SUPPORTED_BENCHMARKS
    except ImportError as e:
        return [f"benchmarks registry not importable: {e}"]

    for name, cfg in SUPPORTED_BENCHMARKS.items():
        if cfg.get("needs_judge", False) and "paper_judge" not in cfg:
            errors.append(
                f"bench {name!r} has needs_judge=True but no paper_judge declared"
            )
    return errors


def main() -> int:
    # Make src/ importable
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

    checks = [
        ("capability_taxonomy sizes", check_capability_taxonomy_sizes),
        ("metric/dim tags in controlled vocab", check_metric_tags_in_vocab),
        ("metric registry required fields", check_metric_required_fields),
        ("needs_judge benches have paper_judge", check_paper_judge_declared),
    ]
    any_failed = False
    for label, fn in checks:
        errors = fn()
        if errors:
            any_failed = True
            print(f"FAIL: {label}", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
        else:
            print(f"OK:   {label}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
