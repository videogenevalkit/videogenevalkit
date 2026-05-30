"""Sanity tests for the shipped vbench/quick_v1.json subset.

Guarantees the property training monitoring depends on: the subset is loadable
via the canonical resolver, its prompt_ids match real vbench prompts, and every
vbench dimension has at least K prompts covered (stratification works).
"""

from __future__ import annotations

from collections import Counter

import pytest

from videvalkit.core.subset import find_subset


def test_quick_v1_loads():
    s = find_subset("vbench", "quick_v1")
    assert s.name == "quick_v1"
    assert s.benchmark == "vbench"
    assert s.n_prompts >= 40       # generous lower bound on the built subset
    assert len(s.hash()) == 64     # sha-256 hex


def test_quick_v1_prompt_ids_are_real_vbench_prompts():
    pytest.importorskip("vbench")
    import importlib
    import json
    from pathlib import Path

    s = find_subset("vbench", "quick_v1")
    m = importlib.import_module("vbench")
    full = json.load(open(Path(m.__file__).parent / "VBench_full_info.json"))
    valid = {e.get("prompt_en") or e.get("prompt") for e in full}
    missing = [pid for pid in s.spec.prompt_ids if pid not in valid]
    assert not missing, f"{len(missing)} prompt_ids not in VBench_full_info: {missing[:3]}"


def test_quick_v1_covers_every_dim():
    """Each of the 16 vbench dimensions must have ≥ 3 prompts in the subset —
    the property that makes the subset usable for cross-dimension monitoring."""
    pytest.importorskip("vbench")
    import importlib
    import json
    from pathlib import Path

    s = find_subset("vbench", "quick_v1")
    m = importlib.import_module("vbench")
    full = json.load(open(Path(m.__file__).parent / "VBench_full_info.json"))
    chosen = set(s.spec.prompt_ids)
    cov: Counter = Counter()
    for e in full:
        text = e.get("prompt_en") or e.get("prompt") or ""
        if text in chosen:
            for d in (e.get("dimension") or []):
                cov[d] += 1
    assert len(cov) == 16, f"only {len(cov)} dims covered: {sorted(cov)}"
    weakest = min(cov.values())
    assert weakest >= 3, f"weakest dim has only {weakest} prompts; coverage: {dict(cov)}"
