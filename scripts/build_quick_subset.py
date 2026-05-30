#!/usr/bin/env python
"""Generate ``quick_v1.json`` subsets for training-time monitoring.

A subset is a **fixed, version-pinned** JSON of prompt_ids — same prompts each
run, so different checkpoints are directly comparable. Selection is
deterministic: stratified-seeded sampling by dimension (K prompts per dim,
dedup, fixed seed) — never random per-run.

    python scripts/build_quick_subset.py --bench vbench
    python scripts/build_quick_subset.py --bench vbench --n 50 --seed 42 \\
        --out src/videvalkit/benchmarks/vbench/subsets/quick_v1.json

Always prints a diversity report:
  - per-dimension coverage (count / fraction)
  - prompt length distribution (subset vs full)
  - lexical diversity (unique-word ratio)
  - optional CLIP-text embedding spread (mean pairwise 1−cos), with --clip

The output JSON validates against ``core.subset.SubsetSpec`` and lands where
``find_subset(bench, "quick_v1")`` looks first (the benchmark's builtin
``subsets/`` dir).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_vbench_prompts() -> list[tuple[str, list[str]]]:
    """Return [(prompt_text, [dims])] from VBench_full_info.json."""
    m = importlib.import_module("vbench")
    p = Path(m.__file__).parent / "VBench_full_info.json"
    data = json.load(open(p))
    out: list[tuple[str, list[str]]] = []
    for e in data:
        text = e.get("prompt_en") or e.get("prompt") or ""
        dims = e.get("dimension", []) or []
        if isinstance(dims, str):
            dims = [dims]
        if text:
            out.append((text, list(dims)))
    return out


LOADERS = {"vbench": load_vbench_prompts}


def stratified_sample(
    entries: list[tuple[str, list[str]]], n_target: int, seed: int,
) -> tuple[list[str], dict]:
    """Deterministic stratified sample with guaranteed per-dim coverage.

    Computes ``k_per_dim = ceil(n_target / n_dims)``, then greedily picks until
    every dim has ≥ K covered prompts (where "covered" = the prompt's dim list
    contains this dim). Because prompts often belong to multiple dims, overlap
    accelerates coverage and the final unique count usually ≤ K × n_dims. Any
    drift above n_target is reported, not silently trimmed — trimming would
    destroy the coverage guarantee.
    """
    import random

    by_dim: dict[str, set[str]] = defaultdict(set)
    prompt_dims: dict[str, list[str]] = {}
    for text, dims in entries:
        prompt_dims.setdefault(text, list(dims))
        for d in dims:
            by_dim[d].add(text)
    dims_sorted = sorted(by_dim)
    n_dims = len(dims_sorted)
    k_per_dim = max(1, math.ceil(n_target / n_dims))

    selected: list[str] = []
    seen: set[str] = set()
    dim_cov: Counter = Counter()
    for i, dim in enumerate(dims_sorted):
        if dim_cov[dim] >= k_per_dim:
            continue
        rng = random.Random(seed * 1000 + i)
        pool = sorted(by_dim[dim])           # determinism: sorted before shuffle
        rng.shuffle(pool)
        for t in pool:
            if dim_cov[dim] >= k_per_dim:
                break
            if t in seen:
                continue
            selected.append(t)
            seen.add(t)
            for d in prompt_dims[t]:
                dim_cov[d] += 1

    info = {
        "by": "dimension",
        "k_per_dim": k_per_dim,
        "n_dims": n_dims,
        "min_dim_coverage": min(dim_cov[d] for d in dims_sorted),
    }
    return selected, info


def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", s.lower())


def diversity_report(
    full: list[tuple[str, list[str]]], chosen: list[str], use_clip: bool,
) -> None:
    chosen_set = set(chosen)
    # Per-dim coverage
    full_per_dim: Counter[str] = Counter()
    sub_per_dim: Counter[str] = Counter()
    for t, dims in full:
        for d in dims:
            full_per_dim[d] += 1
            if t in chosen_set:
                sub_per_dim[d] += 1

    print("\n--- per-dimension coverage ---")
    print(f"  {'dim':<25} {'subset':>8} {'full':>6} {'% of dim':>10}")
    for d in sorted(full_per_dim):
        pct = 100 * sub_per_dim[d] / full_per_dim[d]
        print(f"  {d:<25} {sub_per_dim[d]:>8} {full_per_dim[d]:>6} {pct:>9.1f}%")

    # Length distribution
    def stats(xs):
        toks = [len(tokenize(t)) for t in xs]
        return (
            statistics.mean(toks), statistics.median(toks),
            min(toks), max(toks),
        )
    fmean, fmed, fmin, fmax = stats([t for t, _ in full])
    smean, smed, smin, smax = stats(chosen)
    print("\n--- prompt length (tokens) ---")
    print(f"  {'set':<8} {'mean':>6} {'median':>7} {'min':>4} {'max':>4}")
    print(f"  {'full':<8} {fmean:>6.1f} {fmed:>7.1f} {fmin:>4d} {fmax:>4d}")
    print(f"  {'subset':<8} {smean:>6.1f} {smed:>7.1f} {smin:>4d} {smax:>4d}")

    # Lexical diversity
    sub_toks = [w for t in chosen for w in tokenize(t)]
    full_toks = [w for t, _ in full for w in tokenize(t)]
    sub_vocab = set(sub_toks)
    full_vocab = set(full_toks)
    ttr = len(sub_vocab) / max(1, len(sub_toks))
    vocab_cov = len(sub_vocab) / max(1, len(full_vocab))
    print("\n--- lexical diversity ---")
    print(f"  type-token ratio (subset):     {ttr:.3f}")
    print(f"  vocab covered vs full:         {vocab_cov*100:.1f}%  "
          f"({len(sub_vocab)} / {len(full_vocab)})")

    # Embedding spread (optional)
    if use_clip:
        import numpy as np
        import torch
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load("ViT-L/14", device=device)
        model.eval()
        with torch.no_grad():
            tok = clip.tokenize(chosen, truncate=True).to(device)
            feats = model.encode_text(tok).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            sims = (feats @ feats.T).cpu().numpy()
        n = sims.shape[0]
        iu = np.triu_indices(n, k=1)
        mean_cos = float(sims[iu].mean())
        spread = 1.0 - mean_cos
        print("\n--- CLIP-text embedding spread ---")
        print(f"  mean pairwise cosine:          {mean_cos:.3f}")
        print(f"  spread (1 − mean cos):         {spread:.3f}  (higher = more diverse)")


def write_subset(
    out: Path, bench: str, name: str, prompt_ids: list[str],
    seed: int, strat: dict,
) -> None:
    spec = {
        "schema_version": 1,
        "subset_name": name,
        "benchmark": bench,
        "created": dt.date.today().isoformat(),
        "n_prompts": len(prompt_ids),
        "selection_method": "stratified_seeded",
        "selection_seed": seed,
        "calibration": None,
        "stratification": strat,
        "prompt_ids": prompt_ids,
    }
    # Validate against SubsetSpec before writing.
    from videvalkit.core.subset import SubsetSpec
    SubsetSpec.model_validate(spec)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}  ({len(prompt_ids)} prompts)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="vbench", choices=list(LOADERS))
    ap.add_argument("--n", type=int, default=50, help="target unique prompts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset-name", default="quick_v1")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path; default = src/videvalkit/benchmarks/<bench>/subsets/<name>.json")
    ap.add_argument("--clip", action="store_true",
                    help="compute CLIP-text embedding spread (loads CLIP-ViT-L/14)")
    ap.add_argument("--dry-run", action="store_true",
                    help="diversity report only; don't write the JSON")
    args = ap.parse_args()

    entries = LOADERS[args.bench]()
    print(f"loaded {len(entries)} prompts for {args.bench}")

    chosen, strat = stratified_sample(entries, args.n, args.seed)
    print(f"selected {len(chosen)} prompts (k_per_dim={strat['k_per_dim']}, "
          f"n_dims={strat['n_dims']}, seed={args.seed})")

    diversity_report(entries, chosen, use_clip=args.clip)

    if args.dry_run:
        print("\n--dry-run: not writing")
        return

    out = args.out or (
        Path("src/videvalkit/benchmarks") / args.bench / "subsets"
        / f"{args.subset_name}.json"
    )
    write_subset(out, args.bench, args.subset_name, chosen, args.seed, strat)


if __name__ == "__main__":
    main()
