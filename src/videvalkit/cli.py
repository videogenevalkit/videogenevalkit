"""videvalkit CLI — thin shell over `runner.run`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from videvalkit.configs import (
    BENCHMARK_RELATIONS,
    SUPPORTED_AGGREGATORS,
    SUPPORTED_BENCHMARKS,
    SUPPORTED_JUDGES,
)


@click.group()
def main() -> None:
    """Unified evaluation toolkit for generative video benchmarks."""


@main.command("list")
@click.argument("kind", type=click.Choice(["benchmarks", "judges", "aggregators"]))
@click.option("--no-judge", is_flag=True,
              help="Filter to entries that don't need a VLM/LLM judge "
                   "(useful if you don't have a judge endpoint set up). "
                   "See docs/JUDGE_SELECTION_DESIGN.md §5.3.")
def list_cmd(kind: str, no_judge: bool) -> None:
    """List registered benchmarks / judges / aggregators."""
    if kind == "benchmarks":
        _list_benchmarks(no_judge=no_judge)
        return
    if kind == "judges":
        # --no-judge on the judges list is nonsensical; show all
        if no_judge:
            click.echo("note: --no-judge has no effect on `list judges`")
        registry = SUPPORTED_JUDGES
    else:  # aggregators
        registry = SUPPORTED_AGGREGATORS
    for name, cfg in registry.items():
        click.echo(f"{name:30s} {cfg}")


def _list_benchmarks(no_judge: bool = False) -> None:
    """Pretty-print the benchmark registry with dim counts and superset hints."""
    # Column header — add judge? column for transparency
    click.echo(f"{'name':<16}  {'#dims':>5}  {'gpu':<4}  "
               f"{'judge?':<7}  {'default_judge':<25}  notes")
    click.echo("-" * 88)
    skipped = 0
    for name in SUPPORTED_BENCHMARKS:
        cfg = SUPPORTED_BENCHMARKS[name]
        needs_judge = cfg.get("needs_judge", False)
        if no_judge and needs_judge:
            skipped += 1
            continue
        cls = cfg["cls"]
        # vbench2 lazy-loads dimensions; pull canonical list as fallback.
        n_dims = len(cls.dimensions)
        if n_dims == 0 and name == "vbench2":
            try:
                from videvalkit.benchmarks.vbench2.benchmark import _VBENCH2_CANONICAL_DIMS
                n_dims = len(_VBENCH2_CANONICAL_DIMS)
            except Exception:
                pass
        gpu = "yes" if cfg.get("needs_gpu") else "no"
        judge_mark = "VLM" if needs_judge else "—"
        judge = cfg.get("default_judge", "—") if needs_judge else "—"
        rel = BENCHMARK_RELATIONS.get(name, {})
        notes: list[str] = []
        if "superset_of" in rel:
            notes.append(f"⊃ {rel['superset_of']}")
        click.echo(f"{name:<16}  {n_dims:>5}  {gpu:<4}  {judge_mark:<7}  "
                   f"{judge:<25}  {', '.join(notes)}")
    if no_judge and skipped:
        click.echo(f"\n  [{skipped} bench filtered out: needs VLM/LLM judge. "
                   f"Drop --no-judge to see them.]")


@main.command("doctor")
@click.option("--workspace", type=click.Path(path_type=Path), default=None,
              help="Optional: also probe this workspace's directory health.")
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON instead of human-readable.")
def doctor_cmd(workspace: Path | None, as_json: bool) -> None:
    """Diagnose conda envs, judge endpoints, API keys, and (optional) workspace."""
    from videvalkit.diagnostics import run_all

    rep = run_all(workspace=workspace)
    if as_json:
        click.echo(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
        return

    def _ok(b):
        return "OK" if b is True else ("--" if b is None else "MISS")

    click.echo("== Conda envs ==")
    for b, r in rep["envs"].items():
        click.echo(f"  {b:12s}  {_ok(r['ok'])}  {r.get('reason', '')}")
    click.echo("== Adapter imports ==")
    for b, r in rep["adapters"].items():
        click.echo(f"  {b:12s}  {_ok(r['ok'])}  {r.get('error', '')}")
    click.echo("== Judges ==")
    for name, r in rep["judges"].items():
        click.echo(f"  {name:24s}  reach={_ok(r['reachable']):5s}  "
                   f"key={_ok(r['api_key_present']):5s}  ({r['kind']})")
    if "workspace" in rep:
        click.echo("== Workspace ==")
        for k, d in rep["workspace"]["dirs"].items():
            click.echo(f"  {k:12s}  exists={_ok(d['exists'])}  n={d['n_entries']}")


@main.command("aggregate")
@click.option("--workspace", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", type=click.Path(path_type=Path), default=None,
              help="Where to write the cross-benchmark report JSON.")
def aggregate_cmd(workspace: Path, output: Path | None) -> None:
    """Combine all summary/{benchmark}/{model}.json files into one report."""
    from videvalkit.aggregators import combine_summaries
    from videvalkit.core.types import Summary

    summaries: list[Summary] = []
    summary_root = workspace / "results" / "summary"
    if not summary_root.exists():
        click.echo(f"No summaries under {summary_root}", err=True)
        raise SystemExit(1)
    for jf in summary_root.glob("*/*.json"):
        try:
            summaries.append(Summary.model_validate_json(jf.read_text()))
        except Exception as e:
            click.echo(f"WARN: could not parse {jf}: {e}", err=True)
    report = combine_summaries(summaries)
    if output is None:
        output = workspace / "results" / "leaderboard" / "cross_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    click.echo(f"Wrote {output}")
    for r in report["ranking"]:
        click.echo(f"  #{r['rank']:>2}  {r['model']:30s}  z={r['score']:+.3f}")


@main.command("prepare-workspace")
@click.option("--workspace", required=True, type=click.Path(path_type=Path))
@click.option("--videos", required=True, type=click.Path(exists=True, path_type=Path),
              help="External videos folder to symlink into the workspace.")
def prepare_workspace_cmd(workspace: Path, videos: Path) -> None:
    """Bootstrap a workspace and symlink an external videos folder in."""
    from videvalkit.storage import Workspace

    ws = Workspace(workspace)
    target = ws.layout.videos_dir
    # If the external dir has model subdirs, symlink them in one by one.
    children = [p for p in videos.iterdir() if p.is_dir()]
    if children:
        for child in children:
            link = target / child.name
            if not link.exists():
                link.symlink_to(child.resolve())
                click.echo(f"  linked {child.name}")
    else:
        # Treat the whole dir as one model
        link = target / videos.name
        if not link.exists():
            link.symlink_to(videos.resolve())
            click.echo(f"  linked {videos.name}")
    click.echo(f"Workspace ready: {ws.layout.root}")


@main.command("eval")
@click.option("--bench", "benchmark", required=True, type=click.Choice(list(SUPPORTED_BENCHMARKS)))
@click.option("--videos", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--workspace", required=True, type=click.Path(path_type=Path))
@click.option("--models", multiple=True, help="Model names to evaluate (repeat).")
@click.option("--dimensions", multiple=True, help="Subset of dimensions to run (repeat). Default: all.")
@click.option("--judge", default=None,
              help=("Judge name from SUPPORTED_JUDGES (incl. user yaml), "
                    "or semantic keyword: \"paper\" / \"default\". "
                    "See docs/JUDGE_SELECTION_DESIGN.md §3."))
@click.option("--judge-endpoint", default=None,
              help="Ad-hoc judge endpoint base URL (e.g. http://10.0.0.5:8003/v1). "
                   "Bypasses the registry; pair with --judge-model / --judge-kind. "
                   "Mutually exclusive with --judge.")
@click.option("--judge-model", default=None,
              help="Ad-hoc judge model id (required with --judge-endpoint).")
@click.option("--judge-kind", default=None,
              type=click.Choice(["openai_compatible", "gemini", "anthropic"]),
              help="Ad-hoc judge backend (default: openai_compatible).")
@click.option("--judge-api-key-env", default=None,
              help="Env var holding the bearer token for ad-hoc judge (None = local).")
@click.option("--no-judge", is_flag=True,
              help="Refuse to run any benchmark that needs a VLM/LLM judge. "
                   "Useful for offline / no-API-key workflows.")
@click.option("--aggregator", default=None, type=click.Choice(list(SUPPORTED_AGGREGATORS)))
def eval_cmd(
    benchmark: str,
    videos: Path,
    workspace: Path,
    models: tuple[str, ...],
    dimensions: tuple[str, ...],
    judge: str | None,
    judge_endpoint: str | None,
    judge_model: str | None,
    judge_kind: str | None,
    judge_api_key_env: str | None,
    no_judge: bool,
    aggregator: str | None,
) -> None:
    """Run a single benchmark on a video folder."""
    from videvalkit.runner import run

    # --no-judge: fail-fast if bench needs judge
    bench_cfg = SUPPORTED_BENCHMARKS[benchmark]
    if no_judge:
        if bench_cfg.get("needs_judge", False):
            judge_free = [
                n for n, c in SUPPORTED_BENCHMARKS.items()
                if not c.get("needs_judge", False)
            ]
            raise click.UsageError(
                f"--no-judge: benchmark {benchmark!r} requires a VLM/LLM judge "
                f"and cannot run in judge-free mode.\n"
                f"  Judge-free benches: {judge_free}\n"
                f"  Or drop --no-judge and pass --judge <name>."
            )
        if judge is not None or any(
            f is not None for f in (judge_endpoint, judge_model, judge_kind, judge_api_key_env)
        ):
            raise click.UsageError(
                "--no-judge is mutually exclusive with --judge / --judge-endpoint."
            )

    # Build judge_override from ad-hoc flags, if any are set
    adhoc_flags = (judge_endpoint, judge_model, judge_kind, judge_api_key_env)
    judge_override = None
    if any(f is not None for f in adhoc_flags):
        if judge is not None:
            raise click.UsageError(
                "--judge and --judge-endpoint/--judge-model/--judge-kind are "
                "mutually exclusive. Use --judge <name> OR ad-hoc flags, not both."
            )
        if judge_endpoint is None or judge_model is None:
            raise click.UsageError(
                "--judge-endpoint and --judge-model must both be supplied "
                "for an ad-hoc judge."
            )
        judge_override = {
            "kind": judge_kind or "openai_compatible",
            "endpoint": judge_endpoint,
            "model": judge_model,
            "provider": "adhoc",
            "api_key_env": judge_api_key_env,
        }

    result = run(
        benchmark=benchmark,
        videos=videos,
        workspace=workspace,
        models=list(models) or None,
        dimensions=list(dimensions) or None,
        judge=judge,
        judge_override=judge_override,
        aggregator=aggregator,
    )
    click.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))


# --------------------------------------------------------------------------- #
# fetch-smoke-data — pull videogenevalkit/smoke-data from HuggingFace
# --------------------------------------------------------------------------- #

_SMOKE_REPO = "videogenevalkit/smoke-data"
_CKPT_REPO = "videogenevalkit/checkpoints"


def _default_cache_root() -> Path:
    import os
    return Path(os.environ.get("VIDEVALKIT_CACHE_HOME",
                               Path.home() / ".cache" / "videvalkit"))


@main.command("fetch-smoke-data")
@click.option("--bench", multiple=True,
              type=click.Choice(["vbench", "vbench2", "videobench",
                                 "worldjen", "worldscore", "t2vcompbench",
                                 "semantics_axis", "all"]),
              default=("all",),
              help="One or more benchmarks to fetch. 'all' = everything (~3 GB).")
@click.option("--dest", type=click.Path(path_type=Path), default=None,
              help="Override download directory (default: $VIDEVALKIT_CACHE_HOME/smoke-data).")
@click.option("--dry-run", is_flag=True, help="Report what would download without actually fetching.")
def fetch_smoke_data_cmd(bench: tuple[str, ...], dest: Path | None, dry_run: bool) -> None:
    """Pull smoke-test video samples + prompts from videogenevalkit/smoke-data on HF."""
    benches = set(bench)
    if "all" in benches:
        benches = {"vbench", "vbench2", "videobench", "worldjen", "worldscore",
                   "t2vcompbench", "semantics_axis"}
    dest = dest or (_default_cache_root() / "smoke-data")
    allow = []
    for b in sorted(benches):
        allow.append(f"{b}/*")
    click.echo(f"repo:  {_SMOKE_REPO}")
    click.echo(f"dest:  {dest}")
    click.echo(f"glob:  {allow}")
    if dry_run:
        return
    dest.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=_SMOKE_REPO,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=allow,
    )
    click.echo(f"\n  ✓ smoke-data fetched → {dest}")


# --------------------------------------------------------------------------- #
# fetch-checkpoints — pull videogenevalkit/checkpoints from HuggingFace
# --------------------------------------------------------------------------- #

@main.command("fetch-checkpoints")
@click.option("--bench", multiple=True,
              type=click.Choice(["vbench", "vbench2", "videobench",
                                 "worldjen", "worldscore", "t2vcompbench",
                                 "hf-models", "all"]),
              default=("all",),
              help="One or more benchmark/group keys. 'all' = ~125 GB.")
@click.option("--skip-mllm-upstream", is_flag=True,
              help="Skip the 68 GB LLaVA-1.6-34B (used by T2V-CompBench paper-mode only).")
@click.option("--dest", type=click.Path(path_type=Path), default=None,
              help="Override download directory (default: $VIDEVALKIT_CACHE_HOME/checkpoints).")
@click.option("--dry-run", is_flag=True, help="Report what would download without actually fetching.")
def fetch_checkpoints_cmd(bench: tuple[str, ...], skip_mllm_upstream: bool,
                          dest: Path | None, dry_run: bool) -> None:
    """Pull model weights + prompt registries from videogenevalkit/checkpoints on HF."""
    benches = set(bench)
    if "all" in benches:
        benches = {"vbench", "vbench2", "videobench", "worldjen",
                   "worldscore", "t2vcompbench", "hf-models"}
    dest = dest or (_default_cache_root() / "checkpoints")
    allow: list[str] = []
    for b in sorted(benches):
        allow.append(f"{b}/*")
    ignore: list[str] = []
    if skip_mllm_upstream:
        ignore.append("hf-models/liuhaotian/llava-v1.6-34b/*")
    click.echo(f"repo:    {_CKPT_REPO}")
    click.echo(f"dest:    {dest}")
    click.echo(f"include: {allow}")
    if ignore:
        click.echo(f"exclude: {ignore}")
    if dry_run:
        return
    dest.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=_CKPT_REPO,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=allow,
        ignore_patterns=ignore or None,
    )
    click.echo(f"\n  ✓ checkpoints fetched → {dest}")


# --------------------------------------------------------------------------- #
# fetch-upstream — git-clone the 6 upstream benchmark repos
# --------------------------------------------------------------------------- #

# Each value is (repo_url, branch_or_tag, target_subdir_under_~/.cache/videvalkit/upstream/)
_UPSTREAM_SPECS = {
    "vbench":       ("https://github.com/Vchitect/VBench",                                 "main", "VBench"),
    "vbench2":      ("https://github.com/Vchitect/VBench",                                 "main", "VBench-2.0"),  # VBench-2.0 ships as a subfolder of VBench
    "videobench":   ("https://github.com/Video-Bench/Video-Bench",                         "main", "Video-Bench"),
    "worldjen":     ("https://github.com/moonmath-ai/WorldJen-benchmarking-subsystem",     "main", "WorldJen-benchmarking-subsystem"),
    "worldscore":   ("https://github.com/yhw-yhw/WorldScore",                              "main", "WorldScore"),
    "t2vcompbench": ("https://github.com/KaiyueSun98/T2V-CompBench",                       "V2",   "T2V-CompBench"),
}


@main.command("fetch-upstream")
@click.option("--bench", multiple=True,
              type=click.Choice([*_UPSTREAM_SPECS.keys(), "all"]),
              default=("all",),
              help="One or more upstream repos to clone. 'all' = clone everything (~500 MB total).")
@click.option("--dest", type=click.Path(path_type=Path), default=None,
              help="Override target directory (default: $VIDEVALKIT_CACHE_HOME/upstream).")
@click.option("--depth", type=int, default=1,
              help="Clone depth. Default 1 (shallow) for fastest fetch; pass 0 for full history.")
@click.option("--force", is_flag=True, help="Re-clone even if target dir exists.")
def fetch_upstream_cmd(bench: tuple[str, ...], dest: Path | None, depth: int, force: bool) -> None:
    """Clone the upstream benchmark repos into ~/.cache/videvalkit/upstream/<name>/."""
    import shutil
    import subprocess as _sp

    benches = set(bench)
    if "all" in benches:
        benches = set(_UPSTREAM_SPECS.keys())
    # vbench and vbench2 both point at the same upstream repo; consolidate
    if "vbench2" in benches and "vbench" in benches:
        # vbench2 subfolder is included when we clone the parent VBench repo
        pass
    base = dest or (_default_cache_root() / "upstream")
    base.mkdir(parents=True, exist_ok=True)

    # Dedup: don't clone VBench twice for {vbench, vbench2}
    cloned_repos: set[str] = set()
    for b in sorted(benches):
        url, branch, subdir = _UPSTREAM_SPECS[b]
        if url in cloned_repos:
            click.echo(f"  ✓ {b}: shares clone with sibling; skip")
            continue
        target = base / subdir.split("/")[0]  # parent dir for VBench-2.0 case
        if target.exists() and not force:
            click.echo(f"  ✓ {b}: already at {target} (use --force to re-clone)")
            cloned_repos.add(url)
            continue
        if target.exists() and force:
            shutil.rmtree(target)
        click.echo(f"  cloning {b}: {url} (branch={branch}) -> {target}")
        cmd = ["git", "clone", "--branch", branch]
        if depth > 0:
            cmd += ["--depth", str(depth)]
        cmd += [url, str(target)]
        rc = _sp.call(cmd)
        if rc != 0:
            click.echo(f"  ✗ {b}: git clone failed (exit {rc})", err=True)
            continue
        cloned_repos.add(url)
        click.echo(f"  ✓ {b}: cloned")

    click.echo(f"\n  upstream root: {base}")
    click.echo(f"  set env to override location:")
    click.echo(f"    export VIDEVALKIT_CACHE_HOME={base.parent}")


if __name__ == "__main__":
    main()
