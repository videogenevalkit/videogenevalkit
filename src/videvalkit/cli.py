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

    # Devices
    dev = rep.get("devices", {})
    click.echo("== Devices ==")
    cuda = dev.get("cuda", {})
    if cuda:
        click.echo(f"  cuda          {_ok(cuda.get('available')):5s}  "
                   f"count={cuda.get('count', 0)}")
    if dev.get("npu", {}).get("available"):
        click.echo(f"  npu           OK")
    if dev.get("mps", {}).get("available"):
        click.echo(f"  mps           OK")
    if "torch" in dev:
        click.echo(f"  torch         MISS  {dev['torch'].get('reason', '')}")

    # Benchmarks summary
    bm = rep.get("benchmarks", {})
    click.echo(f"== Benchmarks ({bm.get('total', 0)}) ==")
    click.echo(f"  judge-free:  {bm.get('judge_free', [])}")
    click.echo(f"  needs judge: {bm.get('needs_judge', [])}")

    # Metrics summary
    met = rep.get("metrics", {})
    if "error" not in met:
        click.echo(f"== Metrics ({met.get('total', 0)}; "
                   f"{met.get('judge_free', 0)} judge-free) ==")
        for kind, n in met.get("by_kind", {}).items():
            click.echo(f"  {kind:30s}  {n}")

    # Profiles
    prof = rep.get("profiles", {})
    if "error" not in prof:
        click.echo("== Eval profiles ==")
        for name, p in prof.items():
            click.echo(f"  {name:10s}  subset={str(p.get('subset')):14s}  "
                       f"frames={p.get('frames')}  ~{p.get('est_wallclock_min')}min")

    # Capability coverage
    cap = rep.get("capability_coverage", {})
    if "error" not in cap:
        click.echo("== Capability coverage ==")
        click.echo(f"  {cap.get('covered_tags', 0)}/{cap.get('total_tags', 0)} "
                   f"tags have >=1 contributor")
        if cap.get("uncovered_top_level"):
            click.echo(f"  uncovered top-level: {cap['uncovered_top_level']}")

    # Plugins
    pl = rep.get("plugins", {})
    if "error" not in pl:
        click.echo("== Plugins ==")
        click.echo(f"  disabled_by_env={pl.get('disabled_by_env')}  "
                   f"user_dir_exists={pl.get('user_dir_exists')}")

    click.echo("== Conda envs ==")
    for b, r in rep["envs"].items():
        click.echo(f"  {b:14s}  {_ok(r['ok'])}  {r.get('reason', '')}")
    click.echo("== Adapter imports ==")
    for b, r in rep["adapters"].items():
        click.echo(f"  {b:14s}  {_ok(r['ok'])}  {r.get('error', '')}")
    click.echo("== Judges ==")
    for name, r in rep["judges"].items():
        click.echo(f"  {name:24s}  reach={_ok(r['reachable']):5s}  "
                   f"key={_ok(r['api_key_present']):5s}  ({r['kind']})")
    if "workspace" in rep:
        click.echo("== Workspace ==")
        for k, d in rep["workspace"]["dirs"].items():
            click.echo(f"  {k:14s}  exists={_ok(d['exists'])}  n={d['n_entries']}")


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
@click.option("--profile", default=None,
              type=click.Choice(["quick", "standard", "full"]),
              help="Eval profile: quick [5-10min], standard [30-60min], "
                   "full [hours, paper-faithful]. Default: full.")
@click.option("--subset", "subset_path", default=None,
              type=click.Path(exists=True, path_type=Path),
              help="Path to a subset JSON file. Overrides the profile's "
                   "default subset.")
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
    profile: str | None,
    subset_path: Path | None,
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
        profile=profile,
        subset_path=subset_path,
        aggregator=aggregator,
    )
    click.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))


# --------------------------------------------------------------------------- #
# estimate — read-only cost estimation [QUICK_EVAL_DESIGN §5.3]
# --------------------------------------------------------------------------- #

@main.command("estimate")
@click.option("--bench", "benches", required=True, multiple=True,
              type=click.Choice(list(SUPPORTED_BENCHMARKS)),
              help="One or more benchmarks (repeat).")
@click.option("--profile", default="quick",
              type=click.Choice(["quick", "standard", "full"]),
              help="Eval profile to estimate cost for. Default: quick.")
@click.option("--judge", default=None,
              help="Judge name [for token-cost estimation]. Pricing table in "
                   "videvalkit.pricing [v0.3 follow-up].")
@click.option("--n-models", default=1, type=int,
              help="Number of models to evaluate. Default: 1.")
def estimate_cmd(
    benches: tuple[str, ...],
    profile: str,
    judge: str | None,
    n_models: int,
) -> None:
    """Estimate wall-clock / GPU-hours / judge-call cost before running.

    Read-only — does NOT execute any benchmark. Aggregates profile.estimated
    fields × n_models × len(benches). See QUICK_EVAL_DESIGN.md §5.3.
    """
    from videvalkit.core.profile import resolve_profile

    spec = resolve_profile(profile)
    click.echo(f"Estimating: {len(benches)} bench(es) × {n_models} model(s) "
               f"× profile={profile!r}")
    click.echo()
    click.echo(f"  {'Benchmark':<16}  {'Judge?':<7}  {'Wallclock':>12}  "
               f"{'GPU-h':>8}  {'Judge calls':>12}")
    click.echo("  " + "-" * 64)

    total_wall = 0.0
    total_gpu = 0.0
    total_calls = 0
    for b in benches:
        bcfg = SUPPORTED_BENCHMARKS[b]
        needs_j = bcfg.get("needs_judge", False)
        per_b_wall = spec.estimated.wallclock_min * n_models
        per_b_gpu = spec.estimated.gpu_hours * n_models
        per_b_calls = spec.estimated.judge_calls * n_models if needs_j else 0
        total_wall += per_b_wall
        total_gpu += per_b_gpu
        total_calls += per_b_calls
        click.echo(f"  {b:<16}  {'VLM' if needs_j else '—':<7}  "
                   f"{per_b_wall:>10.1f} min  {per_b_gpu:>7.2f}  "
                   f"{per_b_calls:>12}")

    click.echo("  " + "-" * 64)
    click.echo(f"  {'TOTAL':<16}  {'':<7}  "
               f"{total_wall:>10.1f} min  {total_gpu:>7.2f}  {total_calls:>12}")
    if judge:
        click.echo()
        click.echo(f"  Judge: {judge}  [token cost estimation in v0.3 follow-up]")
    click.echo()
    click.echo("  Note: these are profile-level estimates. Actual wallclock "
               "varies with GPU, video length, network. See "
               "QUICK_EVAL_DESIGN.md §5.3.")


# --------------------------------------------------------------------------- #
# eval-suite — run multiple benchmarks in one shared workspace
# (per QUICK_EVAL_DESIGN.md §5.2)
# --------------------------------------------------------------------------- #

@main.command("eval-suite")
@click.option("--bench", "benches", multiple=True,
              type=click.Choice(list(SUPPORTED_BENCHMARKS)),
              help="Benchmark to run (repeat). Format also accepts bench=profile.")
@click.option("--all-anchored", is_flag=True,
              help="Run all 6 production-ready anchored benchmarks.")
@click.option("--videos", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--workspace", required=True, type=click.Path(path_type=Path))
@click.option("--models", multiple=True)
@click.option("--profile", default="standard",
              type=click.Choice(["quick", "standard", "full"]))
@click.option("--judge", default=None,
              help="Judge for all benches [paper / default / <name>].")
@click.option("--no-judge", is_flag=True,
              help="Skip benches that need a judge.")
def eval_suite_cmd(
    benches: tuple[str, ...],
    all_anchored: bool,
    videos: Path,
    workspace: Path,
    models: tuple[str, ...],
    profile: str,
    judge: str | None,
    no_judge: bool,
) -> None:
    """Run multiple benchmarks into one workspace, then auto-aggregate."""
    from videvalkit.runner import run

    ANCHORED = ["vbench", "vbench2", "videobench", "worldjen",
                "worldscore", "t2vcompbench"]
    selected = list(benches)
    if all_anchored:
        selected = ANCHORED
    if not selected:
        raise click.UsageError("pass --bench <name> (repeat) or --all-anchored")

    # --no-judge filtering
    skipped: list[str] = []
    runnable: list[str] = []
    for b in selected:
        if no_judge and SUPPORTED_BENCHMARKS[b].get("needs_judge", False):
            skipped.append(b)
        else:
            runnable.append(b)

    click.echo(f"Running: {runnable}")
    if skipped:
        click.echo(f"Skipped [need judge]: {skipped}")

    results: dict[str, object] = {}
    for b in runnable:
        click.echo(f"\n=== {b} ===")
        try:
            results[b] = run(
                benchmark=b,
                videos=videos,
                workspace=workspace,
                models=list(models) or None,
                judge=judge,
                profile=profile,
            )
        except Exception as e:
            click.echo(f"  ERROR: {e}", err=True)
            results[b] = {"error": str(e)}

    click.echo(json.dumps(
        {"benches": runnable, "skipped": skipped,
         "results": {k: (v if isinstance(v, dict) else str(v)) for k, v in results.items()}},
        indent=2, ensure_ascii=False, default=str,
    ))


# --------------------------------------------------------------------------- #
# watch — poll a checkpoint dir, run quick eval on each new model
# (per QUICK_EVAL_DESIGN.md §5.4)
# --------------------------------------------------------------------------- #

@main.command("watch")
@click.option("--videos-pattern", required=True,
              help="Glob for checkpoint sample dirs, "
                   "e.g. '/runs/r42/checkpoints/step_*/samples/'.")
@click.option("--workspace", required=True, type=click.Path(path_type=Path))
@click.option("--bench", "benches", multiple=True,
              type=click.Choice(list(SUPPORTED_BENCHMARKS)), required=True)
@click.option("--profile", default="quick",
              type=click.Choice(["quick", "standard", "full"]))
@click.option("--judge", default=None)
@click.option("--interval", default=60, type=int,
              help="Poll interval in seconds. Default 60.")
@click.option("--once", is_flag=True,
              help="Process current matches once and exit [no polling].")
def watch_cmd(
    videos_pattern: str,
    workspace: Path,
    benches: tuple[str, ...],
    profile: str,
    judge: str | None,
    interval: int,
    once: bool,
) -> None:
    """Watch a checkpoint dir; run quick eval on each new model as it appears.

    Appends each result to <workspace>/timeline.jsonl. Polling-based [no
    inotify in v0.2]. Ctrl-C to stop.
    """
    import glob
    import time

    from videvalkit.training import MonitorConfig, monitor

    cfg = MonitorConfig(
        benches=list(benches), profile=profile, judge=judge,
        workspace=str(workspace),
    )
    seen: set[str] = set()
    click.echo(f"watch: pattern={videos_pattern!r} benches={list(benches)} "
               f"profile={profile} interval={interval}s once={once}")

    def _scan_and_run() -> int:
        n_new = 0
        for match in sorted(glob.glob(videos_pattern)):
            if match in seen:
                continue
            seen.add(match)
            model_name = Path(match).parent.name or Path(match).name
            click.echo(f"\n[watch] new checkpoint: {match} → model={model_name}")
            try:
                result = monitor.eval(match, model_name=model_name, cfg=cfg)
                click.echo(f"  overall={result.overall:.4f}")
                n_new += 1
            except Exception as e:
                click.echo(f"  ERROR: {e}", err=True)
        return n_new

    if once:
        n = _scan_and_run()
        click.echo(f"\nwatch --once: processed {n} checkpoint(s).")
        return

    click.echo("watch: polling [Ctrl-C to stop] ...")
    try:
        while True:
            _scan_and_run()
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nwatch: stopped.")


# --------------------------------------------------------------------------- #
# capabilities — list / show capability tags + their contributors
# (per CAPABILITY_TAGS_DESIGN.md §7)
# --------------------------------------------------------------------------- #

@main.group("capabilities")
def capabilities_group() -> None:
    """Inspect capability tags + the metrics/bench-dims that contribute."""


@capabilities_group.command("list")
@click.option("--show-sub", is_flag=True,
              help="Also list sub-tags under each top-level. Default: top-level only.")
def capabilities_list_cmd(show_sub: bool) -> None:
    """List all capability tags + how many contributors each has."""
    from videvalkit.configs.capability_taxonomy import (
        SUB_TAGS_BY_TOP, TOP_LEVEL_TAGS,
    )
    from videvalkit.core.capability import coverage_report

    coverage = coverage_report()
    click.echo(f"{'capability':<28} {'contributors':>14}")
    click.echo("-" * 46)
    for top in TOP_LEVEL_TAGS:
        n_top = len(coverage.get(top, []))
        # Sum across all subs too for top-level rollup
        n_total = n_top + sum(
            len(coverage.get(s, [])) for s in SUB_TAGS_BY_TOP[top]
        )
        click.echo(f"{top:<28} {n_total:>14}")
        if show_sub:
            for sub in SUB_TAGS_BY_TOP[top]:
                n_sub = len(coverage.get(sub, []))
                click.echo(f"  {sub:<26} {n_sub:>14}")


@capabilities_group.command("show")
@click.argument("capability")
def capabilities_show_cmd(capability: str) -> None:
    """Show the contributors covering a specific capability tag.

    capability can be a top-level [e.g. "motion"] OR a sub-tag [e.g.
    "motion.smoothness"]. Top-level matches expand to all sub-tags.
    """
    from videvalkit.configs.capability_taxonomy import (
        TAG_DESCRIPTIONS, expand_capability, is_valid_tag,
    )
    from videvalkit.core.capability import resolve_capability

    if not is_valid_tag(capability):
        click.echo(f"ERROR: unknown capability {capability!r}", err=True)
        click.echo("Run `videvalkit capabilities list` for the controlled vocab.", err=True)
        raise SystemExit(2)

    desc = TAG_DESCRIPTIONS.get(capability, "")
    click.echo(f"{capability}")
    if desc:
        click.echo(f"  {desc}")
    expanded = expand_capability(capability)
    if len(expanded) > 1:
        click.echo(f"  expands to: {expanded}")
    click.echo()

    contributors = resolve_capability(capability)
    if not contributors:
        click.echo("  [no contributors yet — metric/bench-dim registry doesn't "
                   "tag any entry with this capability]")
        return

    click.echo(f"  {'source_kind':<12}  {'name':<40}  {'tags'}")
    click.echo("  " + "-" * 78)
    for c in contributors:
        tags_str = ", ".join(c.tags)
        click.echo(f"  {c.source_kind:<12}  {c.source_name:<40}  {tags_str}")


@capabilities_group.command("eval")
@click.argument("capability")
@click.option("--videos", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--aggregator", default="mean",
              type=click.Choice(["mean", "max", "min"]),
              help="How to combine per-metric normalized scores. Default: mean.")
@click.option("--device", default="auto",
              type=click.Choice(["auto", "cuda", "cpu", "npu"]))
def capabilities_eval_cmd(
    capability: str, videos: Path, aggregator: str, device: str,
) -> None:
    """Run all runnable metrics for a capability on a videos dir [T5].

    Third entry point alongside `eval --bench` and `metric --name`:
    `videvalkit capabilities eval motion --videos X/` runs every judge-free
    per-video metric tagged under `motion`, min-max normalizes, aggregates.

    Metrics needing refs / prompts / judge, or shells, are skipped with a
    reason [use `metric --name` for those].
    """
    from videvalkit.core.capability_eval import run_capability
    from videvalkit.configs.capability_taxonomy import is_valid_tag

    if not is_valid_tag(capability):
        click.echo(f"ERROR: unknown capability {capability!r}", err=True)
        raise SystemExit(2)

    result = run_capability(capability, videos, aggregator=aggregator, device=device)
    click.echo(f"capability: {result.capability}")
    click.echo(f"score:      {result.score:.4f}  "
               f"[aggregator={result.aggregator}, "
               f"{result.n_contributors_run} metric(s) run, "
               f"{result.n_skipped} skipped]")
    click.echo()
    click.echo(f"  {'metric':<28}  {'raw':>8}  {'norm':>6}  status")
    click.echo("  " + "-" * 60)
    for c in result.contributors:
        if c.status == "ok":
            click.echo(f"  {c.name:<28}  {c.raw_score:>8.4f}  "
                       f"{c.normalized:>6.3f}  ok")
        else:
            click.echo(f"  {c.name:<28}  {'—':>8}  {'—':>6}  "
                       f"skip: {c.skip_reason}")


# --------------------------------------------------------------------------- #
# metric — list / show / run standalone metrics
# (per VIDEO_METRICS_DESIGN.md §7)
# --------------------------------------------------------------------------- #

@main.group("metric")
def metric_group() -> None:
    """List / show / run standalone metrics [FVD, CLIP-Score, lifts, ...]."""


@metric_group.command("list")
@click.option("--kind", default=None, help="Filter by kind.")
@click.option("--no-judge", is_flag=True, help="Only judge-free metrics.")
@click.option("--source", default=None, help="Filter by source prefix.")
def metric_list_cmd(kind: str | None, no_judge: bool, source: str | None) -> None:
    """List registered metrics."""
    from videvalkit.metrics import SUPPORTED_METRICS, list_metrics
    names = list_metrics(kind=kind, no_judge=no_judge, source=source)
    click.echo(f"{'metric':<24}  {'kind':<26}  {'judge?':<6}  source")
    click.echo("-" * 86)
    for n in names:
        c = SUPPORTED_METRICS[n]
        jm = "VLM" if c.get("needs_judge") else "—"
        exp = " *" if c.get("experimental") else ""
        click.echo(f"{n + exp:<24}  {c.get('kind', '?'):<26}  {jm:<6}  "
                   f"{c.get('source', '')}")


@metric_group.command("show")
@click.argument("name")
def metric_show_cmd(name: str) -> None:
    """Show a metric's registry entry."""
    from videvalkit.metrics import metric_info
    try:
        info = metric_info(name)
    except KeyError as e:
        click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(2)
    for k, v in info.items():
        click.echo(f"  {k:24s}  {v}")


@metric_group.command("run")
@click.option("--name", required=True)
@click.option("--gen-videos", type=click.Path(exists=True, path_type=Path),
              help="Generated videos dir [or --videos for non-distribution].")
@click.option("--videos", type=click.Path(exists=True, path_type=Path),
              help="Videos dir for per-video / per-prompt metrics.")
@click.option("--ref-videos", type=click.Path(exists=True, path_type=Path),
              help="Reference videos dir [distribution metrics].")
@click.option("--refs", default=None, help="Named reference set [see refs list].")
@click.option("--prompts", type=click.Path(exists=True, path_type=Path),
              help="prompts.jsonl for per-prompt metrics.")
@click.option("--judge", default=None,
              help="Judge name [see `list judges`] for judge-based metrics.")
@click.option("--device", default="auto",
              type=click.Choice(["auto", "cuda", "cpu", "npu"]))
@click.option("--allow-tiny-sample", is_flag=True,
              help="Bypass the small-N error guard for distribution metrics.")
@click.option("--output", type=click.Path(path_type=Path), default=None)
def metric_run_cmd(
    name: str, gen_videos: Path | None, videos: Path | None,
    ref_videos: Path | None, refs: str | None, prompts: Path | None,
    judge: str | None, device: str, allow_tiny_sample: bool, output: Path | None,
) -> None:
    """Run a single metric and print/save its result."""
    from videvalkit.metrics import SUPPORTED_METRICS, get_metric

    if name not in SUPPORTED_METRICS:
        click.echo(f"ERROR: unknown metric {name!r}", err=True)
        raise SystemExit(2)
    cfg = SUPPORTED_METRICS[name]
    kind = cfg.get("kind", "")
    m = get_metric(name)

    def _list_videos(d: Path) -> list[Path]:
        exts = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
        return sorted(p for p in d.rglob("*") if p.suffix.lower() in exts)

    try:
        if kind == "distribution_reference":
            if gen_videos is None:
                raise click.UsageError("distribution metric needs --gen-videos")
            ref_dir = ref_videos
            if ref_dir is None and refs:
                from videvalkit.refs.registry import resolve_ref_path
                ref_dir = resolve_ref_path(refs)
            if ref_dir is None:
                raise click.UsageError("distribution metric needs --ref-videos or --refs")
            result = m.compute(
                gen_videos=_list_videos(gen_videos),
                ref_videos=_list_videos(Path(ref_dir)),
                device=device, allow_tiny_sample=allow_tiny_sample,
            )
        elif kind == "per_prompt_reference_free":
            vdir = videos or gen_videos
            if vdir is None or prompts is None:
                raise click.UsageError(
                    "per-prompt metric needs --videos and --prompts")
            import json as _json
            vids, texts = [], []
            with open(prompts) as f:
                recs = [_json.loads(l) for l in f if l.strip()]
            # naive pairing by order against sorted video files
            vfiles = _list_videos(Path(vdir))
            for vf, rec in zip(vfiles, recs):
                vids.append(vf); texts.append(rec.get("text", ""))
            result = m.compute(videos=vids, prompts=texts)
        elif kind == "per_video_with_vlm_judge":
            vdir = videos or gen_videos
            if vdir is None:
                raise click.UsageError("judge metric needs --videos")
            if not judge:
                raise click.UsageError(
                    f"metric {name!r} needs a judge; pass --judge <name> "
                    "[see `list judges`]")
            from videvalkit.configs import SUPPORTED_JUDGES
            from videvalkit.scorers.vlm_judge import build_judge
            if judge not in SUPPORTED_JUDGES:
                raise click.UsageError(f"unknown judge {judge!r} [see `list judges`]")
            jobj = build_judge(SUPPORTED_JUDGES[judge])
            result = m.compute(_list_videos(Path(vdir)), judge=jobj)
        else:  # per_video_reference_free
            vdir = videos or gen_videos
            if vdir is None:
                raise click.UsageError("per-video metric needs --videos")
            result = m.compute(_list_videos(Path(vdir)))
    except NotImplementedError as e:
        click.echo(f"NOT YET FUNCTIONAL: {e}", err=True)
        raise SystemExit(3)

    payload = result.model_dump() if hasattr(result, "model_dump") else result
    out_str = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if output:
        output.write_text(out_str)
        click.echo(f"wrote {output}")
    else:
        click.echo(out_str)


# --------------------------------------------------------------------------- #
# refs — reference video set management (per VIDEO_METRICS_DESIGN.md §8)
# --------------------------------------------------------------------------- #

@main.group("refs")
def refs_group() -> None:
    """Manage reference video sets for distribution metrics."""


@refs_group.command("list")
def refs_list_cmd() -> None:
    """List built-in + user-registered reference sets."""
    from videvalkit.refs.registry import get_refs
    refs = get_refs()
    click.echo(f"{'name':<26}  {'clips':>7}  source")
    click.echo("-" * 60)
    for name, c in refs.items():
        n = c.get("n_clips", "?")
        src = c.get("path") or f"hf:{c.get('hf_repo', '')}"
        click.echo(f"{name:<26}  {str(n):>7}  {src}")


@refs_group.command("show")
@click.argument("name")
def refs_show_cmd(name: str) -> None:
    from videvalkit.refs.registry import get_refs
    refs = get_refs()
    if name not in refs:
        click.echo(f"ERROR: unknown ref {name!r}", err=True)
        raise SystemExit(2)
    for k, v in refs[name].items():
        click.echo(f"  {k:16s}  {v}")


@refs_group.command("register")
@click.option("--name", required=True)
@click.option("--path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--description", default="")
def refs_register_cmd(name: str, path: Path, description: str) -> None:
    """Register a local directory as a named reference set."""
    from videvalkit.refs.registry import register_ref
    yaml_path = register_ref(name, path, description)
    click.echo(f"registered ref {name!r} → {path}")
    click.echo(f"  saved to {yaml_path}")


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
# fetch-refs — pull a built-in reference video set from HuggingFace
# --------------------------------------------------------------------------- #

@main.command("fetch-refs")
@click.option("--name", "names", multiple=True,
              help="Built-in ref name(s) to fetch (repeatable). Omit to use --all.")
@click.option("--all", "fetch_all", is_flag=True, help="Fetch every built-in ref set.")
@click.option("--dest", type=click.Path(path_type=Path), default=None,
              help="Override cache dir (default: $VIDEVALKIT_CACHE_HOME/refs).")
@click.option("--dry-run", is_flag=True, help="Report what would download without fetching.")
def fetch_refs_cmd(names: tuple[str, ...], fetch_all: bool,
                   dest: Path | None, dry_run: bool) -> None:
    """Download built-in reference video sets for distribution metrics (FVD/VFID/KVD).

    Each set lands in $VIDEVALKIT_CACHE_HOME/refs/<name>, where `metric run
    --refs <name>` and `refs show` find it. For a local set you already have,
    use `refs register --name <name> --path <dir>` instead.
    """
    from videvalkit.refs.registry import BUILTIN_REFS
    wanted = set(BUILTIN_REFS) if fetch_all else set(names)
    if not wanted:
        raise click.UsageError("specify --name <ref> (repeatable) or --all")
    unknown = wanted - set(BUILTIN_REFS)
    if unknown:
        raise click.UsageError(
            f"unknown built-in ref(s): {sorted(unknown)}; "
            f"choose from {sorted(BUILTIN_REFS)}"
        )
    root = dest or (_default_cache_root() / "refs")
    click.echo(f"dest:  {root}")
    for nm in sorted(wanted):
        cfg = BUILTIN_REFS[nm]
        repo = cfg.get("hf_repo")
        subdir = cfg.get("hf_subdir", nm)
        target = root / nm
        click.echo(f"  {nm}: hf:{repo}/{subdir} → {target}  (~{cfg.get('size_gb', '?')} GB)")
        if dry_run:
            continue
        target.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=[f"{subdir}/*"],
        )
    if not dry_run:
        click.echo("\n  ✓ refs fetched")


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
