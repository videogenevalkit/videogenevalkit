"""Semantics Axis Eval adapter — 21 VLM-judged prompt-following axes.

Ported from the in-house ``video_eval`` Semantics Axis Eval. Each axis is a
structured 5-step-CoT system prompt (bundled verbatim under ``prompts/``) that
scores a single video 1-5 on one narrow prompt-following axis, plus a holistic
``overall``. JSON-only output, scored through the toolkit's VLM judge layer —
no upstream backend code is used; any ``SUPPORTED_JUDGES`` endpoint works.

The 21 axes are grouped: entity (9) · spatial (1) · event (7) · cinematic (2)
· modifier (1), plus the top-level ``overall``.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from videvalkit.core.benchmark import BaseBenchmark
from videvalkit.core.layout import WorkspaceLayout
from videvalkit.core.types import PromptItem, RawResult, Summary, VideoSpec
from videvalkit.benchmarks.semantics_axis.parsers import parse_axis_response

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# axis group -> ordered axis names; the .txt lives at prompts/<group>/<axis>.txt
SEMANTICS_AXIS_GROUPS: dict[str, list[str]] = {
    "entity": ["object_class", "multiple_objects", "color", "material", "scene",
               "style", "pose", "emotion", "text_ocr"],
    "spatial": ["spatial_relationship"],
    "event": ["action", "motion_order", "dynamic_attribute",
              "dynamic_spatial_relationship", "human_interaction",
              "complex_plot", "complex_landscape"],
    "cinematic": ["camera_motion", "shot_composition"],
    "modifier": ["temporal_modifier"],
}
# overall is top-level (prompts/overall.txt)
SEMANTICS_AXIS_DIMENSIONS: list[str] = (
    [a for axes in SEMANTICS_AXIS_GROUPS.values() for a in axes] + ["overall"]
)
_AXIS_TO_GROUP: dict[str, str] = {
    a: g for g, axes in SEMANTICS_AXIS_GROUPS.items() for a in axes
}


def _prompt_path(axis: str) -> Path:
    if axis == "overall":
        return _PROMPTS_DIR / "overall.txt"
    group = _AXIS_TO_GROUP.get(axis)
    if group is None:
        raise KeyError(f"unknown semantics_axis dimension: {axis}")
    return _PROMPTS_DIR / group / f"{axis}.txt"


def _render_axis_prompt(axis: str, user_question: str) -> str:
    """Substitute {user_question}; the .txt files double-escape literal braces."""
    tpl = _prompt_path(axis).read_text(encoding="utf-8")
    try:
        return tpl.format(user_question=user_question)
    except (KeyError, ValueError, IndexError):
        # Fallback if a stray single brace slipped past the {{ }} escaping.
        return (tpl.replace("{user_question}", user_question)
                   .replace("{{", "{").replace("}}", "}"))


class SemanticsAxisBenchmark(BaseBenchmark):
    """21-axis VLM-judge prompt-following evaluation (1-5 per axis)."""

    name = "semantics_axis"
    env_name = "videvalkit"
    dimensions = SEMANTICS_AXIS_DIMENSIONS
    video_layout = "{model}/{prompt_id}.mp4"
    SCALE_MIN = 1
    SCALE_MAX = 5

    # ---- prompts ------------------------------------------------------------
    def list_prompts(
        self,
        dimensions: list[str] | None = None,
        prompts_file: str | Path | None = None,
    ) -> Iterator[PromptItem]:
        """Yield ``PromptItem`` rows from a JSONL prompts file.

        Each row: ``{prompt_id, text|prompt, dimensions:[axes]}``. The
        ``dimensions`` list is the curated set of axes in scope for that
        prompt; omit it to evaluate every axis.
        """
        if not prompts_file:
            raise ValueError(
                "SemanticsAxis.list_prompts requires prompts_file=<path/to.jsonl>"
            )
        wanted = set(dimensions) if dimensions else set(self.dimensions)
        with Path(prompts_file).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                dims = [d for d in (e.get("dimensions") or self.dimensions)
                        if d in wanted]
                if not dims:
                    continue
                yield PromptItem(
                    prompt_id=str(e["prompt_id"]),
                    text=e.get("text") or e.get("prompt", ""),
                    dimensions=dims, meta=e,
                )

    def list_required_videos(
        self,
        prompts: list[PromptItem],
        models: list[str],
        layout: WorkspaceLayout,
        samples_per_prompt: int = 1,
    ) -> list[VideoSpec]:
        """One shared video per (model, prompt), scored on every in-scope axis."""
        specs: list[VideoSpec] = []
        for m in models:
            for p in prompts:
                for dim in p.dimensions:
                    specs.append(VideoSpec(
                        path=layout.videos_dir / f"{m}/{p.prompt_id}.mp4",
                        prompt_id=p.prompt_id, model_name=m,
                        dimension=dim, sample_index=0,
                    ))
        return specs

    # ---- evaluate -----------------------------------------------------------
    def evaluate(
        self,
        videos: list[VideoSpec] | None = None,
        layout: WorkspaceLayout | None = None,
        dimensions: list[str] | None = None,
        judge: dict[str, Any] | None = None,
        models: list[str] | None = None,
        prompts_file: str | Path | None = None,
        max_concurrency: int = 4,
        n_frames: int = 8,
        **kwargs: Any,
    ) -> list[RawResult]:
        if layout is None:
            raise ValueError("SemanticsAxis.evaluate requires layout=WorkspaceLayout")
        if judge is None:
            raise ValueError("SemanticsAxis.evaluate requires a judge")
        prompts = list(self.list_prompts(dimensions=dimensions, prompts_file=prompts_file))
        prompt_by_id = {p.prompt_id: p for p in prompts}

        # Build the (video, axis) work list. Each video is scored once per
        # in-scope axis. When the runner passes one VideoSpec per file
        # (dimension=None — videos are not laid out per-axis), expand it.
        specs: list[VideoSpec] = []
        if videos:
            for v in videos:
                p = prompt_by_id.get(v.prompt_id)
                if p is None:
                    continue
                axes = ([v.dimension] if v.dimension in p.dimensions
                        else list(p.dimensions))
                for axis in axes:
                    specs.append(VideoSpec(
                        path=v.path, prompt_id=v.prompt_id,
                        model_name=v.model_name, dimension=axis, sample_index=0,
                    ))
        else:
            if not models:
                raise ValueError("SemanticsAxis.evaluate: need videos or models list")
            specs = self.list_required_videos(prompts, models, layout)

        from videvalkit.storage.workspace import Workspace
        ws = Workspace(layout.root)
        out: list[RawResult] = []
        asyncio.run(self._score_async(
            specs, prompt_by_id, judge, ws, out,
            max_concurrency=max_concurrency, n_frames=n_frames,
        ))
        return out

    async def _score_async(
        self,
        videos: list[VideoSpec],
        prompt_by_id: dict[str, PromptItem],
        judge_cfg: dict[str, Any],
        ws: Any,
        out: list[RawResult],
        max_concurrency: int = 4,
        n_frames: int = 8,
    ) -> None:
        from videvalkit.scheduler.http_pool import HTTPDispatcher
        from videvalkit.scorers.vlm_judge import build_judge

        judge = build_judge(judge_cfg, layout=ws.layout)
        judge.setup()
        dispatcher = HTTPDispatcher(concurrency=max_concurrency)
        if hasattr(judge, "set_dispatcher"):
            judge.set_dispatcher(dispatcher)
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(v: VideoSpec) -> None:
            async with sem:
                dim = v.dimension or ""
                if ws.has_raw(self.name, v.model_name, dim, v.prompt_id):
                    out.append(ws.load_raw(self.name, v.model_name, dim, v.prompt_id))
                    return
                if not v.path.exists():
                    log.warning("semantics_axis: missing video %s", v.path)
                    return
                p = prompt_by_id.get(v.prompt_id)
                if p is None:
                    return
                try:
                    system = _render_axis_prompt(dim, p.text)
                except Exception as e:
                    log.error("semantics_axis: prompt render failed [%s]: %s", dim, e)
                    return
                user = "这是视频的若干采样帧。请严格按系统提示完成评估，只输出 JSON。"
                try:
                    resp = await judge.achat_with_frames(
                        v.path, user, mode="holistic", n_frames=n_frames,
                        max_tokens=2048, temperature=0.0, system=system,
                    )
                except Exception as e:
                    log.error("semantics_axis: [%s][%s/%s] %s",
                              v.model_name, v.prompt_id, dim, e)
                    return
                content = resp.get("content", "") or ""
                score, meta = parse_axis_response(content)
                if score is None:
                    log.warning("semantics_axis: [%s][%s/%s] unparseable response",
                                v.model_name, v.prompt_id, dim)
                    return
                meta["axis_group"] = _AXIS_TO_GROUP.get(dim, "overall")
                r = RawResult(
                    benchmark=self.name, model=v.model_name,
                    dimension=dim, prompt_id=v.prompt_id,
                    score=float(score), judge=getattr(judge, "model", None),
                    scorer="semantics_axis_vlm", video_path=str(v.path), meta=meta,
                )
                ws.write_raw(r)
                out.append(r)

        try:
            await asyncio.gather(*(_one(v) for v in videos))
        finally:
            await dispatcher.aclose()

    # ---- aggregate ----------------------------------------------------------
    def aggregate(
        self, raw: list[RawResult], aggregator: str = "weighted_sum", **kwargs: Any,
    ) -> Summary:
        """Mean per axis, then mean across axes (1-5 scale).

        ``meta.per_group`` additionally reports the mean of each axis group
        (entity / spatial / event / cinematic / modifier) and ``overall``.
        """
        from videvalkit.aggregators.weighted_sum import WeightedSumAggregator
        summary = WeightedSumAggregator().aggregate(raw)
        per_dim = summary.per_dimension
        groups: dict[str, list[float]] = {}
        for dim, val in per_dim.items():
            g = _AXIS_TO_GROUP.get(dim, "overall")
            groups.setdefault(g, []).append(val)
        summary.meta["per_group"] = {
            g: round(sum(v) / len(v), 4) for g, v in groups.items() if v
        }
        summary.aggregator = "semantics_axis"
        return summary
