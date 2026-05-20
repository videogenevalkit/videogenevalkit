"""WorldJen adapter — port of `video_eval/worldjen_local`.

End-to-end pipeline lives entirely inside this adapter (no upstream package
to call into):

  Phase A: text-LLM generates a per-dim VQA question set per prompt.
           Output: `workspace/prompts/worldjen/vqa_questions_{prompt_set}.jsonl`
  Phase B: VLM answers each question per dim per (model, prompt_id) video.
           Output: `workspace/results/raw/worldjen/{model}/{dim}/{prompt_id}.json`

Aggregation: PHASAggregator combines the 16 per-dim mean scores into a
single per-(model, prompt) PHAS, with variance penalty.

The adapter assumes there is a `judge_llm` (text-only, e.g. Qwen3-32B) and
a `judge_vlm` (multimodal, e.g. Gemma-4-31B-IT) configured. Both default
to local vLLM endpoints (see configs/judges.py).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from videvalkit.core.benchmark import BaseBenchmark
from videvalkit.core.layout import WorkspaceLayout
from videvalkit.core.types import PromptItem, RawResult, Summary, VideoSpec
from videvalkit.benchmarks.worldjen.dimensions import (
    WORLDJEN_CATEGORIES,
    WORLDJEN_DEFINITIONS,
    WORLDJEN_DIMENSION_MODES,
    WORLDJEN_DIMENSIONS,
    WORLDJEN_FRAMES_PER_MODE,
)

log = logging.getLogger(__name__)

_BENCH_DIR = Path(__file__).resolve().parent
_PROMPTS_DIR = _BENCH_DIR / "prompts"


class WorldJenBenchmark(BaseBenchmark):
    name = "worldjen"
    env_name = "videvalkit-worldjen"
    dimensions = list(WORLDJEN_DIMENSIONS)
    categories = WORLDJEN_CATEGORIES
    video_layout = "{model}/{prompt_id}.mp4"

    # Concurrency knobs (mirrors config_local.py)
    BATCH_SIZE: int = 2
    MAX_RETRIES: int = 5
    REQUEST_TIMEOUT_S: float = 240.0

    # ---- prompts ------------------------------------------------------------
    def list_prompts(
        self,
        dimensions: list[str] | None = None,
        prompts_file: str | Path | None = None,
    ) -> Iterator[PromptItem]:
        """Yield PromptItem for each entry in the prompts jsonl.

        Layout of an entry::

            {"prompt_id": "000", "prompt": "...", "enhanced_prompt": "...",
             "prompt_set": "all110"}
        """
        if prompts_file is None:
            raise ValueError("WorldJen: list_prompts(prompts_file=...) is required")
        wanted = set(dimensions) if dimensions else set(self.dimensions)
        with Path(prompts_file).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                _text = e.get("enhanced_prompt") or e.get("prompt") or ""
                yield PromptItem(
                    prompt_id=e["prompt_id"],
                    text=_text,
                    dimensions=list(wanted & set(self.dimensions)) or list(self.dimensions),
                    meta={
                        "prompt_set": e.get("prompt_set", "unknown"),
                        "original_prompt": e.get("prompt") or _text,
                    },
                )

    def list_required_videos(
        self,
        prompts: list[PromptItem],
        models: list[str],
        layout: WorkspaceLayout,
        samples_per_prompt: int = 1,
    ) -> list[VideoSpec]:
        """Build the expected (model, prompt) video paths under videos_dir."""
        specs: list[VideoSpec] = []
        for model in models:
            for prompt in prompts:
                for idx in range(samples_per_prompt):
                    pid = prompt.prompt_id
                    rel = self.video_layout.format(
                        model=model, prompt_id=pid, sample_index=idx
                    )
                    specs.append(
                        VideoSpec(
                            path=layout.videos_dir / rel,
                            prompt_id=pid,
                            model_name=model,
                            dimension=None,
                            sample_index=idx,
                        )
                    )
        return specs

    # ---- evaluate (two phases) ---------------------------------------------
    def evaluate(
        self,
        videos: list[VideoSpec] | None = None,
        layout: WorkspaceLayout | None = None,
        dimensions: list[str] | None = None,
        judge: dict[str, Any] | None = None,
        judge_llm: dict[str, Any] | None = None,
        prompts_file: str | Path | None = None,
        vqa_file: str | Path | None = None,
        models: list[str] | None = None,
        max_concurrency: int = 4,
        **kwargs: Any,
    ) -> list[RawResult]:
        """Run Phase A (if vqa_file missing) then Phase B."""
        if layout is None:
            raise ValueError("WorldJen.evaluate requires layout=WorkspaceLayout")
        if prompts_file is None:
            raise ValueError("WorldJen.evaluate requires prompts_file=...")
        if judge is None:
            raise ValueError("WorldJen.evaluate requires judge=<judge cfg dict> (VLM)")
        if judge_llm is None:
            log.warning("WorldJen.evaluate: judge_llm not given; defaulting to judge for VQA gen")
            judge_llm = judge

        prompts = list(self.list_prompts(dimensions=dimensions, prompts_file=prompts_file))
        dims = list(dimensions) if dimensions else list(self.dimensions)

        # Phase A: ensure VQA file
        vqa_path = Path(vqa_file) if vqa_file else (
            layout.prompts_dir / "worldjen" / "vqa_questions.jsonl"
        )
        if not vqa_path.exists():
            log.info("Phase A: generating VQA questions -> %s", vqa_path)
            self._phase_a_generate_vqa(prompts, dims, judge_llm, vqa_path, layout=layout)

        questions_map = self._load_vqa(vqa_path)

        # Phase B: score every (model, prompt) video on every dimension
        if not videos:
            if not models:
                raise ValueError("WorldJen.evaluate: need either videos=[...] or models=[...]")
            videos = self.list_required_videos(prompts, models, layout)

        log.info("Phase B: scoring %d videos × %d dims via VLM=%s",
                 len(videos), len(dims), judge.get("model"))
        raw: list[RawResult] = []
        asyncio.run(self._phase_b_score(videos, questions_map, dims, judge, layout, raw,
                                        max_concurrency=max_concurrency))
        return raw

    # ---- aggregate ----------------------------------------------------------
    def aggregate(
        self,
        raw: list[RawResult],
        aggregator: str = "phas",
        weights: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> Summary:
        from videvalkit.aggregators.phas import PHASAggregator
        from videvalkit.aggregators.weighted_sum import WeightedSumAggregator

        if aggregator == "phas":
            return PHASAggregator(weights=weights).aggregate(raw)
        return WeightedSumAggregator(weights=weights or {}).aggregate(raw)

    # ===== Phase A: VQA generation ==========================================
    def _phase_a_generate_vqa(
        self,
        prompts: list[PromptItem],
        dimensions: list[str],
        judge_llm_cfg: dict[str, Any],
        out_path: Path,
        layout: WorkspaceLayout | None = None,
    ) -> None:
        sys_prompt = (_PROMPTS_DIR / "phase_b_vqa_generator.txt").read_text(encoding="utf-8")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # resume support: load already-generated prompt_ids
        seen: set[str] = set()
        if out_path.exists():
            with out_path.open() as f:
                for line in f:
                    try:
                        seen.add(json.loads(line)["prompt_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue

        wanted = set(dimensions)
        category_chunks: list[list[str]] = []
        for cat_dims in WORLDJEN_CATEGORIES.values():
            chunk = [d for d in cat_dims if d in wanted]
            if not chunk:
                continue
            # Cap chunk size at 3 dims to keep VQA JSON below ~4500 tokens.
            # Larger chunks (e.g. motion_stability's 5 dims) overflow max_tokens.
            for i in range(0, len(chunk), 3):
                category_chunks.append(chunk[i:i + 3])
        pending = [p for p in prompts if p.prompt_id not in seen]
        if not pending:
            return

        asyncio.run(self._phase_a_async(
            pending, category_chunks, judge_llm_cfg, sys_prompt, out_path, layout,
        ))

    async def _phase_a_async(
        self,
        pending: list[PromptItem],
        category_chunks: list[list[str]],
        judge_llm_cfg: dict[str, Any],
        sys_prompt: str,
        out_path: Path,
        layout: WorkspaceLayout | None,
    ) -> None:
        from videvalkit.scheduler.http_pool import HTTPDispatcher
        from videvalkit.utils.video import strip_code_fences

        judge = self._build_judge(judge_llm_cfg, layout=layout)
        judge.setup()
        dispatcher = HTTPDispatcher(
            concurrency=max(4, len(category_chunks)),
            timeout_s=self.REQUEST_TIMEOUT_S + 60,
            max_retries=self.MAX_RETRIES,
        )
        judge.set_dispatcher(dispatcher)
        try:
            with out_path.open("a", encoding="utf-8") as fout:
                for p in pending:
                    log.info("Phase A [%s] starting %d parallel chunks",
                             p.prompt_id, len(category_chunks))
                    coros = [self._phase_a_chunk(judge, sys_prompt, p, chunk)
                             for chunk in category_chunks]
                    results = await asyncio.gather(*coros, return_exceptions=True)
                    vqa: dict[str, list[dict]] = {}
                    for chunk, res in zip(category_chunks, results):
                        if isinstance(res, Exception):
                            log.error("Phase A [%s] %s failed: %s",
                                      p.prompt_id, "/".join(chunk), res)
                            continue
                        vqa.update(res)
                        log.info("Phase A [%s] %s: %d dims ok",
                                 p.prompt_id, "/".join(chunk), len(res))
                    if not vqa:
                        log.error("Phase A: prompt %s produced no VQA", p.prompt_id)
                        continue
                    fout.write(json.dumps({
                        "prompt_id":     p.prompt_id,
                        "prompt":        p.text,
                        "llm_model":     judge_llm_cfg.get("model"),
                        "gen_timestamp": _dt.datetime.utcnow().isoformat() + "Z",
                        "vqa":           vqa,
                    }, ensure_ascii=False) + "\n")
                    fout.flush()
        finally:
            await dispatcher.aclose()

    async def _phase_a_chunk(
        self,
        judge: Any,
        sys_prompt: str,
        p: PromptItem,
        chunk: list[str],
    ) -> dict[str, list[dict]]:
        from videvalkit.utils.video import strip_code_fences
        user_msg = self._format_vqa_user_message(p.text, chunk)
        resp = await judge.achat_text(
            user=user_msg,
            system=sys_prompt,
            max_tokens=4096,
            temperature=0.2,
            extra={"chat_template_kwargs": {"enable_thinking": False}},
        )
        data = json.loads(strip_code_fences(resp["content"]))
        if not isinstance(data, dict):
            raise ValueError(f"expected JSON object, got {type(data).__name__}")
        return data

    @staticmethod
    def _format_vqa_user_message(prompt_text: str, dimensions: list[str]) -> str:
        lines = [f"PROMPT: {prompt_text}", "", "DIMENSIONS TO ANALYZE:"]
        for d in dimensions:
            lines.append(f"- {d}: {WORLDJEN_DEFINITIONS.get(d, '')}")
        lines += ["", "Generate 10 questions per dimension."]
        return "\n".join(lines)

    @staticmethod
    def _load_vqa(path: Path) -> dict[str, dict[str, list[dict]]]:
        out: dict[str, dict[str, list[dict]]] = {}
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out[d["prompt_id"]] = d["vqa"]
        return out

    # ===== Phase B: per-(model, prompt, dim) VLM scoring ====================
    async def _phase_b_score(
        self,
        videos: list[VideoSpec],
        questions_map: dict[str, dict[str, list[dict]]],
        dimensions: list[str],
        judge_cfg: dict[str, Any],
        layout: WorkspaceLayout,
        out: list[RawResult],
        max_concurrency: int = 4,
    ) -> None:
        from videvalkit.scheduler.http_pool import HTTPDispatcher

        judge = self._build_judge(judge_cfg, layout=layout)
        judge.setup()
        dispatcher = HTTPDispatcher(
            concurrency=max_concurrency,
            timeout_s=self.REQUEST_TIMEOUT_S + 30,
            max_retries=self.MAX_RETRIES,
        )
        judge.set_dispatcher(dispatcher)

        from videvalkit.storage.workspace import Workspace
        ws = Workspace(layout.root)

        try:
            sem = asyncio.Semaphore(max_concurrency)

            async def _score_one(video: VideoSpec) -> None:
                vqa = questions_map.get(video.prompt_id)
                if not vqa:
                    log.warning("no VQA for prompt %s; skip %s", video.prompt_id, video.path)
                    return
                async with sem:
                    await self._score_video_all_dims(
                        video, vqa, dimensions, judge, ws, out
                    )

            await asyncio.gather(*(_score_one(v) for v in videos))
        finally:
            await dispatcher.aclose()

    async def _score_video_all_dims(
        self,
        video: VideoSpec,
        vqa: dict[str, list[dict]],
        dimensions: list[str],
        judge: Any,
        ws: Any,
        out: list[RawResult],
    ) -> None:
        if not video.path.exists():
            log.warning("missing video %s; skip", video.path)
            return
        for dim in dimensions:
            if ws.has_raw(self.name, video.model_name, dim, video.prompt_id):
                try:
                    out.append(ws.load_raw(self.name, video.model_name, dim, video.prompt_id))
                except Exception:
                    pass
                continue
            questions = vqa.get(dim) or []
            if not questions:
                continue
            mode = WORLDJEN_DIMENSION_MODES.get(dim, "holistic")
            n_frames = WORLDJEN_FRAMES_PER_MODE[mode]
            user_prompt = self._format_dim_prompt(dim, questions)

            score_list: list[dict] | None = None
            for attempt in range(self.MAX_RETRIES):
                try:
                    resp = await judge.achat_with_frames(
                        video.path, user_prompt, mode=mode, n_frames=n_frames,
                        max_tokens=2048, temperature=0.2,
                    )
                    from videvalkit.utils.video import strip_code_fences
                    score_list = json.loads(strip_code_fences(resp["content"]))
                    break
                except Exception as e:
                    msg = str(e)[:200]
                    wait = (2 ** attempt) * (10 if "429" in msg else 5)
                    log.warning("[%s][%s] %s attempt %d: %s; backoff %ds",
                                video.model_name, video.prompt_id, dim, attempt + 1, msg, wait)
                    await asyncio.sleep(wait)
            if score_list is None:
                log.error("[%s][%s] %s: FAILED after retries", video.model_name, video.prompt_id, dim)
                continue
            mean_score = self._mean_score(score_list)
            r = RawResult(
                benchmark=self.name,
                model=video.model_name,
                dimension=dim,
                prompt_id=video.prompt_id,
                score=mean_score,
                judge=judge.model,
                scorer="worldjen_vlm_judge",
                video_path=str(video.path),
                meta={"questions_n": len(score_list), "mode": mode, "items": score_list},
            )
            ws.write_raw(r)
            out.append(r)

    @staticmethod
    def _format_dim_prompt(dimension: str, questions: list[dict]) -> str:
        lines = [f"You are evaluating a video generated from a prompt. Dimension: {dimension}",
                 "", "Questions to answer:"]
        for i, q in enumerate(questions, 1):
            if not isinstance(q, dict):
                continue
            lines.append(f"{i}. {q.get('question', '')}")
            rubric = q.get("rubric_description") or q.get("rubric") or ""
            if rubric:
                lines.append(f"   Rubric: {rubric}")
        lines += ["",
                  "Answer each question with a score (1-5) and a short justification.",
                  'Return ONLY a JSON list: [{"score": X, "justification": "..."}]']
        return "\n".join(lines)

    @staticmethod
    def _mean_score(items: list[dict]) -> float:
        vals = [item.get("score") for item in items
                if isinstance(item, dict) and isinstance(item.get("score"), (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _build_judge(cfg: dict[str, Any], layout: WorkspaceLayout | None = None) -> Any:
        from videvalkit.scorers.vlm_judge import build_judge
        return build_judge(cfg, layout=layout)

    # ---- export -------------------------------------------------------------
    def export_official(self, summary: Summary, out_path: Path) -> None:
        """WorldJen has no external leaderboard; emit summary_report_unified.json."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(summary.model_dump_json(indent=2))
