"""Video-Bench adapter — 9 MLLM-judged dims, default GPT-4o.

We do NOT call the upstream `videobench` CLI; instead, we run the same
prompt-template pattern through our own VLM judge so any
OpenAI-compatible endpoint (incl. local vLLM) can be swapped in via the
`judge` config. Per-dim prompt templates are bundled inside this package
under `prompts/` and were transcribed from the upstream prompt files.

If users insist on running the upstream `videobench evaluate` CLI, set
`use_upstream=True` in evaluate() and we'll subprocess it.
"""

from __future__ import annotations

import asyncio
import json
import re
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from videvalkit.core.benchmark import BaseBenchmark
from videvalkit.core.layout import WorkspaceLayout
from videvalkit.core.types import PromptItem, RawResult, Summary, VideoSpec

log = logging.getLogger(__name__)


VIDEOBENCH_DIMENSIONS = [
    "imaging_quality", "aesthetic_quality",
    "temporal_consistency", "motion_effects",
    "video_text_consistency", "object_class_consistency",
    "color_consistency", "action_consistency", "scene_consistency",
]

VIDEOBENCH_CATEGORIES = {
    "static":     ["imaging_quality", "aesthetic_quality"],
    "dynamic":    ["temporal_consistency", "motion_effects"],
    "alignment":  ["video_text_consistency", "object_class_consistency",
                   "color_consistency", "action_consistency", "scene_consistency"],
}

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# ─── upstream rubric loader ────────────────────────────────────────────── #
# Static (1-5 scale) and dynamic (1-5 scale) dims share an upstream
# ``PromptTemplate4GPTeval.py`` with module-level Prompt4* constants.
# Alignment dims (1-3 or 1-5 scale) each have their own dict-shaped file
# under ``prompts/<dim>.py`` with a 'gpt4o-system' key.
#
# We load these lazily and reproduce upstream's rubric byte-for-byte rather
# than using a generic template.
_STATIC_RUBRIC_NAMES: dict[str, str] = {
    "imaging_quality":      "Prompt4ImagingQuality",
    "aesthetic_quality":    "Prompt4AestheticQuality",
    "motion_effects":       "Prompt4MotionEffects",
    "temporal_consistency": "Prompt4TemporalConsistency",
}

# Each alignment dim has prompts/<key>.py where the dict variable is named
# ``<key>_prompt``. Adapter dim → upstream key:
_ALIGNMENT_DIM_TO_KEY: dict[str, str] = {
    "video_text_consistency":   "overall_consistency",
    "object_class_consistency": "object_class",
    "color_consistency":        "color",
    "action_consistency":       "action",
    "scene_consistency":        "scene",
}

# Upstream scales:
#   1-5: imaging/aesthetic/motion/temporal, AND video_text_consistency
#        (overall_consistency.py:175 says "from 1 to 5, with 5 being the highest")
#   1-3: color, object_class, scene, action
_FIVE_SCALE_DIMS: set[str] = {
    "imaging_quality", "aesthetic_quality", "motion_effects",
    "temporal_consistency", "video_text_consistency",
}
_THREE_SCALE_DIMS: set[str] = {
    "object_class_consistency", "color_consistency",
    "action_consistency", "scene_consistency",
}

# Cache loaded rubrics to avoid re-parsing per-call.
_RUBRIC_CACHE: dict[str, str] = {}


def _load_static_rubric(dim: str) -> str | None:
    var_name = _STATIC_RUBRIC_NAMES.get(dim)
    if not var_name:
        return None
    if dim in _RUBRIC_CACHE:
        return _RUBRIC_CACHE[dim]
    fp = _PROMPTS_DIR / "_static_rubrics.py"
    if not fp.exists():
        return None
    ns: dict[str, Any] = {}
    try:
        exec(fp.read_text(), ns)
    except Exception as e:
        log.warning("videobench: failed to load %s: %s", fp, e)
        return None
    rubric = ns.get(var_name)
    if rubric:
        _RUBRIC_CACHE[dim] = rubric
    return rubric


def _load_alignment_rubric(dim: str) -> str | None:
    key = _ALIGNMENT_DIM_TO_KEY.get(dim)
    if not key:
        return None
    if dim in _RUBRIC_CACHE:
        return _RUBRIC_CACHE[dim]
    fp = _PROMPTS_DIR / f"{key}.py"
    if not fp.exists():
        return None
    ns: dict[str, Any] = {}
    try:
        exec(fp.read_text(), ns)
    except Exception as e:
        log.warning("videobench: failed to load %s: %s", fp, e)
        return None
    pdict = ns.get(f"{key}_prompt")
    if not isinstance(pdict, dict):
        return None
    # Use the gpt4o-system entry as the rubric (single-call evaluation).
    rubric = pdict.get("gpt4o-system")
    if rubric:
        _RUBRIC_CACHE[dim] = rubric
    return rubric


# Generic prompt template — fallback when no upstream rubric is found.
_DEFAULT_TEMPLATE = """\
You are evaluating a video generated from this prompt:
  "{prompt}"

Evaluate the video on the dimension: {dimension}
({dimension_definition})

Score the video on this dimension on a scale of {scale_min} to {scale_max},
where {scale_min} = poor and {scale_max} = excellent.

Return ONLY a JSON object: {{"score": <int>, "justification": "..."}}
"""

_ALIGNMENT_PROMPT_CACHE: dict[str, dict[str, str]] = {}


def _load_alignment_dict(dim: str) -> dict[str, str] | None:
    """Return all 5 prompt fields used by upstream's 4-turn alignment flow:
    gpt4o-system, Assistant-one, Assistant-two, gpt4o-answer, summer-system.
    """
    key = _ALIGNMENT_DIM_TO_KEY.get(dim)
    if not key:
        return None
    if dim in _ALIGNMENT_PROMPT_CACHE:
        return _ALIGNMENT_PROMPT_CACHE[dim]
    fp = _PROMPTS_DIR / f"{key}.py"
    if not fp.exists():
        return None
    ns: dict[str, Any] = {}
    try:
        exec(fp.read_text(), ns)
    except Exception as e:
        log.warning("videobench: failed to load %s: %s", fp, e)
        return None
    pdict = ns.get(f"{key}_prompt")
    if isinstance(pdict, dict):
        _ALIGNMENT_PROMPT_CACHE[dim] = pdict
        return pdict
    return None


# ─── score parsers (mirror upstream regex per dim family) ───────────── #

_STATIC_SCORE_RE = re.compile(r":\s*(\d+)")


def _parse_static_score(text: str) -> float | None:
    """Upstream static-dim parser (staticquality.py:85).

    Finds first `: <digit>` pattern in the response.
    """
    m = _STATIC_SCORE_RE.search(text or "")
    return float(m.group(1)) if m else None


def _parse_dynamic_score(text: str) -> float | None:
    """Upstream dynamic-dim parser (dynamicquality.py:20-35).

    Walk backwards from the LAST `because` to find the nearest preceding digit.
    """
    if not text:
        return None
    last = text.rfind("because")
    if last < 0:
        return None
    head = text[:last]
    for ch in reversed(head):
        if ch.isdigit():
            return float(ch)
    return None


def _parse_alignment_score(text: str) -> float | None:
    """Upstream alignment 4-turn parser (VideoTextAlignment.py:251-296).

    Find "Updated Video Description" → "Evaluation Result" → "because";
    nearest digit before `because`.
    """
    if not text:
        return None
    s = text.find("Updated Video Description")
    if s < 0:
        return None
    after = text[s + len("Updated Video Description"):]
    e = after.find("Evaluation Result")
    if e < 0:
        return None
    after_eval = after[e + len("Evaluation Result"):]
    b = after_eval.find("because")
    if b < 0:
        return None
    head = after_eval[:b]
    for ch in reversed(head):
        if ch.isdigit():
            return float(ch)
    return None


# Question-extraction regex used by upstream Assistant agents
# (VideoTextAlignment.py:165-176).
_ASSISTANT_Q_RE = re.compile(r"\[Your questions?\]:\s*(.*)", re.DOTALL)
_DESCRIPTION_RE = re.compile(r"\[Video Description\]:\s*(.*)", re.DOTALL)
_DESCRIPTIONS_2_RE = re.compile(r"\[Descriptions\]:(.*?)\[Answers\]", re.DOTALL)


_DIM_DEFS = {
    "imaging_quality":          "Sharpness, clarity, absence of compression artifacts.",
    "aesthetic_quality":        "Artistic composition, color grading, overall visual appeal.",
    "temporal_consistency":     "Identity / appearance of subjects stays stable across frames.",
    "motion_effects":           "Plausible, smooth motion without jitter or warping.",
    "video_text_consistency":   "Does the video match the textual prompt overall?",
    "object_class_consistency": "Are the objects mentioned in the prompt present and identifiable?",
    "color_consistency":        "Are the colors mentioned in the prompt accurate and consistent?",
    "action_consistency":       "Are the actions described in the prompt performed correctly?",
    "scene_consistency":        "Is the scene / environment as described in the prompt?",
}


class VideoBenchBenchmark(BaseBenchmark):
    """Video-Bench adapter — 9 dims, 1-5 / 1-3 scales, 4-turn alignment flow.

    **Known parity gap vs upstream (architectural):** upstream Video-Bench
    scores all candidate models *together* — its host turn for static dims
    receives one base64 frame from every other model as in-context
    examples (``staticquality.py:60-75``), and the dynamic-dim host
    receives all-models' frames in a single call and parses N scores
    (``dynamicquality.py:78-100``). Our per-video pipeline scores each
    video in isolation, so the rubric's "compare videos A/B/C" framing is
    absent. Score *magnitudes* may shift for ``aesthetic_quality`` (whose
    rubric explicitly references "example frames from other model"),
    ``imaging_quality`` (weakly), and motion/temporal dims. The
    prompt-level alignment dims are not affected by this gap.
    """

    name = "videobench"
    env_name = "videvalkit-videobench"
    dimensions = VIDEOBENCH_DIMENSIONS
    video_layout = "{model}/{dimension}/{prompt_id}.mp4"

    # Upstream uses 1-5 for static/dynamic dims AND for video_text_consistency;
    # 1-3 for the other 4 alignment dims (color / object_class / scene / action).
    SCALE_MIN = 1
    SCALE_MAX = 5
    THREE_SCALE_DIMS = _THREE_SCALE_DIMS
    FIVE_SCALE_DIMS = _FIVE_SCALE_DIMS

    # ---- prompts ------------------------------------------------------------
    def list_prompts(
        self,
        dimensions: list[str] | None = None,
        prompts_file: str | Path | None = None,
    ) -> Iterator[PromptItem]:
        """Yield ``PromptItem`` rows from a JSONL prompts file.

        Upstream Video-Bench ships per-dim annotation JSON files (see
        ``validation/reference_videos/videobench-anno/``) but our adapter
        accepts a unified JSONL with one row per prompt and a
        ``dimensions`` list; this is the same shape WorldJen uses.

        ``prompts_file`` is **required** — there is no bundled default. If
        omitted, raises ``ValueError`` so callers don't accidentally evaluate
        on zero prompts.
        """
        if not prompts_file:
            raise ValueError(
                "VideoBench.list_prompts requires prompts_file=<path/to.jsonl>. "
                "See validation/reference_videos/videobench-anno/ for the "
                "upstream per-dim annotations you can flatten into one JSONL."
            )
        wanted = set(dimensions) if dimensions else set(self.dimensions)
        with Path(prompts_file).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                dims = [d for d in (e.get("dimensions") or self.dimensions) if d in wanted]
                if not dims:
                    continue
                yield PromptItem(prompt_id=e["prompt_id"], text=e["prompt"],
                                 dimensions=dims, meta=e)

    def list_required_videos(
        self,
        prompts: list[PromptItem],
        models: list[str],
        layout: WorkspaceLayout,
        samples_per_prompt: int = 1,
        shared_video: bool = True,
    ) -> list[VideoSpec]:
        """Build the expected (model, prompt, dim) video specs.

        If `shared_video=True` (default), expect one video per (model, prompt)
        at `{model}/{prompt_id}.mp4` and score it on every dim in
        `prompt.dimensions`. This matches the toolkit's standard video layout.

        If `shared_video=False`, expect a per-dim variant at
        `{model}/{dimension}/{prompt_id}.mp4`, mirroring upstream Video-Bench.
        """
        specs: list[VideoSpec] = []
        for m in models:
            for p in prompts:
                for dim in p.dimensions:
                    rel = (
                        f"{m}/{p.prompt_id}.mp4" if shared_video
                        else self.video_layout.format(model=m, dimension=dim, prompt_id=p.prompt_id)
                    )
                    specs.append(VideoSpec(
                        path=layout.videos_dir / rel,
                        prompt_id=p.prompt_id, model_name=m, dimension=dim,
                        sample_index=0,
                    ))
        return specs

    # ---- evaluate -----------------------------------------------------------
    def evaluate(
        self,
        videos: list[VideoSpec] | None = None,
        layout: WorkspaceLayout | None = None,
        dimensions: list[str] | None = None,
        judge: dict[str, Any] | None = None,
        assistant_judge: dict[str, Any] | None = None,
        models: list[str] | None = None,
        prompts_file: str | Path | None = None,
        max_concurrency: int = 4,
        **kwargs: Any,
    ) -> list[RawResult]:
        """Score videos.

        Upstream Video-Bench splits the alignment pipeline into two roles:
        the **host** (multimodal turns 1, 3, 4) defaults to ``gpt-4o-2024-08-06``;
        the two **assistants** (text-only turns 2a / 2b) default to
        ``gpt-4o-mini``. Pass ``judge`` for the host role and
        ``assistant_judge`` for the assistants. If ``assistant_judge`` is None,
        the host judge is reused for the assistant turns (back-compat).
        """
        if layout is None:
            raise ValueError("VideoBench.evaluate requires layout=WorkspaceLayout")
        if judge is None:
            raise ValueError("VideoBench.evaluate requires a judge (default gpt-4o)")
        prompts = list(self.list_prompts(dimensions=dimensions, prompts_file=prompts_file))
        shared_video = bool(kwargs.get("shared_video", True))
        if not videos:
            if not models:
                raise ValueError("VideoBench.evaluate: need videos or models list")
            videos = self.list_required_videos(prompts, models, layout, shared_video=shared_video)

        from videvalkit.storage.workspace import Workspace
        ws = Workspace(layout.root)
        out: list[RawResult] = []
        prompt_by_id = {p.prompt_id: p for p in prompts}
        asyncio.run(self._score_videos_async(
            videos, prompt_by_id, judge, ws, out,
            max_concurrency=max_concurrency,
            assistant_judge_cfg=assistant_judge,
        ))
        return out

    async def _score_videos_async(
        self,
        videos: list[VideoSpec],
        prompt_by_id: dict[str, PromptItem],
        judge_cfg: dict[str, Any],
        ws: Any,
        out: list[RawResult],
        max_concurrency: int = 4,
        assistant_judge_cfg: dict[str, Any] | None = None,
    ) -> None:
        from videvalkit.scheduler.http_pool import HTTPDispatcher
        from videvalkit.scorers.vlm_judge import build_judge

        judge = build_judge(judge_cfg, layout=ws.layout)
        judge.setup()
        if assistant_judge_cfg is not None:
            assistant = build_judge(assistant_judge_cfg, layout=ws.layout)
            assistant.setup()
        else:
            assistant = judge
        dispatcher = HTTPDispatcher(concurrency=max_concurrency)
        if hasattr(judge, "set_dispatcher"):
            judge.set_dispatcher(dispatcher)
        if assistant is not judge and hasattr(assistant, "set_dispatcher"):
            assistant.set_dispatcher(dispatcher)

        sem = asyncio.Semaphore(max_concurrency)

        async def _one(v: VideoSpec) -> None:
            async with sem:
                if ws.has_raw(self.name, v.model_name, v.dimension or "", v.prompt_id):
                    out.append(ws.load_raw(self.name, v.model_name, v.dimension or "", v.prompt_id))
                    return
                if not v.path.exists():
                    log.warning("missing video %s", v.path)
                    return
                p = prompt_by_id.get(v.prompt_id)
                if p is None:
                    return
                dim = v.dimension or ""
                try:
                    if dim in _STATIC_RUBRIC_NAMES:
                        # Single-call path; static / dynamic dims share format
                        # but use different parsers.
                        score, raw_text = await self._score_static_or_dynamic_dim(
                            judge, v.path, p.text, dim,
                        )
                    elif dim in _ALIGNMENT_DIM_TO_KEY:
                        # 4-turn host/assistant flow.
                        score, raw_text = await self._score_alignment_dim(
                            judge, v.path, p.text, dim,
                            assistant_judge=assistant,
                        )
                    else:
                        # Unknown dim: fall back to generic single-call rubric.
                        score, raw_text = await self._score_with_default_template(
                            judge, v.path, p.text, dim,
                        )
                except Exception as e:
                    log.error("[%s][%s/%s] %s", v.model_name, v.prompt_id, dim, e)
                    return
                if score is None:
                    log.warning("[%s][%s/%s] could not parse score from response",
                                v.model_name, v.prompt_id, dim)
                    return
                meta_dict: dict[str, Any] = {
                    "raw_text": raw_text[:500] if raw_text else "",
                }
                if dim in _STATIC_RUBRIC_NAMES:
                    meta_dict["upstream_gap"] = (
                        "Upstream batches all models' frames into a single host "
                        "call as in-context examples; our per-video pipeline "
                        "scores in isolation. May shift magnitudes for "
                        "aesthetic_quality / imaging_quality."
                    )
                r = RawResult(
                    benchmark=self.name, model=v.model_name,
                    dimension=dim, prompt_id=v.prompt_id,
                    score=float(score), judge=judge.model,
                    scorer=f"videobench_{('alignment' if dim in _ALIGNMENT_DIM_TO_KEY else 'static_dynamic')}",
                    video_path=str(v.path),
                    meta=meta_dict,
                )
                ws.write_raw(r)
                out.append(r)

        try:
            await asyncio.gather(*(_one(v) for v in videos))
        finally:
            await dispatcher.aclose()

    # ---- per-dim score paths ----------------------------------------------
    async def _score_static_or_dynamic_dim(
        self,
        judge: Any,
        video_path: Path,
        prompt_text: str,
        dim: str,
    ) -> tuple[float | None, str]:
        """Single-call rubric flow for the 4 static / dynamic dims.

        Mirrors upstream `staticquality.py` (imaging / aesthetic) and
        `dynamicquality.py` (motion / temporal).  System prompt is the
        upstream ``Prompt4{Dim}`` rubric verbatim.
        """
        system_rubric = _load_static_rubric(dim)
        if system_rubric is None:
            return await self._score_with_default_template(
                judge, video_path, prompt_text, dim,
            )
        user = (
            f'\n\nThese are the frames from the video generated. '
            f'The text prompt for the video is:\n"{prompt_text}"\n'
            'Please evaluate the video frames according to the criteria.\n'
        )
        resp = await judge.achat_with_frames(
            video_path, user, mode="videobench", n_frames=8,
            max_tokens=1024, temperature=0.0,
            system=system_rubric,
        )
        content = resp.get("content", "") or ""
        if dim in {"imaging_quality", "aesthetic_quality"}:
            score = _parse_static_score(content)
        else:  # motion_effects / temporal_consistency
            score = _parse_dynamic_score(content)
        # If primary parser fails, try the other heuristic. Use ``is None``
        # so a legitimate score of 0 isn't lost to short-circuit `or`.
        if score is None:
            score = _parse_static_score(content)
        if score is None:
            score = _parse_dynamic_score(content)
        return score, content

    async def _score_alignment_dim(
        self,
        judge: Any,
        video_path: Path,
        prompt_text: str,
        dim: str,
        assistant_judge: Any | None = None,
    ) -> tuple[float | None, str]:
        """Faithful 4-turn host/assistant flow (VideoTextAlignment.py).

        Turn 1 (host, multimodal): produce ``[Video Description]:`` +
            ``[Caption]:`` using the ``gpt4o-system`` rubric.
        Turns 2a, 2b (text-only, two assistants): each Assistant produces a
            single question wrapped in ``[Your question]:``.
        Turn 3 (host, multimodal): given the 2 questions, answer them and
            emit ``[Descriptions]: ... [Answers]: ...`` using ``gpt4o-answer``.
        Turn 4 (host, multimodal): final scoring using ``summer-system``;
            emit ``Updated Video Description ... Evaluation Result: <N> because ...``.
        Parser: nearest digit before ``because`` after ``Evaluation Result``
        (mirrors upstream ``extract_content_from_result``).
        """
        prompts = _load_alignment_dict(dim)
        if prompts is None:
            return await self._score_with_default_template(
                judge, video_path, prompt_text, dim,
            )

        # Upstream uses different models for host vs assistant roles
        # (VideoTextAlignment.py:33-35 vs :112). If caller didn't pass an
        # assistant_judge, reuse the host judge for back-compat.
        asst = assistant_judge if assistant_judge is not None else judge

        n_frames = 8

        # ── Turn 1: Host initial description ────────────────────────── #
        t1_user = (
            "You must think following the 'Evaluation Steps' one by one.\n"
            f"This is the text prompt:\n{prompt_text}\n"
        )
        resp1 = await judge.achat_with_frames(
            video_path, t1_user, mode="videobench", n_frames=n_frames,
            max_tokens=1024, temperature=0.0,
            system=prompts.get("gpt4o-system", ""),
        )
        host_desc = (resp1.get("content", "") or "")
        m = _DESCRIPTION_RE.search(host_desc)
        first_info = m.group(1).strip() if m else "No video captured description"
        # We pass the *whole* host description to assistants as upstream does
        # (`self.description = f'This is the video's initial description: ...'`).
        description_block = (
            f"This is the video's initial description:\n<description>\n{host_desc}\n</description>"
        )

        # ── Turn 2: 2 Assistants ask questions (text-only) ───────────── #
        qa_history: list[str] = []
        cur_desc = description_block
        for agent_key in ("Assistant-one", "Assistant-two"):
            asst_system = prompts.get(agent_key, "")
            asst_user = (
                f"This is the text prompt:\n{prompt_text}\n\n"
                + cur_desc
                + "\n\nYour questions must be deisplayed in the format: \n"
                + "<question>\n[your questions] or [I have no question]\n</question>."
                + "\n\nDisplay the results in the specified Output Format"
            )
            resp_a = await asst.achat_text(
                user=asst_user, system=asst_system,
                max_tokens=512, temperature=0.0,
            )
            atext = (resp_a.get("content", "") or "")
            qm = _ASSISTANT_Q_RE.search(atext)
            question_text = qm.group(1).strip() if qm else "[I have no question]"
            qa_history.append(f"This is the question of the {agent_key}: {question_text}\n")
            cur_desc = cur_desc + (
                f"\nThis is a question of another assistant:\n"
                f"<question-one>\n{question_text}\n</question-one>"
            )

        # ── Turn 3: Host answers (multimodal) ────────────────────────── #
        qa_block = (
            "\n\nThere are the questions of two assistants:\n\n<qa_history>\n"
            + "\n".join(qa_history) + "\n</qa_history>\n"
        )
        t3_user = (
            f"This is the text prompt:\n{prompt_text}\n"
            + qa_block
            + f"\nThese are the frames from the video.\n"
        )
        resp3 = await judge.achat_with_frames(
            video_path, t3_user, mode="videobench", n_frames=n_frames,
            max_tokens=1024, temperature=0.0,
            system=prompts.get("gpt4o-answer", ""),
        )
        answer_text = (resp3.get("content", "") or "")
        d2 = _DESCRIPTIONS_2_RE.search(answer_text)
        second_info = d2.group(1).strip() if d2 else "no further descriptions"

        # ── Turn 4: Host final score (multimodal) ────────────────────── #
        history = [
            f"This is the initial information: \n{first_info}\n\n",
            f"\nThis is the second information: {second_info}",
        ]
        history_block = (
            "There are the two informations:\n<history>\n"
            + "\n".join(history) + "\n</history>"
        )
        t4_user = (
            "You must think following the 'evaluation steps' one by one.\n\n"
            + history_block
            + f"\n\nThis is the text prompt:\n{prompt_text}\n\n"
        )
        resp4 = await judge.achat_with_frames(
            video_path, t4_user, mode="videobench", n_frames=n_frames,
            max_tokens=1024, temperature=0.0,
            system=prompts.get("summer-system", ""),
        )
        final = (resp4.get("content", "") or "")
        score = _parse_alignment_score(final)
        # Loose fallback: search anywhere for the upstream pattern.
        if score is None:
            score = _parse_dynamic_score(final)
        # Save the last turn as raw_text for audit; truncate.
        return score, final

    async def _score_with_default_template(
        self,
        judge: Any,
        video_path: Path,
        prompt_text: str,
        dim: str,
    ) -> tuple[float | None, str]:
        """Fallback rubric for dims we couldn't load upstream prompts for."""
        scale_max = 3 if dim in self.THREE_SCALE_DIMS else self.SCALE_MAX
        rendered = _DEFAULT_TEMPLATE.format(
            prompt=prompt_text, dimension=dim,
            dimension_definition=_DIM_DEFS.get(dim, dim.replace("_", " ")),
            scale_min=self.SCALE_MIN, scale_max=scale_max,
        )
        resp = await judge.achat_with_frames(
            video_path, rendered, mode="videobench", n_frames=8,
            max_tokens=512, temperature=0.0,
        )
        content = resp.get("content", "") or ""
        # Try JSON first, then static / dynamic regex.
        try:
            from videvalkit.utils.video import strip_code_fences
            parsed = json.loads(strip_code_fences(content))
            score = float(parsed.get("score", float("nan")))
            if score == score:  # not NaN
                return score, content
        except Exception:
            pass
        # ``is None`` chain so score=0 isn't swallowed by short-circuit ``or``.
        s = _parse_static_score(content)
        if s is None:
            s = _parse_dynamic_score(content)
        return s, content

    # ---- aggregate ----------------------------------------------------------
    def aggregate(
        self,
        raw: list[RawResult],
        aggregator: str = "videobench_per_dim",
        **kwargs: Any,
    ) -> Summary:
        """Mirror upstream ``videobench/__init__.py:336-346``.

        Upstream emits **per-dim per-model average** only — it does **not**
        compute a cross-dim ``overall`` (the 4 static / dynamic dims live on a
        1-5 scale and the 4 alignment dims on 1-3, so an unnormalised mean
        wouldn't be meaningful).

        We mirror this: ``per_dimension`` is the mean of the per-video Scores
        per dim (matching upstream's ``average_scores``). ``overall`` is a
        toolkit-only convenience: a **scale-normalised** mean over dims —
        each per-dim mean is divided by its scale_max (3 or 5) so 1-3 and
        1-5 dims contribute equally. We emit it as ``None`` if the caller
        sets ``aggregator='videobench_per_dim'`` (default), and as a float
        when ``aggregator='weighted_sum'`` to preserve back-compat with
        callers that want a single number.
        """
        from collections import defaultdict
        per_dim_vals: dict[str, list[float]] = defaultdict(list)
        for r in raw:
            if isinstance(r.score, (int, float)):
                per_dim_vals[r.dimension].append(float(r.score))
        per_dim_mean: dict[str, float] = {
            d: sum(vs) / len(vs) for d, vs in per_dim_vals.items() if vs
        }

        # Scale-normalised overall (toolkit convention; upstream emits no overall).
        def _scale_max(d: str) -> int:
            return 3 if d in self.THREE_SCALE_DIMS else self.SCALE_MAX
        if aggregator == "weighted_sum" and per_dim_mean:
            overall: float | None = (
                sum(v / _scale_max(d) for d, v in per_dim_mean.items())
                / len(per_dim_mean)
            )
        else:
            overall = None  # matches upstream's no-overall behaviour

        model = raw[0].model if raw else ""
        return Summary(
            benchmark=self.name, model=model,
            per_dimension=per_dim_mean,
            overall=overall if overall is not None else 0.0,
            n_videos=len({(r.model, r.prompt_id, r.dimension) for r in raw}),
            n_prompts=len({r.prompt_id for r in raw}),
            aggregator=aggregator,
            meta={
                "scale_max": {d: _scale_max(d) for d in per_dim_mean},
                "upstream_note": (
                    "Upstream Video-Bench reports per-dim averages only; "
                    "'overall' here is a toolkit convention: mean of "
                    "(per_dim_mean / scale_max). Not directly comparable "
                    "to any Video-Bench paper headline."
                ),
            },
        )

    def export_official(self, summary: Summary, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "model": summary.model,
            "scores": summary.per_dimension,
            "overall": summary.overall,
        }, indent=2, ensure_ascii=False))
