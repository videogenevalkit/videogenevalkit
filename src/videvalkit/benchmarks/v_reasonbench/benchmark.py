"""V-ReasonBench adapter — 13 reasoning tasks across 4 categories.

Upstream: https://github.com/yangluo7/V-ReasonBench
Paper:    https://arxiv.org/abs/2511.16668
Site:     https://oahzxl.github.io/VReasonBench/

The benchmark differs from VBench / Physics-IQ in two important ways:

1. **Input is an image pair** (initial frame + final frame), not a free-text
   prompt. The model under test must generate the intermediate video that
   transforms the initial state into the final state.

2. **Scoring is programmatic** (deterministic, no VLM-as-judge). The
   authors tried VLM-as-judge and abandoned it because the VLM frequently
   misreads small grid cells and fine structural differences. Each task
   has a per-task verifier (``verifiers/<task>.py`` upstream) that
   inspects the generated video's final frame (or a sequence of key frames)
   and returns a binary Pass / Unpass.

3. **Pass@5**: the model generates 5 samples per task instance; the
   instance counts as a pass if **any** of the 5 verifies.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from videvalkit.core.benchmark import BaseBenchmark
from videvalkit.core.layout import WorkspaceLayout
from videvalkit.core.types import PromptItem, RawResult, Summary, VideoSpec

log = logging.getLogger(__name__)


VREASONBENCH_CATEGORIES = {
    "structured_problem_solving": ["tic_tac_toe", "sudoku", "n_queens", "tower_of_hanoi"],
    "spatial_cognition":          ["color_connection", "maze_navigation", "shape_matching"],
    "pattern_inference":          ["rule_following", "sequence_continuation"],
    "physical_dynamics":          ["block_sliding", "domino_chain", "pendulum", "projectile"],
}

VREASONBENCH_TASKS: list[str] = [t for ts in VREASONBENCH_CATEGORIES.values() for t in ts]

# Pass@k convention (upstream uses k=5).
VREASONBENCH_K: int = 5


class VReasonBenchBenchmark(BaseBenchmark):
    name = "v_reasonbench"
    env_name = "videvalkit"
    dimensions = VREASONBENCH_TASKS

    # Image-pair conditioned; ``input_pairs/{task}/{instance_id}/{0,1}.png``
    # lives next to the generated videos. We stash the pair paths in meta.
    video_layout = "{model}/{prompt_id}-{sample_index}.mp4"
    SAMPLES_PER_PROMPT: int = VREASONBENCH_K

    # ---- prompts ------------------------------------------------------------
    def list_prompts(
        self,
        dimensions: list[str] | None = None,
        prompts_file: str | Path | None = None,
    ) -> Iterator[PromptItem]:
        """Each prompt is an (initial image, final image) pair.

        Expected JSONL row::

            {"prompt_id": "tic_tac_toe_0001", "task": "tic_tac_toe",
             "initial_image": "input_pairs/tic_tac_toe/0001/0.png",
             "final_image":   "input_pairs/tic_tac_toe/0001/1.png",
             "text":          "complete the tic-tac-toe board"}
        """
        wanted = set(dimensions) if dimensions else set(self.dimensions)
        if prompts_file is None:
            # Smoke-test fallback: one synthetic example per task.
            for i, t in enumerate(VREASONBENCH_TASKS):
                if t not in wanted:
                    continue
                yield PromptItem(
                    prompt_id=f"{t}_{i:04d}",
                    text=f"[{t}] placeholder reasoning instance",
                    dimensions=[t],
                    meta={
                        "category": self._category_of(t),
                        "initial_image": None,
                        "final_image":   None,
                    },
                )
            return
        with Path(prompts_file).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                t = e.get("task")
                if t not in wanted:
                    continue
                yield PromptItem(
                    prompt_id=str(e["prompt_id"]),
                    text=e.get("text", ""),
                    dimensions=[t],
                    meta={
                        "category": self._category_of(t),
                        "initial_image": e.get("initial_image"),
                        "final_image":   e.get("final_image"),
                    },
                )

    def list_required_videos(
        self,
        prompts: list[PromptItem],
        models: list[str],
        layout: WorkspaceLayout,
        samples_per_prompt: int | None = None,
    ) -> list[VideoSpec]:
        n = samples_per_prompt if samples_per_prompt is not None else self.SAMPLES_PER_PROMPT
        specs: list[VideoSpec] = []
        for m in models:
            for p in prompts:
                for idx in range(n):
                    rel = self.video_layout.format(
                        model=m, prompt_id=p.prompt_id, sample_index=idx,
                    )
                    specs.append(VideoSpec(
                        path=layout.videos_dir / rel,
                        prompt_id=p.prompt_id,
                        model_name=m,
                        dimension=(p.dimensions[0] if p.dimensions else None),
                        sample_index=idx,
                        meta=dict(p.meta),
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
        **kwargs: Any,
    ) -> list[RawResult]:
        """Run per-task deterministic verifiers on each generated clip.

        Wiring sketch:
          1. Group videos by (model, prompt_id) — 5 samples per instance.
          2. For each sample, extract the final frame (and any task-specific
             intermediate frames) with ``utils.video.extract_frames``.
          3. Look up the per-task verifier in
             ``v_reasonbench/verifiers/<task>.py`` and call it on the
             extracted frame(s); the verifier returns ``{"pass": bool}``.
          4. Aggregate per instance with pass@5 (instance passes if any
             sample passes) → one RawResult per instance with score in
             {0.0, 1.0}.
        """
        raise NotImplementedError(
            "VReasonBenchBenchmark.evaluate() is a stub. Verifiers live in "
            "https://github.com/yangluo7/V-ReasonBench — port each to "
            "scorers/metric/v_reasonbench/<task>.py and dispatch on `task`."
        )

    # ---- aggregate ----------------------------------------------------------
    def aggregate(
        self,
        raw: list[RawResult],
        aggregator: str = "weighted_sum",
        **kwargs: Any,
    ) -> Summary:
        """Pass-rate per task → mean across tasks → overall pass rate."""
        from collections import defaultdict
        per_task: dict[str, list[float]] = defaultdict(list)
        for r in raw:
            if isinstance(r.score, (int, float)):
                per_task[r.dimension].append(float(r.score))
        per_task_mean = {t: sum(vs) / len(vs) for t, vs in per_task.items() if vs}

        per_category: dict[str, list[float]] = defaultdict(list)
        for t, v in per_task_mean.items():
            per_category[self._category_of(t)].append(v)
        per_category_mean = {c: sum(vs) / len(vs) for c, vs in per_category.items() if vs}
        overall = (sum(per_category_mean.values()) / len(per_category_mean)
                   if per_category_mean else 0.0)

        model = raw[0].model if raw else ""
        return Summary(
            benchmark=self.name,
            model=model,
            per_dimension=per_task_mean,
            overall=overall,
            n_videos=len({(r.model, r.prompt_id, r.dimension) for r in raw}),
            n_prompts=len({r.prompt_id for r in raw}),
            aggregator=aggregator,
            meta={"per_category_mean": per_category_mean,
                  "pass_at_k": VREASONBENCH_K,
                  "scale": "0-1 (pass rate)"},
        )

    # ---- helpers ------------------------------------------------------------
    @classmethod
    def _category_of(cls, task: str) -> str:
        for cat, ts in VREASONBENCH_CATEGORIES.items():
            if task in ts:
                return cat
        return "unknown"
