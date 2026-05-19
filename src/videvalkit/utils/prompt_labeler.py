"""LLM-based prompt labeler — auto-generate VBench-compatible auxiliary_info.

VBench v1 and VBench-2.0 expect each prompt to carry per-dim ground-truth
labels in their `*_full_info.json` files. If you have custom prompts
(narrative scenes, e.g. WorldJen's 110 prompts) you can't just plug them
in — the prompt-dependent dims need those labels.

This module uses a local LLM (default: Qwen3-32B at :8004 via vLLM) to
read each prompt and emit structured auxiliary_info per dim, then merges
everything into a drop-in `{benchmark}_full_info.json`.

Usage::

    from videvalkit.utils.prompt_labeler import label_prompts_vbench_v1, label_prompts_vbench2

    out_path = label_prompts_vbench_v1(
        prompts_file="workspace/prompts/worldjen/prompts.jsonl",
        out_path="workspace/prompts/auto_labels/vbench_full_info.json",
        judge_name="qwen3-32b-local",
        dims=["color", "object_class", "scene", "spatial_relationship",
              "multiple_objects", "appearance_style", "human_action"],
    )

Each dim's schema follows VBench's existing format (verified by reading
the upstream `VBench_full_info.json` / `VBench2_full_info.json`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────────
# Per-dim extraction schemas. Each entry has:
#   "instruction"  : what to ask the LLM
#   "format"       : JSON shape returned (used in system prompt + parsing)
#   "to_aux"       : function (parsed_json -> auxiliary_info shape upstream wants)
#                    Return None if the dim isn't applicable to this prompt.
# ────────────────────────────────────────────────────────────────────────────────


VBENCH_V1_DIM_SCHEMAS = {
    "color": {
        "instruction": (
            "Identify ONE primary (object, color) pair explicitly stated in the prompt. "
            "If multiple, pick the most prominent. If no color is mentioned, return null."
        ),
        "format": '{"object": "<noun>" | null, "color": "<color word>" | null}',
        "to_aux": lambda d: (
            {"color": {"color": d["color"]}} if d.get("color") else None
        ),
    },
    "object_class": {
        "instruction": (
            "Identify ONE primary subject/object class (a single noun) mentioned in the prompt."
        ),
        "format": '{"object": "<singular noun>" | null}',
        "to_aux": lambda d: (
            {"object_class": {"object": d["object"]}} if d.get("object") else None
        ),
    },
    "multiple_objects": {
        "instruction": (
            "List 2-4 distinct object classes that the prompt says should appear together. "
            "Use singular nouns. Return null if only one object class is mentioned."
        ),
        "format": '{"objects": ["<noun>", "<noun>", ...] | null}',
        "to_aux": lambda d: (
            {"multiple_objects": {"object": " and ".join(d["objects"])}}
            if d.get("objects") and len(d["objects"]) >= 2 else None
        ),
    },
    "spatial_relationship": {
        "instruction": (
            "Identify ONE spatial relationship: (object_a, relationship, object_b). "
            "relationship must be one of: 'on the left of', 'on the right of', "
            "'above', 'below', 'in front of', 'behind', 'next to', 'on'. "
            "Return null if no clear spatial relation is stated."
        ),
        "format": '{"object_a": "<noun>", "relationship": "<rel>", "object_b": "<noun>"} | null',
        "to_aux": lambda d: (
            {"spatial_relationship": {"spatial_relationship": {
                "object_a": d["object_a"], "object_b": d["object_b"],
                "relationship": d["relationship"]}}}
            if d and d.get("object_a") and d.get("object_b") else None
        ),
    },
    "scene": {
        "instruction": (
            "Identify the primary scene/location (e.g. 'alley', 'beach', 'school hallway'). "
            "Use a short noun phrase. Return null if no clear location is named."
        ),
        "format": '{"scene": "<location>" | null}',
        "to_aux": lambda d: (
            {"scene": {"scene": {"scene": d["scene"]}}} if d.get("scene") else None
        ),
    },
    "appearance_style": {
        "instruction": (
            "Identify the artistic/visual style (e.g. 'Van Gogh style', 'anime', 'photorealistic'). "
            "Return null if no style is mentioned."
        ),
        "format": '{"style": "<style>" | null}',
        "to_aux": lambda d: (
            {"appearance_style": {"appearance_style": d["style"]}} if d.get("style") else None
        ),
    },
    "human_action": {
        # VBench's human_action uses Kinetics-400 class names, but upstream's
        # full_info entries don't carry per-prompt auxiliary_info — the model
        # matches the prompt against the Kinetics class space at runtime.
        # We still mark `dimension=["human_action"]` if humans + verbs are present.
        "instruction": (
            "Does the prompt describe a clear human action (e.g., dancing, slapping, hugging)? "
            "If yes, list the action verbs in present participle form."
        ),
        "format": '{"actions": ["<verb-ing>", ...] | null}',
        "to_aux": lambda d: ({} if d.get("actions") else None),   # tag only, no aux
    },
}


VBENCH2_DIM_SCHEMAS = {
    "Motion_Order_Understanding": {
        "instruction": (
            "Extract the sequence of distinct actions in temporal order. "
            "Each action should be a short phrase (e.g., 'sitting', 'getting up')."
        ),
        "format": '{"sequence": ["<action>", "<action>", ...]}',
        "to_aux": lambda d: (
            d["sequence"] if d.get("sequence") and len(d["sequence"]) >= 2 else None
        ),
    },
    "Motion_Rationality": {
        "instruction": (
            "Generate 3-5 yes/no questions a viewer should answer after watching the video "
            "to verify the motion is physically plausible and matches the prompt. "
            "Each question should end with '(yes or no)'."
        ),
        "format": '{"questions": ["<question> (yes or no)", ...]}',
        "to_aux": lambda d: (
            d["questions"] if d.get("questions") else None
        ),
    },
    "Human_Interaction": {
        # No auxiliary_info needed for this dim — LLaVA reads prompt at eval time.
        "instruction": (
            "Does the prompt describe an interaction between TWO OR MORE humans "
            "(e.g., handshake, hug, fight)? Answer true/false."
        ),
        "format": '{"has_human_interaction": true | false}',
        "to_aux": lambda d: ({} if d.get("has_human_interaction") else None),
    },
    "Human_Anatomy": {
        # Always applicable when humans are present; no auxiliary_info needed.
        "instruction": (
            "Does the prompt feature one or more humans? Answer true/false."
        ),
        "format": '{"has_human": true | false}',
        "to_aux": lambda d: ({} if d.get("has_human") else None),
    },
    "Human_Clothes": {
        "instruction": (
            "Does the prompt mention clothing or attire (color, type, style)? "
            "If yes, list the clothing items per person."
        ),
        "format": '{"clothing": [{"person": "str", "items": ["str"]}] | null}',
        "to_aux": lambda d: ({} if d.get("clothing") else None),
    },
}


SYS_PROMPT_TEMPLATE = """\
You are a structured information extractor. Given a video-generation prompt,
extract specific information for the dimension below.

DIMENSION: {dim}
TASK: {instruction}
OUTPUT FORMAT: Return a JSON object matching exactly this schema:
{format}

Return only the JSON object, no markdown fences or commentary.
If the dimension does not apply to the prompt, return the null variant.
"""


def _strip_fences(text: str) -> str:
    s = text.strip()
    if "```json" in s:
        return s.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in s:
        return s.split("```", 1)[1].split("```", 1)[0].strip()
    return s


def _label_one(judge: Any, prompt_text: str, dim: str, schema: dict) -> dict | None:
    """Ask LLM for this (prompt, dim). Return auxiliary_info subdict or None."""
    sys_p = SYS_PROMPT_TEMPLATE.format(
        dim=dim, instruction=schema["instruction"], format=schema["format"],
    )
    try:
        resp = judge.chat_text(
            user=f"PROMPT: {prompt_text}",
            system=sys_p,
            max_tokens=512,
            temperature=0.1,
            extra={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = _strip_fences(resp["content"])
        if not raw or raw.lower() == "null":
            return None
        data = json.loads(raw)
        return schema["to_aux"](data)
    except Exception as e:
        log.warning("label %s on %r failed: %s", dim, prompt_text[:60], e)
        return None


def _iter_prompts(prompts_file: Path):
    for line in prompts_file.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        yield rec


def label_prompts_vbench_v1(
    prompts_file: str | Path,
    out_path: str | Path,
    judge_cfg: dict[str, Any],
    dims: list[str] | None = None,
) -> Path:
    """VBench v1 format: ONE entry per prompt, auxiliary_info is dict keyed by dim.

    Output drop-in for `VBench_full_info.json`::
        [{"prompt_en": "...", "dimension": [...], "auxiliary_info": {dim: {...}}}, ...]
    """
    from videvalkit.scorers.vlm_judge import build_judge

    schemas = (VBENCH_V1_DIM_SCHEMAS if dims is None
               else {d: VBENCH_V1_DIM_SCHEMAS[d] for d in dims
                     if d in VBENCH_V1_DIM_SCHEMAS})
    prompts_file = Path(prompts_file)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    judge = build_judge(judge_cfg, layout=None); judge.setup()

    seen: set[str] = set()
    out_entries: list[dict] = []
    if out_path.exists():
        try:
            out_entries = json.loads(out_path.read_text())
            seen = {str(e.get("prompt_id", "")) for e in out_entries}
        except Exception:
            out_entries, seen = [], set()

    prompts = list(_iter_prompts(prompts_file))
    log.info("VBench v1: labeling %d prompts × %d dims using %s",
             len(prompts), len(schemas), judge_cfg.get("model"))

    for i, rec in enumerate(prompts):
        pid = str(rec["prompt_id"])
        if pid in seen:
            continue
        text = rec.get("enhanced_prompt") or rec["prompt"]
        active_dims: list[str] = []
        aux: dict[str, Any] = {}
        for dim, schema in schemas.items():
            res = _label_one(judge, text, dim, schema)
            if res is not None:
                active_dims.append(dim)
                if res:
                    aux.update(res)
        entry = {"prompt_id": pid, "prompt_en": text, "dimension": active_dims}
        if aux:
            entry["auxiliary_info"] = aux
        out_entries.append(entry)
        log.info("[%d/%d] %s -> dims=%s", i + 1, len(prompts), pid, active_dims)
        out_path.write_text(json.dumps(out_entries, indent=2, ensure_ascii=False))

    log.info("wrote %d entries to %s", len(out_entries), out_path)
    return out_path


def label_prompts_vbench2(
    prompts_file: str | Path,
    out_path: str | Path,
    judge_cfg: dict[str, Any],
    dims: list[str] | None = None,
) -> Path:
    """VBench-2.0 format: ONE entry per (prompt, dim), auxiliary_info is bare value.

    Upstream's `VBench2_full_info.json` uses entries like::
        {"prompt_en": "...", "dimension": ["Camera_Motion"], "auxiliary_info": "zoom_in"}
        {"prompt_en": "...", "dimension": ["Motion_Order_Understanding"],
         "auxiliary_info": ["sitting", "get up"]}

    So we emit a separate entry per dim per prompt (when the dim is applicable).
    Dims that need no auxiliary_info (Human_Anatomy, Human_Interaction, etc.)
    just get `dimension: [...]` with no auxiliary_info key.
    """
    from videvalkit.scorers.vlm_judge import build_judge

    schemas = (VBENCH2_DIM_SCHEMAS if dims is None
               else {d: VBENCH2_DIM_SCHEMAS[d] for d in dims
                     if d in VBENCH2_DIM_SCHEMAS})
    prompts_file = Path(prompts_file)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    judge = build_judge(judge_cfg, layout=None); judge.setup()

    seen: set[tuple[str, str]] = set()
    out_entries: list[dict] = []
    if out_path.exists():
        try:
            out_entries = json.loads(out_path.read_text())
            for e in out_entries:
                pid = str(e.get("prompt_id", ""))
                for d in e.get("dimension", []):
                    seen.add((pid, d))
        except Exception:
            out_entries, seen = [], set()

    prompts = list(_iter_prompts(prompts_file))
    log.info("VBench-2.0: labeling %d prompts × %d dims using %s",
             len(prompts), len(schemas), judge_cfg.get("model"))

    for i, rec in enumerate(prompts):
        pid = str(rec["prompt_id"])
        text = rec.get("enhanced_prompt") or rec["prompt"]
        tagged_dims: list[str] = []
        for dim, schema in schemas.items():
            if (pid, dim) in seen:
                tagged_dims.append(dim); continue
            res = _label_one(judge, text, dim, schema)
            if res is None:
                continue
            entry = {"prompt_id": pid, "prompt_en": text, "dimension": [dim]}
            # res is the auxiliary_info value (could be {}, [...], "...", etc.)
            if res not in ({}, [], None):
                entry["auxiliary_info"] = res
            out_entries.append(entry)
            tagged_dims.append(dim)
        log.info("[%d/%d] %s -> dims=%s", i + 1, len(prompts), pid, tagged_dims)
        out_path.write_text(json.dumps(out_entries, indent=2, ensure_ascii=False))

    log.info("wrote %d entries to %s", len(out_entries), out_path)
    return out_path
