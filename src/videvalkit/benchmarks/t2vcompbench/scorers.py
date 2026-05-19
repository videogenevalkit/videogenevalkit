"""T2V-CompBench MLLM-dim scorers.

Four of the seven T2V-CompBench V2 dimensions are MLLM-judged. We implement
them as **single VLM-direct calls** (rubric scoring) rather than calling
upstream's Grid-LLaVA / D-LLaVA two-step pipeline. The per-dim prompts
below capture the same semantics:

  * consistent_attribute  — attribute-entity binding stability across frames
  * dynamic_attribute     — continuous attribute evolution over time
  * action_binding        — correct subject ↔ action assignment
  * object_interactions   — interactions occur and are physically plausible

Each call goes through ``judge.chat_with_frames(...)`` (an
``OpenAICompatibleVLMJudge`` constructed by ``scorers.vlm_judge.factory``),
so the same endpoint that VBench-2.0 or WorldJen use is reused — no
checkpoint download required (it's a remote HTTP call).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from videvalkit.utils.video import strip_code_fences

log = logging.getLogger(__name__)


T2VCOMPBENCH_MLLM_DIMS = (
    "consistent_attribute",
    "dynamic_attribute",
    "action_binding",
    "object_interactions",
)

# Per-dim VLM prompt templates. Each asks the VLM to return ONE JSON object
# with {"score": 0..1, "justification": "..."}.
DIMENSION_PROMPT_TEMPLATES: dict[str, str] = {
    "consistent_attribute": (
        "You are evaluating a text-to-video generation for the "
        "**consistent attribute binding** dimension.\n\n"
        "User prompt:\n  \"{prompt}\"\n\n"
        "Decompose the prompt into independent attribute-entity phrases "
        "(e.g. 'a blue car', 'a white fence'). For each phrase, check "
        "whether the described attribute is correctly bound to the "
        "correct subject across the frames shown, AND whether that "
        "binding stays stable over time (no colour-swap, no shape drift).\n\n"
        "Penalise: wrong colour, wrong shape, attribute-subject swap, "
        "attribute drift across frames.\n\n"
        "Output **only** one JSON object on a single line:\n"
        "{{\"score\": <float in [0,1]>, \"justification\": \"<short reason>\"}}\n"
    ),
    "dynamic_attribute": (
        "You are evaluating a text-to-video generation for the "
        "**dynamic attribute** dimension (state machine evolution).\n\n"
        "User prompt:\n  \"{prompt}\"\n\n"
        "Identify the attribute that is supposed to EVOLVE over time "
        "(e.g. 'leaf gradually turning yellow' → leaf colour green→yellow). "
        "Score three things, then take their geometric mean:\n"
        " (a) Initial state matches the early frames.\n"
        " (b) Final state matches the late frames.\n"
        " (c) Transition between them is continuous, not abrupt.\n"
        "If the prompt has no time-evolving attribute, return score=0.5 "
        "and say so.\n\n"
        "Output **only** one JSON object:\n"
        "{{\"score\": <float in [0,1]>, \"justification\": \"<short reason>\"}}\n"
    ),
    "action_binding": (
        "You are evaluating a text-to-video generation for the "
        "**action binding** dimension.\n\n"
        "User prompt:\n  \"{prompt}\"\n\n"
        "Identify each subject and the specific action it should perform "
        "(e.g. 'the cat on the LEFT climbs a tree' AND 'the dog on the "
        "RIGHT runs'). Score whether each action is bound to the correct "
        "subject. Penalise: subject swap, missing action, the wrong "
        "subject performing an action, action absent.\n\n"
        "Output **only** one JSON object:\n"
        "{{\"score\": <float in [0,1]>, \"justification\": \"<short reason>\"}}\n"
    ),
    "object_interactions": (
        "You are evaluating a text-to-video generation for the "
        "**object interactions** dimension.\n\n"
        "User prompt:\n  \"{prompt}\"\n\n"
        "Identify each interaction described between objects/people "
        "(handshake, pour from cup, push door, etc.) and judge whether "
        "the video shows it executing physically correctly: contact "
        "occurs, causal feedback is realistic, no 'fake' / pass-through "
        "actions.\n\n"
        "Score 1.0 if every interaction is clear and correct, 0.0 if "
        "absent or wrong, partial scores in between.\n\n"
        "Output **only** one JSON object:\n"
        "{{\"score\": <float in [0,1]>, \"justification\": \"<short reason>\"}}\n"
    ),
}


def _extract_content(resp: dict[str, Any]) -> str:
    """Pull text out of a judge response.

    ``OpenAICompatibleVLMJudge._post_sync`` already normalises responses to
    ``{"content": <str>, "usage": ..., "model": ...}``, so the flat key is
    the common case. Fall back to OpenAI raw shape for safety.
    """
    if isinstance(resp, dict):
        if isinstance(resp.get("content"), str):
            return resp["content"]
        try:
            msg = resp["choices"][0]["message"]["content"]
            if isinstance(msg, list):
                return "".join(
                    (p.get("text", "") if isinstance(p, dict) else str(p))
                    for p in msg
                )
            return str(msg)
        except (KeyError, IndexError, TypeError):
            pass
    raise RuntimeError(f"unexpected judge response shape: {str(resp)[:200]}")


_SCORE_REGEX = re.compile(
    r'"?score"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE,
)


def _parse_score(content: str) -> tuple[float, str]:
    """Parse a {"score": float, "justification": str} object out of `content`.

    Robust to:
      * ``` ```json ... ``` ``` fences (handled by strip_code_fences)
      * leading prose before the JSON (search for first ``{``)
      * the VLM only writing ``score: 0.7`` without proper JSON
    """
    s = strip_code_fences(content).strip()
    # Try strict JSON parse first.
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "score" in obj:
            return float(obj["score"]), str(obj.get("justification", ""))[:300]
    except Exception:
        pass
    # Try to find a JSON-ish substring after the first '{'.
    if "{" in s and "}" in s:
        cand = s[s.index("{"):s.rindex("}") + 1]
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and "score" in obj:
                return float(obj["score"]), str(obj.get("justification", ""))[:300]
        except Exception:
            pass
    # Last resort: regex.
    m = _SCORE_REGEX.search(s)
    if m:
        return float(m.group(1)), s[:200]
    raise ValueError(f"could not parse score from response: {s[:200]!r}")


def score_video_dim(
    judge: Any,
    video_path: str | Path,
    prompt_text: str,
    dim: str,
    n_frames: int = 8,
    mode: str = "holistic",
) -> dict[str, Any]:
    """Score one (video, dim) by issuing one multimodal call to ``judge``.

    Returns a dict with at least ``score`` (float in [0,1]) and
    ``justification`` (string). On parse failure, raises ValueError.
    """
    if dim not in DIMENSION_PROMPT_TEMPLATES:
        raise KeyError(f"no MLLM template for dim={dim!r}")
    vlm_prompt = DIMENSION_PROMPT_TEMPLATES[dim].format(prompt=prompt_text)
    resp = judge.chat_with_frames(
        video_path=str(video_path), prompt=vlm_prompt,
        mode=mode, n_frames=n_frames, max_tokens=512, temperature=0.0,
    )
    content = _extract_content(resp)
    score, justification = _parse_score(content)
    score = max(0.0, min(1.0, score))
    return {
        "score":         score,
        "justification": justification,
        "raw_response":  content[:500],
        "judge_model":   getattr(judge, "model", "unknown"),
        "n_frames":      n_frames,
        "mode":          mode,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# CV-based scorers                                                            #
# ─────────────────────────────────────────────────────────────────────────── #
#
# Each scorer's ``score_one(frames, meta)`` takes the **LLM-extracted
# metadata** (objects / numbers / spatial / d_1 / d_2 — see extract.py)
# rather than the raw prompt. The metadata schema matches upstream
# T2V-CompBench V2 exactly, so scorers can mirror the upstream formulas
# 1:1.
#
#   * generative_numeracy   ─ box_threshold=0.4, text_threshold=0.25;
#                             single obj exact-match {0, 1}; two-obj 0.5
#                             per correct match. (upstream
#                             compbench_eval_numeracy.py)
#   * spatial_relationships ─ box_threshold=0.35, text_threshold=0.25;
#                             2D via bbox-center axis comparison; 3D via
#                             Depth-Anything mean-depth comparison;
#                             frame score combines spatial_score + prob,
#                             with the upstream {-2,-1,0,>0} bucket
#                             mapping for the video score. (upstream
#                             compbench_eval_spatial_relationships.py)
#   * motion_binding        ─ ``net = back_motion - fore_motion`` per
#                             object, direction match against d_1/d_2,
#                             two-obj 0.5 each, ``raw*0.8+0.2`` mapping.
#                             (upstream compbench_eval_motion_binding.py)
#                             We use GroundingDINO bbox + RAFT flow as a
#                             proxy for upstream's GroundingSAM + DOT
#                             (those checkpoints aren't reachable from
#                             this env); the **scoring structure** is
#                             unchanged. Scorer tag explicitly says
#                             "raft_proxy".

T2VCOMPBENCH_CV_DIMS = (
    "generative_numeracy",
    "spatial_relationships",
    "motion_binding",
)

_GD_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# Words → integers for generative_numeracy prompt parsing.
_WORD_NUMBERS = {
    "no": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a": 1, "an": 1, "single": 1, "lone": 1,
    "few": 3, "several": 3, "many": 5, "couple": 2, "pair": 2,
}

# Phrase relations the parser recognises (spatial_relationships).
_REL_PATTERNS = [
    # 2D
    ("left",   re.compile(r"(.+?)\s+(?:to\s+the\s+)?left(?:\s+side)?\s+of\s+(.+)", re.I)),
    ("right",  re.compile(r"(.+?)\s+(?:to\s+the\s+)?right(?:\s+side)?\s+of\s+(.+)", re.I)),
    ("above",  re.compile(r"(.+?)\s+(?:directly\s+)?above\s+(.+)", re.I)),
    ("below",  re.compile(r"(.+?)\s+(?:directly\s+)?below\s+(.+)", re.I)),
    ("above",  re.compile(r"(.+?)\s+on\s+top\s+of\s+(.+)", re.I)),
    ("below",  re.compile(r"(.+?)\s+under(?:neath)?\s+(.+)", re.I)),
    # 3D
    ("front",  re.compile(r"(.+?)\s+in\s+front\s+of\s+(.+)", re.I)),
    ("behind", re.compile(r"(.+?)\s+behind\s+(.+)", re.I)),
]


# ----- shared helpers ------------------------------------------------------ #

def _gd_detect(processor, model, image, queries: list[str],
               box_threshold: float = 0.30, text_threshold: float = 0.25,
               device: str = "cuda") -> dict[str, Any]:
    """Run GroundingDINO once on one PIL image with a list of phrase queries.

    Returns ``{"boxes": Tensor(N,4), "labels": list[str], "scores": Tensor(N)}``
    where boxes are in xyxy pixel coords at the input resolution.
    """
    import torch                        # noqa: PLC0415

    text_query = ". ".join(queries) + "."
    inputs = processor(images=image, text=text_query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]
    return {
        "boxes":  results.get("boxes"),
        "labels": [str(l).lower() for l in results.get("labels", [])],
        "scores": results.get("scores"),
    }


def _bbox_center(box) -> tuple[float, float]:
    """Convert [x1, y1, x2, y2] tensor / list to (cx, cy)."""
    return float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)


def _label_matches(label: str, noun: str) -> bool:
    """Loose match: any noun token appears in the label string."""
    label_l = label.lower()
    return any(tok in label_l for tok in noun.lower().split() if len(tok) >= 3)


# ----- generative_numeracy ------------------------------------------------ #

def _parse_count_noun(prompt: str) -> list[tuple[int, str]]:
    """Extract (count, noun-phrase) pairs from a prompt.

    Patterns matched (case-insensitive):
      * "<num> <noun>" — e.g. "three yellow ducks"
      * "<word-num> <noun>" — e.g. "two cats"
    Heuristic stops at conjunctions / commas to avoid run-on phrases.
    """
    tokens = re.findall(r"[A-Za-z]+|\d+", prompt.lower())
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        count: int | None = None
        if t.isdigit():
            count = int(t)
        elif t in _WORD_NUMBERS:
            count = _WORD_NUMBERS[t]
        if count is not None and count >= 1:
            # Take up to 3 following content words as the noun phrase.
            stop_words = {"the", "a", "an", "of", "and", "or", "in", "on",
                          "with", "next", "to"}
            noun_toks: list[str] = []
            j = i + 1
            while j < len(tokens) and len(noun_toks) < 3:
                tj = tokens[j]
                if tj in _NOUN_STOPWORDS_LITE or tj in stop_words or tj.isdigit():
                    break
                noun_toks.append(tj)
                j += 1
            if noun_toks:
                out.append((count, " ".join(noun_toks)))
            i = j
        else:
            i += 1
    return out


_NOUN_STOPWORDS_LITE: frozenset[str] = frozenset({
    "the", "of", "and", "or", "but", "in", "on", "at", "to", "with",
    "for", "from", "by", "as", "is", "are", "was", "were", "be",
})


class GenerativeNumeracyScorer:
    """Upstream-faithful per-frame:

      * 1 object  → 1.0 if detected count exactly equals expected, else 0.0
      * 2 objects → 0.5 per object whose detected count exactly equals expected

    Then mean over frames. Thresholds match upstream
    ``compbench_eval_numeracy.py`` (box=0.4, text=0.25).

    ``meta`` is the LLM-extracted dict
    ``{"objects": [...], "numbers": [...]}`` (see extract.py).
    """

    name = "generative_numeracy"
    BOX_THRESHOLD = 0.4
    TEXT_THRESHOLD = 0.25

    def __init__(self, device: str = "cuda"):
        from transformers import (  # noqa: PLC0415
            AutoModelForZeroShotObjectDetection, AutoProcessor,
        )
        self.device = device
        log.info("[t2vcompbench] loading Grounding-DINO for generative_numeracy")
        self.processor = AutoProcessor.from_pretrained(_GD_MODEL_ID)
        self.model = (AutoModelForZeroShotObjectDetection
                      .from_pretrained(_GD_MODEL_ID).to(device).eval())

    def score_one(self, frames: list, meta: dict[str, Any]) -> dict[str, Any]:
        objects = meta.get("objects") or []
        numbers = meta.get("numbers") or []
        if not objects or len(objects) != len(numbers):
            return {
                "generative_numeracy": 0.0,   # upstream scores absent objects as 0
                "objects": list(objects),
                "numbers": list(numbers),
                "skip_reason": "LLM extracted no (object, count) pair",
            }

        per_frame: list[float] = []
        per_frame_detail: list[dict] = []
        for f in frames:
            det = _gd_detect(self.processor, self.model, f, list(objects),
                             self.BOX_THRESHOLD, self.TEXT_THRESHOLD, self.device)
            # Count detections per object, using upstream label-matching.
            per_obj_hits = []
            for obj in objects:
                n_hits = sum(1 for lbl in det["labels"] if _label_matches(lbl, obj))
                per_obj_hits.append(n_hits)
            # Upstream score per frame:
            if len(objects) == 1:
                frame_score = 1.0 if per_obj_hits[0] == int(numbers[0]) else 0.0
            else:  # 2 objects
                ok = sum(1 for h, n in zip(per_obj_hits, numbers) if h == int(n))
                frame_score = 0.5 * ok
            per_frame.append(frame_score)
            per_frame_detail.append({
                "objects": list(objects),
                "expected": [int(n) for n in numbers],
                "detected": per_obj_hits,
                "frame_score": frame_score,
            })

        return {
            "generative_numeracy": float(sum(per_frame) / len(per_frame)),
            "objects": list(objects),
            "numbers": [int(n) for n in numbers],
            "per_frame": per_frame_detail,
        }


# ----- spatial_relationships ---------------------------------------------- #

def _parse_spatial_triple(prompt: str) -> tuple[str, str, str] | None:
    """Return (subject_a, relation, subject_b) for the first matching pattern."""
    for rel_name, pat in _REL_PATTERNS:
        m = pat.search(prompt)
        if m:
            a = m.group(1).strip().strip(".,;:!?")
            b = m.group(2).strip().strip(".,;:!?")
            # Trim noise words from start/end of each side
            a = re.sub(r"^(a|an|the|some)\s+", "", a, flags=re.I).strip()
            b = re.sub(r"^(a|an|the|some)\s+", "", b, flags=re.I).strip()
            # Cut B at first conjunction / comma
            b = re.split(r",|\sand\s|\sor\s", b, maxsplit=1)[0].strip()
            if a and b:
                return a, rel_name, b
    return None


class SpatialRelationshipsScorer:
    """Upstream-faithful spatial scoring.

    Per-frame score (matches ``compbench_eval_spatial_relationships.py``):
      * Both objects undetected → ``-2``
      * Exactly one object undetected → ``-1``
      * Both detected, **wrong** relation → ``0``
      * Both detected, **correct** relation →
          ``0.5 * spatial_score + 0.5 * prob_score`` where
            - 2D rel: ``spatial_score = 1 - IoU``
            - 3D rel: ``spatial_score = IoU``
            - ``prob_score = 0.5 * conf_A + 0.5 * conf_B``

    Per-video aggregation (also upstream): mean of per-frame scores, then
    bucket map:
      * score < -1  →  0
      * -1 ≤ score ≤ 0 → 0.2 + 0.2 * (score + 1)   (linear in [-1, 0] → [0.2, 0.4])
      * score > 0   →  0.6 * score + 0.4

    Thresholds: ``box_threshold=0.35``, ``text_threshold=0.25``.

    ``meta`` is the LLM-extracted dict
    ``{"object_1": ..., "spatial": ..., "object_2": ...}``.
    """

    name = "spatial_relationships"
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25

    def __init__(self, device: str = "cuda"):
        from transformers import (  # noqa: PLC0415
            AutoModelForZeroShotObjectDetection, AutoProcessor,
        )
        self.device = device
        log.info("[t2vcompbench] loading Grounding-DINO for spatial_relationships")
        self.gd_processor = AutoProcessor.from_pretrained(_GD_MODEL_ID)
        self.gd_model = (AutoModelForZeroShotObjectDetection
                         .from_pretrained(_GD_MODEL_ID).to(device).eval())
        self._depth_pipeline = None

    def _depth(self, frame):
        if self._depth_pipeline is None:
            from transformers import pipeline  # noqa: PLC0415
            log.info("[t2vcompbench] loading Depth-Anything-V2 (lazy)")
            self._depth_pipeline = pipeline(
                "depth-estimation", model=_DEPTH_MODEL_ID, device=self.device,
            )
        return self._depth_pipeline(frame)["depth"]

    def score_one(self, frames: list, meta: dict[str, Any]) -> dict[str, Any]:
        a = meta.get("object_1")
        rel = meta.get("spatial")
        b = meta.get("object_2")
        if not (a and b and rel):
            return {
                "spatial_relationships": 0.0,
                "triple": {"object_1": a, "spatial": rel, "object_2": b},
                "skip_reason": "LLM extracted no spatial triple",
            }

        per_frame_raw: list[float] = []
        per_frame_detail: list[dict] = []
        for f in frames:
            raw, detail = self._score_frame(f, a, rel, b)
            per_frame_raw.append(raw)
            per_frame_detail.append(detail)
        mean_raw = float(sum(per_frame_raw) / len(per_frame_raw))
        video_score = _spatial_bucket_map(mean_raw)
        return {
            "spatial_relationships": video_score,
            "mean_raw":              mean_raw,
            "triple":                {"object_1": a, "spatial": rel, "object_2": b},
            "per_frame":             per_frame_detail,
        }

    def _score_frame(self, frame, a: str, rel: str, b: str):
        det = _gd_detect(self.gd_processor, self.gd_model, frame, [a, b],
                         self.BOX_THRESHOLD, self.TEXT_THRESHOLD, self.device)
        boxes = det["boxes"]
        labels = det["labels"]
        scores = det["scores"]

        def pick(noun: str):
            cands = [(i, lbl) for i, lbl in enumerate(labels)
                     if _label_matches(lbl, noun)]
            if not cands:
                return None
            cands.sort(key=lambda t: -float(scores[t[0]]))
            return int(cands[0][0])

        ia = pick(a) if labels else None
        ib = pick(b) if labels else None

        if ia is None and ib is None:
            return -2.0, {"reason": "both objects undetected"}
        if ia is None or ib is None:
            return -1.0, {"reason": "one object undetected",
                          "has_a": ia is not None, "has_b": ib is not None}

        box_a = boxes[ia].tolist()
        box_b = boxes[ib].tolist()
        ax, ay = _bbox_center(box_a)
        bx, by = _bbox_center(box_b)
        iou = _bbox_iou(box_a, box_b)
        prob_score = 0.5 * float(scores[ia]) + 0.5 * float(scores[ib])

        # 2D relations
        relation_correct = None
        if rel == "left":
            relation_correct = ax < bx
            spatial_score = 1 - iou
        elif rel == "right":
            relation_correct = ax > bx
            spatial_score = 1 - iou
        elif rel == "above":
            relation_correct = ay < by
            spatial_score = 1 - iou
        elif rel == "below":
            relation_correct = ay > by
            spatial_score = 1 - iou
        elif rel in {"front", "behind"}:
            depth_map = self._depth(frame)
            import numpy as np            # noqa: PLC0415
            d_arr = np.asarray(depth_map, dtype="float32")
            da = float(_mean_depth_in_box(d_arr, box_a))
            db = float(_mean_depth_in_box(d_arr, box_b))
            # Depth Anything: larger = closer to camera (in front)
            relation_correct = (da > db) if rel == "front" else (da < db)
            spatial_score = iou       # 3D uses IoU directly per upstream
        else:
            return 0.0, {"reason": f"unknown relation {rel!r}"}

        if not relation_correct:
            return 0.0, {"reason": "wrong relation", "rel": rel,
                         "iou": iou, "ax_ay": [ax, ay], "bx_by": [bx, by]}
        return 0.5 * spatial_score + 0.5 * prob_score, {
            "rel": rel, "iou": iou, "spatial_score": spatial_score,
            "prob_score": prob_score,
        }


def _spatial_bucket_map(mean_raw: float) -> float:
    """Upstream's piecewise map from per-frame mean raw → video score."""
    if mean_raw < -1:
        return 0.0
    if mean_raw <= 0:
        # [-1, 0] → [0.2, 0.4]
        return 0.2 + 0.2 * (mean_raw + 1)
    return 0.6 * mean_raw + 0.4


def _bbox_iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0


def _mean_depth_in_box(depth_arr, box) -> float:
    """Mean depth inside an xyxy bounding box (clipped to image)."""
    H, W = depth_arr.shape[:2]
    x1, y1, x2, y2 = (int(max(0, min(W - 1, v))) for v in box)
    if x2 <= x1 or y2 <= y1:
        return float(depth_arr.mean())
    return float(depth_arr[y1:y2, x1:x2].mean())


# ----- motion_binding ----------------------------------------------------- #

def _first_content_noun(prompt: str) -> str | None:
    """Heuristic: first content word that isn't a stopword / common verb."""
    tokens = re.findall(r"[A-Za-z]+", prompt.lower())
    BLOCK = _NOUN_STOPWORDS_LITE | {
        "small", "big", "large", "tall", "short", "old", "young", "red", "blue",
        "green", "yellow", "white", "black",  # adjectives
    }
    for t in tokens:
        if len(t) < 3 or t in BLOCK:
            continue
        if t in _WORD_NUMBERS:
            continue
        return t
    return None


_SAM_MODEL_ID = "facebook/sam-vit-base"
_COTRACKER_REPO = "facebookresearch/co-tracker"
_COTRACKER_MODEL = "cotracker3_offline"


class MotionBindingScorer:
    """Upstream-faithful motion_binding: GroundingDINO bbox → SAM mask →
    Co-Tracker3 dense tracking → ``net = back - fore`` per object →
    direction match → upstream ``raw * 0.8 + 0.2`` mapping.

    Mirrors ``compbench_eval_motion_binding.py``:
      * Frame 0: detect each object with GroundingDINO; use the bbox as
        a prompt for SAM to get a precise binary mask per object.
      * Whole 16-frame clip: Co-Tracker3 (DOT's backbone) tracks a regular
        grid of query points across time.
      * For each object: split track endpoints into foreground (inside
        the SAM mask) and background (outside). Compute mean (Δx, Δy).
        ``net_left = back_x - fore_x`` and ``net_up = back_y - fore_y``
        (positive = subject moved left / up in pixel space, with camera
        motion cancelled).
      * Direction-aware score:
            left:  max(0,  net_left) / W
            right: max(0, -net_left) / W
            up:    max(0,  net_up)   / H
            down:  max(0, -net_up)   / H
      * Multi-object: average per-object scores.
      * Final: ``score_tmp = raw_score * 0.8 + 0.2`` (upstream model_score).

    Why this differs from a full DOT clone: DOT layers extra optical-flow
    refinement on top of Co-Tracker3. We use Co-Tracker3 standalone (its
    HF-hub'd ``cotracker3_offline`` model) — semantics unchanged, fidelity
    slightly lower than DOT but within the same regime. Upstream's tag is
    preserved as ``cv:gd_sam_cotracker3``.
    """

    name = "motion_binding"
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.20
    GRID_SIZE = 20   # 400 query points across the frame
    DOWNSAMPLE_H = 360
    DOWNSAMPLE_W = 640

    def __init__(self, device: str = "cuda"):
        from transformers import (  # noqa: PLC0415
            AutoModelForZeroShotObjectDetection, AutoProcessor,
            SamModel, SamProcessor,
        )
        import torch                    # noqa: PLC0415
        self.device = device

        log.info("[t2vcompbench] loading Grounding-DINO for motion_binding")
        self.gd_processor = AutoProcessor.from_pretrained(_GD_MODEL_ID)
        self.gd_model = (AutoModelForZeroShotObjectDetection
                         .from_pretrained(_GD_MODEL_ID).to(device).eval())

        log.info("[t2vcompbench] loading SAM for motion_binding masks")
        self.sam_processor = SamProcessor.from_pretrained(_SAM_MODEL_ID)
        self.sam_model = SamModel.from_pretrained(_SAM_MODEL_ID).to(device).eval()

        log.info("[t2vcompbench] loading Co-Tracker3 (offline)")
        # torch.hub caches the python source under ~/.cache/torch/hub/<repo>;
        # the .pth checkpoint also lands in ~/.cache/torch/hub/checkpoints/.
        self.tracker = torch.hub.load(
            _COTRACKER_REPO, _COTRACKER_MODEL,
            source="github", trust_repo=True,
        ).to(device).eval()

    # ----- per-frame helpers ----------------------------------------- #

    def _gd_box_for(self, frame, noun):
        det = _gd_detect(self.gd_processor, self.gd_model, frame, [noun],
                         self.BOX_THRESHOLD, self.TEXT_THRESHOLD, self.device)
        if det["boxes"] is None or len(det["boxes"]) == 0:
            return None
        idx = int(det["scores"].argmax().item())
        return det["boxes"][idx].tolist()

    def _sam_mask_for(self, frame, box):
        """Return a (H, W) bool ndarray mask for the given xyxy box."""
        import torch                    # noqa: PLC0415
        import numpy as np              # noqa: PLC0415

        inputs = self.sam_processor(
            frame, input_boxes=[[box]], return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0][0]   # (3, H, W) — SAM returns 3 candidate masks
        scores = outputs.iou_scores[0, 0]
        best = int(scores.argmax().item())
        return masks[best].numpy().astype(bool)

    # ----- video tracking -------------------------------------------- #

    def _track_video(self, frames):
        """Co-Tracker3 offline: dense grid tracks over the whole clip.

        Returns:
          tracks (T, N, 2)  — pixel coords of N grid points across T frames
          visibility (T, N) — bool visibility per point per frame
          (down_h, down_w) — resolution at which tracking ran
        """
        import torch                    # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
        from torchvision import transforms as T  # noqa: PLC0415

        to_tensor = T.ToTensor()
        stack = torch.stack([to_tensor(f) for f in frames])  # (T, 3, H, W) in [0,1]
        stack = F.interpolate(
            stack, size=(self.DOWNSAMPLE_H, self.DOWNSAMPLE_W),
            mode="bilinear", align_corners=False,
        )
        video = (stack * 255.0).unsqueeze(0).to(self.device)  # (1, T, 3, H', W')
        with torch.no_grad():
            tracks, vis = self.tracker(video, grid_size=self.GRID_SIZE)
        # tracks: (1, T, N, 2), vis: (1, T, N)
        return tracks[0].cpu().numpy(), vis[0].cpu().numpy()

    # ----- score one (object, direction) pair ------------------------ #

    def _object_score(self, frame0, obj, direction, tracks, vis):
        """Mirror of upstream object_score() for one (obj, d) pair."""
        import numpy as np              # noqa: PLC0415

        box0 = self._gd_box_for(frame0, obj)
        if box0 is None:
            return None, {"reason": "GD did not detect object in frame 0",
                          "object": obj}
        mask0 = self._sam_mask_for(frame0, box0)
        # Resize mask to tracker resolution.
        from PIL import Image as _Img    # noqa: PLC0415
        mask_pil = _Img.fromarray((mask0 * 255).astype("uint8"))
        mask_small = np.asarray(
            mask_pil.resize((self.DOWNSAMPLE_W, self.DOWNSAMPLE_H), _Img.NEAREST)
        ) > 127

        # Assign each query point to fg/bg by its frame-0 location.
        pt0 = tracks[0]                                  # (N, 2): (x, y)
        pt_last = tracks[-1]                             # (N, 2)
        valid = vis[0] & vis[-1]                          # (N,)
        # bound-check
        xs = pt0[:, 0].astype(int).clip(0, self.DOWNSAMPLE_W - 1)
        ys = pt0[:, 1].astype(int).clip(0, self.DOWNSAMPLE_H - 1)
        in_mask = mask_small[ys, xs]                      # (N,) bool
        fg = valid & in_mask
        bg = valid & ~in_mask
        if not fg.any() or not bg.any():
            return None, {"reason": "fg or bg query set empty",
                          "n_fg": int(fg.sum()), "n_bg": int(bg.sum())}

        # Mean displacement vectors.
        d_fg = (pt_last[fg] - pt0[fg]).mean(axis=0)      # (2,) [dx, dy]
        d_bg = (pt_last[bg] - pt0[bg]).mean(axis=0)
        # Upstream: net_left = back_x - fore_x ; net_up = back_y - fore_y.
        # (image y grows downward, but the sign convention used by upstream
        # treats positive net_up = subject moved 'up' in pixel space.)
        net_left = float(d_bg[0] - d_fg[0])
        net_up = float(d_bg[1] - d_fg[1])
        # Note: pixel y increases downward, so subject "moving up" in
        # world space means d_fg[1] < d_bg[1] → net_up > 0.

        W, H = self.DOWNSAMPLE_W, self.DOWNSAMPLE_H
        d = direction
        if d == "left":
            raw = max(0.0, net_left) / W
        elif d == "right":
            raw = max(0.0, -net_left) / W
        elif d == "up":
            raw = max(0.0, net_up) / H
        elif d == "down":
            raw = max(0.0, -net_up) / H
        else:
            return None, {"reason": f"unknown direction {d!r}", "object": obj}
        return float(min(1.0, raw)), {
            "object": obj, "d": d,
            "net_left": net_left, "net_up": net_up,
            "n_fg_pts": int(fg.sum()), "n_bg_pts": int(bg.sum()),
        }

    # ----- main entry ------------------------------------------------ #

    def score_one(self, frames: list, meta: dict[str, Any]) -> dict[str, Any]:
        obj1, d1 = meta.get("object_1"), meta.get("d_1")
        obj2, d2 = meta.get("object_2"), meta.get("d_2")

        if not obj1 or not d1:
            # No motion target at all -> upstream returns -1 for undetected;
            # we keep 0 as the headline (after final mapping → 0.2).
            return {
                "motion_binding": 0.2,
                "skip_reason": "LLM extracted no (object, direction) pair",
                "meta_used":    {"object_1": obj1, "d_1": d1,
                                 "object_2": obj2, "d_2": d2},
            }

        if len(frames) < 4:
            raise ValueError("motion_binding needs >= 4 frames; got "
                             f"{len(frames)}")

        # Track once for the whole clip.
        tracks, vis = self._track_video(frames)

        # Score per (object, direction).
        scores: list[float] = []
        details: list[dict] = []
        s1, d1_detail = self._object_score(frames[0], obj1, d1, tracks, vis)
        details.append(d1_detail)
        if s1 is not None:
            scores.append(s1)
        if obj2 and d2:
            s2, d2_detail = self._object_score(frames[0], obj2, d2, tracks, vis)
            details.append(d2_detail)
            if s2 is not None:
                scores.append(s2)

        if not scores:
            raw_score = 0.0
        elif obj2 and d2 and len(scores) == 2:
            raw_score = float(sum(scores) / 2)
        else:
            raw_score = float(scores[0])

        # Upstream model_score mapping: raw_score * 0.8 + 0.2
        final = raw_score * 0.8 + 0.2
        return {
            "motion_binding": float(final),
            "raw_score":      raw_score,
            "per_object":     details,
            "meta_used":      {"object_1": obj1, "d_1": d1,
                               "object_2": obj2, "d_2": d2},
        }


def score_cv_dim(
    cv_scorer: Any, frames: list, prompt_text: str, dim: str,
) -> dict[str, Any]:
    """Dispatch one frames+prompt call into the appropriate CV scorer."""
    if dim == "generative_numeracy":
        return cv_scorer.score_one(frames, prompt_text)
    if dim == "spatial_relationships":
        return cv_scorer.score_one(frames, prompt_text)
    if dim == "motion_binding":
        return cv_scorer.score_one(frames, prompt_text)
    raise KeyError(f"no CV scorer for dim={dim!r}")
