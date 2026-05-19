"""LLM-based extraction of T2V-CompBench's meta_data structure from a prompt.

Upstream T2V-CompBench V2 ships **structured metadata** alongside each
prompt:

  * ``meta_data/generative_numeracy.json``   → ``objects`` + ``numbers``
  * ``meta_data/spatial_relationships.json`` → ``object_1`` + ``spatial`` + ``object_2``
  * ``meta_data/motion_binding.json``        → ``object_1`` + ``d_1`` + (``object_2`` + ``d_2``)

The upstream evaluators **never parse the prompt themselves** — they only
read from these files. For user-supplied prompts that lack pre-built
metadata, we recreate the same structure by issuing **one LLM call per
(prompt, dim) pair**. The returned schema matches upstream exactly so the
downstream scorers can stay 1:1 with the upstream formulas.

The LLM is whichever judge the adapter is given (we recommend Qwen3-32B
text endpoint at :8004; Gemma-4-31B-IT can also handle text-only chats).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from videvalkit.utils.video import strip_code_fences

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Prompt templates  (one per CV dim)
# --------------------------------------------------------------------------- #

_NUMERACY_TMPL = """\
You are a strict prompt analyzer for a text-to-video benchmark.

Given the prompt below, extract every concrete countable object together
with its expected count. Map number words to integers
(one->1, two->2, three->3, ..., a/an/single->1, several/few->3, many->5).

Prompt: "{prompt}"

Output **one** JSON object on a single line, no extra text:
{{"objects": ["<noun1>", "<noun2>"], "numbers": [<int1>, <int2>]}}

Rules:
- Up to 2 distinct objects (upstream benchmark scores 1 or 2 obj).
- Each "noun" is a SHORT noun phrase usable as a Grounding-DINO query
  (e.g. "yellow duck", "red apple"). Avoid full clauses.
- Skip abstract / mass nouns (water, light, weather, ambience).
- If the prompt has no countable noun at all, return empty arrays.
"""

_SPATIAL_TMPL = """\
You are a strict prompt analyzer for a text-to-video benchmark.

Extract the FIRST spatial relation in the prompt: object_1 in the
specified relation to object_2.

Prompt: "{prompt}"

Output **one** JSON object on a single line, no extra text:
{{"object_1": "<noun phrase>", "spatial": "<one of: left|right|above|below|front|behind>", "object_2": "<noun phrase>"}}

Rules:
- "spatial" MUST be one of: "left", "right", "above", "below", "front",
  "behind". Map "on top of"→"above", "underneath"→"below", "in front
  of"→"front".
- If the prompt has no spatial relation, output
  {{"object_1": null, "spatial": null, "object_2": null}}.
- Each noun phrase is short and usable as a Grounding-DINO query.
"""

_MOTION_TMPL = """\
You are a strict prompt analyzer for a text-to-video benchmark.

Extract the moving subject(s) and the direction each one moves. Direction
is from the viewer's perspective.

Prompt: "{prompt}"

Output **one** JSON object on a single line, no extra text:
{{"object_1": "<noun phrase>", "d_1": "<left|right|up|down>",
  "object_2": "<noun phrase or null>", "d_2": "<left|right|up|down or null>"}}

Rules:
- d must be one of "left", "right", "up", "down" (or null if no second object).
- "moves from left to right" -> "right". "rises" -> "up". "falls" -> "down".
  "moves to the right" -> "right".
- If no clear motion direction in the prompt, output
  {{"object_1": null, "d_1": null, "object_2": null, "d_2": null}}.
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _content(resp: Any) -> str:
    """OpenAICompatibleVLMJudge / vLLM responses both surface 'content' at top."""
    if isinstance(resp, dict):
        if isinstance(resp.get("content"), str):
            return resp["content"]
        try:
            msg = resp["choices"][0]["message"]["content"]
            return str(msg) if not isinstance(msg, list) else "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in msg
            )
        except (KeyError, IndexError, TypeError):
            pass
    raise RuntimeError(f"unexpected llm response shape: {str(resp)[:200]}")


def _json_from(content: str) -> dict[str, Any]:
    """Robust JSON extraction (handles code fences, leading prose)."""
    s = strip_code_fences(content).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    if "{" in s and "}" in s:
        cand = s[s.index("{"):s.rindex("}") + 1]
        return json.loads(cand)
    raise ValueError(f"no JSON in LLM response: {s[:200]!r}")


def _ask_llm(llm_judge: Any, template: str, prompt: str) -> dict[str, Any]:
    user = template.format(prompt=prompt)
    resp = llm_judge.chat_text(user=user, max_tokens=256, temperature=0.0)
    return _json_from(_content(resp))


# ─── Deterministic heuristic extractors (no-LLM fallback) ────────────── #
# Used when judge=None: produce upstream-shaped meta from raw prompt text
# via regex so the CV scorers can still run.

_WORD_NUMS = {
    "no": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a": 1, "an": 1, "single": 1, "lone": 1, "couple": 2, "pair": 2,
    "few": 3, "several": 3, "many": 5,
}
_STOPWORDS = {"the", "of", "and", "or", "in", "on", "with", "next", "to",
              "for", "from", "by", "as", "at"}


def _heuristic_numeracy(prompt: str) -> dict[str, Any]:
    """Heuristic ``{objects, numbers}`` parsing from prompt text."""
    tokens = re.findall(r"[A-Za-z]+|\d+", prompt.lower())
    objs: list[str] = []
    nums: list[int] = []
    i = 0
    while i < len(tokens) and len(objs) < 2:
        t = tokens[i]
        cnt: int | None = None
        if t.isdigit():
            cnt = int(t)
        elif t in _WORD_NUMS:
            cnt = _WORD_NUMS[t]
        if cnt is not None and cnt >= 1:
            j = i + 1
            noun_toks: list[str] = []
            while j < len(tokens) and len(noun_toks) < 3:
                tj = tokens[j]
                if tj in _STOPWORDS or tj.isdigit():
                    break
                noun_toks.append(tj)
                j += 1
            if noun_toks:
                objs.append(" ".join(noun_toks))
                nums.append(cnt)
            i = j
        else:
            i += 1
    return {"objects": objs, "numbers": nums}


_SPATIAL_PATTERNS = [
    ("left",   re.compile(r"(.+?)\s+(?:to\s+the\s+)?left(?:\s+side)?\s+of\s+(.+)", re.I)),
    ("right",  re.compile(r"(.+?)\s+(?:to\s+the\s+)?right(?:\s+side)?\s+of\s+(.+)", re.I)),
    ("above",  re.compile(r"(.+?)\s+(?:directly\s+)?above\s+(.+)", re.I)),
    ("below",  re.compile(r"(.+?)\s+(?:directly\s+)?below\s+(.+)", re.I)),
    ("above",  re.compile(r"(.+?)\s+on\s+top\s+of\s+(.+)", re.I)),
    ("below",  re.compile(r"(.+?)\s+under(?:neath)?\s+(.+)", re.I)),
    ("front",  re.compile(r"(.+?)\s+in\s+front\s+of\s+(.+)", re.I)),
    ("behind", re.compile(r"(.+?)\s+behind\s+(.+)", re.I)),
]


def _heuristic_spatial(prompt: str) -> dict[str, Any]:
    for rel, pat in _SPATIAL_PATTERNS:
        m = pat.search(prompt)
        if m:
            a = re.sub(r"^(a|an|the|some)\s+", "", m.group(1).strip(),
                       flags=re.I).strip().strip(".,;:!?")
            b = re.split(r",|\sand\s|\sor\s", m.group(2), maxsplit=1)[0]
            b = re.sub(r"^(a|an|the|some)\s+", "", b.strip(),
                       flags=re.I).strip().strip(".,;:!?")
            if a and b:
                return {"object_1": a, "spatial": rel, "object_2": b}
    return {"object_1": None, "spatial": None, "object_2": None}


_DIRECTION_PATTERNS = {
    "left":  re.compile(r"\b(left|leftward|leftwards)\b", re.I),
    "right": re.compile(r"\b(right|rightward|rightwards)\b", re.I),
    "up":    re.compile(r"\b(up|upward|upwards|rises|rising|ascending)\b", re.I),
    "down":  re.compile(r"\b(down|downward|downwards|falls|falling|descending)\b", re.I),
}


def _heuristic_motion(prompt: str) -> dict[str, Any]:
    """First content noun + direction word."""
    tokens = re.findall(r"[A-Za-z]+", prompt.lower())
    BLOCK = _STOPWORDS | _WORD_NUMS.keys() | {
        "small", "big", "large", "tall", "short", "old", "young",
        "red", "blue", "green", "yellow", "white", "black",
    }
    obj = next((t for t in tokens if len(t) >= 3 and t not in BLOCK), None)
    direction = None
    for d, pat in _DIRECTION_PATTERNS.items():
        if pat.search(prompt):
            direction = d; break
    return {
        "object_1": obj, "d_1": direction,
        "object_2": None, "d_2": None,
    }


# --------------------------------------------------------------------------- #
# Public extractors
# --------------------------------------------------------------------------- #

# Synonym map for spatial relations from prompt to upstream canon.
_SPATIAL_CANON = {
    "left": "left", "right": "right",
    "above": "above", "on": "above", "on_top_of": "above", "ontop": "above",
    "below": "below", "under": "below", "underneath": "below",
    "front": "front", "in_front": "front", "in_front_of": "front",
    "behind": "behind",
}

_DIRECTION_CANON = {"left", "right", "up", "down"}


def extract_numeracy(llm_judge: Any, prompt: str) -> dict[str, Any]:
    """Returns ``{"objects": [...], "numbers": [...]}``, matching upstream.

    Falls back to a deterministic regex-based extractor when ``llm_judge``
    is ``None`` or the LLM call fails.
    """
    if llm_judge is None:
        return _heuristic_numeracy(prompt)
    try:
        meta = _ask_llm(llm_judge, _NUMERACY_TMPL, prompt)
    except Exception as e:
        log.warning("t2vcompbench/extract.numeracy LLM failed: %s; falling back to heuristic", e)
        return _heuristic_numeracy(prompt)
    objs = [str(x).strip() for x in (meta.get("objects") or []) if x]
    nums = []
    for x in (meta.get("numbers") or []):
        try:
            nums.append(int(x))
        except (TypeError, ValueError):
            continue
    # Pair-align: keep only as many objects as we have numbers (and vice versa)
    k = min(len(objs), len(nums))
    return {"objects": objs[:k], "numbers": nums[:k]}


def extract_spatial(llm_judge: Any, prompt: str) -> dict[str, Any]:
    """Returns ``{"object_1": ..., "spatial": ..., "object_2": ...}``.

    Heuristic fallback used when no LLM judge is available.
    """
    if llm_judge is None:
        return _heuristic_spatial(prompt)
    try:
        meta = _ask_llm(llm_judge, _SPATIAL_TMPL, prompt)
    except Exception as e:
        log.warning("t2vcompbench/extract.spatial LLM failed: %s; falling back to heuristic", e)
        return _heuristic_spatial(prompt)
    rel_raw = meta.get("spatial")
    rel = _SPATIAL_CANON.get(
        str(rel_raw).lower().strip().replace(" ", "_") if rel_raw else "",
        None,
    )
    return {
        "object_1": (meta.get("object_1") or None),
        "spatial":  rel,
        "object_2": (meta.get("object_2") or None),
    }


def extract_motion(llm_judge: Any, prompt: str) -> dict[str, Any]:
    """Returns ``{"object_1": ..., "d_1": ..., "object_2": ..., "d_2": ...}``.

    Heuristic fallback used when no LLM judge is available.
    """
    if llm_judge is None:
        return _heuristic_motion(prompt)
    try:
        meta = _ask_llm(llm_judge, _MOTION_TMPL, prompt)
    except Exception as e:
        log.warning("t2vcompbench/extract.motion LLM failed: %s; falling back to heuristic", e)
        return _heuristic_motion(prompt)
    def _dir(d):
        if d is None:
            return None
        d = str(d).lower().strip()
        return d if d in _DIRECTION_CANON else None
    return {
        "object_1": (meta.get("object_1") or None),
        "d_1":      _dir(meta.get("d_1")),
        "object_2": (meta.get("object_2") or None),
        "d_2":      _dir(meta.get("d_2")),
    }
