"""Parse the JSON response of a semantics-axis VLM judge prompt.

Each axis prompt emits a JSON object keyed by a Chinese axis name; the score
lives at ``score_5`` (1-5 integer). Narrow axes also carry ``不适用`` (scope
mismatch / N/A) and a ``汇总`` block with S1/S2/S3 severity counts.
"""
from __future__ import annotations

import json
import re
from typing import Any


def _extract_json(text: str) -> dict | None:
    """Pull the first valid JSON object out of a (possibly fenced) response."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start : i + 1])
                except Exception:
                    return None
    return None


def _find_key(obj: Any, key: str) -> Any:
    """Depth-first search for the first occurrence of ``key``."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def parse_axis_response(text: str) -> tuple[float | None, dict]:
    """Return ``(score_5, meta)``.

    ``score_5`` is a 1-5 float, or ``None`` if the response is unparseable.
    ``meta`` carries the N/A flag, severity counts and the verdict reason.
    """
    meta: dict = {"raw_text": (text or "")[:600]}
    data = _extract_json(text)
    if data is None:
        m = re.search(r'score_5["\s:]+([1-5])', text or "")
        return (float(m.group(1)), meta) if m else (None, meta)

    score = _find_key(data, "score_5")
    na = _find_key(data, "不适用")
    summary = _find_key(data, "汇总")
    reason = _find_key(data, "总评理由")
    if na is not None:
        meta["not_applicable"] = bool(na)
    if isinstance(summary, dict):
        meta["severity"] = {k: summary.get(k) for k in ("S1_count", "S2_count", "S3_count")}
    if reason:
        meta["reason"] = str(reason)[:300]
    if score is None:
        return None, meta
    try:
        s = float(score)
    except Exception:
        return None, meta
    return (s if 1.0 <= s <= 5.0 else None), meta
