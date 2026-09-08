"""PRD-238 S3 — one honest line per tool call.

The stream's ``tool-end`` frame used to carry only success/duration, so the
chat could show *that* a tool ran but never *what came back*. This turns a
tool's result into a short, single-line summary for the activity trail.
Never raw payloads: the summary is derived from the result's own headline
fields and capped, so nothing large or sensitive rides the wire by accident.

Pure, stdlib only.
"""
from __future__ import annotations

import re
from typing import Any, Optional

SUMMARY_MAX_CHARS = 120
SKIPPED_SUMMARY = "Skipped — already ran with the same input"

#: Fields that usually hold a human-readable headline, in preference order.
_HEADLINE_KEYS = ("summary", "message", "error", "detail", "title", "status_text")
#: Fields that hold a countable collection.
_COLLECTION_KEYS = ("results", "items", "tasks", "agents", "documents", "matches", "rows", "data")


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _cap(text: str, limit: int) -> str:
    text = _one_line(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def tool_result_summary(result: Any, limit: int = SUMMARY_MAX_CHARS) -> Optional[str]:
    """A ≤``limit``-char line describing ``result``, or None when nothing honest fits."""
    if result is None:
        return None
    if isinstance(result, str):
        return _cap(result, limit) or None
    if not isinstance(result, dict):
        return None
    for key in _HEADLINE_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return _cap(value, limit)
    for key in _COLLECTION_KEYS:
        value = result.get(key)
        if isinstance(value, (list, tuple)):
            noun = key if len(value) != 1 else key.rstrip("s")
            return _cap(f"{len(value)} {noun}", limit)
    for key in ("count", "total"):
        value = result.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return _cap(f"{value} {key}", limit)
    if result.get("success") is False:
        return "Failed"
    if result.get("success") is True:
        return "Done"
    return None
