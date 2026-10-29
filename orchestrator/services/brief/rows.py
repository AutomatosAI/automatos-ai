"""PRD-268 §6.11.2 — what every brief section's rows share: the fold, the cuts, the actions.

* **The fold**: a section shows 5 rows and folds the rest behind "Show N more"; a section only
  one row over shows whole (fold at 5, never at 6). A reader reads ``ROWS_READ`` rows, enough
  to know which.
* **The cuts**: a title is its first line, at most ``TITLE_CHARS``; a reason at most
  ``WHY_CHARS``; each also at most ``BYTES_PER_CHAR`` UTF-8 bytes a character, so a title of
  emoji or CJK is cut shorter than one of letters and a brief of the most rows still fits its
  ``CHAT_PART_MAX_BYTES`` part (``services/chat_parts.py``). Titles and reasons that came from
  a model or a person are data: they are quoted on the card, never followed.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from services.brief.model import BriefAction, BriefOpens, BriefRow
from services.brief.strings import say

__all__ = [
    "BYTES_PER_CHAR", "FOLD_AT", "ROWS_READ", "TITLE_CHARS", "WHY_CHARS", "action", "cut", "fold", "iso", "opens",
    "title_of",
]

FOLD_AT = 5
ROWS_READ = FOLD_AT + 2
TITLE_CHARS = 120
WHY_CHARS = 200
BYTES_PER_CHAR = 2
ELLIPSIS = "…"
ELLIPSIS_BYTES = len(ELLIPSIS.encode("utf-8"))


def fold(rows: Sequence[BriefRow], count: int) -> Tuple[List[BriefRow], int]:
    """The rows a section shows, and how many of its ``count`` it folds away."""
    shown = list(rows) if len(rows) <= FOLD_AT + 1 else list(rows[:FOLD_AT])
    return shown, max(int(count) - len(shown), 0)


def cut(text: Any, limit: int) -> str:
    """``text`` on one line, its spaces collapsed, cut to ``limit`` characters and to
    ``limit * BYTES_PER_CHAR`` UTF-8 bytes (a character is never split)."""
    words = " ".join(str(text or "").split())
    budget = limit * BYTES_PER_CHAR
    if len(words) <= limit and len(words.encode("utf-8")) <= budget:
        return words
    kept = words[: limit - len(ELLIPSIS)].encode("utf-8")[: budget - ELLIPSIS_BYTES]
    return kept.decode("utf-8", "ignore").rstrip() + ELLIPSIS


def title_of(text: Any, language: str) -> str:
    """A row's title: the first line of ``text``, cut, or "Untitled" when there is none."""
    lines = str(text or "").strip().splitlines()
    return cut(lines[0], TITLE_CHARS) if lines and lines[0].strip() else say(language, "untitled")


def iso(value: Any) -> Optional[str]:
    """A time as ISO 8601; a string as it is; None as None."""
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def action(verb: str, language: str, *, done: bool = False) -> BriefAction:
    """The row action ``verb`` in ``language``; ``done`` adds what it says once pressed."""
    done_label = say(language, f"action_{verb}_done") if done else None
    return BriefAction(verb=verb, label=say(language, f"action_{verb}"), done_label=done_label)


def opens(kind: str, ident: Any) -> Optional[BriefOpens]:
    """Where a row goes, or None when it has no id to go to."""
    return BriefOpens(type=kind, id=str(ident)) if ident not in (None, "") else None
