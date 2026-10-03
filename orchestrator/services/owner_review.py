"""F180 (reopened 2 Oct, night 6): when the owner says the work waits for them, it does.

Night 6, iterations 11-14, on fb675857b. "Drafts only - nothing gets sent without
me" (#1205, #1206) and "set to wait for me" (#1209) were filed with no review mode,
so each closed Done on its own while Auto told the owner it would wait. When Auto
did ask, it said "human_review", which the tool refused (#1214's first try). The
persona: "lets it finish without you, however clearly you said you wanted to see
it first."

This module reads the owner's own words for a request to see the work before it
is done. A request the same words decline ("no need to wait for me") is not one.
Pure: the caller decides which words are the owner's.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

# Ways night 6's owner, and owners generally, ask to see the work first.
_ASKS = re.compile(
    r"\bwait(?:s|ing)?\s+for\s+me\b"
    r"|\b(?:set|put)\s+(?:it\s+|them\s+)?to\s+wait\b"
    r"|\bwithout\s+me\b"
    r"|\bdrafts?\s+only\b"
    r"|\b(?:for|needs?|wants?|with|after)\s+my\s+(?:review|approval|sign[\s-]?off|ok|okay|go[\s-]?ahead)\b"
    r"|\b(?:before|until)\s+(?:it['’]?s|it\s+is|they['’]?re|they\s+are|it\s+gets|anything\s+(?:is|gets))\s+"
    r"(?:done|finished|marked|sent|posted|printed|published|final|live)\b"
    # Wanting to see it counts only with "first" or "before": "I want to see the report" is a request to read.
    r"|\b(?:i\s+(?:want|need|['’]d\s+like|would\s+like)\s+to|let\s+me)\s+(?:see|look\s+at|check|read)\s+"
    r"(?:it|them|that|this)\s+(?:first|before)\b"
    r"|\bi\s+(?:want|need|['’]d\s+like|would\s+like)\s+to\s+(?:review|approve)\s+(?:it|them|that|this)\b",
    re.IGNORECASE,
)
# The same words declining it: "don't wait for me", "no need to review it".
_DECLINES = re.compile(
    r"\b(?:don['’]?t|do\s+not|no\s+need\s+to|doesn['’]?t\s+need\s+to|does\s+not\s+need\s+to|needn['’]?t)\s+"
    r"(?:\w+\s+){0,2}(?:wait|review|check|approve|see|look)\b",
    re.IGNORECASE,
)


def asks_to_see_it_first(text: Optional[str]) -> bool:
    """True when ``text`` asks for the work to wait for its owner before it is done."""
    words = str(text or "")
    return bool(_ASKS.search(words)) and not _DECLINES.search(words)


def owner_asked_to_review(messages: Iterable[Optional[str]]) -> bool:
    """True when any of the owner's ``messages`` (newest first) asks to see the work first."""
    return any(asks_to_see_it_first(text) for text in messages)


__all__ = ["asks_to_see_it_first", "owner_asked_to_review"]
