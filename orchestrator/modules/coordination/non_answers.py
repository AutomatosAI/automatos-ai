"""F286 (night 8): an answer that is not the work fails the mission's check.

Mission #0400 needed the owner's Gmail, which is not connected. Its first step's whole
answer was "Now let me inject my findings into the mission field and provide the
complete list with draft replies:", and its second's "I need the list of club
members… from "task_1"… This information is still missing." Both passed, the mission
"completed" with that sentence as its result, and nothing said Gmail isn't connected.
#0383.3 answered "I cannot locate the specific approved content from cards #0383.1
and #0383.2" and, once, "Now let me look for the specific "Front page line: Rwanda
Nyamasheke" task:".

Two kinds of answer are not the work (``not_the_work``). Each fails the check quoting
its sentence: the step is asked once more for the work, and if its answer is still
not the work it fails, and its mission says why (F283):
- a note the agent left itself: a short answer that ends with a colon, or with an
  ellipsis after "I", "me" or "my", or that starts "Now let me", "Let me" or "I'll now";
- an answer that says it could not do the work: the information is missing, it needs
  the list, it has no access, something is not connected, it cannot locate or access
  what it was meant to use, or it cannot find the file, sheet, task or data. "I
  couldn't find any pause requests" is a finding, not a failure, and a drafted reply's
  "I couldn't find your order" is the draft's own words: neither fails. Nor does a
  line that ends with an ellipsis for effect ("Bright, sweet and floral…").
"""
from __future__ import annotations

import re
from typing import Optional

# How a step's answer that is not the work is told so (mission verification).
A_NOTE = "Its answer is a note, not the work: "
CANNOT = "Its answer says it could not do the work: "
NOT_THE_WORK = (A_NOTE, CANNOT)
# A note the agent left itself is short: #0400.1's whole answer was 18 words.
NOTE_MAX_WORDS = 30
QUOTE_MAX_CHARS = 200
ELLIPSES = ("…", "...")

_NOTE_START = re.compile(r"^(?:now\s+let\s+me|let\s+me|i'?ll\s+now|i\s+will\s+now)\b", re.IGNORECASE)
_FIRST_PERSON = re.compile(r"\b(?:i|me|my)\b", re.IGNORECASE)
_CANNOT = re.compile(
    r"\binformation\s+(?:is|was)\s+(?:still\s+)?missing\b"
    r"|\bI\s+(?:still\s+)?need\s+the\s+(?:full\s+|exact\s+|complete\s+)?list\s+of\b"
    r"|\bI\s+(?:do\s+not|don't|did\s+not|didn't)\s+have\s+access\s+to\b"
    r"|\b(?:is|are|was)\s+not\s+connected\b|\b(?:isn't|aren't|wasn't)\s+connected\b"
    r"|\bI\s+(?:cannot|can't|could\s+not|couldn't|am\s+unable\s+to|was\s+unable\s+to)\s+(?:\w+ly\s+)?(?:locate|access)\b"
    r"|\b(?:cannot|can't|could\s+not|couldn't)\s+(?:\w+ly\s+)?find\s+(?:the|this|that|these|those|its|my)\s+"
    r"(?:[\w'-]+\s+){0,3}?(?:file|document|sheet|table|list|data|information|content|details|task|card|step|mission"
    r"|results?)\b"
    r"|\bnot\s+found\s+in\s+(?:this|the|your)\s+workspace\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")


def not_the_work(output: object) -> Optional[str]:
    """Why ``output`` is not the work, quoting it, or None when it may be the work."""
    text = str(output or "").strip()
    if not text:
        return None
    if _is_a_note(text):
        return A_NOTE + _quoted(text)
    said = _CANNOT.search(text)
    return CANNOT + _quoted(_sentence_at(text, said.start())) if said else None


def _is_a_note(text: str) -> bool:
    if len(text.split()) > NOTE_MAX_WORDS:
        return False
    if text.endswith(":") or _NOTE_START.match(text):
        return True
    return text.endswith(ELLIPSES) and _FIRST_PERSON.search(text) is not None


def _sentence_at(text: str, at: int) -> str:
    """The sentence of ``text`` that holds position ``at``."""
    start = 0
    for boundary in _SENTENCE_END.finditer(text):
        if boundary.start() >= at:
            return text[start:boundary.start()].strip()
        start = boundary.end()
    return text[start:].strip()


def _quoted(sentence: str) -> str:
    cut = sentence if len(sentence) <= QUOTE_MAX_CHARS else sentence[:QUOTE_MAX_CHARS].rstrip() + "…"
    return f'"{cut}"'


__all__ = ["A_NOTE", "CANNOT", "NOT_THE_WORK", "not_the_work"]
