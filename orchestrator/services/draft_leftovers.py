"""F327 (night 9b): a card's answer that still holds a placeholder, or a line about a
tool or a skill, says so.

The Ops agent's reorder email to the importer was signed "[Your name]" through three
rounds (#1971; earlier #0095), though the sign-off is in the owner's brand voice paper,
and the Club newsletter helper's draft opened "It appears the 'Harbourline voice' skill
is not available" (#0107). The owner would have sent both on as written.

A mission step with a placeholder already fails its check (F248,
``core.services.placeholders``); a plain board card had no check. ``left_in_note`` is
the cheap one, beside F199's and F304's (``services.pasted_data.unverified_figures_note``,
``services.answer_sources``): a card's answer that still holds a template placeholder,
or says a skill, plugin or toolkit is not available or that it saved to the
scratchpad, gets a line naming what was left in. Like theirs, it is a line for the
owner, not a verdict: the card ends as it would have.
"""
from __future__ import annotations

import functools
import re
from typing import Callable, Iterable, Optional

PLACEHOLDERS_NOTE = "\n\nCheck before using this answer: it still has placeholders where its content belongs: {slots}."
PLUMBING_NOTE = "\n\nCheck before using this answer: it has a line about a tool or a skill, not the work: \"{said}\"."
SLOTS_SHOWN = 6
QUOTE_MAX_CHARS = 160

_STATUS = (r"(?:(?:is|are|was|were)(?:\s+not|n't)\s+(?:currently\s+)?(?:available|installed|loaded|enabled|found)"
           r"|not\s+(?:available|installed|loaded|enabled|found)|unavailable)")
_PLUMBING = re.compile(rf"\b(?:skill|plugin|toolkit)s?\b[^.!?\n]{{0,40}}?\b{_STATUS}\b"
                       r"|\bsaved\s+(?:it\s+|this\s+)?to\s+(?:the\s+|my\s+|your\s+)?scratchpad\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentence_at(text: str, at: int) -> str:
    """The sentence of ``text`` that holds position ``at``, cut to QUOTE_MAX_CHARS."""
    start = 0
    for boundary in _SENTENCE_END.finditer(text):
        if boundary.start() >= at:
            return _cut(text[start:boundary.start()])
        start = boundary.end()
    return _cut(text[start:])


def _cut(sentence: str) -> str:
    sentence = sentence.strip()
    return sentence if len(sentence) <= QUOTE_MAX_CHARS else sentence[:QUOTE_MAX_CHARS].rstrip() + "…"


def left_in_note(result: object) -> Optional[str]:
    """The owner's lines for an answer with placeholders or a tool or skill's status left
    in it, else None."""
    from core.services.placeholders import template_placeholders

    text = str(result or "")
    notes = []
    slots = template_placeholders(text)
    if slots:
        notes.append(PLACEHOLDERS_NOTE.format(slots=", ".join(slots[:SLOTS_SHOWN])))
    said = _PLUMBING.search(text)
    if said:
        notes.append(PLUMBING_NOTE.format(said=_sentence_at(text, said.start())))
    return "".join(notes) or None


def an_answer_leaves_nothing_in(
        check: Callable[[object, Iterable[str]], Optional[str]]) -> Callable[[object, Iterable[str]], Optional[str]]:
    """Wrap a board card's answer check that takes the answer and the names of the
    actions that worked in its run (F199's): its note, then F327's."""
    @functools.wraps(check)
    def wrapped(result: object, succeeded: Iterable[str]) -> Optional[str]:
        notes = [check(result, succeeded), left_in_note(result)]
        return "".join(note for note in notes if note) or None
    return wrapped


__all__ = ["PLACEHOLDERS_NOTE", "PLUMBING_NOTE", "an_answer_leaves_nothing_in", "left_in_note"]
