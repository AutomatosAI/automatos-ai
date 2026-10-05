"""F334 (night 10): a card whose own answer says the work isn't done is not done.

With review_mode ``auto`` the owner does not see every card, so a card's answer is the
only account of the run. Night 10's #2084 opened "I couldn't make the invoice you
asked for." and said "I won't call it done.", and #2093 opened "Invoice HL-2026-0142
isn't done." Both closed ``done``. An answer that says, in so many words, that the
agent could not do the work or will not call it done now goes to review with that
sentence quoted (``finalize_board_task_run`` reads it through
``services.result_substance.nothing_done_note``).

It stays narrow, because a negative sentence about the business is not a verdict on
the work ("the delivery isn't done on Sundays"):

* anywhere in the answer: the agent declines to call the work done ("I won't call it
  done", "I can't mark this complete"), or says the card, ticket, task or work is not
  done, with the clause ending there ("So this card isn't done.");
* the answer's opening sentence, where agents give their verdict: it is "<the
  deliverable> isn't done." and nothing more, starts "Not done", or starts "I
  couldn't make / produce / create … the …".

"The card isn't done until you approve it" names a step after the work and does not
trip it.
"""
from __future__ import annotations

import re
from typing import Optional

from core.services.ticket_reasons import SAYS_NOT_DONE_NOTE_PREFIX

SAYS_NOT_DONE_NOTE = SAYS_NOT_DONE_NOTE_PREFIX + ' "{words}" Sent to review instead of done.'
QUOTED_CHARS = 200
OPENING_SUBJECT_CHARS = 120

_DONE = r"(?:done|finished|complete)"
_CLAUSE_ENDS = r"(?=\s*(?:[.!;:,)]|\u2014|\u2013|$))"
_WORK = r"(?:this card|this ticket|this task|the card|the ticket|the task|the work|the job)"
_DECLINES = re.compile(
    r"\bI (?:can't|cannot|can not|couldn't|could not|won't|will not|wouldn't|would not) "
    rf"(?:call|mark|count|consider|report) (?:it|this|{_WORK}) (?:as )?{_DONE}\b", re.IGNORECASE)
_WORK_NOT_DONE = re.compile(
    rf"\b{_WORK} (?:isn't|is not|wasn't|was not|is still not|still isn't) (?:yet )?{_DONE}(?: yet)?{_CLAUSE_ENDS}",
    re.IGNORECASE | re.MULTILINE)
_OPENS_NOT_DONE = re.compile(
    rf"^(?:(?P<subject>[^.!?]{{1,{OPENING_SUBJECT_CHARS}}}) (?:isn't|is not|is still not|still isn't) {_DONE}[.!]?"
    rf"|not (?:yet )?{_DONE}{_CLAUSE_ENDS}.*)$", re.IGNORECASE)
_OPENS_COULD_NOT = re.compile(
    r"^(?:(?:sorry|unfortunately),?\s+)?I (?:couldn't|could not|can't|cannot|wasn't able to|was not able to|"
    r"was unable to|am unable to|haven't been able to|have not been able to|failed to) "
    r"(?:make|produce|create|generate|build|write|draft|prepare|finish|complete|do|deliver) "
    r"(?:the|a|an|your|this|that|it|these|those|any)\b", re.IGNORECASE)
# Someone else's words, not the agent's verdict: "You said this card isn't done, so I redid it."
_REPORTED = re.compile(r"\b(?:said|says|say|told|wrote|writes|noted|mentioned|reported|flagged)\b", re.IGNORECASE)
_MARKUP = re.compile(r"[*`]")
_LINE_LEAD = re.compile(r"^[#>\-\s]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
_SENTENCE_START = re.compile(r"[.!?\n]")


def _plain(answer: str) -> str:
    """The answer with curly apostrophes made straight and bold or code marks taken off."""
    return _MARKUP.sub("", (answer or "").replace("\u2019", "'"))


def _opening_sentence(text: str) -> str:
    """The first sentence of the answer's first non-empty line, list or heading marks off."""
    line = next((ln for ln in text.splitlines() if ln.strip()), "")
    return _SENTENCE_END.split(_LINE_LEAD.sub("", line), maxsplit=1)[0].strip()


def _opening_verdict(opening: str) -> bool:
    """The opening sentence is the agent's own "not done" or "I couldn't make …"."""
    if _OPENS_COULD_NOT.match(opening):
        return True
    found = _OPENS_NOT_DONE.match(opening)
    return bool(found) and not _REPORTED.search(found.group("subject") or "")


def _own_words(text: str, start: int) -> bool:
    """The sentence before ``start`` reports no one else's words."""
    boundary = max((m.end() for m in _SENTENCE_START.finditer(text, 0, start)), default=0)
    return not _REPORTED.search(text[boundary:start])


def said_not_done_note(answer: str) -> Optional[str]:
    """The review note for an answer that says its work isn't done, quoting where it
    says so; None for any other answer."""
    text = _plain(answer)
    opening = _opening_sentence(text)
    if opening and _opening_verdict(opening):
        return SAYS_NOT_DONE_NOTE.format(words=opening[:QUOTED_CHARS])
    for pattern in (_DECLINES, _WORK_NOT_DONE):
        found = next((m for m in pattern.finditer(text) if _own_words(text, m.start())), None)
        if found is not None:
            return SAYS_NOT_DONE_NOTE.format(words=found.group(0)[:QUOTED_CHARS])
    return None


__all__ = ["SAYS_NOT_DONE_NOTE", "said_not_done_note"]
