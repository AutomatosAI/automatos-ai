"""The saved conversation as the model reads it: a turn that did not answer is a marker, not a reply.

A turn that fails stays in the conversation as an assistant note, "⚠️ <plain sentence>"
with source origin ``turn_error`` (``consumers.chatbot.turn_notes``), so the person still
sees it after a reload. Handed to the model as it is, that note reads as Auto's own reply
and leaves the request before it looking open. On 23 Oct the person asked for agentfactory
research (the turn failed), said "Hi" three times (each failed), then "Hey" once the
provider worked, and Auto answered the agentfactory research.

Every turn's history is built here, from the saved rows: the chat route (and its detached
turn), the public widget, the voice call, and the thread checkpoint's distiller. The saved
rows and what the UI shows are unchanged.

Which row answers which request: every turn the person starts leaves one assistant row,
its reply or its note. A post a background producer made (a watcher, a scheduled task, a
mission's narration, a click's result: PRD-205, its ``source`` names its origin) answers
none. Counting the person's messages against those rows:

* a turn's note while a request waits for its answer is ``FAILED_TURN_MARKER``;
* a note when no request waits is the tail of a turn that had already answered (its reply
  was saved, then a step after the save failed): it is left out, so a saved answer is
  never followed by "did not come through";
* a reply the turn saved in place of an answer it never got (two blank completions, a
  blank after the tools ran: ``NO_ANSWER_SENTENCES``) is the marker too.

The marker never claims nothing happened: a turn can fail after its tools ran (the forced
synthesis, a later round), so it says to check before doing anything again (F185's rule).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from consumers.chatbot.empty_completion import EMPTY_COMPLETION_FALLBACK, NOTHING_SAID
from consumers.chatbot.turn_errors import TURN_ERROR_ORIGIN
from modules.voice.call_binding import VOICE_ORIGIN

USER = "user"
FAILED_TURN_MARKER = (
    "[Auto's reply to this message did not come through: the turn failed or was stopped. Anything it "
    "did before that may already be done, so check before doing it again. Answer the person's newest "
    "message; offer to retry this one only if it still matters.]"
)
# What a turn saves in place of an answer it did not get (empty_completion): no answer either.
NO_ANSWER_SENTENCES = (EMPTY_COMPLETION_FALLBACK, NOTHING_SAID)
# The origins of rows a turn writes itself, beside the chat's and the widget's (no source at all):
# a live-voice turn's, and a failed turn's note.
_TURN_ORIGINS = frozenset({VOICE_ORIGIN, TURN_ERROR_ORIGIN})

Row = Any
Entry = Dict[str, Any]


def _source(row: Row) -> Optional[Mapping[str, Any]]:
    """The row's provenance; None when it has none (an empty one included)."""
    source = getattr(row, "source", None)
    return source if isinstance(source, Mapping) and source else None


def is_failed_turn_note(source: Any) -> bool:
    """Whether a saved message's ``source`` marks it as the note a failed turn left."""
    return isinstance(source, Mapping) and source.get("origin") == TURN_ERROR_ORIGIN


def is_turn_reply(row: Row) -> bool:
    """Whether a saved assistant row is a turn's own (its reply or its note), never a post a
    background producer made."""
    if getattr(row, "role", None) == USER:
        return False
    source = _source(row)
    return source is None or source.get("origin") in _TURN_ORIGINS


def _text(parts: Any) -> str:
    return "\n".join(str(part.get("text") or "") for part in parts or ()
                     if isinstance(part, Mapping) and part.get("type") == "text")


def _no_answer(row: Row) -> bool:
    """A failed turn's note, or a reply saved in place of an answer the turn never got."""
    if is_failed_turn_note(_source(row)):
        return True
    said = _text(getattr(row, "parts", None))
    return any(sentence in said for sentence in NO_ANSWER_SENTENCES)


def _as_saved(row: Row) -> Entry:
    return {"role": row.role, "parts": row.parts}


def _marker(row: Row) -> Entry:
    return {"role": row.role, "parts": [{"type": "text", "text": FAILED_TURN_MARKER}]}


def _read(row: Row, waiting: int) -> Tuple[Optional[Entry], int]:
    """What the model reads for ``row`` (None: left out), and how many of the person's
    requests wait for an answer after it."""
    if row.role == USER:
        return _as_saved(row), waiting + 1
    if not is_turn_reply(row):
        return _as_saved(row), waiting
    if waiting == 0:
        return (None if is_failed_turn_note(_source(row)) else _as_saved(row)), 0
    return (_marker(row) if _no_answer(row) else _as_saved(row)), waiting - 1


def history_for_model(rows: Iterable[Row]) -> List[Entry]:
    """The saved messages (oldest first) as a turn's history, one new ``{role, parts}``
    dict each; the rows themselves are never changed."""
    history: List[Entry] = []
    waiting = 0
    for row in rows:
        entry, waiting = _read(row, waiting)
        if entry is not None:
            history.append(entry)
    return history


def answered_after(rows: Sequence[Row], message_id: Any) -> Optional[bool]:
    """Whether a turn's row (a reply or a note) follows the person's message ``message_id``;
    None when the message is not among ``rows``."""
    at = next((i for i, row in enumerate(rows) if getattr(row, "id", None) == message_id), None)
    if at is None:
        return None
    return any(is_turn_reply(row) for row in rows[at + 1:])


def rows_for_the_turn(rows: Sequence[Row], message_id: Any) -> List[Row]:
    """The saved rows as the turn that saved ``message_id`` reads them once it runs, after the
    turns queued before it ended: the person's later messages belong to later turns and are left
    out, and what was saved since its message (those turns' replies, a background post) comes
    before it, so its message is the newest. ``rows`` as they are when the message is not there."""
    at = next((i for i, row in enumerate(rows) if message_id is not None
               and getattr(row, "id", None) == message_id), None)
    if at is None:
        return list(rows)
    since = [row for row in rows[at + 1:] if row.role != USER]
    return [*rows[:at], *since, rows[at]]


__all__ = ["FAILED_TURN_MARKER", "NO_ANSWER_SENTENCES", "answered_after", "history_for_model",
           "is_failed_turn_note", "is_turn_reply", "rows_for_the_turn"]
