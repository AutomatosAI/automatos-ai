"""The saved conversation as the model reads it: a failed turn is a marker, not a reply.

A turn that fails at the AI provider stays in the conversation as an assistant note,
"⚠️ <plain sentence>" with source origin ``turn_error``
(``StreamingChatService._persist_turn_error``), so the person still sees it after a
reload. Handed to the model as it is, that note reads as Auto's own reply and leaves the
request before it looking open. On 23 Oct the person asked for agentfactory research
(the turn failed), said "Hi" three times (each failed), then "Hey" once the provider
worked, and Auto answered the agentfactory research.

Every turn's history is built here, from the saved rows: the chat route (and its
detached turn), the public widget and the voice call. In that history a failed turn's
note is ``FAILED_TURN_MARKER``. The saved row and what the UI shows are unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from consumers.chatbot.turn_errors import TURN_ERROR_ORIGIN

FAILED_TURN_MARKER = (
    "[This request was not answered: the turn failed. The person may have moved on — answer their "
    "newest message, and offer to retry this one only if it is still relevant.]"
)


def is_failed_turn_note(source: Any) -> bool:
    """Whether a saved message's ``source`` marks it as the note a failed turn left."""
    return isinstance(source, Mapping) and source.get("origin") == TURN_ERROR_ORIGIN


def _model_message(row: Any) -> Dict[str, Any]:
    """One saved message as ``{role, parts}``; a failed turn's note carries the marker."""
    if is_failed_turn_note(getattr(row, "source", None)):
        return {"role": row.role, "parts": [{"type": "text", "text": FAILED_TURN_MARKER}]}
    return {"role": row.role, "parts": row.parts}


def history_for_model(rows: Iterable[Any]) -> List[Dict[str, Any]]:
    """The saved messages (oldest first) as a turn's history, one new ``{role, parts}``
    dict each. A message without a ``turn_error`` source keeps its parts as saved; the
    rows themselves are never changed."""
    return [_model_message(row) for row in rows]


__all__ = ["FAILED_TURN_MARKER", "history_for_model", "is_failed_turn_note"]
