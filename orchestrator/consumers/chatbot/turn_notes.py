"""The note a turn that did not answer leaves in the conversation (PRD-239 S4).

A failed turn says so on the stream and keeps a note in the conversation: an assistant
message "⚠️ <sentence>" with source origin ``turn_error``. The client shows it as the
failure after a reload (frontend lib/chat/message-text.ts, ``savedTurnError``), and the
model's history reads it as a marker, never as Auto's reply (``model_history``).

* ``save_turn_note``: the streaming service saves it when its turn fails
  (``StreamingChatService._persist_turn_error``), and the chat route for the session-agent
  lane. A failure that left the session's transaction aborted (a refused query nobody rolled
  back) made the save fail too, and the note was lost with a debug line: the session is now
  rolled back and the note saved once more, and a note that still does not save is logged.
* ``note_an_unanswered_turn``: a turn that ended before it saved anything (Stop, a failure
  outside the streaming service: the producer, the session queue) leaves its note after the
  turn, in a session of its own, unless a reply or a note already follows the person's
  message (Stop pressed while the answered turn was still storing its memory).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from consumers.chatbot.model_history import answered_after
from consumers.chatbot.turn_errors import STOPPED, TURN_ERROR_ORIGIN, UNFINISHED, TurnError

logger = logging.getLogger(__name__)

WARNING_MARK = "⚠️"
NOTE_LABEL = "Error"
ASSISTANT = "assistant"


def note_parts(err: TurnError) -> List[Dict[str, Any]]:
    """The note's parts: the failure's sentence after the warning mark."""
    return [{"type": "text", "text": f"{WARNING_MARK} {err.message}"}]


def note_source(err: TurnError) -> Dict[str, Any]:
    """The note's provenance: what the client and the model's history read it by."""
    return {"origin": TURN_ERROR_ORIGIN, "label": NOTE_LABEL, "code": err.code}


def _save(chat_service: Any, chat_id: str, workspace_id: Any, err: TurnError) -> None:
    chat_service.save_message(chat_id=chat_id, role=ASSISTANT, parts=note_parts(err),
                              workspace_id=workspace_id, source=note_source(err))


def save_turn_note(chat_service: Any, chat_id: str, workspace_id: Any, err: TurnError) -> bool:
    """Keep ``err`` in the conversation as the turn's note. False (logged with its trace)
    when it could not be saved, even after rolling the session back."""
    try:
        _save(chat_service, chat_id, workspace_id, err)
        return True
    except Exception:  # noqa: BLE001 — the turn's failure may have left the transaction aborted: roll back, retry
        logger.warning("[Chat] the turn error note did not save in chat %s; rolling back to save it again",
                       chat_id, exc_info=True)
    try:
        chat_service.db.rollback()
        _save(chat_service, chat_id, workspace_id, err)
        return True
    except Exception:  # noqa: BLE001 — the note is lost: logged with its trace, and the caller is told
        logger.exception("[Chat] could not persist the turn error note in chat %s", chat_id)
        return False


def _note_if_unanswered(workspace_id: Any, chat_id: str, message_id: Any, cancelled: bool) -> bool:
    from consumers.chatbot.service import ChatService
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        chat_service = ChatService(db)
        if answered_after(chat_service.get_messages_by_chat_id(chat_id), message_id) is not False:
            return False
        return save_turn_note(chat_service, chat_id, workspace_id, STOPPED if cancelled else UNFINISHED)
    except Exception:  # noqa: BLE001 — after the turn: logged with its trace, never raised into the turn's end
        logger.exception("[Chat] could not note the unfinished turn in chat %s", chat_id)
        return False
    finally:
        db.close()


async def note_an_unanswered_turn(workspace_id: Any, chat_id: str, message_id: Any, *, cancelled: bool) -> bool:
    """After a turn that did not complete: its note (stopped, or did not finish), unless a reply
    or a note already follows the person's message ``message_id``. True when one was saved.
    The database work runs off the event loop."""
    return await asyncio.to_thread(_note_if_unanswered, workspace_id, chat_id, message_id, cancelled)


__all__ = ["note_an_unanswered_turn", "note_parts", "note_source", "save_turn_note"]
