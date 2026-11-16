"""A product's server key's chat turn runs once per ``Idempotency-Key`` (PRD-274 Wave 1, §5.2).

``POST /api/agents/{id}/execute`` made a retried run safe; a retried chat turn ran again, a second model
call, and the first turn's outcome could not be read back (a key reads no conversations). Now a server
key's turn that sends ``Idempotency-Key`` (``api.chat_server_key.answers_a_repeated_key_once``) holds
one ``chat_turn_keys`` row for ``KEY_WINDOW_HOURS``:

* **The first request** inserts the row (running) before its turn starts; the unique index on (key,
  Idempotency-Key) lets one of two racing requests hold it. A request refused before its turn starts
  (a 422, a 403, a 409 for consent) deletes it again: nothing ran, so the key is free.
* **When the turn ends,** whatever the client did, the row is marked done with its answer's message
  id, or failed (``record_end``).
* **A repeat of the same request:** a done turn is answered with its saved answer, read from the
  message by its id and framed as the turn framed it (its chat id, its text, its finish), with
  ``Idempotent-Replayed: true``; nothing runs. A turn still running is a 409. A failed turn, a turn
  running longer than ``services.chat_turns.INFLIGHT_TTL_S`` (presumed dead, as the turn registry
  presumes it), and a done turn whose answer is no longer kept (its chat was deleted, its end user's
  erase has started) run again.
* **The same key on a different request** is a 409, as ``/execute`` answers it. The request is kept as
  its hash (``request_hash``); the row holds no words of the turn.
* **Past the window** the key runs as new: each keyed claim first deletes the key's rows older than it.

Every step commits on a session of its own; the callers run them off the event loop.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Iterable, Mapping, Optional, Tuple, Union

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.agent_execute import IDEMPOTENCY_HEADER, REPLAYED_HEADER
from core.models.chat_turn_keys import DONE, FAILED, RUNNING, ChatTurnKey
from services.chat_turns import INFLIGHT_TTL_S

logger = logging.getLogger(__name__)

KEY_WINDOW_HOURS = 24
# Left out of the request's hash: the conversation, which the server reads from its own store, and a
# voice turn's timing tag.
NOT_THE_TURN = frozenset({"messages", "turnRef"})
CLAIM_ATTEMPTS = 3
USER = "user"
DATA_STREAM_MEDIA_TYPE = "text/plain; charset=utf-8"  # the chat route's own (api.chat.stream_chat)
STILL_RUNNING = f"The turn this {IDEMPOTENCY_HEADER} started is still running: retry once it has answered."
USED_FOR_ANOTHER = f"This {IDEMPOTENCY_HEADER} was used for a different request."


@dataclass(frozen=True)
class Replay:
    """A done turn's answer: the chat it answered in, and its text parts."""

    chat_id: str
    texts: Tuple[str, ...]


def request_hash(request: Any) -> str:
    """A sha256 of what defines the turn: the request's body without ``NOT_THE_TURN``. Its message, agent,
    model, app context, end user, chat id and locale are all in it, so a change to any one of them is a
    different request."""
    fields = request.model_dump(mode="json", exclude=set(NOT_THE_TURN))
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def claim(db: Session, ctx: Any, key: str, wanted: str) -> Union[Replay, uuid.UUID]:
    """The turn ``key`` names for the server key ``ctx`` carries: a ``Replay`` of its saved answer, or the
    id of the row this request now holds, when it is to run the turn. 409 while another request runs it,
    or when ``key`` was used for a request whose hash is not ``wanted``. Commits."""
    workspace_id, key_id = ctx.workspace_id, _as_uuid(ctx.api_key_id)
    forget_expired(db, workspace_id, key_id)
    for _ in range(CLAIM_ATTEMPTS):
        earlier = _row(db, workspace_id, key_id, key)
        replay = _decided(db, earlier, wanted) if earlier is not None else None
        if replay is not None:
            return replay
        mine = ChatTurnKey(id=uuid.uuid4(), workspace_id=workspace_id, api_key_id=key_id, idempotency_key=key,
                           request_hash=wanted, state=RUNNING)
        held = _hold(db, mine, earlier)
        if held is not None:
            return held
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=STILL_RUNNING)  # every attempt lost a race


def forget_expired(db: Session, workspace_id: Any, key_id: Any) -> int:
    """Deletes the key's rows made longer ago than the window, whichever Idempotency-Key they hold: the
    cleanup, on use, scoped to the key. How many were deleted. Commits."""
    before = _now() - timedelta(hours=KEY_WINDOW_HOURS)
    gone = (db.query(ChatTurnKey)
            .filter(ChatTurnKey.workspace_id == workspace_id, ChatTurnKey.api_key_id == key_id,
                    ChatTurnKey.created_at < before)
            .delete(synchronize_session=False))
    db.commit()
    return gone


def _row(db: Session, workspace_id: Any, key_id: uuid.UUID, key: str) -> Optional[ChatTurnKey]:
    return (db.query(ChatTurnKey)
            .filter(ChatTurnKey.workspace_id == workspace_id, ChatTurnKey.api_key_id == key_id,
                    ChatTurnKey.idempotency_key == key)
            .first())


def _decided(db: Session, earlier: ChatTurnKey, wanted: str) -> Optional[Replay]:
    """What an earlier request's row answers this one: its saved answer, a 409, or None when this request
    runs the turn again (the earlier one failed or died, or its answer is no longer kept)."""
    if earlier.request_hash != wanted:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=USED_FOR_ANOTHER)
    if earlier.state == RUNNING and not _presumed_dead(earlier):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=STILL_RUNNING)
    return saved_answer(db, earlier) if earlier.state == DONE else None


def _presumed_dead(row: ChatTurnKey) -> bool:
    """Running longer than the turn registry's bound: its process died, or its response was never read,
    so nothing will mark how it ended."""
    return row.created_at < _now() - timedelta(seconds=INFLIGHT_TTL_S)


def _hold(db: Session, mine: ChatTurnKey, earlier: Optional[ChatTurnKey]) -> Optional[uuid.UUID]:
    """Inserts ``mine``, after deleting ``earlier`` only if it is still as it was read: the unique index is
    the one arbiter, so of two requests taking over one row, one runs. None (rolled back) when another
    request changed the row or inserted first; the caller reads again."""
    if earlier is not None and not _deleted_as_read(db, earlier):
        db.rollback()
        return None
    row_id = mine.id
    db.add(mine)
    try:
        db.commit()
    except IntegrityError:  # a request with the same key holds it first
        db.rollback()
        return None
    return row_id


def _deleted_as_read(db: Session, earlier: ChatTurnKey) -> bool:
    gone = (db.query(ChatTurnKey)
            .filter(ChatTurnKey.id == earlier.id, ChatTurnKey.state == earlier.state,
                    ChatTurnKey.created_at == earlier.created_at)
            .delete(synchronize_session=False))
    return gone == 1


def saved_answer(db: Session, row: Any) -> Optional[Replay]:
    """A done turn's answer, read from its saved message by id in the row's workspace. None when it is no
    longer kept (its chat was deleted) or is an end user's whose consent no longer counts (their erase has
    started): the turn then runs as new, and the route's own checks answer it."""
    from core.models import Chat, Message
    from core.models.end_users import consented

    if row.answer_message_id is None or row.chat_id is None:
        return None
    found = (db.query(Message.parts, Chat.end_user_id)
             .join(Chat, Chat.id == Message.chat_id)
             .filter(Message.id == row.answer_message_id, Message.chat_id == row.chat_id,
                     Chat.workspace_id == row.workspace_id)
             .first())
    if found is None:
        return None
    parts, end_user_id = found
    if end_user_id and not consented(db, row.workspace_id, end_user_id):
        return None
    return Replay(chat_id=str(row.chat_id), texts=answer_texts(parts))


def answer_texts(parts: Any) -> Tuple[str, ...]:
    """A saved reply's answer: its ``text`` parts, in order. Its receipts, reasoning and narration are
    parts of other types (``consumers.chatbot.narration.reply_parts``)."""
    return tuple(str(part.get("text") or "") for part in parts or ()
                 if isinstance(part, Mapping) and part.get("type") == "text")


def release(db: Session, row_id: Any) -> int:
    """Deletes the row of a request refused before its turn started: nothing ran, so a corrected request
    may use the key. Commits."""
    gone = (db.query(ChatTurnKey)
            .filter(ChatTurnKey.id == row_id, ChatTurnKey.state == RUNNING)
            .delete(synchronize_session=False))
    db.commit()
    return gone


def record_end(db: Session, row_id: Any, chat_id: str, message_id: Any, completed: bool) -> str:
    """Marks the turn done with its answer's message id, or failed: it did not complete (Stop, a failure),
    or it completed with no answer saved after the person's message (a failed turn's note). Commits; the
    state written."""
    answer = answer_after(db, chat_id, message_id) if completed else None
    state = DONE if answer is not None else FAILED
    (db.query(ChatTurnKey)
     .filter(ChatTurnKey.id == row_id, ChatTurnKey.state == RUNNING)
     .update({"state": state, "chat_id": _as_uuid(chat_id), "answer_message_id": answer},
             synchronize_session=False))
    db.commit()
    logger.info("[chat_turn_keys] keyed turn %s ended %s", row_id, state)
    return state


def answer_after(db: Session, chat_id: str, message_id: Any) -> Optional[Any]:
    """The id of the turn's answer: the newest row a turn saved after the person's message ``message_id``,
    unless it is a failed turn's note. The session queue runs one turn of a chat at a time, so when this
    turn ends its own reply is the newest; a post a background producer made answers no turn and is
    passed over (``consumers.chatbot.model_history.is_turn_reply``)."""
    from consumers.chatbot.model_history import is_failed_turn_note, is_turn_reply
    from core.models import Message

    chat_uuid = _as_uuid(chat_id)
    asked = db.query(Message.created_at).filter(Message.id == message_id, Message.chat_id == chat_uuid).scalar()
    if asked is None:
        return None
    rows = (db.query(Message.id, Message.role, Message.source)
            .filter(Message.chat_id == chat_uuid, Message.created_at > asked, Message.role != USER)
            .order_by(Message.created_at.desc())
            .all())
    newest = next((row for row in rows if is_turn_reply(row)), None)
    if newest is None or is_failed_turn_note(newest.source):
        return None
    return newest.id


def replayed(replay: Replay) -> StreamingResponse:
    """The first turn's answer, framed as the turn framed it: the chat-id frame with the chat it answered
    in, its text frames, then the finish frame, with ``Idempotent-Replayed: true``. Nothing runs."""
    from api.chat_dispatch import response_headers
    from consumers.chatbot.streaming import get_streaming_handler

    handler = get_streaming_handler()
    frames = (handler.format_aisdk_chat_id(replay.chat_id), *map(handler.format_aisdk_text, replay.texts),
              handler.format_aisdk_finish())
    return StreamingResponse(_streamed(frames), media_type=DATA_STREAM_MEDIA_TYPE,
                             headers={**response_headers(None), REPLAYED_HEADER: "true"})


async def _streamed(frames: Iterable[str]) -> AsyncIterator[str]:
    for frame in frames:
        yield frame


def _as_uuid(value: Any) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _now() -> datetime:
    return datetime.now(timezone.utc)
